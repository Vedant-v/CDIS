"""
domain_decay.py
---------------
Manages per-domain temporal decay rates (lambda) with:
  - In-process TTL cache (avoids per-request DB round-trips)
  - Distributed Redis lock for epoch-consistent lambda auto-tuning
  - Volatility event recording when shock magnitude crosses threshold
"""

import time
import os
from datetime import date, timedelta
from uuid import UUID

import redis
from celery import shared_task

from app.db.cassandra import get_session

# ---------------------------------------------------------------------------
# CQL Statements
# ---------------------------------------------------------------------------

_FETCH_LAMBDA = """
SELECT lambda AS lam FROM domain_decay_config
WHERE domain = %s
"""

_UPDATE_LAMBDA = """
UPDATE domain_decay_config
SET lambda = %s
WHERE domain = %s
"""

_INSERT_VOLATILITY_EVENT = """
INSERT INTO domain_volatility_events
    (domain, detection_week, attribution_start, attribution_end)
VALUES (%s, %s, %s, %s)
"""

# ---------------------------------------------------------------------------
# In-Process Cache
# ---------------------------------------------------------------------------

_CACHE_TTL: float = 300.0          # seconds
_cache: dict = {}                  # { domain: (lambda_value, expiry_timestamp) }

# ---------------------------------------------------------------------------
# Redis Client (lazy — only instantiated when the task path is hit)
# ---------------------------------------------------------------------------

_redis_client = None

def _get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "redis"), port=6379, db=0
        )
    return _redis_client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_domain_lambda(domain: str) -> float:
    """
    Return the decay rate λ for *domain*.

    Source priority:
      1. In-process cache (TTL = 5 min)
      2. Cassandra domain_decay_config
      3. Hardcoded default (0.08) if no row exists

    λ is clamped to [0.001, 1.0] at both read and write paths to
    prevent edge-case vulnerabilities from extreme values.
    """
    now = time.monotonic()

    if domain in _cache:
        val, expiry = _cache[domain]
        if now < expiry:
            return val

    row = get_session().execute(_FETCH_LAMBDA, (domain,)).one()
    val = float(row.lam) if row else 0.08
    val = max(0.001, min(val, 1.0))

    _cache[domain] = (val, now + _CACHE_TTL)
    return val


def _invalidate_cache(domain: str) -> None:
    _cache.pop(domain, None)


# ---------------------------------------------------------------------------
# Celery Task — Phase 4F Lambda Auto-Tuner
# ---------------------------------------------------------------------------

@shared_task(name="app.services.domain_decay.auto_tune_lambda_task", expires=432_000)
def auto_tune_lambda_task(domain: str, epoch_str: str) -> None:
    """
    Epoch-consistent lambda update.

    Triggered exclusively by DOMAIN_PROJECTION_COMMITTED events.
    Runs as a logical singleton via a distributed Redis lock; any
    concurrent invocation for the same domain either waits (up to
    5 s) or exits silently.

    Algorithm:
      1. Monotonic epoch guard — skip if we already tuned a newer epoch.
      2. Pull ST and LT fusion projections for this epoch.
      3. Fetch ground-truth outcomes for each claim (concurrent CQL).
      4. Compute ST and LT confident-prediction error rates.
      5. shock = (ST_error - LT_error) × volume_weight
      6. EMA update: λ_new = λ_old + η·shock - γ·(λ_old - λ_base)
      7. If shock is severe → record a domain_volatility_event.
    """
    from cassandra.concurrent import execute_concurrent

    redis_client = _get_redis()
    lock_key     = f"lambda_tune_lock:{domain}"

    try:
        with redis_client.lock(lock_key, timeout=120, blocking_timeout=5):
            session   = get_session()
            db_epoch  = UUID(epoch_str)

            # ---- 1. Monotonic epoch guard --------------------------------
            last_str = redis_client.get(f"last_tuned_epoch:{domain}")
            if last_str:
                last_epoch = UUID(last_str.decode())
                if last_epoch.time >= db_epoch.time:
                    return   # Already processed a newer/equal epoch

            # Invalidate cache BEFORE re-reading to prevent EMA skew
            _invalidate_cache(domain)

            # Record this epoch as processed
            redis_client.set(f"last_tuned_epoch:{domain}", str(db_epoch))

            # ---- 2. Fetch ST/LT projections for this epoch ---------------
            proj_rows = list(session.execute(
                """
                SELECT claim_id, st_alpha, st_beta, lt_alpha, lt_beta
                FROM claim_fusion_projection_by_epoch
                WHERE domain = %s AND projection_epoch = %s
                """,
                (domain, db_epoch),
            ))

            if not proj_rows:
                return

            # ---- 3. Bulk-fetch outcomes ----------------------------------
            claim_ids    = [r.claim_id for r in proj_rows]
            outcome_stmt = session.prepare(
                "SELECT claim_id, outcome FROM claim_outcomes WHERE claim_id = ?"
            )
            results = execute_concurrent(
                session,
                [(outcome_stmt, (cid,)) for cid in claim_ids],
                concurrency=100,
            )

            outcomes: dict = {}
            for success, result in results:
                if success and result:
                    row = result[0]
                    outcomes[row.claim_id] = row.outcome

            # ---- 4. Score ST vs LT error rates --------------------------
            st_conf_count = st_conf_wrong = 0
            lt_conf_count = lt_conf_wrong = 0

            for r in proj_rows:
                outcome = outcomes.get(r.claim_id)
                if outcome is None:
                    continue

                st_sum = (r.st_alpha or 0.0) + (r.st_beta or 0.0)
                if st_sum > 0:
                    st_p = r.st_alpha / st_sum
                    if st_p > 0.8 or st_p < 0.2:
                        st_conf_count += 1
                        if (1.0 if st_p > 0.5 else 0.0) != outcome:
                            st_conf_wrong += 1

                lt_sum = (r.lt_alpha or 0.0) + (r.lt_beta or 0.0)
                if lt_sum > 0:
                    lt_p = r.lt_alpha / lt_sum
                    if lt_p > 0.8 or lt_p < 0.2:
                        lt_conf_count += 1
                        if (1.0 if lt_p > 0.5 else 0.0) != outcome:
                            lt_conf_wrong += 1

            # Minimum sample sizes for statistical reliability
            if st_conf_count < 5 or lt_conf_count < 10:
                return

            # ---- 5. Compute shock ----------------------------------------
            st_error_rate = st_conf_wrong / st_conf_count
            lt_error_rate = lt_conf_wrong / lt_conf_count

            raw_shock     = st_error_rate - lt_error_rate
            volume_weight = st_conf_count / (st_conf_count + lt_conf_count)
            shock         = raw_shock * volume_weight

            # ---- 6. EMA update ------------------------------------------
            current_lam = get_domain_lambda(domain)
            eta         = 0.05    # learning rate
            gamma       = 0.01    # mean-reversion force
            lambda_base = 0.08    # long-run neutral lambda

            new_lam = current_lam + (eta * shock) - (gamma * (current_lam - lambda_base))
            new_lam = max(0.001, min(new_lam, 1.0))

            session.execute(_UPDATE_LAMBDA, (new_lam, domain))
            _invalidate_cache(domain)

            # ---- 7. Volatility event recording ---------------------------
            if st_error_rate > 0.40 and shock > 0.15:
                today      = date.today()
                attr_start = today - timedelta(weeks=2)
                attr_end   = today - timedelta(weeks=1)
                session.execute(_INSERT_VOLATILITY_EVENT, (domain, today, attr_start, attr_end))

    except redis.exceptions.LockNotOwnedError:
        # Another worker already holds the lock — skip gracefully
        pass
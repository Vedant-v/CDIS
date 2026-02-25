import time
from app.db.cassandra import get_session
from datetime import date, timedelta

FETCH_LAMBDA = """
SELECT lambda AS lam FROM domain_decay_config
WHERE domain = %s
"""

UPDATE_LAMBDA = """
UPDATE domain_decay_config
SET lambda = %s
WHERE domain = %s
"""

from celery import shared_task
import redis
import os
from uuid import UUID

redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "redis"), port=6379, db=0)

FETCH_LAMBDA = """
SELECT lambda AS lam FROM domain_decay_config
WHERE domain = %s
"""

UPDATE_LAMBDA = """
UPDATE domain_decay_config
SET lambda = %s
WHERE domain = %s
"""

# Phase 4F: Bounded event lookups 
INSERT_VOLATILITY_EVENT = """
INSERT INTO domain_volatility_events (domain, detection_week, attribution_start, attribution_end)
VALUES (%s, %s, %s, %s)
"""

_cache = {}
CACHE_TTL = 300  # 5 minutes cache for domain decay configs

def get_domain_lambda(domain: str) -> float:
    now = time.time()
    
    # Return from cache if valid
    if domain in _cache:
        val, expiry = _cache[domain]
        if now < expiry:
            return val

    # Fetch from Cassandra
    row = get_session().execute(FETCH_LAMBDA, (domain,)).one()
    val = row.lam if row else 0.08
    val = max(0.001, min(val, 1.0)) # Clamp lambda to prevent edge case vulnerabilities
    
    # Store in cache
    _cache[domain] = (val, now + CACHE_TTL)
    return val

@shared_task(name="app.services.domain_decay.auto_tune_lambda_task", expires=432000)
def auto_tune_lambda_task(domain: str, epoch_str: str):
    """
    Phase 4F: Epoch-Consistent Lambda Update Window
    Runs as a singleton strictly triggered by DOMAIN_PROJECTION_COMMITTED.
    """
    lock_key = f"lambda_tune_lock:{domain}"
    from cassandra.concurrent import execute_concurrent
    
    # 1. Acquire Distributed Lock
    with redis_client.lock(lock_key, timeout=120, blocking_timeout=5):
        session = get_session()
        
        # 1. Monotonic Epoch Verification
        db_row = session.execute("SELECT projection_epoch FROM domain_projection_epoch WHERE domain = %s", (domain,)).one()
        if not db_row:
            return
        db_epoch = db_row.projection_epoch
        
        last_tuned_str = redis_client.get(f"last_tuned_epoch:{domain}")
        if last_tuned_str:
            last_tuned_epoch = UUID(last_tuned_str.decode())
            if last_tuned_epoch.time >= db_epoch.time:
                # Abort: this epoch is older than or equal to the last tuned epoch
                return
                
        # Invalidate Cache BEFORE fetching to prevent EMA skew
        if domain in _cache:
            del _cache[domain]
            
        # Mark as tuned
        redis_client.set(f"last_tuned_epoch:{domain}", str(db_epoch))

        # 2. Projection-Synchronous Shock Evaluation
        FETCH_PROJ = """
        SELECT claim_id, st_alpha, st_beta, lt_alpha, lt_beta
        FROM claim_fusion_projection_by_epoch
        WHERE domain = %s AND projection_epoch = %s
        """
        proj_rows = list(session.execute(FETCH_PROJ, (domain, db_epoch)))
        
        if not proj_rows:
            return

        # Fetch outcomes in bulk concurrently
        claim_ids = [r.claim_id for r in proj_rows]
        
        outcome_stmt = session.prepare("SELECT claim_id, outcome FROM claim_outcomes WHERE claim_id = ?")
        results = execute_concurrent(session, [(outcome_stmt, (cid,)) for cid in claim_ids], concurrency=100)

        outcomes = {}
        for success, result in results:
            if success and result:
                row = result[0]
                outcomes[row.claim_id] = row.outcome

        st_conf_count = 0
        st_conf_wrong = 0
        lt_conf_count = 0
        lt_conf_wrong = 0

        for r in proj_rows:
            outcome = outcomes.get(r.claim_id)
            if outcome is None:
                continue
                
            # ST Eval
            st_sum = (r.st_alpha or 0) + (r.st_beta or 0)
            if st_sum > 0:
                st_p = r.st_alpha / st_sum
                if st_p > 0.8 or st_p < 0.2:
                    st_conf_count += 1
                    pred = 1.0 if st_p > 0.5 else 0.0
                    if pred != outcome:
                        st_conf_wrong += 1
                        
            # LT Eval
            lt_sum = (r.lt_alpha or 0) + (r.lt_beta or 0)
            if lt_sum > 0:
                lt_p = r.lt_alpha / lt_sum
                if lt_p > 0.8 or lt_p < 0.2:
                    lt_conf_count += 1
                    pred = 1.0 if lt_p > 0.5 else 0.0
                    if pred != outcome:
                        lt_conf_wrong += 1

        if st_conf_count < 5 or lt_conf_count < 10:
            return
            
        st_error_rate = st_conf_wrong / st_conf_count
        lt_error_rate = lt_conf_wrong / lt_conf_count
        
        raw_shock = st_error_rate - lt_error_rate
        volume_weight = st_conf_count / (st_conf_count + lt_conf_count)
        shock = raw_shock * volume_weight
        
        # EMA Update
        current_lam = get_domain_lambda(domain)
        eta = 0.05
        gamma = 0.01
        lambda_base = 0.08
        
        new_lam = current_lam + (eta * shock) - (gamma * (current_lam - lambda_base))
        new_lam = max(0.001, min(new_lam, 1.0))
        
        session.execute(UPDATE_LAMBDA, (new_lam, domain))
        
        # Invalidate cache again just to be safe after write
        if domain in _cache:
            del _cache[domain]
            
        if st_error_rate > 0.40 and shock > 0.15:
            today = date.today()
            # Event is attributed to [T-2, T-1] relative to the projection trigger
            attr_start = today - timedelta(weeks=2)
            attr_end = today - timedelta(weeks=1)
            session.execute(INSERT_VOLATILITY_EVENT, (domain, today, attr_start, attr_end))

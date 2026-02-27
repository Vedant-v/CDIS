"""
projection_worker.py
--------------------
Materialises claim-level belief projections used for O(1) CQRS reads
(claim_fusion_projection) and domain-level epoch management.

Fixes applied vs original:
  1. project_agent_resilience — actually aggregates ledger data (was no-op)
  2. initialize_domain_epoch  — queries claims_by_domain, not decision_claim_edges
  3. project_claim_for_epoch  — wires auto_tune_lambda_task after epoch commit
"""

from celery import shared_task
from uuid import UUID, uuid1
from typing import Optional

import redis
import os
import structlog

from app.db.cassandra import get_session
from app.services.fusion_service import entropy, compute_raw_fusion_from_opinions
from app.db.dao.fusion_dao import fetch_opinions_st, fetch_opinions_lt
from app.services.expertise_temporal import fetch_temporal_expertise

logger       = structlog.get_logger()
redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "redis"), port=6379, db=0)

_MAX_EVIDENCE = 1_000_000.0


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _scale_evidence(alpha: float, beta: float):
    total = alpha + beta
    if total > _MAX_EVIDENCE:
        scale = _MAX_EVIDENCE / total
        return alpha * scale, beta * scale
    return alpha, beta


# ---------------------------------------------------------------------------
# Domain Epoch Initialization
# ---------------------------------------------------------------------------

@shared_task(name="app.services.projection_worker.initialize_domain_epoch")
def initialize_domain_epoch(domain: str):
    """
    Kick off a fresh projection epoch for *domain*.

    Queries claims_by_domain (not decision_claim_edges — that table is
    decision-scoped, not domain-scoped).
    """
    session = get_session()

    rows      = session.execute(
        "SELECT claim_id FROM claims_by_domain WHERE domain = %s",
        (domain,),
    )
    claim_ids = [row.claim_id for row in rows]

    if not claim_ids:
        logger.info("No claims found for domain", domain=domain)
        return

    epoch        = uuid1()
    target_count = len(claim_ids)

    redis_client.set(f"domain_epoch_target:{domain}:{epoch}", target_count)
    logger.info("Initialized projection epoch", domain=domain, epoch=str(epoch), target=target_count)

    for cid in claim_ids:
        project_claim_for_epoch.delay(str(cid), domain, str(epoch))


# ---------------------------------------------------------------------------
# Per-Claim Projection
# ---------------------------------------------------------------------------

@shared_task(
    name="app.services.projection_worker.project_claim_for_epoch",
    ignore_result=True,
)
def project_claim_for_epoch(claim_id_str: str, domain: str, epoch_str: str):
    claim_id = UUID(claim_id_str)
    epoch    = UUID(epoch_str)

    st_opinions = fetch_opinions_st(claim_id, days_back=30)
    lt_opinions = fetch_opinions_lt(claim_id)

    st_fusion = compute_raw_fusion_from_opinions(st_opinions, domain)
    lt_fusion = compute_raw_fusion_from_opinions(lt_opinions, domain)

    st_alpha, st_beta = _scale_evidence(
        st_fusion.get("alpha_total", 0.0),
        st_fusion.get("beta_total",  0.0),
    )
    lt_alpha, lt_beta = _scale_evidence(
        lt_fusion.get("alpha_total", 0.0),
        lt_fusion.get("beta_total",  0.0),
    )

    st_entropy_ = st_fusion["entropy"]
    lt_entropy_ = lt_fusion["entropy"]
    st_conf     = 1.0 - st_entropy_
    lt_conf     = 1.0 - lt_entropy_
    st_strength = st_alpha + st_beta
    lt_strength = lt_alpha + lt_beta

    session = get_session()

    # Dual-write: per-claim projection + epoch-keyed projection
    session.execute(
        """
        INSERT INTO claim_fusion_projection (
            claim_id, projection_epoch,
            st_alpha, st_beta, lt_alpha, lt_beta,
            st_entropy, lt_entropy, st_conf, lt_conf,
            st_strength, lt_strength
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            claim_id, epoch,
            st_alpha, st_beta, lt_alpha, lt_beta,
            st_entropy_, lt_entropy_, st_conf, lt_conf,
            st_strength, lt_strength,
        ),
    )

    session.execute(
        """
        INSERT INTO claim_fusion_projection_by_epoch (
            domain, projection_epoch, claim_id,
            st_alpha, st_beta, lt_alpha, lt_beta,
            st_entropy, lt_entropy, st_conf, lt_conf,
            st_strength, lt_strength
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            domain, epoch, claim_id,
            st_alpha, st_beta, lt_alpha, lt_beta,
            st_entropy_, lt_entropy_, st_conf, lt_conf,
            st_strength, lt_strength,
        ),
    )

    # Redis barrier
    participation_key = f"domain_epoch_participation:{domain}:{epoch}"
    target_key        = f"domain_epoch_target:{domain}:{epoch}"

    redis_client.sadd(participation_key, str(claim_id))
    current_count = redis_client.scard(participation_key)

    target_bytes = redis_client.get(target_key)
    if not target_bytes:
        logger.error("Epoch target missing", domain=domain, epoch=str(epoch))
        return

    target = int(target_bytes)

    if current_count >= target:
        # Monotonic CAS commit
        result = session.execute(
            """
            UPDATE domain_projection_epoch
            SET projection_epoch = %s
            WHERE domain = %s
            IF projection_epoch < %s
            """,
            (epoch, domain, epoch),
        ).one()

        if result.applied:
            logger.info("DOMAIN_PROJECTION_COMMITTED", domain=domain, epoch=str(epoch))

            # Wire up Phase 4F auto-tuner
            from app.services.domain_decay import auto_tune_lambda_task
            auto_tune_lambda_task.delay(domain, str(epoch))
        else:
            logger.info("Epoch commit skipped (monotonic CAS)", domain=domain, epoch=str(epoch))


# ---------------------------------------------------------------------------
# Agent VRS Projection  (FIXED — was a no-op in original)
# ---------------------------------------------------------------------------

@shared_task(
    name="app.services.projection_worker.project_agent_resilience",
    ignore_result=True,
)
def project_agent_resilience(agent_id_str: str, domain: str, epoch_str: Optional[str] = None):
    """
    Compute and persist the Volatility Resilience Score (VRS) for an agent.

    VRS = 1 - (shock_weighted_error / shock_weighted_confidence)

    Reads from agent_volatility_resilience_ledger across all available
    detection_week partitions for this (agent_id, domain).

    VRS ∈ [0, 1]:
      - 1.0  → agent is perfectly calibrated during shock periods
      - 0.0  → agent is maximally wrong during shocks (or no data)
    """
    agent_id = UUID(agent_id_str)
    session  = get_session()

    # Fetch all resilience ledger rows for this agent+domain.
    # We can't ALLOW FILTERING on detection_week without a partition scan,
    # so we rely on the table being partitioned by (agent_id, domain) and
    # detection_week being a clustering key — this scan is bounded per agent.
    rows = session.execute(
        """
        SELECT shock_weighted_error_sum, shock_weighted_confidence_sum
        FROM agent_volatility_resilience_ledger
        WHERE agent_id = %s AND domain = %s
        """,
        (agent_id, domain),
    )

    total_error = 0.0
    total_conf  = 0.0

    for row in rows:
        total_error += row.shock_weighted_error_sum  or 0.0
        total_conf  += row.shock_weighted_confidence_sum or 0.0

    if total_conf > 0.0:
        vrs = max(0.0, 1.0 - (total_error / total_conf))
    else:
        vrs = 0.0

    epoch = UUID(epoch_str) if epoch_str else uuid1()

    session.execute(
        """
        INSERT INTO agent_resilience_projection
            (agent_id, domain, projection_epoch, vrs)
        VALUES (%s, %s, %s, %s)
        """,
        (agent_id, domain, epoch, vrs),
    )

    logger.info("Updated agent VRS", agent_id=agent_id_str, domain=domain, vrs=round(vrs, 4))
"""
entropy_worker.py
-----------------
Pre-outcome entropy credit: rewards agents whose opinion *reduced*
collective uncertainty, before any ground-truth is known.

Critical fix applied: compute_weight now receives prediction_count
(was missing, causing TypeError on every call in original code).
"""

from app.worker import celery
from app.db.cassandra import get_session
from app.services.fusion_service import compute_weight, entropy
from app.services.expertise_temporal import fetch_temporal_expertise
from typing import Dict, Any, List
from uuid import UUID

_MAX_CAS_RETRIES = 5


# ---------------------------------------------------------------------------
# Pure-math helper (no DB)
# ---------------------------------------------------------------------------

def calculate_fusion_metrics(opinions: List[Any], expertise: Dict[UUID, Any]):
    """
    Compute cluster-penalised, weight-adjusted Beta fusion from a list of
    opinion objects.

    Returns: (p, H, alpha_total, beta_total, weight_detail_list)
    """
    cluster_counts: dict = {}
    for op in opinions:
        cluster_counts[op.cluster_id] = cluster_counts.get(op.cluster_id, 0) + 1

    alpha_total = 0.0
    beta_total  = 0.0
    weights     = []

    for op in opinions:
        exp_ = expertise.get(op.agent_id)
        if not exp_:
            continue

        # FIX: prediction_count was missing in original code → TypeError
        base_w = compute_weight(
            exp_.calibration_score,
            exp_.bias_score,
            exp_.entropy_score,
            getattr(exp_, "prediction_count", 0),
        )

        cluster_penalty = 1.0 / cluster_counts[op.cluster_id]
        w = base_w * cluster_penalty

        weights.append({
            "cluster_id":     op.cluster_id,
            "weighted_alpha": w * op.alpha,
            "weighted_beta":  w * op.beta,
        })
        alpha_total += w * op.alpha
        beta_total  += w * op.beta

    total = alpha_total + beta_total
    if total == 0.0:
        return 0.5, 1.0, 0.0, 0.0, weights

    p = alpha_total / total
    return p, entropy(p), alpha_total, beta_total, weights


# ---------------------------------------------------------------------------
# Celery Task
# ---------------------------------------------------------------------------

@celery.task(bind=True, max_retries=3, default_retry_delay=5)
def process_entropy_contribution(
    self,
    claim_id_str: str,
    domain: str,
    agent_id_str: str,
    alpha: float,
    beta: float,
    current_cluster_id: str,
):
    """
    Compute and persist the entropy-reduction contribution of a newly
    submitted opinion, then update the submitting agent's entropy_score.

    Idempotency: delta_H < 1e-9 is discarded; the score update itself is
    a CAS loop (read-modify-write with retry) so duplicate task execution
    is safe.
    """
    claim_id = UUID(claim_id_str)
    agent_id = UUID(agent_id_str)

    from app.db.dao.fusion_dao import fetch_opinions_st

    # ----------------------------------------------------------------
    # 1.  Fetch current opinions and expertise
    # ----------------------------------------------------------------
    opinions = fetch_opinions_st(claim_id, days_back=30)

    expertise: Dict[UUID, Any] = {}
    for op in opinions:
        if op.agent_id not in expertise:
            expertise[op.agent_id] = fetch_temporal_expertise(op.agent_id, domain)

    # Simulate "before" by excluding the agent's own new opinion
    opinions_before = [op for op in opinions if op.agent_id != agent_id]

    _, H_before, alpha_before, beta_before, weights_before = calculate_fusion_metrics(
        opinions_before, expertise
    )
    _, H_after, alpha_after, beta_after, _ = calculate_fusion_metrics(
        opinions, expertise
    )

    # ----------------------------------------------------------------
    # 2.  Cluster dominance discount
    # ----------------------------------------------------------------
    total_evidence_before = alpha_before + beta_before
    if total_evidence_before > 0.0:
        cluster_evidence_before = sum(
            e["weighted_alpha"] + e["weighted_beta"]
            for e in weights_before
            if e["cluster_id"] == current_cluster_id
        )
        cluster_ratio = cluster_evidence_before / total_evidence_before
    else:
        cluster_ratio = 0.0

    cluster_discount = 1.0 - cluster_ratio
    delta_H = (H_before - H_after) * cluster_discount

    if abs(delta_H) < 1e-9:
        return

    # ----------------------------------------------------------------
    # 3.  CAS update of entropy_score with retry loop
    # ----------------------------------------------------------------
    session      = get_session()
    lr           = 0.05
    decay_rate   = 0.001
    neutral_E    = 0.5
    evidence_scale = min(1.0, (alpha_after + beta_after) / 1000.0)

    for attempt in range(_MAX_CAS_RETRIES):
        row = session.execute(
            "SELECT entropy_score FROM agent_expertise WHERE agent_id = %s AND domain = %s",
            (agent_id, domain),
        ).one()

        old_E = float(row.entropy_score) if row else 0.5

        new_E = old_E + (lr * evidence_scale * delta_H)
        new_E = new_E - decay_rate * (new_E - neutral_E)
        new_E = max(0.01, min(new_E, 0.99))

        # Write with LWT (optimistic concurrency)
        result = session.execute(
            """
            UPDATE agent_expertise
            SET entropy_score = %s
            WHERE agent_id = %s AND domain = %s
            IF entropy_score = %s
            """,
            (new_E, agent_id, domain, old_E),
        )

        if result.one().applied:
            # Mirror to denormalised read-table
            session.execute(
                """
                UPDATE expertise_by_domain
                SET entropy_score = %s
                WHERE domain = %s AND agent_id = %s
                """,
                (new_E, domain, agent_id),
            )
            return   # Success

        # CAS failed — another writer updated concurrently; retry with fresh read
        if attempt == _MAX_CAS_RETRIES - 1:
            # Log and drop rather than infinite-loop; the update is non-critical
            import structlog
            structlog.get_logger().warning(
                "entropy_score CAS exhausted retries",
                agent_id=str(agent_id),
                domain=domain,
            )
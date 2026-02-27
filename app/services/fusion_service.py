"""
fusion_service.py
-----------------
Core math engine: weight computation, entropy, and belief fusion.

All pure-math functions (entropy, compute_weight, compute_raw_fusion_from_opinions,
weighted_logit_aggregate) have NO side effects and NO database calls — they can be
unit-tested in full isolation.

DB-touching functions (fuse_claim, compute_raw_claim_fusion) are separated clearly.
"""

import math
from math import log2
from uuid import UUID
from typing import Dict, Any, List

from app.db.cassandra import get_session
from app.db.dao.fusion_dao import fetch_opinions_st
from app.services.expertise_temporal import fetch_temporal_expertise


# =====================================================
# Entropy  (binary / bits)
# =====================================================

def entropy(p: float) -> float:
    """Shannon entropy in bits.  Returns value in [0, 1]."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * log2(p) - (1.0 - p) * log2(1.0 - p)


# =====================================================
# Weight Computation — Probation / Trust-Growth Model
# =====================================================

def compute_weight(
    C: float,           # Calibration score  ∈ [0, 1]
    B: float,           # Bias score         ∈ [0, 1]
    E: float,           # Entropy-contribution score ∈ [0, 1]
    prediction_count: int,  # Historical prediction count ≥ 0
) -> float:
    """
    Returns an agent's influence weight in [0, 1].

    Trust growth curve:
      - prediction_count = 0   → weight ≈ 0   (Sybil defence)
      - prediction_count → ∞  → weight → base_w (asymptotes below 1)

    Logarithmic growth is intentional: the curve is concave, so the
    marginal benefit of each additional prediction decays.  This makes
    reputation-farming increasingly expensive.
    """
    bias_factor = 1.0 - B

    base_w = (
        0.4 * C
        + 0.2 * E
        + 0.4 * bias_factor
    )

    k = 5.0  # tuning constant — log(1+148)/5 ≈ 1.0, so ~148 predictions for full trust
    trust_factor = min(1.0, math.log1p(prediction_count) / k)

    return max(0.0, min(base_w * trust_factor, 1.0))


# =====================================================
# Raw Fusion  (pure math — no DB)
# =====================================================

def compute_raw_fusion_from_opinions(
    opinions: list,
    domain: str,
    expertise_map: Dict[UUID, Any] = None,
) -> Dict[str, Any]:
    """
    Fuse a list of opinion objects into a probability estimate.

    opinions: objects with .agent_id, .alpha, .beta, .cluster_id
    expertise_map: pre-fetched {agent_id: AgentExpertise}; if None the
                   function fetches from DB (needed for live path).

    Returns:
        p_success, entropy, alpha_total, beta_total, evidence_strength
    """

    if expertise_map is None:
        expertise_map = {}
        for op in opinions:
            if op.agent_id not in expertise_map:
                expertise_map[op.agent_id] = fetch_temporal_expertise(op.agent_id, domain)

    # Cluster density — one effective vote per cluster regardless of size
    cluster_counts: Dict[str, int] = {}
    for op in opinions:
        cluster_counts[op.cluster_id] = cluster_counts.get(op.cluster_id, 0) + 1

    alpha_total = 0.0
    beta_total  = 0.0

    for op in opinions:
        exp_ = expertise_map.get(op.agent_id)
        if not exp_:
            continue

        base_w = compute_weight(
            exp_.calibration_score,
            exp_.bias_score,
            exp_.entropy_score,
            getattr(exp_, "prediction_count", 0),
        )

        cluster_penalty = 1.0 / cluster_counts[op.cluster_id]

        alpha_total += base_w * op.alpha * cluster_penalty
        beta_total  += base_w * op.beta  * cluster_penalty

    total_evidence = alpha_total + beta_total

    if total_evidence == 0.0:
        return {
            "p_success": 0.5,
            "entropy": 1.0,
            "alpha_total": 0.0,
            "beta_total": 0.0,
            "evidence_strength": 0.0,
        }

    # Defensive clamp — should never be needed if inputs are valid
    alpha_total = max(0.0, alpha_total)
    beta_total  = max(0.0, beta_total)

    p = alpha_total / total_evidence
    H = entropy(p)

    return {
        "p_success": p,
        "entropy": H,
        "alpha_total": alpha_total,
        "beta_total": beta_total,
        "evidence_strength": total_evidence,
    }


# =====================================================
# O(1) CQRS Projection Read
# =====================================================

def fuse_claim(claim_id: UUID, domain: str) -> Dict[str, Any]:
    """
    Phase 5 CQRS: O(1) read from materialised claim_fusion_projection.
    Falls back to neutral prior if projection row is absent.
    """
    session = get_session()

    row = session.execute(
        """
        SELECT st_alpha, st_beta, st_entropy, st_strength
        FROM claim_fusion_projection
        WHERE claim_id = %s
        """,
        (claim_id,),
    ).one()

    if row is not None:
        total = (row.st_alpha or 0.0) + (row.st_beta or 0.0)
        if total > 0.0:
            p = row.st_alpha / total
            return {
                "p_success": p,
                "entropy": row.st_entropy if row.st_entropy is not None else entropy(p),
                "strength": row.st_strength or 0.0,
            }

    return {"p_success": 0.5, "entropy": 1.0, "strength": 0.0}


# =====================================================
# Live Raw-Fusion Wrapper  (DB-touching)
# =====================================================

def compute_raw_claim_fusion(claim_id: UUID, domain: str) -> Dict[str, Any]:
    opinions = fetch_opinions_st(claim_id, days_back=30)
    return compute_raw_fusion_from_opinions(opinions, domain)
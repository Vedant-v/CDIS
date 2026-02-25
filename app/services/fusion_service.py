from math import log2
from uuid import UUID
from typing import Dict, Any
from app.db.cassandra import get_session
from app.db.dao.fusion_dao import fetch_recent_opinions
from app.services.expertise_temporal import fetch_temporal_expertise

def entropy(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0
    return -p * log2(p) - (1-p) * log2(1-p)

def compute_weight(C: float, B: float, E: float) -> float:
    bias_factor = (1 - B)
    w = (
        0.4 * C +
        0.2 * E +
        0.4 * bias_factor
    )
    return max(w, 0.01)

def fuse_claim(claim_id: UUID, domain: str) -> Dict[str, Any]:
    """
    Phase 5 CQRS: O(1) read from claim_fusion_projection.
    """
    session = get_session()
    query = "SELECT st_alpha, st_beta, st_entropy FROM claim_fusion_projection WHERE claim_id = %s"
    row = session.execute(query, (claim_id,)).one()
    
    if row and (row.st_alpha + row.st_beta) > 0:
        p = row.st_alpha / (row.st_alpha + row.st_beta)
        return {
            "p_success": p,
            "entropy": row.st_entropy
        }
        
    return {
        "p_success": 0.5,
        "entropy": 1.0
    }

def compute_raw_claim_fusion(claim_id: UUID, domain: str) -> Dict[str, Any]:
    opinions = fetch_recent_opinions(claim_id)
    
    expertise = {}
    for op in opinions:
        if op.agent_id not in expertise:
            expertise[op.agent_id] = fetch_temporal_expertise(op.agent_id, domain)

    cluster_counts = {}
    for op in opinions:
        cluster = op.cluster_id
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1

    alpha_total = 0.0
    beta_total = 0.0
    weights = []

    for op in opinions:
        exp = expertise.get(op.agent_id)

        if not exp:
            continue

        base_w = compute_weight(
            exp.calibration_score,
            exp.bias_score,
            exp.entropy_score
        )
        
        cluster_penalty = 1 / (1 + cluster_counts[op.cluster_id])
        effective_a = op.alpha * cluster_penalty
        effective_b = op.beta  * cluster_penalty
        
        weights.append((base_w, effective_a, effective_b))

    total_w = sum(w for w,_,_ in weights)
    
    if total_w == 0:
        raw_alpha = sum(op.alpha for op in opinions)
        raw_beta = sum(op.beta for op in opinions)
        if raw_alpha + raw_beta > 0:
            p = raw_alpha / (raw_alpha + raw_beta)
            return {
                "p_success": p,
                "entropy": entropy(p)
            }
        return {
            "p_success": 0.5, # undefined
            "entropy": 1.0
        }

    for w, a, b in weights:
        wn = w / total_w
        alpha_total += wn * a
        beta_total  += wn * b

    if alpha_total + beta_total == 0:
        return {
            "p_success": 0.5,
            "entropy": 1.0
        }

    p = alpha_total / (alpha_total + beta_total)
    H = entropy(p)

    return {
        "p_success": p,
        "entropy": H
    }

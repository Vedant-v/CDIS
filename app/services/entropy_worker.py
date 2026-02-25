from app.worker import celery
from app.db.cassandra import get_session
from app.services.fusion_service import fetch_recent_opinions, compute_weight, entropy
from app.services.expertise_temporal import fetch_temporal_expertise
from typing import Dict, Any, List
from uuid import UUID

def calculate_fusion_metrics(opinions: List[Any], expertise: Dict[UUID, Any]):
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
        base_w = compute_weight(exp.calibration_score, exp.bias_score, exp.entropy_score)
        
        cluster_penalty = 1 / (1 + cluster_counts[op.cluster_id])
        w = base_w * cluster_penalty
        
        weights.append((w, op.alpha, op.beta))

    total_w = sum(w for w,_,_ in weights)
    
    if total_w == 0:
        return 0.5, 1.0

    for w, a, b in weights:
        wn = w / total_w
        alpha_total += wn * a
        beta_total  += wn * b

    if alpha_total + beta_total == 0:
        return 0.5, 1.0

    p = alpha_total / (alpha_total + beta_total)
    return p, entropy(p)


@celery.task
def process_entropy_contribution(claim_id_str: str, domain: str, agent_id_str: str, alpha: float, beta: float, current_cluster_id: str):
    claim_id = UUID(claim_id_str)
    agent_id = UUID(agent_id_str)

    # 1. Fetch current posterior
    opinions = fetch_recent_opinions(claim_id)
    
    expertise = {}
    for op in opinions:
        if op.agent_id not in expertise:
            expertise[op.agent_id] = fetch_temporal_expertise(op.agent_id, domain)
    
    # Exclude the newly submitted opinion to find H_before
    # Note: the new opinion is already in the DB by the time this task runs, 
    # so we filter it out to simulate "before".
    opinions_before = [op for op in opinions if op.agent_id != agent_id]
    
    _, H_before = calculate_fusion_metrics(opinions_before, expertise)

    # 2. Simulate fusion with the new opinion (H_after)
    # Since the new opinion is in `opinions`, `calculate_fusion_metrics` naturally calculates H_after
    _, H_after = calculate_fusion_metrics(opinions, expertise)

    # 3. Compute Contribution
    # Cluster Penalty: Discount entropy credit if the agent's cluster already dominates the baseline belief.
    total_opinions_before = len(opinions_before)
    if total_opinions_before > 0:
        same_cluster_count = sum(1 for op in opinions_before if op.cluster_id == current_cluster_id)
        cluster_ratio = same_cluster_count / total_opinions_before
    else:
        cluster_ratio = 0.0
        
    cluster_discount = 1.0 - cluster_ratio
    
    delta_H = (H_before - H_after) * cluster_discount

    # 4. Update Entropy Score
    row = get_session().execute("""
    SELECT entropy_score FROM agent_expertise
    WHERE agent_id = %s AND domain = %s
    """, (agent_id, domain)).one()

    old_E = row.entropy_score if row else 0.5

    # Same exponential update rule: old + lr * (signal - old)
    lr = 0.05
    # Since we want delta_H > 0 to be good, we apply the raw delta.
    # We might cap delta_H broadly between [-1, 1], but mathematically it's a difference of max 1.0.
    new_E = old_E + lr * (delta_H - old_E)
    new_E = max(0.01, new_E) # Clamp

    # Write back
    get_session().execute("""
    UPDATE agent_expertise
    SET entropy_score = %s
    WHERE agent_id = %s
    AND domain = %s
    """, (new_E, agent_id, domain))

    get_session().execute("""
    UPDATE expertise_by_domain
    SET entropy_score = %s
    WHERE domain = %s
    AND agent_id = %s
    """, (new_E, domain, agent_id))

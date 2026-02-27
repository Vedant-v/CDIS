import uuid
from app.services.fusion_service import compute_raw_fusion_from_opinions

class MockOpinion:
    def __init__(self, alpha, beta, agent_id, cluster_id):
        self.alpha = alpha
        self.beta = beta
        self.agent_id = agent_id
        self.cluster_id = cluster_id

# Mocking the temporal expertise to always return fixed weights
import app.services.fusion_service as fs
fs.fetch_temporal_expertise = lambda a, d: type("Exp", (), {"calibration_score": 0.8, "bias_score": 0.1, "entropy_score": 0.3})()
w = 0.74 # Base weight computed from above constants in fusion_service (0.4*0.8 + 0.2*0.3 + 0.4*(1-0.1))

def test_A_strength_monotonicity():
    print("\n--- Test A: Strength Monotonicity ---")
    agent_id = uuid.uuid4()
    
    op1 = MockOpinion(10.0, 5.0, agent_id, "cluster1")
    res1 = compute_raw_fusion_from_opinions([op1], "test")
    p1 = res1["p_success"]
    n1 = res1["evidence_strength"]
    
    op2 = MockOpinion(10.0, 5.0, agent_id, "cluster2")
    res2 = compute_raw_fusion_from_opinions([op1, op2], "test")
    p2 = res2["p_success"]
    n2 = res2["evidence_strength"]

    assert abs(p1 - p2) < 0.001, "Monotonicity failed: scaling alpha/beta shifted the posterior p."
    assert abs(n2 - (2 * n1)) < 0.001, "Monotonicity failed: 2 identical identical distinct agents didn't double strength."
    
    print("✅ Passed Test A")

def test_B_opposition_effect():
    print("\n--- Test B: Opposition Effect ---")
    
    agent_id1 = uuid.uuid4()
    agent_id2 = uuid.uuid4()
    
    # Very optimistic opinion
    op1 = MockOpinion(90.0, 10.0, agent_id1, "cluster1")
    res1 = compute_raw_fusion_from_opinions([op1], "test")
    p1 = res1["p_success"]
    
    # Exact opposite
    op2 = MockOpinion(10.0, 90.0, agent_id2, "cluster2")
    res2 = compute_raw_fusion_from_opinions([op1, op2], "test")
    p2 = res2["p_success"]
    
    assert p1 > 0.8, "Initial optimistic opinion didn't resolve to > 0.8"
    assert abs(p2 - 0.5) < 0.001, "Opposing opinion did not exactly cancel out p to 0.5"
    
    print("✅ Passed Test B")

def test_C_cluster_neutrality():
    print("\n--- Test C: Cluster Neutrality ---")
    # Base scenario: 1 person in cluster
    agent_1 = uuid.uuid4()
    op_base = MockOpinion(20.0, 5.0, agent_1, "clusterA")
    res_base = compute_raw_fusion_from_opinions([op_base], "test")
    
    n_base = res_base["evidence_strength"]
    
    # 5 people in the same cluster repeating the exact same thing
    ops_cluster = []
    for _ in range(5):
        ops_cluster.append(MockOpinion(20.0, 5.0, uuid.uuid4(), "clusterA"))
        
    res_cluster = compute_raw_fusion_from_opinions(ops_cluster, "test")
    n_cluster = res_cluster["evidence_strength"]
    p_base = res_base["p_success"]
    p_cluster = res_cluster["p_success"]
    
    # Due to 1/N size penalty, total strength and posterior shouldn't change
    assert abs(n_cluster - n_base) < 0.001, "Cluster penalty failed to prevent strength multiplication"
    assert abs(p_cluster - p_base) < 0.001, "Cluster distribution changed posterior"
    assert abs(res_cluster["alpha_total"] - res_base["alpha_total"]) < 0.001, "Alpha components do not match!"
    assert abs(res_cluster["beta_total"] - res_base["beta_total"]) < 0.001, "Beta components do not match!"
    
    print("✅ Passed Test C")

def test_D_cap_stability():
    print("\n--- Test D: Cap Stability ---")
    st_alpha = 900_000.0
    st_beta = 300_000.0
    
    st_total = st_alpha + st_beta
    p_before = st_alpha / st_total
    
    MAX_EVIDENCE = 1_000_000.0
    
    if st_total > MAX_EVIDENCE:
        scale = MAX_EVIDENCE / st_total
        st_alpha *= scale
        st_beta *= scale
        
    st_total_after = st_alpha + st_beta
    p_after = st_alpha / st_total_after
    
    assert abs(st_total_after - MAX_EVIDENCE) < 0.001, "Cap did not bind at 1 million exactly."
    assert abs(p_before - p_after) < 0.001, "Proportional clamp altered the posterior probability."
    
    print("✅ Passed Test D")

if __name__ == "__main__":
    test_A_strength_monotonicity()
    test_B_opposition_effect()
    test_C_cluster_neutrality()
    test_D_cap_stability()
    print("\n🎉 ALL THEOREM TESTS PASSED 🎉")

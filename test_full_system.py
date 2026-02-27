import uuid
import random
import math

from app.services.fusion_service import compute_raw_fusion_from_opinions
from app.services.dag_service import evaluate_decision_graph
import app.services.dag_service as dag_module


# ================================
# MOCK OBJECTS
# ================================

class MockOpinion:
    def __init__(self, alpha, beta, agent_id, cluster_id):
        self.alpha = alpha
        self.beta = beta
        self.agent_id = agent_id
        self.cluster_id = cluster_id


# Mock expertise to constant values
import app.services.fusion_service as fs
fs.fetch_temporal_expertise = lambda a, d: type(
    "Exp", (), {"calibration_score": 0.8, "bias_score": 0.1, "entropy_score": 0.2}
)()


# ================================
# TEST 1 — Bayesian Opposition Cancellation
# ================================

def test_opposition_cancellation():
    print("Test 1: Opposition cancellation")

    a1 = MockOpinion(90, 10, uuid.uuid4(), "A")
    a2 = MockOpinion(10, 90, uuid.uuid4(), "B")

    res = compute_raw_fusion_from_opinions([a1, a2], "test")
    p = res["p_success"]

    assert abs(p - 0.5) < 0.01
    print("✔ Passed")


# ================================
# TEST 2 — Cluster Neutrality
# ================================

def test_cluster_neutrality():
    print("Test 2: Cluster neutrality")

    ops = []
    for _ in range(50):
        ops.append(MockOpinion(20, 5, uuid.uuid4(), "same_cluster"))

    res = compute_raw_fusion_from_opinions(ops, "test")

    assert abs(res["p_success"] - (20/25)) < 0.01
    print("✔ Passed")


# ================================
# TEST 3 — Strength Monotonicity
# ================================

def test_strength_monotonicity():
    print("Test 3: Strength monotonicity")

    op = MockOpinion(10, 5, uuid.uuid4(), "A")
    res1 = compute_raw_fusion_from_opinions([op], "test")

    ops = [MockOpinion(10, 5, uuid.uuid4(), str(i)) for i in range(10)]
    res2 = compute_raw_fusion_from_opinions(ops, "test")

    assert res2["evidence_strength"] > res1["evidence_strength"]
    print("✔ Passed")


# ================================
# TEST 4 — DAG Aggregation Stability
# ================================

def test_dag_aggregation():
    print("Test 4: DAG aggregation stability")

    # Monkeypatch fuse_claim
    fake_data = {}

    def fake_fuse(claim_id, domain):
        return fake_data[claim_id]

    dag_module.fuse_claim = fake_fuse

    # Build fake graph manually
    import networkx as nx
    G = nx.DiGraph()

    leaf1 = uuid.uuid4()
    leaf2 = uuid.uuid4()
    root = uuid.uuid4()

    G.add_edge(leaf1, root, weight=1.0, is_root=True)
    G.add_edge(leaf2, root, weight=1.0, is_root=True)

    dag_module.load_decision_graph = lambda x: G

    fake_data[leaf1] = {"p_success": 0.9, "entropy": 0.2, "strength": 200}
    fake_data[leaf2] = {"p_success": 0.9, "entropy": 0.2, "strength": 200}

    res = evaluate_decision_graph(uuid.uuid4(), {})

    assert res["p_success"] > 0.85
    assert res["confidence"] > 0.3
    print("✔ Passed")


# ================================
# TEST 5 — Adversarial Burst Resistance
# ================================

def test_adversarial_cluster_attack():
    print("Test 5: Adversarial cluster attack resistance")

    ops = []

    # Honest minority
    for _ in range(5):
        ops.append(MockOpinion(90, 10, uuid.uuid4(), str(uuid.uuid4())))

    # 500 bots in same cluster
    for _ in range(500):
        ops.append(MockOpinion(10, 90, uuid.uuid4(), "bot_cluster"))

    res = compute_raw_fusion_from_opinions(ops, "test")

    # Bots should not fully dominate due to cluster penalty
    assert res["p_success"] > 0.4
    print("✔ Passed")


# ================================
# TEST 6 — Cycle Detection
# ================================

def test_cycle_detection():
    print("Test 6: Cycle detection")

    import networkx as nx

    G = nx.DiGraph()
    a = uuid.uuid4()
    b = uuid.uuid4()

    G.add_edge(a, b, weight=1.0, is_root=False)
    G.add_edge(b, a, weight=1.0, is_root=True)

    dag_module.load_decision_graph = lambda x: G

    res = evaluate_decision_graph(uuid.uuid4(), {})

    assert "error" in res
    print("✔ Passed")


# ================================
# RUN ALL
# ================================

if __name__ == "__main__":
    test_opposition_cancellation()
    test_cluster_neutrality()
    test_strength_monotonicity()
    test_dag_aggregation()
    test_adversarial_cluster_attack()
    test_cycle_detection()

    print("\nALL SYSTEM TESTS PASSED")
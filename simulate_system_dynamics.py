import uuid
import random
import math
from statistics import mean
import app.services.fusion_service as fs
from app.services.fusion_service import compute_raw_fusion_from_opinions
from app.services.entropy_worker import calculate_fusion_metrics

# ==============================================
# MOCKING EXPERTISE
# ==============================================

class MockOpinion:
    def __init__(self, alpha, beta, agent_id, cluster_id):
        self.alpha = alpha
        self.beta = beta
        self.agent_id = agent_id
        self.cluster_id = cluster_id


def mock_expertise(agent_id, domain="test"):
    """
    Honest agents have history.
    Bots have zero history.
    """
    if "bot" in str(agent_id):
        return type("Exp", (), {
            "calibration_score": 0.4,
            "bias_score": 0.8,
            "entropy_score": 0.8,
            "prediction_count": 0
        })()

    return type("Exp", (), {
        "calibration_score": 0.8,
        "bias_score": 0.1,
        "entropy_score": 0.2,
        "prediction_count": 50
    })()


fs.fetch_temporal_expertise = mock_expertise

# ==============================================
# 1️⃣ MONTE CARLO ADVERSARIAL TEST
# ==============================================

def monte_carlo_adversarial_test():
    print("\n--- Simulation 1: Monte Carlo Adversarial Test ---")

    honest = [
        MockOpinion(90, 10, f"honest_{i}", f"h_cluster_{i}")
        for i in range(10)
    ]

    bots = [
        MockOpinion(10, 90, f"bot_{i}", "BOT_CLUSTER")
        for i in range(500)
    ]

    result = compute_raw_fusion_from_opinions(honest + bots, "test")

    print("p_success:", round(result["p_success"], 4))
    print("strength:", round(result["evidence_strength"], 2))

    assert result["p_success"] > 0.8
    print("✔ Adversarial cluster attack neutralized.")

# ==============================================
# 2️⃣ LONG-HORIZON ENTROPY DRIFT
# ==============================================

def entropy_drift_simulation(iterations=5000):
    print("\n--- Simulation 2: Long-Horizon Entropy Drift ---")

    E = 0.5
    history = []

    for _ in range(iterations):

        background = [
            MockOpinion(70, 30, f"other_{i}", f"c_{i}")
            for i in range(10)
        ]

        noisy_alpha = 70 + random.uniform(-10, 10)
        noisy_beta = 30 + random.uniform(-10, 10)

        agent_op = MockOpinion(noisy_alpha, noisy_beta, "honest_agent", "agent_cluster")

        opinions_after = background + [agent_op]

        expertise = {
            op.agent_id: mock_expertise(op.agent_id)
            for op in opinions_after
        }

        _, H_before, alpha_before, beta_before, weights_before = calculate_fusion_metrics(background, expertise)
        _, H_after, alpha_after, beta_after, weights_after = calculate_fusion_metrics(opinions_after, expertise)

        delta_H = H_before - H_after

        if abs(delta_H) > 1e-9:
            E += 0.05 * delta_H
            E = max(0.01, min(E, 0.99))

        history.append(E)

    print("Final Entropy Score:", round(E, 4))
    print("Mean Entropy Score:", round(mean(history), 4))

    assert 0.01 < E < 0.99
    print("✔ Entropy stable.")

# ==============================================
# 3️⃣ DOMAIN SHOCK TEST
# ==============================================

def domain_shock_simulation():
    print("\n--- Simulation 3: Domain Shock ---")

    base = [
        MockOpinion(90, 10, f"honest_{i}", f"c_{i}")
        for i in range(50)
    ]

    res1 = compute_raw_fusion_from_opinions(base, "test")
    print("Initial p_success:", round(res1["p_success"], 4))

    shock = [
        MockOpinion(10, 90, f"bot_{i}", f"shock_{i}")
        for i in range(200)
    ]

    res2 = compute_raw_fusion_from_opinions(base + shock, "test")

    print("After shock p_success:", round(res2["p_success"], 4))

    assert res2["p_success"] < 0.5
    print("✔ Domain shock handled.")

# ==============================================
# 4️⃣ CLUSTER SPLITTING TEST
# ==============================================

def sim_cluster_splitting_attack():
    print("\n--- Simulation 4: Cluster Splitting Attack ---")

    honest = [
        MockOpinion(90, 10, f"honest_{i}", f"h_cluster_{i}")
        for i in range(10)
    ]

    bots = [
        MockOpinion(5, 95, f"bot_{i}", f"bot_cluster_{i}")
        for i in range(500)
    ]

    result = compute_raw_fusion_from_opinions(honest + bots, "test")

    print("p_success:", round(result["p_success"], 4))
    print("strength:", round(result["evidence_strength"], 2))

    assert result["p_success"] > 0.5
    print("✔ Cluster splitting resisted.")

# ==============================================
# MAIN
# ==============================================

if __name__ == "__main__":
    try:
        monte_carlo_adversarial_test()
        entropy_drift_simulation()
        domain_shock_simulation()
        sim_cluster_splitting_attack()
        print("\n🎉 ALL CORE SYSTEM TESTS PASSED")
    except AssertionError as e:
        print("\n❌ SIMULATION FAILED:", e)
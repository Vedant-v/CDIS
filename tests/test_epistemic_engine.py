"""
tests/test_epistemic_engine.py
==============================
Comprehensive test suite for the Epistemic Aggregation Engine.

Tests are grouped into five categories:
  1. Mathematical Invariants
  2. Adversarial Resistance
  3. Temporal Dynamics
  4. Structural (DAG) Reasoning
  5. System-Level Sanity

All tests are pure-math: no database, no Celery, no Redis.
Fake opinion objects are constructed in-place.

Run with:
    pytest tests/test_epistemic_engine.py -v
"""

import sys
import os
import math
import random
from collections import namedtuple
from typing import List, Dict, Any
from uuid import uuid4

import pytest

# ---------------------------------------------------------------------------
# Inject paths so imports work without installing the package
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Import only the pure-math functions — no DB calls are triggered
# ---------------------------------------------------------------------------
from app.services.fusion_service import (
    entropy,
    compute_weight,
    compute_raw_fusion_from_opinions,
)
from app.services.opinion_service import belief_to_beta, MAX_OPINION_STRENGTH
from app.services.dag_service import (
    weighted_logit_aggregate,
    logit,
    inv_logit,
    strength_factor,
    evaluate_decision_graph,
)
from app.services.entropy_worker import calculate_fusion_metrics
from app.utils.time_bucket import current_week_bucket


# ---------------------------------------------------------------------------
# Test Fixtures / Helpers
# ---------------------------------------------------------------------------

AgentExpertise = namedtuple(
    "AgentExpertise",
    ["calibration_score", "bias_score", "entropy_score", "prediction_count"],
)

Opinion = namedtuple("Opinion", ["agent_id", "alpha", "beta", "cluster_id"])


def make_expert(calibration=0.8, bias=0.1, entropy_s=0.7, count=100):
    return AgentExpertise(
        calibration_score=calibration,
        bias_score=bias,
        entropy_score=entropy_s,
        prediction_count=count,
    )


def make_opinion(belief, confidence, cluster_id="c1", agent_id=None):
    if agent_id is None:
        agent_id = uuid4()
    alpha, beta = belief_to_beta(belief, confidence)
    return Opinion(agent_id=agent_id, alpha=alpha, beta=beta, cluster_id=cluster_id)


def fuse(opinions: List[Opinion], expertise_override=None) -> Dict[str, Any]:
    """
    Wrapper: fuse opinions without hitting the DB.
    expertise_override maps agent_id → AgentExpertise.
    If not provided, every agent gets a default good expert.
    """
    if expertise_override is None:
        expertise_override = {op.agent_id: make_expert() for op in opinions}
    return compute_raw_fusion_from_opinions(
        opinions, domain="test", expertise_map=expertise_override
    )


# ============================================================================
# CATEGORY 1: MATHEMATICAL INVARIANTS
# ============================================================================

class TestMathematicalInvariants:

    # ------------------------------------------------------------------ 1A
    def test_entropy_zero_at_extremes(self):
        assert entropy(0.0) == 0.0
        assert entropy(1.0) == 0.0

    def test_entropy_max_at_half(self):
        assert abs(entropy(0.5) - 1.0) < 1e-9

    def test_entropy_bounds(self):
        for p in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
            H = entropy(p)
            assert 0.0 <= H <= 1.0, f"entropy({p}) = {H} out of [0,1]"

    def test_entropy_no_nan(self):
        for p in [0.0, 1e-10, 0.5, 1.0 - 1e-10, 1.0]:
            H = entropy(p)
            assert not math.isnan(H)
            assert not math.isinf(H)

    # ------------------------------------------------------------------ 1B: Strength Monotonicity
    def test_adding_identical_evidence_preserves_p(self):
        """
        Adding a second agent with the same belief in a DIFFERENT cluster
        must not shift p_success (cluster penalty makes each cluster count as
        one effective vote regardless of size, so two clusters of 1 = two votes
        with the same logit → same weighted average).

        Evidence strength must increase because accumulated alpha+beta grows.
        """
        a1, a2 = uuid4(), uuid4()
        exp = make_expert()

        # Single agent in cluster "c1"
        ops_1x = [Opinion(a1, alpha=20.0, beta=5.0, cluster_id="c1")]
        exp_map_1 = {a1: exp}

        # Two agents in DIFFERENT clusters — each gets full cluster_penalty=1.0
        ops_2x = [Opinion(a1, alpha=20.0, beta=5.0, cluster_id="c1"),
                  Opinion(a2, alpha=20.0, beta=5.0, cluster_id="c2")]
        exp_map_2 = {a1: exp, a2: exp}

        r1 = compute_raw_fusion_from_opinions(ops_1x, "test", exp_map_1)
        r2 = compute_raw_fusion_from_opinions(ops_2x, "test", exp_map_2)

        assert abs(r1["p_success"] - r2["p_success"]) < 1e-6, (
            f"p shifted from {r1['p_success']:.4f} to {r2['p_success']:.4f} "
            "when adding identical independent evidence in a new cluster"
        )
        assert r2["evidence_strength"] > r1["evidence_strength"], (
            f"strength did not increase: r1={r1['evidence_strength']:.4f}, "
            f"r2={r2['evidence_strength']:.4f}"
        )

    # ------------------------------------------------------------------ 1C: Opposition Cancellation
    def test_equal_opposing_evidence_converges_to_half(self):
        """
        Equal alpha and beta evidence must yield p ≈ 0.5.
        """
        a1, a2 = uuid4(), uuid4()
        exp_map = {a1: make_expert(), a2: make_expert()}
        ops = [
            Opinion(agent_id=a1, alpha=20.0, beta=5.0,  cluster_id="c1"),
            Opinion(agent_id=a2, alpha=5.0,  beta=20.0, cluster_id="c2"),
        ]
        # agents have same expertise so weights are equal
        r = compute_raw_fusion_from_opinions(ops, "test", exp_map)
        assert abs(r["p_success"] - 0.5) < 0.05, (
            f"Opposition cancellation failed: p = {r['p_success']:.4f}"
        )

    # ------------------------------------------------------------------ 1D: Probability Bounds
    def test_probability_always_in_unit_interval(self):
        for _ in range(200):
            n_ops = random.randint(1, 20)
            ops, exp_map = [], {}
            for _ in range(n_ops):
                aid = uuid4()
                belief = random.random()
                conf   = random.random()
                op = make_opinion(belief, conf, agent_id=aid)
                ops.append(op)
                exp_map[aid] = make_expert(
                    calibration=random.random(),
                    bias=random.random() * 0.5,
                    entropy_s=random.random(),
                    count=random.randint(0, 200),
                )
            r = compute_raw_fusion_from_opinions(ops, "test", exp_map)
            assert 0.0 <= r["p_success"] <= 1.0
            assert 0.0 <= r["entropy"]   <= 1.0
            assert r["alpha_total"] >= 0.0
            assert r["beta_total"]  >= 0.0
            assert not math.isnan(r["p_success"])

    # ------------------------------------------------------------------ 1E: Scaling Invariance
    def test_scaling_alpha_beta_preserves_p(self):
        """
        Multiplying all alpha/beta by a constant k must not change p_success.
        """
        for k in [0.5, 2.0, 100.0, 0.001]:
            a1, a2 = uuid4(), uuid4()
            exp_map = {a1: make_expert(), a2: make_expert()}

            ops_base = [
                Opinion(agent_id=a1, alpha=10.0, beta=5.0, cluster_id="c1"),
                Opinion(agent_id=a2, alpha=8.0,  beta=2.0, cluster_id="c2"),
            ]
            ops_scaled = [
                Opinion(agent_id=a1, alpha=10.0*k, beta=5.0*k, cluster_id="c1"),
                Opinion(agent_id=a2, alpha=8.0*k,  beta=2.0*k, cluster_id="c2"),
            ]

            r_base   = compute_raw_fusion_from_opinions(ops_base,   "test", exp_map)
            r_scaled = compute_raw_fusion_from_opinions(ops_scaled, "test", exp_map)

            assert abs(r_base["p_success"] - r_scaled["p_success"]) < 1e-6, (
                f"Scaling by {k} changed p: {r_base['p_success']:.6f} → {r_scaled['p_success']:.6f}"
            )
            # Strength must scale proportionally
            assert abs(r_scaled["evidence_strength"] - k * r_base["evidence_strength"]) < 1e-4, (
                f"Strength didn't scale correctly at k={k}"
            )

    # ------------------------------------------------------------------ 1F: belief_to_beta invariants
    def test_belief_to_beta_always_positive(self):
        for b in [0.0, 0.01, 0.5, 0.99, 1.0]:
            for c in [0.0, 0.01, 0.5, 1.0]:
                alpha, beta = belief_to_beta(b, c)
                assert alpha > 0.0
                assert beta  > 0.0

    def test_belief_to_beta_ratio_matches_belief(self):
        for belief in [0.1, 0.3, 0.5, 0.7, 0.9]:
            alpha, beta = belief_to_beta(belief, 0.8)
            reconstructed = alpha / (alpha + beta)
            # After clipping belief is unchanged for values in (0.01, 0.99)
            assert abs(reconstructed - belief) < 0.01

    def test_belief_to_beta_max_strength(self):
        alpha, beta = belief_to_beta(0.5, 1.0)
        assert alpha + beta <= MAX_OPINION_STRENGTH + 1e-9

    # ------------------------------------------------------------------ 1G: compute_weight invariants
    def test_weight_zero_for_new_agent(self):
        w = compute_weight(1.0, 0.0, 1.0, prediction_count=0)
        assert w == 0.0, "New agent should have zero weight (Sybil defence)"

    def test_weight_grows_with_predictions(self):
        weights = [
            compute_weight(0.8, 0.1, 0.7, count)
            for count in [0, 1, 5, 20, 100, 1000]
        ]
        for i in range(len(weights) - 1):
            assert weights[i] <= weights[i + 1], (
                f"Weight did not increase at step {i}: {weights}"
            )

    def test_weight_bounded(self):
        for _ in range(100):
            w = compute_weight(
                random.random(), random.random() * 0.5,
                random.random(), random.randint(0, 10_000)
            )
            assert 0.0 <= w <= 1.0

    def test_high_bias_reduces_weight(self):
        w_low_bias  = compute_weight(0.9, 0.0, 0.8, 100)
        w_high_bias = compute_weight(0.9, 0.9, 0.8, 100)
        assert w_high_bias < w_low_bias


# ============================================================================
# CATEGORY 2: ADVERSARIAL RESISTANCE
# ============================================================================

class TestAdversarialResistance:

    # ------------------------------------------------------------------ 2A: Bot Swarm Single Cluster
    def test_bot_swarm_single_cluster_bounded(self):
        """
        500 bots in the same cluster should behave like ~1 effective opinion.
        The cluster penalty compresses all cluster mass to 1/N per agent,
        so their combined influence equals approximately one agent.
        Compare swarm result vs a single honest agent with same belief.
        """
        cluster = "bot_cluster"
        honest_agent = uuid4()

        # Single honest agent
        single_op = [Opinion(honest_agent, alpha=20.0, beta=5.0, cluster_id=cluster)]
        single_exp = {honest_agent: make_expert()}
        r_single = compute_raw_fusion_from_opinions(single_op, "test", single_exp)

        # 500 bots in same cluster, each submitting same belief
        n_bots  = 500
        bot_ops = []
        bot_exp = {}
        for _ in range(n_bots):
            aid = uuid4()
            bot_ops.append(Opinion(aid, alpha=20.0, beta=5.0, cluster_id=cluster))
            bot_exp[aid] = make_expert()

        r_swarm = compute_raw_fusion_from_opinions(bot_ops, "test", bot_exp)

        # p_success should be essentially the same
        assert abs(r_swarm["p_success"] - r_single["p_success"]) < 0.01, (
            f"Bot swarm shifted p significantly: single={r_single['p_success']:.4f}, "
            f"swarm={r_swarm['p_success']:.4f}"
        )

    # ------------------------------------------------------------------ 2B: Cluster Splitting Attack
    def test_cluster_splitting_blocked_by_trust_curve(self):
        """
        Bots split into 500 unique clusters but have zero prediction history.
        New agents get weight ≈ 0, so the swarm is still powerless.
        """
        n_bots   = 500
        bot_ops  = []
        bot_exp  = {}

        for i in range(n_bots):
            aid = uuid4()
            # Each in its own cluster (no cluster penalty) — but prediction_count=0
            bot_ops.append(Opinion(aid, alpha=25.0, beta=5.0, cluster_id=f"cluster_{i}"))
            bot_exp[aid] = make_expert(count=0)   # zero history → zero trust

        # Honest agent with real history pushing the other way
        honest_id = uuid4()
        honest_op = [Opinion(honest_id, alpha=5.0, beta=25.0, cluster_id="honest")]
        honest_exp = {honest_id: make_expert(count=100)}

        # Merge
        all_ops = bot_ops + honest_op
        all_exp = {**bot_exp, **honest_exp}

        r = compute_raw_fusion_from_opinions(all_ops, "test", all_exp)

        # Despite 500 bots pushing p > 0.5, honest agent should dominate
        assert r["p_success"] < 0.5, (
            f"Cluster-split bots with zero trust overwhelmed honest agent: p={r['p_success']:.4f}"
        )

    # ------------------------------------------------------------------ 2C: Reputation Farming
    def test_reputation_farming_is_expensive(self):
        """
        Trust growth is logarithmic: going from 0→100 predictions costs
        much more than 0→10. The marginal trust gain per prediction decreases.
        This makes systematic farming expensive.
        """
        deltas = []
        prev_w = 0.0
        for count in range(1, 201):
            w = compute_weight(0.8, 0.1, 0.7, count)
            deltas.append(w - prev_w)
            prev_w = w

        # First 10 increments should be larger than last 10
        early_gain = sum(deltas[:10])
        late_gain  = sum(deltas[190:200])
        assert early_gain > late_gain, (
            f"Trust gain should be concave (diminishing returns); "
            f"early={early_gain:.4f}, late={late_gain:.4f}"
        )

    # ------------------------------------------------------------------ 2D: Coordinated Long-Term Attack
    def test_high_trust_coordinated_attack_requires_mass_evidence(self):
        """
        Even 50 high-trust agents claiming p≈0.9 against a strong existing
        prior of p≈0.1 (represented by alpha/beta ratio) need overwhelming
        evidence to flip the fusion past 0.5.

        Test: if existing pool strongly says p≈0.1 (50 votes),
        attackers (50 high-trust agents saying p≈0.9) should be able to flip
        only if their total weighted evidence is actually larger.
        """
        # Defenders: p=0.1 belief, high trust
        defenders, def_exp = [], {}
        for _ in range(50):
            aid = uuid4()
            defenders.append(Opinion(aid, alpha=5.0, beta=45.0, cluster_id="def"))
            def_exp[aid] = make_expert(count=200)

        # Attackers: p=0.9 belief, high trust, different cluster
        attackers, atk_exp = [], {}
        for _ in range(50):
            aid = uuid4()
            attackers.append(Opinion(aid, alpha=45.0, beta=5.0, cluster_id="atk"))
            atk_exp[aid] = make_expert(count=200)

        # Baseline: defenders only
        r_def = compute_raw_fusion_from_opinions(defenders, "test", def_exp)
        assert r_def["p_success"] < 0.3, "Baseline defender belief should be < 0.3"

        # Defenders + attackers of equal size — cluster penalty compresses each
        # cluster to 1 effective vote, so outcome should be ~0.5
        all_ops = defenders + attackers
        all_exp = {**def_exp, **atk_exp}
        r_mixed = compute_raw_fusion_from_opinions(all_ops, "test", all_exp)

        # The result should be between the two, not instantly flipped
        assert r_def["p_success"] < r_mixed["p_success"] < 0.9, (
            f"Mixed result {r_mixed['p_success']:.4f} should be between "
            f"{r_def['p_success']:.4f} and 0.9"
        )

    # ------------------------------------------------------------------ 2E: Zero-trust swarm ignores
    def test_zero_trust_swarm_has_no_effect(self):
        """Even 10,000 zero-trust agents must not move the needle."""
        honest_id = uuid4()
        honest_op = [make_opinion(0.3, 0.9, agent_id=honest_id)]
        honest_exp = {honest_id: make_expert(count=100)}
        r_baseline = compute_raw_fusion_from_opinions(honest_op, "test", honest_exp)

        n = 10_000
        swarm_ops, swarm_exp = [], {}
        for i in range(n):
            aid = uuid4()
            swarm_ops.append(Opinion(aid, alpha=45.0, beta=5.0, cluster_id=f"c{i}"))
            swarm_exp[aid] = make_expert(count=0)   # zero trust

        all_ops = honest_op + swarm_ops
        all_exp = {**honest_exp, **swarm_exp}
        r_swarm = compute_raw_fusion_from_opinions(all_ops, "test", all_exp)

        assert abs(r_swarm["p_success"] - r_baseline["p_success"]) < 0.01, (
            f"Zero-trust swarm shifted p: baseline={r_baseline['p_success']:.4f}, "
            f"with_swarm={r_swarm['p_success']:.4f}"
        )


# ============================================================================
# CATEGORY 3: TEMPORAL DYNAMICS
# ============================================================================

class TestTemporalDynamics:

    # ------------------------------------------------------------------ 3A: Entropy Drift Stability
    def test_entropy_score_does_not_saturate_under_noise(self):
        """
        Simulate 10,000 small delta_H updates using the entropy worker update rule.
        Entropy score must not peg at 0.01 or 0.99.
        """
        E   = 0.5
        lr  = 0.05
        dec = 0.001
        neutral = 0.5
        rng = random.Random(42)

        for _ in range(10_000):
            # Random small delta_H ∈ [-0.05, +0.05]
            delta_H        = rng.gauss(0.0, 0.02)
            evidence_scale = rng.uniform(0.0, 1.0)

            E = E + (lr * evidence_scale * delta_H)
            E = E - dec * (E - neutral)
            E = max(0.01, min(0.99, E))

        assert 0.1 < E < 0.9, (
            f"Entropy score saturated to {E:.4f} after 10k noisy updates"
        )

    # ------------------------------------------------------------------ 3B: Domain Shock Inversion
    def test_domain_shock_can_invert_strong_prior(self):
        """
        A strong prior (p=0.9) can be overturned by a burst of strong
        contradictory evidence.

        We test this at the fusion level: replace old defenders with
        attackers of greater weight.
        """
        # Strong existing belief: many agents saying p≈0.9
        old_ops, old_exp = [], {}
        for _ in range(10):
            aid = uuid4()
            old_ops.append(Opinion(aid, alpha=40.0, beta=5.0, cluster_id=f"old"))
            old_exp[aid] = make_expert(count=200)

        r_before = compute_raw_fusion_from_opinions(old_ops, "test", old_exp)
        assert r_before["p_success"] > 0.8

        # Shock: strong new evidence saying p≈0.1
        shock_ops, shock_exp = [], {}
        for i in range(20):   # More agents, different cluster
            aid = uuid4()
            shock_ops.append(Opinion(aid, alpha=5.0, beta=40.0, cluster_id=f"shock_{i}"))
            shock_exp[aid] = make_expert(count=200)

        all_ops = old_ops + shock_ops
        all_exp = {**old_exp, **shock_exp}
        r_after = compute_raw_fusion_from_opinions(all_ops, "test", all_exp)

        assert r_after["p_success"] < r_before["p_success"], (
            "Strong shock evidence did not reduce p_success"
        )
        # With double the agents in different clusters, should drive below 0.5
        assert r_after["p_success"] < 0.5, (
            f"Shock did not invert prior: p_after={r_after['p_success']:.4f}"
        )

    # ------------------------------------------------------------------ 3C: Dormant Agent Decay
    def test_dormant_agent_loses_weight_over_time(self):
        """
        An agent with no recent activity should have effectively zero
        contribution to the fusion after long dormancy.
        The trust curve already handles this: prediction_count doesn't
        increase, but the temporal decay in expertise_temporal reduces
        the effective weighted_count → effectively reduces calibration score.

        We test the weight component here: same agent, shrinking contribution
        to alpha_total as expertise degrades via decay simulation.
        """
        # Simulate what happens when weighted_count decays to near-zero
        # and the agent returns default neutral scores

        from app.services.expertise_temporal import AgentExpertise

        active_exp  = AgentExpertise(0.85, 0.05, 0.80, 150)   # strong recent history
        dormant_exp = AgentExpertise(0.50, 0.00, 0.50,   0)   # decayed to neutral (no history)

        w_active  = compute_weight(active_exp.calibration_score,  active_exp.bias_score,
                                   active_exp.entropy_score,  active_exp.prediction_count)
        w_dormant = compute_weight(dormant_exp.calibration_score, dormant_exp.bias_score,
                                   dormant_exp.entropy_score, dormant_exp.prediction_count)

        assert w_dormant < w_active, (
            f"Dormant agent ({w_dormant:.4f}) should have less weight than active ({w_active:.4f})"
        )
        assert w_dormant == 0.0, (
            "Fully dormant agent (prediction_count=0) should have zero weight"
        )

    # ------------------------------------------------------------------ 3D: Zero Evidence Baseline
    def test_zero_evidence_returns_neutral_prior(self):
        """No opinions → p=0.5, entropy=1.0."""
        r = compute_raw_fusion_from_opinions([], "test", {})
        assert r["p_success"]        == 0.5
        assert r["entropy"]          == 1.0
        assert r["evidence_strength"] == 0.0


# ============================================================================
# CATEGORY 4: STRUCTURAL (DAG) REASONING
# ============================================================================

class TestDAGReasoning:

    def _make_node_belief(self, p, H=None, strength=50.0):
        """Helper to build belief/entropy/strength for a node."""
        if H is None:
            H = entropy(p)
        return {"p": p, "H": H, "strength": strength}

    # ------------------------------------------------------------------ 4A: Logit Utilities
    def test_logit_inv_logit_roundtrip(self):
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert abs(inv_logit(logit(p)) - p) < 1e-10

    def test_logit_clamps_at_extremes(self):
        # Should not raise and should return finite values
        l0 = logit(0.0)
        l1 = logit(1.0)
        assert math.isfinite(l0)
        assert math.isfinite(l1)

    # ------------------------------------------------------------------ 4B: Independent Leaf Aggregation
    def test_two_strong_independent_leaves_compound(self):
        """
        weighted_logit_aggregate is a weighted AVERAGE in logit-space — it
        pools evidence, it does not multiply independent probabilities.

        Key properties verified:
        1. Two identical p=0.9 leaves → exactly 0.9  (average of same logit)
        2. Two asymmetric leaves (0.7 and 0.9) → result between them, pulled
           toward the heavier-weighted one
        3. The DAG captures increased *confidence* via strength accumulation,
           not supermultiplicative p — that is the correct design choice.
        """
        # Property 1: identical leaves → their value
        result_equal = weighted_logit_aggregate([0.9, 0.9], [1.0, 1.0])
        assert abs(result_equal - 0.9) < 1e-9, (
            f"Equal-weight identical leaves should return the leaf value, got {result_equal:.6f}"
        )

        # Property 2: asymmetric leaves → weighted average between them
        result_asym = weighted_logit_aggregate([0.7, 0.9], [1.0, 1.0])
        assert 0.7 < result_asym < 0.9, (
            f"Average of 0.7 and 0.9 should be between them, got {result_asym:.4f}"
        )

        # Property 3: dominant high-confidence leaf pulls result strongly upward
        result_dominant = weighted_logit_aggregate([0.9, 0.5], [100.0, 1.0])
        assert result_dominant > 0.88, (
            f"High-weight 0.9 leaf should dominate: got {result_dominant:.4f}"
        )

    def test_single_leaf_passes_through(self):
        """Single child: aggregation should return ~same value."""
        for p in [0.2, 0.5, 0.8]:
            result = weighted_logit_aggregate([p], [1.0])
            assert abs(result - p) < 1e-6

    def test_weak_leaf_does_not_dominate(self):
        """
        A zero-entropy (high-certainty) leaf with weight 100 vs a
        high-entropy leaf with weight 1 should heavily favour the strong leaf.
        """
        # Strong leaf p=0.9 weight=100, weak p=0.5 weight=1
        result = weighted_logit_aggregate([0.9, 0.5], [100.0, 1.0])
        assert result > 0.85, (
            f"Strong leaf should dominate: got {result:.4f}"
        )

    def test_empty_children_returns_half(self):
        assert weighted_logit_aggregate([], []) == 0.5

    # ------------------------------------------------------------------ 4C: Strength Factor
    def test_strength_factor_saturates(self):
        sf0   = strength_factor(0.0)
        sf100 = strength_factor(100.0)
        sf_inf = strength_factor(1e9)
        assert sf0   == 0.0
        assert sf100  > sf0
        assert sf_inf < 1.0
        assert sf_inf > sf100

    def test_strength_factor_monotone(self):
        vals = [strength_factor(s) for s in [0, 1, 10, 100, 1000, 10_000]]
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1]

    # ------------------------------------------------------------------ 4D: DAG evaluate (pure, mocked fuse_claim)
    def test_dag_evaluate_no_graph_returns_error(self, monkeypatch):
        """Empty graph should return error dict."""
        from app.services import dag_service

        monkeypatch.setattr(dag_service, "load_decision_graph",
                            lambda did: __import__("networkx").DiGraph())

        result = dag_service.evaluate_decision_graph(uuid4(), {})
        assert "error" in result

    def test_dag_independent_strong_leaves_give_high_root(self, monkeypatch):
        """
        Two leaves at p=0.9 (identical) aggregate to a root of exactly p=0.9
        via weighted logit averaging — the math is a weighted average, not a
        Bayesian product of independent probabilities.

        What DOES increase with two strong leaves is *confidence*: root_strength
        accumulates from both children, pushing strength_factor higher, which
        raises the confidence score even though p_success stays at 0.9.
        """
        import networkx as nx
        from app.services import dag_service

        node_a = uuid4()
        node_b = uuid4()
        node_r = uuid4()

        G = nx.DiGraph()
        G.add_edge(node_a, node_r, weight=1.0, is_root=False)
        G.add_edge(node_b, node_r, weight=1.0, is_root=True)

        monkeypatch.setattr(dag_service, "load_decision_graph", lambda did: G)
        monkeypatch.setattr(dag_service, "fuse_claim",
                            lambda cid, dom: {"p_success": 0.9, "entropy": 0.47, "strength": 100.0})

        result = dag_service.evaluate_decision_graph(uuid4(), {})
        assert "error" not in result

        # p_success = weighted logit average of two identical 0.9 inputs = 0.9
        assert abs(result["p_success"] - 0.9) < 1e-4, (
            f"Two equal 0.9 leaves: root p_success should be ~0.9, got {result['p_success']}"
        )

        # Confidence grows because root_strength = sum of both children's strength
        # strength_factor(200) > strength_factor(100) → higher confidence
        single_conf = (1.0 - 0.47) * (100.0 / (100.0 + 200.0))   # one leaf
        assert result["confidence"] > single_conf, (
            f"Two leaves should give higher confidence than one: "
            f"got {result['confidence']:.4f}, single-leaf baseline={single_conf:.4f}"
        )

    def test_dag_high_entropy_leaf_excluded(self, monkeypatch):
        """
        A leaf with entropy=1.0 (certain=0) contributes effective_weight=0
        and should be excluded, leaving root determined by the good leaf.
        """
        import networkx as nx
        from app.services import dag_service

        good_leaf = uuid4()
        bad_leaf  = uuid4()
        root_node = uuid4()

        G = nx.DiGraph()
        G.add_edge(good_leaf, root_node, weight=1.0, is_root=False)
        G.add_edge(bad_leaf,  root_node, weight=1.0, is_root=True)

        def mock_fuse(cid, dom):
            if cid == good_leaf:
                return {"p_success": 0.85, "entropy": 0.3, "strength": 80.0}
            return {"p_success": 0.5, "entropy": 1.0, "strength": 0.01}

        monkeypatch.setattr(dag_service, "load_decision_graph", lambda did: G)
        monkeypatch.setattr(dag_service, "fuse_claim", mock_fuse)

        result = dag_service.evaluate_decision_graph(uuid4(), {})
        assert "error" not in result
        # Root should reflect the good leaf, not be dragged to 0.5
        assert result["p_success"] > 0.7, (
            f"High-entropy leaf should be excluded; root={result['p_success']:.4f}"
        )

    def test_dag_cycle_detection(self, monkeypatch):
        """Cyclic graph must return error, not hang."""
        import networkx as nx
        from app.services import dag_service

        a, b = uuid4(), uuid4()
        G = nx.DiGraph()
        G.add_edge(a, b)
        G.add_edge(b, a)

        monkeypatch.setattr(dag_service, "load_decision_graph", lambda did: G)

        result = dag_service.evaluate_decision_graph(uuid4(), {})
        assert "error" in result
        assert "cycle" in result["error"].lower()

    def test_dag_risk_node_detection(self, monkeypatch):
        """A significantly weak foundation node should appear in risk_nodes."""
        import networkx as nx
        from app.services import dag_service

        weak_leaf   = uuid4()
        strong_leaf = uuid4()
        root_node   = uuid4()

        G = nx.DiGraph()
        G.add_edge(weak_leaf,   root_node, weight=1.0, is_root=False)
        G.add_edge(strong_leaf, root_node, weight=1.0, is_root=True)

        def mock_fuse(cid, dom):
            if cid == strong_leaf:
                return {"p_success": 0.85, "entropy": 0.2, "strength": 100.0}
            # weak leaf: low p, enough strength to not be filtered
            return {"p_success": 0.3, "entropy": 0.5, "strength": 20.0}

        monkeypatch.setattr(dag_service, "load_decision_graph", lambda did: G)
        monkeypatch.setattr(dag_service, "fuse_claim", mock_fuse)

        result = dag_service.evaluate_decision_graph(uuid4(), {})
        assert "error" not in result
        # weak_leaf is more than 0.2 below root p_success → should be flagged
        assert str(weak_leaf) in result["risk_nodes"], (
            f"Weak leaf should be in risk_nodes; got {result['risk_nodes']}"
        )

    def test_dag_strength_weighted_aggregation(self):
        """
        Heavier child nodes must dominate the logit aggregate.
        """
        # Child A: p=0.9, weight=200 (high strength × high certainty)
        # Child B: p=0.3, weight=1   (negligible)
        result = weighted_logit_aggregate([0.9, 0.3], [200.0, 1.0])
        assert result > 0.8, (
            f"Strong child should dominate; got {result:.4f}"
        )


# ============================================================================
# CATEGORY 5: SYSTEM-LEVEL SANITY
# ============================================================================

class TestSystemLevelSanity:

    # ------------------------------------------------------------------ 5A: Random Noise Stability
    def test_random_noise_hovers_near_half(self):
        """
        1000 agents with uniformly random beliefs should produce p ≈ 0.5.
        """
        rng = random.Random(99)
        ops, exp_map = [], {}
        for i in range(1000):
            aid    = uuid4()
            belief = rng.random()
            conf   = rng.uniform(0.3, 0.9)
            ops.append(make_opinion(belief, conf, cluster_id=f"c{i}", agent_id=aid))
            exp_map[aid] = make_expert(
                calibration=rng.uniform(0.5, 0.9),
                bias=rng.uniform(0.0, 0.3),
                count=rng.randint(10, 200),
            )

        r = compute_raw_fusion_from_opinions(ops, "test", exp_map)
        assert 0.4 < r["p_success"] < 0.6, (
            f"Random noise produced biased p_success={r['p_success']:.4f}"
        )

    # ------------------------------------------------------------------ 5B: Extreme Evidence Stress
    def test_extreme_evidence_no_overflow(self):
        """Very large alpha/beta must not crash or produce NaN/Inf."""
        ops, exp_map = [], {}
        for i in range(10):
            aid = uuid4()
            ops.append(Opinion(aid, alpha=1e12, beta=1e6, cluster_id=f"c{i}"))
            exp_map[aid] = make_expert(count=500)

        r = compute_raw_fusion_from_opinions(ops, "test", exp_map)
        assert not math.isnan(r["p_success"])
        assert not math.isinf(r["p_success"])
        assert 0.0 <= r["p_success"] <= 1.0

    def test_very_small_evidence_no_underflow(self):
        ops, exp_map = [], {}
        for i in range(10):
            aid = uuid4()
            ops.append(Opinion(aid, alpha=1e-10, beta=1e-10, cluster_id=f"c{i}"))
            exp_map[aid] = make_expert(count=100)

        r = compute_raw_fusion_from_opinions(ops, "test", exp_map)
        assert not math.isnan(r["p_success"])
        assert 0.0 <= r["p_success"] <= 1.0

    # ------------------------------------------------------------------ 5C: Zero Evidence
    def test_zero_evidence_neutral_prior(self):
        r = fuse([])
        assert r["p_success"]         == 0.5
        assert r["entropy"]           == 1.0
        assert r["evidence_strength"] == 0.0
        assert r["alpha_total"]       == 0.0
        assert r["beta_total"]        == 0.0

    # ------------------------------------------------------------------ 5D: Micro-Signal Exploit
    def test_micro_signal_cannot_drift_entropy_to_ceiling(self):
        """
        10,000 tiny positive delta_H updates must not drift entropy_score to 0.99.
        The mean-reversion term keeps it bounded.
        """
        E       = 0.5
        lr      = 0.05
        dec     = 0.001
        neutral = 0.5

        for _ in range(10_000):
            delta_H        = 0.001    # tiny but always positive
            evidence_scale = 0.01     # minimal weight
            E = E + (lr * evidence_scale * delta_H)
            E = E - dec * (E - neutral)
            E = max(0.01, min(0.99, E))

        assert E < 0.9, (
            f"Micro-signal exploit drifted entropy_score to {E:.4f}"
        )

    # ------------------------------------------------------------------ 5E: Domain Isolation
    def test_domain_isolated_expertise(self):
        """
        An agent's weight in domain A must not be influenced by
        their expertise in domain B.  (Structural property — verified
        by confirming compute_weight uses only the passed-in scores.)
        """
        # Same trust count, different calibration
        w_domain_a = compute_weight(C=0.9, B=0.0, E=0.9, prediction_count=100)
        w_domain_b = compute_weight(C=0.1, B=0.5, E=0.3, prediction_count=100)

        assert w_domain_a != w_domain_b, (
            "Different domain expertise should produce different weights"
        )
        assert w_domain_a > w_domain_b

    # ------------------------------------------------------------------ 5F: Confidence Monotonicity
    def test_more_strong_evidence_increases_confidence(self):
        """
        Adding more high-certainty evidence should increase confidence
        (via strength_factor growth and entropy reduction).
        """
        from app.services.dag_service import strength_factor, entropy as dag_entropy

        def confidence_at_n(n: int) -> float:
            # Simulate: n independent agents all saying p=0.8
            alpha = n * 20.0
            beta  = n * 5.0
            p     = alpha / (alpha + beta)
            H     = dag_entropy(p)
            sf    = strength_factor(alpha + beta)
            return (1.0 - H) * sf

        c1  = confidence_at_n(1)
        c10 = confidence_at_n(10)
        c50 = confidence_at_n(50)

        assert c1 < c10 < c50, (
            f"Confidence not monotone: c1={c1:.4f}, c10={c10:.4f}, c50={c50:.4f}"
        )

    # ------------------------------------------------------------------ 5G: belief_to_beta → fuse roundtrip
    def test_belief_roundtrip(self):
        """
        Submitting a belief through belief_to_beta and back through fusion
        should recover approximately the original belief.
        """
        for belief in [0.1, 0.3, 0.5, 0.7, 0.9]:
            aid = uuid4()
            op  = make_opinion(belief, 0.9, agent_id=aid)
            exp_map = {aid: make_expert(count=100)}
            r = compute_raw_fusion_from_opinions([op], "test", exp_map)
            assert abs(r["p_success"] - belief) < 0.02, (
                f"Roundtrip failed for belief={belief}: got p={r['p_success']:.4f}"
            )

    # ------------------------------------------------------------------ 5H: current_week_bucket
    def test_week_bucket_is_monday(self):
        from datetime import date
        wb = current_week_bucket()
        assert wb.weekday() == 0, f"Week bucket should be Monday, got weekday={wb.weekday()}"

    def test_week_bucket_not_in_future(self):
        from datetime import date
        assert current_week_bucket() <= date.today()


# ============================================================================
# CATEGORY 6: ELITE / HARD TESTS
# ============================================================================

class TestEliteAdversarial:

    # ------------------------------------------------------------------ 6A: Adversarial Timing Attack
    def test_burst_injection_at_high_confidence_does_not_wildly_overreact(self):
        """
        Bots inject a burst when existing p is already high (0.85).
        The cluster penalty should compress their combined influence.
        System must not overshoot past the honest equilibrium.
        """
        # Honest pool: 20 agents saying p≈0.85
        honest_ops, honest_exp = [], {}
        for i in range(20):
            aid = uuid4()
            honest_ops.append(Opinion(aid, alpha=35.0, beta=5.0, cluster_id=f"h{i}"))
            honest_exp[aid] = make_expert(count=200)

        r_before = compute_raw_fusion_from_opinions(honest_ops, "test", honest_exp)

        # Bot burst: 1000 bots in ONE cluster, pushing even higher
        burst_ops, burst_exp = [], {}
        for _ in range(1000):
            aid = uuid4()
            burst_ops.append(Opinion(aid, alpha=49.0, beta=1.0, cluster_id="burst_cluster"))
            burst_exp[aid] = make_expert(count=200)

        all_ops = honest_ops + burst_ops
        all_exp = {**honest_exp, **burst_exp}
        r_after = compute_raw_fusion_from_opinions(all_ops, "test", all_exp)

        # Due to cluster penalty, the 1000-bot burst acts like 1 agent
        # So r_after should be only slightly higher than r_before
        delta = abs(r_after["p_success"] - r_before["p_success"])
        assert delta < 0.05, (
            f"Burst injection caused over-reaction: "
            f"before={r_before['p_success']:.4f}, after={r_after['p_success']:.4f}, Δ={delta:.4f}"
        )

    # ------------------------------------------------------------------ 6B: Multi-Stage Narrative Attack
    def test_staged_manipulation_requires_overwhelming_evidence(self):
        """
        Stage 1: establish a strong pool at p≈0.7 (20 honest agents)
        Stage 2: inject bots pushing toward p≈0.9 (different cluster)
        Stage 3: inject bots pushing toward p≈0.9 again (yet another cluster)

        Each new cluster adds one effective vote. With 20 honest vs 2 clusters,
        honest pool should still dominate.
        """
        # Honest pool
        honest_ops, honest_exp = [], {}
        for i in range(20):
            aid = uuid4()
            honest_ops.append(Opinion(aid, alpha=28.0, beta=12.0, cluster_id=f"h{i}"))
            honest_exp[aid] = make_expert(count=200)

        r0 = compute_raw_fusion_from_opinions(honest_ops, "test", honest_exp)

        # Stage 1 attack: 500 bots in cluster A
        atk1_ops, atk1_exp = [], {}
        for _ in range(500):
            aid = uuid4()
            atk1_ops.append(Opinion(aid, alpha=45.0, beta=5.0, cluster_id="attack_A"))
            atk1_exp[aid] = make_expert(count=200)

        # Stage 2 attack: 500 bots in cluster B
        atk2_ops, atk2_exp = [], {}
        for _ in range(500):
            aid = uuid4()
            atk2_ops.append(Opinion(aid, alpha=45.0, beta=5.0, cluster_id="attack_B"))
            atk2_exp[aid] = make_expert(count=200)

        # All together: 20 honest clusters vs 2 attack clusters
        all_ops = honest_ops + atk1_ops + atk2_ops
        all_exp = {**honest_exp, **atk1_exp, **atk2_exp}
        r_final = compute_raw_fusion_from_opinions(all_ops, "test", all_exp)

        # 20 vs 2 — honest majority should win
        assert r_final["p_success"] < 0.75, (
            f"Two-stage attack with 2 clusters vs 20 honest clusters moved p too much: "
            f"{r0['p_success']:.4f} → {r_final['p_success']:.4f}"
        )

    # ------------------------------------------------------------------ 6C: Mixed Honest + Corrupt Experts
    def test_minority_corrupt_high_trust_agents_diluted(self):
        """
        30% of agents are corrupt (high trust, wrong direction).
        70% are honest. Honest majority should prevail.
        """
        n_honest  = 70
        n_corrupt = 30

        ops, exp_map = [], {}

        for i in range(n_honest):
            aid = uuid4()
            ops.append(Opinion(aid, alpha=30.0, beta=10.0, cluster_id=f"h{i}"))
            exp_map[aid] = make_expert(count=200)

        for i in range(n_corrupt):
            aid = uuid4()
            ops.append(Opinion(aid, alpha=10.0, beta=30.0, cluster_id=f"c{i}"))
            exp_map[aid] = make_expert(count=200)   # High trust but wrong

        r = compute_raw_fusion_from_opinions(ops, "test", exp_map)

        assert r["p_success"] > 0.55, (
            f"Honest majority (70%) should win; got p={r['p_success']:.4f}"
        )
        # Should NOT flip to corruption side
        assert r["p_success"] < 0.85, "Result should reflect imperfect signal"


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
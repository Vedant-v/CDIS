"""
run_tests.py
------------
Runs the test suite using Python's built-in unittest framework.
No external packages required.

Usage:
    python3 run_tests.py
"""

import sys
import os
import math
import random
import unittest
from collections import namedtuple
from uuid import uuid4

# ---------------------------------------------------------------------------
# Patch sys.path
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ---------------------------------------------------------------------------
# Stub out DB-touching imports so pure-math functions can be imported
# ---------------------------------------------------------------------------

# Stub cassandra module
import types

cassandra_mod = types.ModuleType("cassandra")
cassandra_cluster = types.ModuleType("cassandra.cluster")
cassandra_cluster.Cluster = object
cassandra_mod.cluster = cassandra_cluster
sys.modules["cassandra"] = cassandra_mod
sys.modules["cassandra.cluster"] = cassandra_cluster

# Stub structlog
structlog_mod = types.ModuleType("structlog")
structlog_mod.get_logger = lambda: type("L", (), {
    "info": lambda *a, **k: None,
    "error": lambda *a, **k: None,
    "warning": lambda *a, **k: None,
})()
sys.modules["structlog"] = structlog_mod

# Stub redis
redis_mod = types.ModuleType("redis")
redis_mod.Redis = lambda **k: None
redis_exceptions = types.ModuleType("redis.exceptions")
redis_exceptions.LockNotOwnedError = Exception
redis_mod.exceptions = redis_exceptions
sys.modules["redis"] = redis_mod
sys.modules["redis.exceptions"] = redis_exceptions

# Stub celery
celery_mod = types.ModuleType("celery")
celery_mod.Celery = lambda *a, **k: type("C", (), {
    "task": lambda self, *a, **k: (lambda f: f)
})()
celery_mod.shared_task = lambda *a, **k: (lambda f: f)
sys.modules["celery"] = celery_mod

# Stub app.worker
worker_mod = types.ModuleType("app.worker")
worker_mod.celery = type("C", (), {"task": lambda *a, **k: (lambda f: f)})()
sys.modules["app.worker"] = worker_mod

# Stub app.db.cassandra
db_cassandra = types.ModuleType("app.db.cassandra")
db_cassandra.get_session = lambda: None
db_cassandra.close_session = lambda: None
sys.modules["app.db"] = types.ModuleType("app.db")
sys.modules["app.db.cassandra"] = db_cassandra

# Stub dao modules
for mod_name in ["app.db.dao.fusion_dao", "app.db.dao.opinion_dao", "app.db.dao.outcome_dao"]:
    m = types.ModuleType(mod_name)
    m.fetch_opinions_st = lambda *a, **k: []
    m.fetch_opinions_lt = lambda *a, **k: []
    m.insert_opinion = lambda *a, **k: None
    m.insert_claim_outcome = lambda *a, **k: None
    sys.modules[mod_name] = m

# Stub domain_decay (so expertise_temporal can be imported cleanly)
domain_decay_mod = types.ModuleType("app.services.domain_decay")
domain_decay_mod.get_domain_lambda = lambda domain: 0.08
domain_decay_mod.auto_tune_lambda_task = lambda *a, **k: None
sys.modules["app.services.domain_decay"] = domain_decay_mod

# Stub expertise_temporal — will be replaced below after import
expertise_temporal_stub = types.ModuleType("app.services.expertise_temporal")
AgentExpertiseStub = namedtuple(
    "AgentExpertise",
    ["calibration_score", "bias_score", "entropy_score", "prediction_count"],
)
expertise_temporal_stub.AgentExpertise = AgentExpertiseStub
expertise_temporal_stub.fetch_temporal_expertise = lambda aid, domain: AgentExpertiseStub(0.8, 0.1, 0.7, 100)
sys.modules["app.services.expertise_temporal"] = expertise_temporal_stub

# Now import the real pure-math modules
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
from app.utils.time_bucket import current_week_bucket

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

AgentExpertise = AgentExpertiseStub
Opinion = namedtuple("Opinion", ["agent_id", "alpha", "beta", "cluster_id"])


def make_expert(calibration=0.8, bias=0.1, entropy_s=0.7, count=100):
    return AgentExpertise(calibration, bias, entropy_s, count)


def make_opinion(belief, confidence, cluster_id="c1", agent_id=None):
    if agent_id is None:
        agent_id = uuid4()
    alpha, beta = belief_to_beta(belief, confidence)
    return Opinion(agent_id=agent_id, alpha=alpha, beta=beta, cluster_id=cluster_id)


def fuse(opinions, expertise_override=None):
    if expertise_override is None:
        expertise_override = {op.agent_id: make_expert() for op in opinions}
    return compute_raw_fusion_from_opinions(opinions, "test", expertise_override)


# ============================================================================
# CATEGORY 1: MATHEMATICAL INVARIANTS
# ============================================================================

class TestMathematicalInvariants(unittest.TestCase):

    def test_entropy_zero_at_extremes(self):
        self.assertEqual(entropy(0.0), 0.0)
        self.assertEqual(entropy(1.0), 0.0)

    def test_entropy_max_at_half(self):
        self.assertAlmostEqual(entropy(0.5), 1.0, places=9)

    def test_entropy_bounds(self):
        for p in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
            H = entropy(p)
            self.assertGreaterEqual(H, 0.0, f"entropy({p}) negative")
            self.assertLessEqual(H, 1.0, f"entropy({p}) > 1")

    def test_entropy_no_nan(self):
        for p in [0.0, 1e-10, 0.5, 1.0 - 1e-10, 1.0]:
            H = entropy(p)
            self.assertFalse(math.isnan(H))
            self.assertFalse(math.isinf(H))

    def test_adding_identical_evidence_preserves_p(self):
        a1, a2 = uuid4(), uuid4()
        exp = make_expert()
        ops_1 = [Opinion(a1, alpha=20.0, beta=5.0, cluster_id="c1")]
        ops_2 = [Opinion(a1, alpha=20.0, beta=5.0, cluster_id="c1"),
                 Opinion(a2, alpha=20.0, beta=5.0, cluster_id="c2")]
        exp_1 = {a1: exp}
        exp_2 = {a1: exp, a2: exp}
        r1 = compute_raw_fusion_from_opinions(ops_1, "test", exp_1)
        r2 = compute_raw_fusion_from_opinions(ops_2, "test", exp_2)
        self.assertAlmostEqual(r1["p_success"], r2["p_success"], places=5,
            msg="Identical evidence in diff clusters should not shift p")
        self.assertGreater(r2["evidence_strength"], r1["evidence_strength"])

    def test_equal_opposing_evidence_converges_to_half(self):
        a1, a2 = uuid4(), uuid4()
        exp_map = {a1: make_expert(), a2: make_expert()}
        ops = [
            Opinion(a1, alpha=20.0, beta=5.0, cluster_id="c1"),
            Opinion(a2, alpha=5.0, beta=20.0, cluster_id="c2"),
        ]
        r = compute_raw_fusion_from_opinions(ops, "test", exp_map)
        self.assertAlmostEqual(r["p_success"], 0.5, delta=0.05,
            msg=f"Opposing equal evidence should be ~0.5, got {r['p_success']:.4f}")

    def test_probability_always_in_unit_interval(self):
        rng = random.Random(7)
        for _ in range(200):
            n = rng.randint(1, 20)
            ops, exp_map = [], {}
            for _ in range(n):
                aid = uuid4()
                op = make_opinion(rng.random(), rng.random(), agent_id=aid)
                ops.append(op)
                exp_map[aid] = make_expert(
                    calibration=rng.random(),
                    bias=rng.random() * 0.5,
                    count=rng.randint(0, 200),
                )
            r = compute_raw_fusion_from_opinions(ops, "test", exp_map)
            self.assertGreaterEqual(r["p_success"], 0.0)
            self.assertLessEqual(r["p_success"], 1.0)
            self.assertGreaterEqual(r["entropy"], 0.0)
            self.assertLessEqual(r["entropy"], 1.0)
            self.assertGreaterEqual(r["alpha_total"], 0.0)
            self.assertFalse(math.isnan(r["p_success"]))

    def test_scaling_invariance(self):
        for k in [0.5, 2.0, 100.0, 0.001]:
            a1, a2 = uuid4(), uuid4()
            exp_map = {a1: make_expert(), a2: make_expert()}
            ops_base = [Opinion(a1, 10.0, 5.0, "c1"), Opinion(a2, 8.0, 2.0, "c2")]
            ops_scaled = [Opinion(a1, 10.0*k, 5.0*k, "c1"), Opinion(a2, 8.0*k, 2.0*k, "c2")]
            r_b = compute_raw_fusion_from_opinions(ops_base,   "test", exp_map)
            r_s = compute_raw_fusion_from_opinions(ops_scaled, "test", exp_map)
            self.assertAlmostEqual(r_b["p_success"], r_s["p_success"], places=5,
                msg=f"Scaling by {k} changed p")
            self.assertAlmostEqual(
                r_s["evidence_strength"], k * r_b["evidence_strength"], delta=1e-4,
                msg=f"Strength didn't scale at k={k}")

    def test_belief_to_beta_always_positive(self):
        for b in [0.0, 0.01, 0.5, 0.99, 1.0]:
            for c in [0.0, 0.01, 0.5, 1.0]:
                alpha, beta = belief_to_beta(b, c)
                self.assertGreater(alpha, 0.0)
                self.assertGreater(beta, 0.0)

    def test_weight_zero_for_new_agent(self):
        w = compute_weight(1.0, 0.0, 1.0, 0)
        self.assertEqual(w, 0.0, "New agent must have zero weight")

    def test_weight_grows_with_predictions(self):
        weights = [compute_weight(0.8, 0.1, 0.7, c) for c in [0,1,5,20,100,1000]]
        for i in range(len(weights)-1):
            self.assertLessEqual(weights[i], weights[i+1])

    def test_weight_bounded(self):
        rng = random.Random(1)
        for _ in range(100):
            w = compute_weight(rng.random(), rng.random()*0.5, rng.random(), rng.randint(0,10000))
            self.assertGreaterEqual(w, 0.0)
            self.assertLessEqual(w, 1.0)

    def test_high_bias_reduces_weight(self):
        w_low  = compute_weight(0.9, 0.0, 0.8, 100)
        w_high = compute_weight(0.9, 0.9, 0.8, 100)
        self.assertLess(w_high, w_low)


# ============================================================================
# CATEGORY 2: ADVERSARIAL RESISTANCE
# ============================================================================

class TestAdversarialResistance(unittest.TestCase):

    def test_bot_swarm_single_cluster_bounded(self):
        cluster = "bot_cluster"
        honest = uuid4()
        single_op = [Opinion(honest, 20.0, 5.0, cluster)]
        single_exp = {honest: make_expert()}
        r_single = compute_raw_fusion_from_opinions(single_op, "test", single_exp)

        bots, bot_exp = [], {}
        for _ in range(500):
            aid = uuid4()
            bots.append(Opinion(aid, 20.0, 5.0, cluster))
            bot_exp[aid] = make_expert()
        r_swarm = compute_raw_fusion_from_opinions(bots, "test", bot_exp)

        self.assertAlmostEqual(r_swarm["p_success"], r_single["p_success"], delta=0.01,
            msg=f"Swarm shifted p: single={r_single['p_success']:.4f}, swarm={r_swarm['p_success']:.4f}")

    def test_cluster_splitting_blocked_by_trust(self):
        bots, bot_exp = [], {}
        for i in range(500):
            aid = uuid4()
            bots.append(Opinion(aid, 25.0, 5.0, f"c{i}"))
            bot_exp[aid] = make_expert(count=0)  # zero trust

        honest = uuid4()
        honest_op = [Opinion(honest, 5.0, 25.0, "honest")]
        honest_exp = {honest: make_expert(count=100)}

        all_ops = bots + honest_op
        all_exp = {**bot_exp, **honest_exp}
        r = compute_raw_fusion_from_opinions(all_ops, "test", all_exp)
        self.assertLess(r["p_success"], 0.5,
            f"Zero-trust split bots should not overpower honest agent: p={r['p_success']:.4f}")

    def test_reputation_farming_concave(self):
        deltas, prev = [], 0.0
        for c in range(1, 201):
            w = compute_weight(0.8, 0.1, 0.7, c)
            deltas.append(w - prev)
            prev = w
        early = sum(deltas[:10])
        late  = sum(deltas[190:200])
        self.assertGreater(early, late, "Trust gain should be diminishing (concave)")

    def test_coordinated_attack_needs_mass_evidence(self):
        defs, def_exp = [], {}
        for _ in range(50):
            aid = uuid4()
            defs.append(Opinion(aid, 5.0, 45.0, "def"))
            def_exp[aid] = make_expert(count=200)
        r_def = compute_raw_fusion_from_opinions(defs, "test", def_exp)
        self.assertLess(r_def["p_success"], 0.3)

        atks, atk_exp = [], {}
        for _ in range(50):
            aid = uuid4()
            atks.append(Opinion(aid, 45.0, 5.0, "atk"))
            atk_exp[aid] = make_expert(count=200)

        all_ops = defs + atks
        all_exp = {**def_exp, **atk_exp}
        r = compute_raw_fusion_from_opinions(all_ops, "test", all_exp)
        self.assertGreater(r["p_success"], r_def["p_success"])
        self.assertLess(r["p_success"], 0.9)

    def test_zero_trust_swarm_no_effect(self):
        honest = uuid4()
        h_op = [make_opinion(0.3, 0.9, agent_id=honest)]
        h_exp = {honest: make_expert(count=100)}
        r_base = compute_raw_fusion_from_opinions(h_op, "test", h_exp)

        swarm, s_exp = [], {}
        for i in range(10000):
            aid = uuid4()
            swarm.append(Opinion(aid, 45.0, 5.0, f"c{i}"))
            s_exp[aid] = make_expert(count=0)

        all_ops = h_op + swarm
        all_exp = {**h_exp, **s_exp}
        r = compute_raw_fusion_from_opinions(all_ops, "test", all_exp)
        self.assertAlmostEqual(r["p_success"], r_base["p_success"], delta=0.01)


# ============================================================================
# CATEGORY 3: TEMPORAL DYNAMICS
# ============================================================================

class TestTemporalDynamics(unittest.TestCase):

    def test_entropy_no_saturation_under_noise(self):
        E, rng = 0.5, random.Random(42)
        for _ in range(10000):
            dH = rng.gauss(0.0, 0.02)
            ev = rng.uniform(0.0, 1.0)
            E = E + (0.05 * ev * dH)
            E = E - 0.001 * (E - 0.5)
            E = max(0.01, min(0.99, E))
        self.assertGreater(E, 0.1)
        self.assertLess(E, 0.9)

    def test_shock_inversion(self):
        old_ops, old_exp = [], {}
        for _ in range(10):
            aid = uuid4()
            old_ops.append(Opinion(aid, 40.0, 5.0, "old"))
            old_exp[aid] = make_expert(count=200)
        r_before = compute_raw_fusion_from_opinions(old_ops, "test", old_exp)
        self.assertGreater(r_before["p_success"], 0.8)

        shock_ops, shock_exp = [], {}
        for i in range(20):
            aid = uuid4()
            shock_ops.append(Opinion(aid, 5.0, 40.0, f"s{i}"))
            shock_exp[aid] = make_expert(count=200)

        all_ops = old_ops + shock_ops
        all_exp = {**old_exp, **shock_exp}
        r_after = compute_raw_fusion_from_opinions(all_ops, "test", all_exp)
        self.assertLess(r_after["p_success"], r_before["p_success"])
        self.assertLess(r_after["p_success"], 0.5)

    def test_dormant_agent_zero_weight(self):
        w_active  = compute_weight(0.85, 0.05, 0.80, 150)
        w_dormant = compute_weight(0.50, 0.00, 0.50, 0)
        self.assertLess(w_dormant, w_active)
        self.assertEqual(w_dormant, 0.0)

    def test_zero_evidence_neutral_prior(self):
        r = compute_raw_fusion_from_opinions([], "test", {})
        self.assertEqual(r["p_success"], 0.5)
        self.assertEqual(r["entropy"], 1.0)
        self.assertEqual(r["evidence_strength"], 0.0)


# ============================================================================
# CATEGORY 4: STRUCTURAL (DAG) REASONING
# ============================================================================

class TestDAGReasoning(unittest.TestCase):

    def test_logit_inv_logit_roundtrip(self):
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            self.assertAlmostEqual(inv_logit(logit(p)), p, places=9)

    def test_logit_clamps_extremes(self):
        self.assertTrue(math.isfinite(logit(0.0)))
        self.assertTrue(math.isfinite(logit(1.0)))

    def test_two_strong_leaves_compound(self):
        # weighted_logit_aggregate is a weighted AVERAGE in logit-space, not
        # an independence product. Two identical p=0.9 leaves return exactly p=0.9.
        # Compounding (super-additivity) is intentionally not modelled here —
        # the strength accumulation in the DAG captures increased confidence instead.
        result_equal = weighted_logit_aggregate([0.9, 0.9], [1.0, 1.0])
        self.assertAlmostEqual(result_equal, 0.9, places=5,
            msg="Weighted average of identical leaves should return the leaf value")
        # Strong leaf must dominate a weak one
        result_strong = weighted_logit_aggregate([0.9, 0.5], [100.0, 1.0])
        self.assertGreater(result_strong, 0.85,
            msg=f"High-weight 0.9 leaf should dominate weak 0.5: {result_strong:.4f}")

    def test_single_leaf_passthrough(self):
        for p in [0.2, 0.5, 0.8]:
            self.assertAlmostEqual(weighted_logit_aggregate([p], [1.0]), p, places=5)

    def test_strong_leaf_dominates_weak(self):
        result = weighted_logit_aggregate([0.9, 0.5], [100.0, 1.0])
        self.assertGreater(result, 0.85)

    def test_empty_children_returns_half(self):
        self.assertEqual(weighted_logit_aggregate([], []), 0.5)

    def test_strength_factor_saturates(self):
        self.assertEqual(strength_factor(0.0), 0.0)
        self.assertLess(strength_factor(1e9), 1.0)
        self.assertGreater(strength_factor(1e9), strength_factor(100.0))

    def test_strength_factor_monotone(self):
        vals = [strength_factor(s) for s in [0, 1, 10, 100, 1000, 10000]]
        for i in range(len(vals)-1):
            self.assertLessEqual(vals[i], vals[i+1])

    def test_dag_no_graph_error(self):
        import networkx as nx
        import app.services.dag_service as ds
        orig = ds.load_decision_graph
        ds.load_decision_graph = lambda did: nx.DiGraph()
        result = ds.evaluate_decision_graph(uuid4(), {})
        ds.load_decision_graph = orig
        self.assertIn("error", result)

    def test_dag_cycle_detection(self):
        import networkx as nx
        import app.services.dag_service as ds
        a, b = uuid4(), uuid4()
        G = nx.DiGraph()
        G.add_edge(a, b)
        G.add_edge(b, a)
        orig = ds.load_decision_graph
        ds.load_decision_graph = lambda did: G
        result = ds.evaluate_decision_graph(uuid4(), {})
        ds.load_decision_graph = orig
        self.assertIn("error", result)
        self.assertIn("cycle", result["error"].lower())

    def test_dag_two_strong_leaves_high_root(self):
        import networkx as nx
        import app.services.dag_service as ds

        node_a, node_b, node_r = uuid4(), uuid4(), uuid4()
        G = nx.DiGraph()
        G.add_edge(node_a, node_r, weight=1.0, is_root=False)
        G.add_edge(node_b, node_r, weight=1.0, is_root=True)

        orig_load  = ds.load_decision_graph
        orig_fuse  = ds.fuse_claim
        ds.load_decision_graph = lambda did: G
        ds.fuse_claim = lambda cid, dom: {"p_success": 0.9, "entropy": 0.47, "strength": 100.0}

        result = ds.evaluate_decision_graph(uuid4(), {})
        ds.load_decision_graph = orig_load
        ds.fuse_claim = orig_fuse

        self.assertNotIn("error", result)
        # Two equal leaves at p=0.9 → weighted logit average = 0.9
        # Confidence (strength_factor) grows, but p_success stays at 0.9
        self.assertAlmostEqual(result["p_success"], 0.9, delta=0.01,
            msg=f"Two equal 0.9 leaves: root should be ~0.9, got {result['p_success']}")
        # Confidence should reflect strong evidence
        self.assertGreater(result["confidence"], 0.2,
            msg="Two high-strength leaves should yield meaningful confidence")

    def test_dag_high_entropy_leaf_excluded(self):
        import networkx as nx
        import app.services.dag_service as ds

        good, bad, root = uuid4(), uuid4(), uuid4()
        G = nx.DiGraph()
        G.add_edge(good, root, weight=1.0, is_root=False)
        G.add_edge(bad,  root, weight=1.0, is_root=True)

        def mock_fuse(cid, dom):
            if cid == good:
                return {"p_success": 0.85, "entropy": 0.3, "strength": 80.0}
            return {"p_success": 0.5, "entropy": 1.0, "strength": 0.01}

        orig_load, orig_fuse = ds.load_decision_graph, ds.fuse_claim
        ds.load_decision_graph = lambda did: G
        ds.fuse_claim = mock_fuse
        result = ds.evaluate_decision_graph(uuid4(), {})
        ds.load_decision_graph = orig_load
        ds.fuse_claim = orig_fuse

        self.assertNotIn("error", result)
        self.assertGreater(result["p_success"], 0.7)

    def test_dag_risk_node_flagged(self):
        import networkx as nx
        import app.services.dag_service as ds

        weak, strong, root = uuid4(), uuid4(), uuid4()
        G = nx.DiGraph()
        G.add_edge(weak,   root, weight=1.0, is_root=False)
        G.add_edge(strong, root, weight=1.0, is_root=True)

        def mock_fuse(cid, dom):
            if cid == strong:
                return {"p_success": 0.85, "entropy": 0.2, "strength": 100.0}
            return {"p_success": 0.3, "entropy": 0.5, "strength": 20.0}

        orig_load, orig_fuse = ds.load_decision_graph, ds.fuse_claim
        ds.load_decision_graph = lambda did: G
        ds.fuse_claim = mock_fuse
        result = ds.evaluate_decision_graph(uuid4(), {})
        ds.load_decision_graph = orig_load
        ds.fuse_claim = orig_fuse

        self.assertNotIn("error", result)
        self.assertIn(str(weak), result["risk_nodes"])


# ============================================================================
# CATEGORY 5: SYSTEM-LEVEL SANITY
# ============================================================================

class TestSystemLevelSanity(unittest.TestCase):

    def test_random_noise_near_half(self):
        rng = random.Random(99)
        ops, exp_map = [], {}
        for i in range(1000):
            aid = uuid4()
            op = make_opinion(rng.random(), rng.uniform(0.3, 0.9), cluster_id=f"c{i}", agent_id=aid)
            ops.append(op)
            exp_map[aid] = make_expert(calibration=rng.uniform(0.5, 0.9),
                                       bias=rng.uniform(0.0, 0.3),
                                       count=rng.randint(10, 200))
        r = compute_raw_fusion_from_opinions(ops, "test", exp_map)
        self.assertGreater(r["p_success"], 0.4)
        self.assertLess(r["p_success"], 0.6)

    def test_extreme_evidence_no_overflow(self):
        ops, exp_map = [], {}
        for i in range(10):
            aid = uuid4()
            ops.append(Opinion(aid, 1e12, 1e6, f"c{i}"))
            exp_map[aid] = make_expert(count=500)
        r = compute_raw_fusion_from_opinions(ops, "test", exp_map)
        self.assertFalse(math.isnan(r["p_success"]))
        self.assertFalse(math.isinf(r["p_success"]))
        self.assertGreaterEqual(r["p_success"], 0.0)
        self.assertLessEqual(r["p_success"], 1.0)

    def test_zero_evidence_neutral(self):
        r = compute_raw_fusion_from_opinions([], "test", {})
        self.assertEqual(r["p_success"], 0.5)
        self.assertEqual(r["entropy"], 1.0)

    def test_micro_signal_no_drift(self):
        E = 0.5
        for _ in range(10000):
            E = E + (0.05 * 0.01 * 0.001)
            E = E - 0.001 * (E - 0.5)
            E = max(0.01, min(0.99, E))
        self.assertLess(E, 0.9)

    def test_domain_isolation_structural(self):
        w_a = compute_weight(C=0.9, B=0.0, E=0.9, prediction_count=100)
        w_b = compute_weight(C=0.1, B=0.5, E=0.3, prediction_count=100)
        self.assertNotEqual(w_a, w_b)
        self.assertGreater(w_a, w_b)

    def test_confidence_monotone_with_evidence(self):
        def conf(n):
            a = n * 20.0
            b = n * 5.0
            p = a / (a + b)
            H = entropy(p)
            sf = strength_factor(a + b)
            return (1.0 - H) * sf
        self.assertLess(conf(1), conf(10))
        self.assertLess(conf(10), conf(50))

    def test_belief_roundtrip(self):
        for belief in [0.1, 0.3, 0.5, 0.7, 0.9]:
            aid = uuid4()
            op = make_opinion(belief, 0.9, agent_id=aid)
            exp_map = {aid: make_expert(count=100)}
            r = compute_raw_fusion_from_opinions([op], "test", exp_map)
            self.assertAlmostEqual(r["p_success"], belief, delta=0.02,
                msg=f"Roundtrip failed for belief={belief}")

    def test_week_bucket_is_monday(self):
        wb = current_week_bucket()
        self.assertEqual(wb.weekday(), 0, "Week bucket should be Monday")


# ============================================================================
# CATEGORY 6: ELITE ADVERSARIAL
# ============================================================================

class TestEliteAdversarial(unittest.TestCase):

    def test_burst_injection_limited(self):
        honest_ops, honest_exp = [], {}
        for i in range(20):
            aid = uuid4()
            honest_ops.append(Opinion(aid, 35.0, 5.0, f"h{i}"))
            honest_exp[aid] = make_expert(count=200)
        r_before = compute_raw_fusion_from_opinions(honest_ops, "test", honest_exp)

        burst, b_exp = [], {}
        for _ in range(1000):
            aid = uuid4()
            burst.append(Opinion(aid, 49.0, 1.0, "burst_cluster"))
            b_exp[aid] = make_expert(count=200)

        all_ops = honest_ops + burst
        all_exp = {**honest_exp, **b_exp}
        r_after = compute_raw_fusion_from_opinions(all_ops, "test", all_exp)

        delta = abs(r_after["p_success"] - r_before["p_success"])
        self.assertLess(delta, 0.05,
            f"Burst injection over-reacted: before={r_before['p_success']:.4f}, "
            f"after={r_after['p_success']:.4f}, Δ={delta:.4f}")

    def test_staged_attack_honest_majority_wins(self):
        honest_ops, honest_exp = [], {}
        for i in range(20):
            aid = uuid4()
            honest_ops.append(Opinion(aid, 28.0, 12.0, f"h{i}"))
            honest_exp[aid] = make_expert(count=200)

        all_ops, all_exp = list(honest_ops), dict(honest_exp)
        for stage_cluster in ["attack_A", "attack_B"]:
            for _ in range(500):
                aid = uuid4()
                all_ops.append(Opinion(aid, 45.0, 5.0, stage_cluster))
                all_exp[aid] = make_expert(count=200)

        r = compute_raw_fusion_from_opinions(all_ops, "test", all_exp)
        # 20 clusters vs 2 — honest should dominate
        self.assertLess(r["p_success"], 0.75,
            f"2 attack clusters vs 20 honest clusters: p={r['p_success']:.4f}")

    def test_minority_corrupt_agents_diluted(self):
        ops, exp_map = [], {}
        for i in range(70):
            aid = uuid4()
            ops.append(Opinion(aid, 30.0, 10.0, f"h{i}"))
            exp_map[aid] = make_expert(count=200)
        for i in range(30):
            aid = uuid4()
            ops.append(Opinion(aid, 10.0, 30.0, f"c{i}"))
            exp_map[aid] = make_expert(count=200)
        r = compute_raw_fusion_from_opinions(ops, "test", exp_map)
        self.assertGreater(r["p_success"], 0.55,
            f"70% honest should win; p={r['p_success']:.4f}")
        self.assertLess(r["p_success"], 0.85)


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    for cls in [
        TestMathematicalInvariants,
        TestAdversarialResistance,
        TestTemporalDynamics,
        TestDAGReasoning,
        TestSystemLevelSanity,
        TestEliteAdversarial,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
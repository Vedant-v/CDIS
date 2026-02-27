"""
dag_service.py
--------------
Decision Graph evaluation layer.

A decision graph is a DAG where:
  - Foundation nodes (in-degree 0) are terminal claims backed by
    direct agent opinions (fetched via fuse_claim CQRS read).
  - Internal nodes aggregate their children via weighted logit fusion.
  - One root node (out-degree 0, or flagged is_decision_root) represents
    the overall decision claim whose probability we want to evaluate.

Propagation is topological (bottom-up): leaves first, root last.
"""

import networkx as nx
from uuid import UUID
from math import log, exp
from typing import Dict, Any, List

from app.db.cassandra import get_session
from app.services.fusion_service import fuse_claim, entropy


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_GRAPH_STRENGTH   = 1_000.0   # Clamp accumulated strength (prevents overflow)
STRENGTH_CONF_SCALE  = 200.0     # Half-saturation constant for confidence
MIN_RISK_STRENGTH    = 5.0       # Ignore near-empty foundation nodes in risk scan
RISK_GAP_THRESHOLD   = 0.2       # Flag foundation nodes weaker than root by this margin


# ---------------------------------------------------------------------------
# Graph Loading
# ---------------------------------------------------------------------------

def load_decision_graph(decision_id: UUID) -> nx.DiGraph:
    """
    Load the decision sub-graph from Cassandra.
    Edges are stored as (child → parent) direction.
    """
    rows = get_session().execute(
        """
        SELECT parent_claim_id, child_claim_id, weight, is_decision_root
        FROM decision_claim_edges
        WHERE decision_id = %s
        """,
        (decision_id,),
    )

    G = nx.DiGraph()
    for row in rows:
        G.add_edge(
            row.child_claim_id,
            row.parent_claim_id,
            weight=row.weight if row.weight is not None else 1.0,
            is_root=bool(row.is_decision_root) if row.is_decision_root is not None else False,
        )

    return G


# ---------------------------------------------------------------------------
# Numeric Utilities  (natural-log logit — different base from entropy bits,
# but these are used purely for aggregation, not compared to entropy values)
# ---------------------------------------------------------------------------

def logit(p: float) -> float:
    """Natural-log logit with epsilon clamp."""
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return log(p / (1.0 - p))


def inv_logit(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))


def weighted_logit_aggregate(child_ps: List[float], weights: List[float]) -> float:
    """
    Weighted average in logit-space, then mapped back via sigmoid.

    This is mathematically sounder than averaging raw probabilities:
    two independent p=0.9 claims should compound above 0.9, not stay at 0.9.
    """
    if not weights:
        return 0.5

    total_w = sum(weights)
    if total_w <= 0.0:
        return 0.5

    total = sum(w * logit(p) / total_w for p, w in zip(child_ps, weights))
    return inv_logit(total)


def strength_factor(strength: float) -> float:
    """Saturating sigmoid: maps [0, ∞) → [0, 1)."""
    return strength / (strength + STRENGTH_CONF_SCALE)


# ---------------------------------------------------------------------------
# Core DAG Evaluation
# ---------------------------------------------------------------------------

def evaluate_decision_graph(
    decision_id: UUID,
    domain_map: Dict[UUID, str],
) -> Dict[str, Any]:
    """
    Evaluate a decision graph and return:
      - p_success   : root-node probability
      - confidence  : (1 - entropy) × strength_factor(root_strength)
      - risk_nodes  : foundation nodes significantly weaker than the root
      - beliefs     : per-node probability map
    """
    G = load_decision_graph(decision_id)

    if len(G) == 0:
        return {"error": "no decision graph found"}

    try:
        topo_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        return {"error": "cycle detected in decision graph"}

    beliefs:    Dict[UUID, float] = {}
    entropies:  Dict[UUID, float] = {}
    strengths:  Dict[UUID, float] = {}

    # ----------------------------------------------------------------
    # Forward (bottom-up) propagation
    # ----------------------------------------------------------------
    for node in topo_order:
        predecessors = list(G.predecessors(node))

        # Foundation node — backed directly by agent opinions
        if not predecessors:
            domain = domain_map.get(node, "general")
            result = fuse_claim(node, domain)

            beliefs[node]   = result["p_success"]
            entropies[node] = result["entropy"]
            strengths[node] = min(result.get("strength", 1.0), MAX_GRAPH_STRENGTH)
            continue

        # Internal node — aggregate children
        child_ps:    List[float] = []
        child_ws:    List[float] = []
        accum_strength = 0.0

        for child in predecessors:
            if child not in beliefs:
                continue

            certainty      = 1.0 - entropies[child]
            edge_weight    = G[child][node].get("weight", 1.0)
            child_strength = strengths.get(child, 1.0)
            eff_weight     = edge_weight * child_strength * certainty

            if eff_weight <= 0.0:
                continue   # Skip zero-weight / maximum-entropy children

            child_ps.append(beliefs[child])
            child_ws.append(eff_weight)
            accum_strength += child_strength

        if not child_ps:
            beliefs[node]   = 0.5
            entropies[node] = 1.0
            strengths[node] = 0.0
            continue

        p_node = weighted_logit_aggregate(child_ps, child_ws)

        beliefs[node]   = p_node
        entropies[node] = entropy(p_node)
        strengths[node] = min(accum_strength, MAX_GRAPH_STRENGTH)

    # ----------------------------------------------------------------
    # Root resolution
    # ----------------------------------------------------------------
    explicit_roots = {
        v for _, v, data in G.edges(data=True) if data.get("is_root")
    }
    if not explicit_roots:
        explicit_roots = {n for n, d in G.out_degree() if d == 0}

    if len(explicit_roots) == 0:
        return {"error": "no root node found"}
    if len(explicit_roots) > 1:
        return {"error": "multiple root nodes detected"}

    root         = next(iter(explicit_roots))
    p_success    = beliefs[root]
    root_entropy = entropies[root]
    root_strength = strengths[root]

    # ----------------------------------------------------------------
    # Risk detection: foundation nodes that are much weaker than root
    # ----------------------------------------------------------------
    # Foundation nodes = nodes with in-degree 0 (no children feeding them)
    foundation_nodes = [n for n, d in G.in_degree() if d == 0]
    risk_nodes = []

    for node in foundation_nodes:
        node_p      = beliefs.get(node, 0.5)
        node_str    = strengths.get(node, 0.0)

        if node_str < MIN_RISK_STRENGTH:
            continue   # Too little evidence to be meaningful

        if node_p < (p_success - RISK_GAP_THRESHOLD):
            risk_nodes.append(str(node))

    # ----------------------------------------------------------------
    # Confidence
    # ----------------------------------------------------------------
    confidence = (1.0 - root_entropy) * strength_factor(root_strength)

    return {
        "decision_id": str(decision_id),
        "p_success":   round(p_success, 4),
        "confidence":  round(confidence, 4),
        "risk_nodes":  risk_nodes,
        "beliefs":     {str(k): round(v, 4) for k, v in beliefs.items()},
    }
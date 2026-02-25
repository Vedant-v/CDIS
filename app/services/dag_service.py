import networkx as nx
from uuid import UUID
from math import log, exp
from typing import Dict, Any, List
from app.db.cassandra import get_session
from app.services.fusion_service import fuse_claim, entropy

FETCH_EDGES = """
SELECT parent_claim_id, child_claim_id, weight, is_decision_root
FROM decision_claim_edges
WHERE decision_id = %s
"""

def load_decision_graph(decision_id: UUID) -> nx.DiGraph:
    rows = get_session().execute(FETCH_EDGES, (decision_id,))
    G = nx.DiGraph()
    for row in rows:
        weight = row.weight if row.weight is not None else 1.0
        is_root = row.is_decision_root if row.is_decision_root is not None else False
        G.add_edge(row.child_claim_id, row.parent_claim_id, weight=weight, is_root=is_root)
    return G

def logit(p: float) -> float:
    return log(p / (1 - p))

def inv_logit(x: float) -> float:
    return 1 / (1 + exp(-x))

def weighted_logit_aggregate(child_ps: List[float], weights: List[float]) -> float:
    total = 0.0
    for p, w in zip(child_ps, weights):
        p = min(max(p, 0.001), 0.999)
        total += w * logit(p)
    return inv_logit(total)

def evaluate_decision_graph(decision_id: UUID, domain_map: Dict[UUID, str]) -> Dict[str, Any]:
    G = load_decision_graph(decision_id)
    
    if len(G) == 0:
        return {"error": "no decision graph found"}

    beliefs = {}
    entropies = {}

    # Topological order: processing leaves to root
    # Note: the edges we added: child -> parent. 
    # Therefore, topological sort correctly evaluates dependencies first.
    topo_order = list(nx.topological_sort(G))
    
    for node in topo_order:
        predecessors = list(G.predecessors(node))
        
        if not predecessors:
            # Leaf node: needs raw claim fusion
            domain = domain_map.get(node, "general")
            fusion_result = fuse_claim(node, domain)
            beliefs[node] = fusion_result["p_success"]
            entropies[node] = fusion_result["entropy"]
        else:
            # Internal node: aggregate child beliefs via noisy-AND
            child_ps = []
            weights = []
            for child in predecessors:
                # Ignore nodes with no information (entropy == 1.0) to prevent 0.5 dilution
                if child in beliefs and entropies.get(child, 1.0) < 1.0:
                    child_ps.append(beliefs[child])
                    weights.append(G[child][node].get("weight", 1.0))
            
            if not child_ps:
                beliefs[node] = 0.5
                entropies[node] = 1.0
                continue

            p_node = weighted_logit_aggregate(child_ps, weights)
            beliefs[node] = p_node
            entropies[node] = entropy(p_node)

    # The root node(s) are explicitly defined by is_decision_root from edges
    explicit_roots = set()
    for u, v, data in G.edges(data=True):
        if data.get("is_root"):
            explicit_roots.add(v)
            
    if not explicit_roots:
        # Fallback for older schemas/graphs without explicit root flags
        explicit_roots = {n for n, d in G.out_degree() if d == 0}
        
    if not explicit_roots:
        return {"error": "no root node found (cycle detected or empty graph?)"}
    
    root = list(explicit_roots)[0]
    p_success = beliefs[root]
    
    # Identify Risk Nodes: any leaf claim falling significantly below expectation (e.g., < 0.5)
    risk_nodes = []
    leaves = [n for n, d in G.in_degree() if d == 0]
    for leaf in leaves:
        if beliefs[leaf] < 0.5:
            risk_nodes.append(str(leaf))

    # Overall Decision Confidence (Invert of root entropy)
    confidence = max(0.0, 1.0 - entropies[root])

    return {
        "decision_id": str(decision_id),
        "p_success": round(p_success, 3),
        "risk_nodes": risk_nodes,
        "confidence": round(confidence, 3),
        "beliefs": {str(k): round(v, 3) for k, v in beliefs.items()}
    }

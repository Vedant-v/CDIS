from fastapi import APIRouter as Router, HTTPException
from uuid import UUID
from pydantic import BaseModel
from typing import Dict
import networkx as nx
from app.services.dag_service import evaluate_decision_graph, load_decision_graph
from app.db.cassandra import get_session

router = Router()

class DecisionEdgeInput(BaseModel):
    parent_claim_id: UUID
    child_claim_id: UUID
    weight: float = 1.0
    is_decision_root: bool = False

class DecisionEvaluateInput(BaseModel):
    domain_map: Dict[UUID, str]  # Map of claim_id to its respective evaluation domain

@router.post("/decision/{decision_id}/edges")
async def add_decision_edge(decision_id: UUID, req: DecisionEdgeInput):
    # Phase 5A: Synchronous DAG Cycle Validation
    G = load_decision_graph(decision_id)
    
    # networkx requires child -> parent to evaluate correctly in the current logic
    G.add_edge(req.child_claim_id, req.parent_claim_id, weight=req.weight, is_root=req.is_decision_root)
    
    if not nx.is_directed_acyclic_graph(G):
        raise HTTPException(status_code=400, detail="Cycle detected. Edge rejected to prevent NetworkXUnfeasible.")

    # Safe to write to Cassandra
    session = get_session()
    INSERT_EDGE = """
    INSERT INTO decision_claim_edges (decision_id, parent_claim_id, child_claim_id, weight, is_decision_root)
    VALUES (%s, %s, %s, %s, %s)
    """
    session.execute(INSERT_EDGE, (decision_id, req.parent_claim_id, req.child_claim_id, req.weight, req.is_decision_root))
    
    return {"status": "success", "message": "Edge added safely without cycles."}

@router.post("/decision/{decision_id}/evaluate")
async def evaluate_decision(decision_id: UUID, req: DecisionEvaluateInput):
    result = evaluate_decision_graph(decision_id, req.domain_map)
    return result


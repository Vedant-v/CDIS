from fastapi import APIRouter as Router
from uuid import UUID
from pydantic import BaseModel

from app.services.opinion_service import belief_to_beta
from app.db.dao.opinion_dao import insert_opinion
from app.services.entropy_worker import process_entropy_contribution

router = Router()

class OpinionInput(BaseModel):
    claim_id: UUID
    agent_id: UUID
    belief: float
    confidence: float
    domain: str
    stakeholder_type: str
    cluster_id: str


@router.post("/submit")
async def submit_opinion(op: OpinionInput):

    alpha, beta = belief_to_beta(
        op.belief,
        op.confidence
    )
    print("DEBUG:", alpha, beta)

    insert_opinion(
        op.claim_id,
        op.agent_id,
        alpha,
        beta,
        op.domain,
        op.stakeholder_type,
        op.cluster_id
    )

    # Trigger Async Pre-Outcome Credit
    process_entropy_contribution.apply_async(
        args=[
            str(op.claim_id),
            op.domain,
            str(op.agent_id),
            alpha,
            beta,
            op.cluster_id
        ],
        countdown=1
    )

    return {"status": "stored", "fusion": "entropy_learning_started"}

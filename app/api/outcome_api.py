from fastapi import APIRouter as Router
from pydantic import BaseModel
from uuid import UUID
from app.db.dao.outcome_dao import insert_claim_outcome
from app.services.expertise_worker import update_expertise

class OutcomeSubmit(BaseModel):
    outcome: float
    observed_value: float = 0.0

router = Router()

@router.post("/outcome/{claim_id}")
async def record_outcome(claim_id: UUID, payload: OutcomeSubmit):
    insert_claim_outcome(claim_id, payload.outcome, payload.observed_value)
    update_expertise.delay(str(claim_id))
    return {"status": "stored", "fusion": "learning started"}

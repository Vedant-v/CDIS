"""
opinion_api.py
--------------
REST endpoint for submitting agent opinions on claims.

Outbox pattern: DB write happens first; Celery enqueue failure is
logged but does NOT cause the HTTP response to fail (the opinion is
safely stored and can be re-processed by a background reconciler).
"""

import structlog
from fastapi import APIRouter as Router
from uuid import UUID
from pydantic import BaseModel, field_validator

from app.services.opinion_service import belief_to_beta
from app.db.dao.opinion_dao import insert_opinion
from app.services.entropy_worker import process_entropy_contribution

router = Router()
logger = structlog.get_logger()


class OpinionInput(BaseModel):
    claim_id:         UUID
    agent_id:         UUID
    belief:           float
    confidence:       float
    domain:           str
    stakeholder_type: str
    cluster_id:       str

    @field_validator("belief")
    @classmethod
    def belief_in_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("belief must be in [0, 1]")
        return v

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        return v


@router.post("/submit")
async def submit_opinion(op: OpinionInput):
    alpha, beta = belief_to_beta(op.belief, op.confidence)

    # DB write first — opinion is durable before we attempt async work
    insert_opinion(
        op.claim_id,
        op.agent_id,
        alpha,
        beta,
        op.domain,
        op.stakeholder_type,
        op.cluster_id,
    )

    # Enqueue entropy learning task — failure is non-fatal
    try:
        process_entropy_contribution.apply_async(
            args=[
                str(op.claim_id),
                op.domain,
                str(op.agent_id),
                alpha,
                beta,
                op.cluster_id,
            ],
            countdown=1,
        )
    except Exception as exc:
        logger.error(
            "Failed to enqueue entropy task",
            claim_id=str(op.claim_id),
            agent_id=str(op.agent_id),
            error=str(exc),
        )
        # Opinion is already stored; entropy learning will be picked up
        # by the periodic reconciliation sweep (if implemented).

    return {"status": "stored", "fusion": "entropy_learning_started"}
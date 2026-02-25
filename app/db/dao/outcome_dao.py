from uuid import UUID
from datetime import datetime
from app.db.cassandra import get_session

INSERT_OUTCOME = """
INSERT INTO claim_outcomes (claim_id, outcome, observed_value, resolved_at)
VALUES (%s, %s, %s, %s)
"""

def insert_claim_outcome(claim_id: UUID, outcome: float, observed_value: float):
    get_session().execute(
        INSERT_OUTCOME,
        (claim_id, outcome, observed_value, datetime.utcnow())
    )

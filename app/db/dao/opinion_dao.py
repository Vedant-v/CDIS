from app.db.cassandra import get_session
from uuid import UUID
from datetime import datetime
import redis
import os

redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "redis"), port=6379, db=0)

INSERT_OPINION = """
INSERT INTO opinions_by_claim
(claim_id, day_bucket, submission_time, agent_id,
 alpha, beta, domain, stakeholder_type, cluster_id)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

def insert_opinion(
    claim_id: UUID,
    agent_id: UUID,
    alpha: float,
    beta: float,
    domain: str,
    stakeholder_type: str,
    cluster_id: str
):
    now = datetime.utcnow()
    day = now.date()

    get_session().execute(
        INSERT_OPINION,
        (
            claim_id,
            day,
            now,
            agent_id,
            alpha,
            beta,
            domain,
            stakeholder_type,
            cluster_id
        )
    )

    get_session().execute("""
    INSERT INTO agent_expertise (agent_id, domain, calibration_score, bias_score, entropy_score)
    VALUES (%s, %s, %s, %s, %s)
    IF NOT EXISTS
    """, (agent_id, domain, 0.5, 0.0, 0.5))

    get_session().execute("""
    INSERT INTO expertise_by_domain (domain, agent_id, calibration_score, bias_score, entropy_score)
    VALUES (%s, %s, %s, %s, %s)
    IF NOT EXISTS
    """, (domain, agent_id, 0.5, 0.0, 0.5))

    # Strict domain epoch batching - option B
    # Mark the domain as dirty so the periodic epoch coordinator scheduler can pick it up
    try:
        redis_client.set(f"domain_dirty:{domain}", 1)
    except Exception as e:
        pass # In production, log this correctly

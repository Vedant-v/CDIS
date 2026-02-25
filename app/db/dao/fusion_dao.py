from app.db.cassandra import get_session
from datetime import datetime, timedelta
from uuid import UUID

FETCH_OPINIONS = """
SELECT agent_id, alpha, beta, domain, cluster_id
FROM opinions_by_claim
WHERE claim_id = %s
AND day_bucket = %s
"""

FETCH_EXPERTISE = """
SELECT agent_id, calibration_score, bias_score, entropy_score
FROM expertise_by_domain
WHERE domain = %s
"""

def fetch_recent_opinions(claim_id: UUID):
    today = datetime.utcnow().date()
    buckets = [
        today,
        today - timedelta(days=1),
        today - timedelta(days=2)
    ]

    all_rows = []

    for b in buckets:
        rows = get_session().execute(
            FETCH_OPINIONS,
            (claim_id, b)
        )
        all_rows.extend(rows)

    return all_rows

def fetch_expertise(domain: str):
    rows = get_session().execute(
        FETCH_EXPERTISE,
        (domain,)
    )
    return {
        row.agent_id: row
        for row in rows
    }

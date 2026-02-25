from math import exp
from datetime import date
from collections import namedtuple
from app.db.cassandra import get_session

FETCH_LEDGER = """
SELECT week_bucket,
prediction_count,
brier_sum,
bias_sum,
overconfidence_count
FROM expertise_ledger_by_agent
WHERE agent_id = %s
AND domain = %s
LIMIT 52
"""

FETCH_EVENTS = """
SELECT detection_week, attribution_start, attribution_end
FROM domain_volatility_events
WHERE domain = %s AND detection_week >= %s
"""

FETCH_RESILIENCE = """
SELECT vrs
FROM agent_resilience_projection
WHERE agent_id = %s AND domain = %s
"""

AgentExpertise = namedtuple("AgentExpertise", ["calibration_score", "bias_score", "entropy_score"])

from app.services.domain_decay import get_domain_lambda

def fetch_temporal_expertise(agent_id, domain) -> AgentExpertise:
    lam = get_domain_lambda(domain)
    session = get_session()
    
    today = date.today()
    from datetime import timedelta
    cutoff_date = today - timedelta(weeks=12)
    
    # Phase 4F: Bounded Time Event Lookups pushed to DB
    events_rows = session.execute(FETCH_EVENTS, (domain, cutoff_date))
    events = []
    
    for ev in events_rows:
        ev_start = ev.attribution_start.date() if hasattr(ev.attribution_start, "date") and callable(ev.attribution_start.date) else ev.attribution_start
        ev_end = ev.attribution_end.date() if hasattr(ev.attribution_end, "date") and callable(ev.attribution_end.date) else ev.attribution_end
        ev_detection = ev.detection_week.date() if hasattr(ev.detection_week, "date") and callable(ev.detection_week.date) else ev.detection_week
        
        # Bounded event lookup: handled by DB now, just append
        events.append({"start": ev_start, "end": ev_end})
            
    resilience_row = session.execute(FETCH_RESILIENCE, (agent_id, domain)).one()
    agent_vrs = resilience_row.vrs if resilience_row else 0.0

    rows = session.execute(
        FETCH_LEDGER,
        (agent_id, domain)
    )

    today = date.today()

    weighted_count = 0.0
    weighted_brier = 0.0
    weighted_bias = 0.0
    weighted_over = 0.0

    for r in rows:
        week_bucket_date = r.week_bucket.date() if hasattr(r.week_bucket, "date") and callable(r.week_bucket.date) else r.week_bucket
        weeks_ago = (today - week_bucket_date).days / 7.0
        
        # Standard historical decay
        decay = exp(-lam * weeks_ago)
        
        # Phase 4D/4F: Event-scoped Decay Modulation (Non-stacking)
        best_eff_lam = lam
        for ev in events:
            if ev["start"] <= week_bucket_date <= ev["end"]:
                # Compute effective lambda with floor to prevent Elite Freezing
                eff_lam = max(lam * (1.0 - agent_vrs), 0.005)
                # Track the strongest protection (lowest lambda)
                best_eff_lam = min(best_eff_lam, eff_lam)
                
        # Modulate the base decay factor ONCE per bucket
        decay *= exp((lam - best_eff_lam) * weeks_ago)

        weighted_count += decay * (r.prediction_count or 0)
        weighted_brier += decay * (r.brier_sum or 0.0)
        weighted_bias += decay * (r.bias_sum or 0.0)
        weighted_over += decay * (r.overconfidence_count or 0)

    if weighted_count == 0:
        return AgentExpertise(0.5, 0.0, 0.5)

    calibration = max(0.01, min(1.0 - (weighted_brier / weighted_count), 1.0))
    bias = max(0.0, min(abs(weighted_bias / weighted_count), 1.0))
    entropy = max(0.01, min(1.0 - (weighted_over / weighted_count), 1.0))

    return AgentExpertise(calibration, bias, entropy)

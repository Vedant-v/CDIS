"""
expertise_temporal.py
---------------------
Computes time-decayed expertise scores for an agent in a given domain.

Key design decisions:
  - Exponential decay controlled by domain lambda (λ)
  - Volatility events can reduce effective λ for weeks inside their
    attribution window — predictions during chaotic periods are judged
    less harshly
  - Agent VRS (Volatility Resilience Score) further modulates the
    event-period decay: resilient agents retain credit from shock periods
  - All returned scores are clamped; the function never returns NaN or
    out-of-range values
"""

from math import exp
from datetime import date, timedelta
from collections import namedtuple
from typing import List

from app.db.cassandra import get_session
from app.services.domain_decay import get_domain_lambda


# ---------------------------------------------------------------------------
# CQL Statements
# ---------------------------------------------------------------------------

_FETCH_LEDGER = """
SELECT week_bucket,
       prediction_count,
       brier_sum,
       bias_sum,
       overconfidence_count
FROM expertise_ledger_by_agent
WHERE agent_id = %s
  AND domain   = %s
LIMIT 52
"""

_FETCH_EVENTS = """
SELECT detection_week, attribution_start, attribution_end
FROM domain_volatility_events
WHERE domain        = %s
  AND detection_week >= %s
"""

_FETCH_RESILIENCE = """
SELECT vrs
FROM agent_resilience_projection
WHERE agent_id = %s
  AND domain   = %s
"""


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

AgentExpertise = namedtuple(
    "AgentExpertise",
    ["calibration_score", "bias_score", "entropy_score", "prediction_count"],
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_temporal_expertise(agent_id, domain: str) -> AgentExpertise:
    """
    Compute time-decayed expertise scores from the weekly ledger.

    Returns AgentExpertise with sensible neutral defaults when the agent
    has no prediction history, ensuring the system degrades gracefully.
    """
    lam     = get_domain_lambda(domain)
    session = get_session()
    today   = date.today()

    # We look back 12 weeks for volatility events (attribution windows
    # can stretch up to 2 weeks behind detection, plus some headroom)
    cutoff = today - timedelta(weeks=12)

    # ----------------------------------------------------------------
    # Fetch volatility events (used to modulate decay per week bucket)
    # ----------------------------------------------------------------
    events: List[dict] = []
    for ev in session.execute(_FETCH_EVENTS, (domain, cutoff)):
        ev_start = ev.attribution_start.date() if hasattr(ev.attribution_start, "date") else ev.attribution_start
        ev_end   = ev.attribution_end.date()   if hasattr(ev.attribution_end,   "date") else ev.attribution_end
        events.append({"start": ev_start, "end": ev_end})

    # ----------------------------------------------------------------
    # Fetch agent VRS
    # ----------------------------------------------------------------
    resilience_row = session.execute(_FETCH_RESILIENCE, (agent_id, domain)).one()
    agent_vrs      = float(resilience_row.vrs) if resilience_row else 0.0
    agent_vrs      = max(0.0, min(agent_vrs, 1.0))

    # ----------------------------------------------------------------
    # Accumulate decay-weighted ledger
    # ----------------------------------------------------------------
    weighted_count = 0.0
    weighted_brier = 0.0
    weighted_bias  = 0.0
    weighted_over  = 0.0

    for r in session.execute(_FETCH_LEDGER, (agent_id, domain)):
        wb = r.week_bucket.date() if hasattr(r.week_bucket, "date") else r.week_bucket
        weeks_ago = (today - wb).days / 7.0

        # Determine effective lambda for this week bucket.
        # If the bucket falls inside a volatility attribution window, use
        # a reduced λ so that predictions made during chaotic periods are
        # weighted less aggressively discounted.
        eff_lam = lam
        for ev in events:
            if ev["start"] <= wb <= ev["end"]:
                shock_lam = max(lam * (1.0 - agent_vrs), 0.005)
                eff_lam   = min(eff_lam, shock_lam)

        decay = exp(-eff_lam * weeks_ago)
        count = r.prediction_count or 0

        weighted_count += decay * count
        weighted_brier += decay * (r.brier_sum          or 0.0)
        weighted_bias  += decay * (r.bias_sum            or 0.0)
        weighted_over  += decay * (r.overconfidence_count or 0)

    # ----------------------------------------------------------------
    # No history → neutral priors
    # ----------------------------------------------------------------
    if weighted_count == 0.0:
        return AgentExpertise(
            calibration_score=0.5,
            bias_score=0.0,
            entropy_score=0.5,
            prediction_count=0,
        )

    calibration = max(0.01, min(1.0 - (weighted_brier / weighted_count), 1.0))
    bias        = max(0.00, min(abs(weighted_bias  / weighted_count), 1.0))
    entropy_s   = max(0.01, min(1.0 - (weighted_over  / weighted_count), 1.0))

    return AgentExpertise(
        calibration_score=calibration,
        bias_score=bias,
        entropy_score=entropy_s,
        prediction_count=int(weighted_count),
    )
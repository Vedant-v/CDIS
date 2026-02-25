from app.worker import celery
from app.db.cassandra import get_session
from datetime import datetime, timedelta
from uuid import UUID
from cassandra.query import BatchStatement
from math import log2

FETCH_OPINIONS = """
SELECT agent_id, alpha, beta, domain
FROM opinions_by_claim
WHERE claim_id = %s
AND day_bucket = %s
"""

FETCH_OUTCOME = """
SELECT outcome FROM claim_outcomes
WHERE claim_id = %s
"""

def brier(p: float, outcome: float) -> float:
    return (p - outcome)**2

def signed_error(p: float, outcome: float) -> float:
    return p - outcome

def entropy(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0
    return -p * log2(p) - (1-p) * log2(1-p)

@celery.task
def update_expertise(claim_id_str: str):
    claim_id = UUID(claim_id_str)
    from datetime import date
    today = date.today()

    buckets = [
        today,
        today - timedelta(days=1),
        today - timedelta(days=2)
    ]

    opinions = []
    for b in buckets:
        rows = get_session().execute(
            FETCH_OPINIONS,
            (claim_id, b)
        )
        opinions.extend(list(rows))

    outcome_row = get_session().execute(
        FETCH_OUTCOME,
        (claim_id,)
    ).one()

    if not outcome_row:
        return

    outcome = float(outcome_row.outcome)
    session = get_session()

    from app.utils.time_bucket import current_week_bucket
    week_bucket = current_week_bucket()

    from cassandra import ConsistencyLevel

    mark_scored_stmt = session.prepare("""
    INSERT INTO agent_predictions_by_domain
    (agent_id, domain, claim_id, evaluation_time, predicted_prob, outcome)
    VALUES (?, ?, ?, ?, ?, ?)
    IF NOT EXISTS
    """)
    mark_scored_stmt.consistency_level = ConsistencyLevel.QUORUM
    mark_scored_stmt.serial_consistency_level = ConsistencyLevel.SERIAL

    read_ledger_stmt = session.prepare("""
    SELECT prediction_count, brier_sum, bias_sum, overconfidence_count
    FROM expertise_ledger_by_agent
    WHERE agent_id=? AND domain=? AND week_bucket=?
    """)

    insert_ledger_stmt = session.prepare("""
    INSERT INTO expertise_ledger_by_agent 
    (agent_id, domain, week_bucket, prediction_count, brier_sum, bias_sum, overconfidence_count)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    IF NOT EXISTS
    """)
    insert_ledger_stmt.consistency_level = ConsistencyLevel.QUORUM
    insert_ledger_stmt.serial_consistency_level = ConsistencyLevel.SERIAL

    update_ledger_stmt = session.prepare("""
    UPDATE expertise_ledger_by_agent
    SET prediction_count = ?, brier_sum = ?, bias_sum = ?, overconfidence_count = ?
    WHERE agent_id = ? AND domain = ? AND week_bucket = ?
    IF prediction_count = ?
    """)
    update_ledger_stmt.consistency_level = ConsistencyLevel.QUORUM
    update_ledger_stmt.serial_consistency_level = ConsistencyLevel.SERIAL

    # --- Phase 4C: Domain Volatility Ledger ---
    read_domain_stmt = session.prepare("""
    SELECT prediction_count, brier_sum, confident_prediction_count, confident_wrong_count
    FROM domain_volatility_ledger
    WHERE domain=? AND week_bucket=?
    """)
    
    insert_domain_stmt = session.prepare("""
    INSERT INTO domain_volatility_ledger 
    (domain, week_bucket, prediction_count, brier_sum, confident_prediction_count, confident_wrong_count)
    VALUES (?, ?, ?, ?, ?, ?)
    IF NOT EXISTS
    """)
    insert_domain_stmt.consistency_level = ConsistencyLevel.QUORUM
    insert_domain_stmt.serial_consistency_level = ConsistencyLevel.SERIAL

    update_domain_stmt = session.prepare("""
    UPDATE domain_volatility_ledger
    SET prediction_count = ?, brier_sum = ?, confident_prediction_count = ?, confident_wrong_count = ?
    WHERE domain = ? AND week_bucket = ?
    IF prediction_count = ?
    """)
    update_domain_stmt.consistency_level = ConsistencyLevel.QUORUM
    update_domain_stmt.serial_consistency_level = ConsistencyLevel.SERIAL

    # --- Phase 4D: Agent Resilience Ledger ---
    read_resilience_stmt = session.prepare("""
    SELECT shock_weighted_error_sum, shock_weighted_confidence_sum
    FROM agent_volatility_resilience_ledger
    WHERE agent_id=? AND domain=? AND detection_week=?
    """)
    
    insert_resilience_stmt = session.prepare("""
    INSERT INTO agent_volatility_resilience_ledger 
    (agent_id, domain, detection_week, shock_weighted_error_sum, shock_weighted_confidence_sum)
    VALUES (?, ?, ?, ?, ?)
    IF NOT EXISTS
    """)
    insert_resilience_stmt.consistency_level = ConsistencyLevel.QUORUM
    insert_resilience_stmt.serial_consistency_level = ConsistencyLevel.SERIAL

    update_resilience_stmt = session.prepare("""
    UPDATE agent_volatility_resilience_ledger
    SET shock_weighted_error_sum = ?, shock_weighted_confidence_sum = ?
    WHERE agent_id = ? AND domain = ? AND detection_week = ?
    IF shock_weighted_confidence_sum = ?
    """)
    update_resilience_stmt.consistency_level = ConsistencyLevel.QUORUM
    update_resilience_stmt.serial_consistency_level = ConsistencyLevel.SERIAL
    
    FETCH_EVENTS = "SELECT detection_week, attribution_start, attribution_end FROM domain_volatility_events WHERE domain = %s"
    domain_events_cache = {}

    for op in opinions:
        if op.alpha + op.beta == 0:
            continue
            
        p = op.alpha / (op.alpha + op.beta)
        
        # Step 1: Exactly-once LWT Event Gate
        result_mark = session.execute(mark_scored_stmt, (
            op.agent_id, op.domain, claim_id, datetime.utcnow(), p, outcome
        ))
        
        if not result_mark.one().applied:
            continue
            
        bs = brier(p, outcome)
        error = signed_error(p, outcome)
        sharpness = abs(p - 0.5) * 2
        
        is_overconfident = 1 if (sharpness > 0.6 and bs > 0.25) else 0

        # Step 2 — read ledger snapshot
        row = session.execute(read_ledger_stmt, (op.agent_id, op.domain, week_bucket)).one()

        if not row:
            # Step 3 — CAS insert for new row
            session.execute(insert_ledger_stmt, (
                op.agent_id, op.domain, week_bucket,
                1, bs, error, is_overconfident
            ))
        else:
            # Step 3 — CAS update for existing row
            current_count = row.prediction_count or 0
            current_brier = row.brier_sum or 0.0
            current_bias  = row.bias_sum or 0.0
            current_over = row.overconfidence_count or 0
            
            session.execute(update_ledger_stmt, (
                current_count + 1,
                current_brier + bs,
                current_bias + error,
                current_over + is_overconfident,
                op.agent_id, op.domain, week_bucket,
                current_count
            ))

        # --- Phase 4C: Update Domain Volatility Ledger ---
        is_confident = 1 if (p > 0.7 or p < 0.3) else 0
        is_confident_wrong = 1 if (is_confident and bs > 0.25) else 0

        d_row = session.execute(read_domain_stmt, (op.domain, week_bucket)).one()
        if not d_row:
            session.execute(insert_domain_stmt, (
                op.domain, week_bucket,
                1, bs, is_confident, is_confident_wrong
            ))
        else:
            d_count = d_row.prediction_count or 0
            d_brier = d_row.brier_sum or 0.0
            d_conf_count = d_row.confident_prediction_count or 0
            d_conf_wrong = d_row.confident_wrong_count or 0
            
            session.execute(update_domain_stmt, (
                d_count + 1,
                d_brier + bs,
                d_conf_count + is_confident,
                d_conf_wrong + is_confident_wrong,
                op.domain, week_bucket,
                d_count
            ))

        # --- Phase 4D: Agent Volatility Resilience Ledger ---
        if op.domain not in domain_events_cache:
            domain_events_cache[op.domain] = list(session.execute(FETCH_EVENTS, (op.domain,)))
            
        events = domain_events_cache[op.domain]
        if events:
            # Quadratic Confidence Scaling
            conf_weight = (2.0 * abs(p - 0.5)) ** 2
            weighted_error = conf_weight * bs
            
            wb_date = week_bucket.date() if hasattr(week_bucket, "date") and callable(week_bucket.date) else week_bucket
            
            for ev in events:
                ev_start = ev.attribution_start.date() if hasattr(ev.attribution_start, "date") and callable(ev.attribution_start.date) else ev.attribution_start
                ev_end = ev.attribution_end.date() if hasattr(ev.attribution_end, "date") and callable(ev.attribution_end.date) else ev.attribution_end
                
                if ev_start <= wb_date <= ev_end:
                    ev_detection = ev.detection_week.date() if hasattr(ev.detection_week, "date") and callable(ev.detection_week.date) else ev.detection_week
                    
                    # CAS Update for Resilience Ledger
                    r_row = session.execute(read_resilience_stmt, (op.agent_id, op.domain, ev_detection)).one()
                    
                    if not r_row:
                        session.execute(insert_resilience_stmt, (
                            op.agent_id, op.domain, ev_detection,
                            weighted_error, conf_weight
                        ))
                    else:
                        curr_err = r_row.shock_weighted_error_sum or 0.0
                        curr_conf = r_row.shock_weighted_confidence_sum or 0.0
                        
                        session.execute(update_resilience_stmt, (
                            curr_err + weighted_error,
                            curr_conf + conf_weight,
                            op.agent_id, op.domain, ev_detection,
                            curr_conf
                        ))

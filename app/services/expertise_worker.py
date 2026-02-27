"""
expertise_worker.py
-------------------
Post-outcome expertise scoring.  Triggered when a claim resolves.

For each opinion on the resolved claim:
  1. Exactly-once LWT gate (agent_predictions_by_domain IF NOT EXISTS)
  2. Compute Brier score, signed error, overconfidence flag
  3. CAS-update expertise_ledger_by_agent  (with retry)
  4. CAS-update domain_volatility_ledger   (with retry)
  5. CAS-update agent_volatility_resilience_ledger for shock periods (with retry)

CAS retry pattern: read → compute new value → attempt IF old_value = X.
On conflict, re-read and retry up to MAX_CAS_RETRIES times.
"""

from app.worker import celery
from app.db.cassandra import get_session
from datetime import datetime, timedelta, date
from uuid import UUID
from math import log2

import structlog

logger = structlog.get_logger()

_MAX_CAS_RETRIES = 5

FETCH_OPINIONS = """
SELECT agent_id, alpha, beta, domain
FROM opinions_by_claim
WHERE claim_id  = %s
  AND day_bucket = %s
"""

FETCH_OUTCOME = """
SELECT outcome FROM claim_outcomes
WHERE claim_id = %s
"""


# ---------------------------------------------------------------------------
# Pure-math helpers
# ---------------------------------------------------------------------------

def brier(p: float, outcome: float) -> float:
    return (p - outcome) ** 2


def signed_error(p: float, outcome: float) -> float:
    return p - outcome


def local_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * log2(p) - (1.0 - p) * log2(1.0 - p)


# ---------------------------------------------------------------------------
# CAS helper
# ---------------------------------------------------------------------------

def _cas_update(session, read_stmt, insert_stmt, update_stmt,
                read_args, insert_fn, update_fn,
                current_field_name: str, retries: int = _MAX_CAS_RETRIES):
    """
    Generic CAS read-modify-write with insert-if-absent fallback.

    read_stmt    → SELECT ... IF EXISTS
    insert_stmt  → INSERT ... IF NOT EXISTS
    update_stmt  → UPDATE ... IF <current_field_name> = ?
    insert_fn(row=None)  → tuple of values to INSERT
    update_fn(row)       → tuple of values to UPDATE  (new_vals + pk + old_guard)
    """
    for attempt in range(retries):
        row = session.execute(read_stmt, read_args).one()

        if row is None:
            result = session.execute(insert_stmt, insert_fn(None))
            if result.one().applied:
                return
            # Another worker inserted concurrently — loop and update
            continue

        result = session.execute(update_stmt, update_fn(row))
        if result.one().applied:
            return

        if attempt == retries - 1:
            logger.warning("CAS exhausted retries", read_args=read_args)


# ---------------------------------------------------------------------------
# Celery Task
# ---------------------------------------------------------------------------

@celery.task(bind=True, max_retries=3, default_retry_delay=10)
def update_expertise(self, claim_id_str: str):
    claim_id = UUID(claim_id_str)
    today    = date.today()

    # Gather opinions from last 3 day-buckets (claim may have been submitted yesterday)
    opinions = []
    for delta in (0, 1, 2):
        rows = get_session().execute(FETCH_OPINIONS, (claim_id, today - timedelta(days=delta)))
        opinions.extend(list(rows))

    outcome_row = get_session().execute(FETCH_OUTCOME, (claim_id,)).one()
    if not outcome_row:
        return

    outcome  = float(outcome_row.outcome)
    session  = get_session()

    from app.utils.time_bucket import current_week_bucket
    from cassandra import ConsistencyLevel

    week_bucket = current_week_bucket()

    # ----------------------------------------------------------------
    # Prepare statements once
    # ----------------------------------------------------------------
    def _prep(cql, cl=ConsistencyLevel.QUORUM, serial=True):
        stmt = session.prepare(cql)
        stmt.consistency_level = cl
        if serial:
            stmt.serial_consistency_level = ConsistencyLevel.SERIAL
        return stmt

    mark_scored_stmt = _prep("""
        INSERT INTO agent_predictions_by_domain
            (agent_id, domain, claim_id, evaluation_time, predicted_prob, outcome)
        VALUES (?, ?, ?, ?, ?, ?)
        IF NOT EXISTS
    """)

    # --- Agent ledger ---
    read_ledger_stmt = _prep("""
        SELECT prediction_count, brier_sum, bias_sum, overconfidence_count
        FROM expertise_ledger_by_agent
        WHERE agent_id=? AND domain=? AND week_bucket=?
    """, serial=False)

    insert_ledger_stmt = _prep("""
        INSERT INTO expertise_ledger_by_agent
            (agent_id, domain, week_bucket, prediction_count, brier_sum, bias_sum, overconfidence_count)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        IF NOT EXISTS
    """)

    update_ledger_stmt = _prep("""
        UPDATE expertise_ledger_by_agent
        SET prediction_count = ?, brier_sum = ?, bias_sum = ?, overconfidence_count = ?
        WHERE agent_id=? AND domain=? AND week_bucket=?
        IF prediction_count = ?
    """)

    # --- Domain volatility ledger ---
    read_domain_stmt = _prep("""
        SELECT prediction_count, brier_sum, confident_prediction_count, confident_wrong_count
        FROM domain_volatility_ledger
        WHERE domain=? AND week_bucket=?
    """, serial=False)

    insert_domain_stmt = _prep("""
        INSERT INTO domain_volatility_ledger
            (domain, week_bucket, prediction_count, brier_sum, confident_prediction_count, confident_wrong_count)
        VALUES (?, ?, ?, ?, ?, ?)
        IF NOT EXISTS
    """)

    update_domain_stmt = _prep("""
        UPDATE domain_volatility_ledger
        SET prediction_count=?, brier_sum=?, confident_prediction_count=?, confident_wrong_count=?
        WHERE domain=? AND week_bucket=?
        IF prediction_count=?
    """)

    # --- Resilience ledger ---
    read_resilience_stmt = _prep("""
        SELECT shock_weighted_error_sum, shock_weighted_confidence_sum
        FROM agent_volatility_resilience_ledger
        WHERE agent_id=? AND domain=? AND detection_week=?
    """, serial=False)

    insert_resilience_stmt = _prep("""
        INSERT INTO agent_volatility_resilience_ledger
            (agent_id, domain, detection_week, shock_weighted_error_sum, shock_weighted_confidence_sum)
        VALUES (?, ?, ?, ?, ?)
        IF NOT EXISTS
    """)

    update_resilience_stmt = _prep("""
        UPDATE agent_volatility_resilience_ledger
        SET shock_weighted_error_sum=?, shock_weighted_confidence_sum=?
        WHERE agent_id=? AND domain=? AND detection_week=?
        IF shock_weighted_confidence_sum=?
    """)

    FETCH_EVENTS_CQL = (
        "SELECT detection_week, attribution_start, attribution_end "
        "FROM domain_volatility_events WHERE domain = %s"
    )
    domain_events_cache: dict = {}

    # ----------------------------------------------------------------
    # Per-opinion processing
    # ----------------------------------------------------------------
    for op in opinions:
        if (op.alpha + op.beta) == 0:
            continue

        p = op.alpha / (op.alpha + op.beta)

        # Step 1: Exactly-once gate
        result_mark = session.execute(
            mark_scored_stmt,
            (op.agent_id, op.domain, claim_id, datetime.utcnow(), p, outcome),
        )
        if not result_mark.one().applied:
            continue   # Already scored this opinion

        bs            = brier(p, outcome)
        error         = signed_error(p, outcome)
        sharpness     = abs(p - 0.5) * 2
        is_overconf   = 1 if (sharpness > 0.6 and bs > 0.25) else 0

        # Step 2: Agent ledger CAS
        for attempt in range(_MAX_CAS_RETRIES):
            row = session.execute(read_ledger_stmt, (op.agent_id, op.domain, week_bucket)).one()
            if row is None:
                res = session.execute(insert_ledger_stmt, (
                    op.agent_id, op.domain, week_bucket,
                    1, bs, error, is_overconf,
                ))
                if res.one().applied:
                    break
            else:
                c_count = row.prediction_count or 0
                res = session.execute(update_ledger_stmt, (
                    c_count + 1,
                    (row.brier_sum or 0.0) + bs,
                    (row.bias_sum  or 0.0) + error,
                    (row.overconfidence_count or 0) + is_overconf,
                    op.agent_id, op.domain, week_bucket,
                    c_count,
                ))
                if res.one().applied:
                    break
            if attempt == _MAX_CAS_RETRIES - 1:
                logger.warning("agent ledger CAS exhausted", agent_id=str(op.agent_id))

        # Step 3: Domain volatility ledger CAS
        is_confident      = 1 if (p > 0.7 or p < 0.3) else 0
        is_confident_wrong = 1 if (is_confident and bs > 0.25) else 0

        for attempt in range(_MAX_CAS_RETRIES):
            d_row = session.execute(read_domain_stmt, (op.domain, week_bucket)).one()
            if d_row is None:
                res = session.execute(insert_domain_stmt, (
                    op.domain, week_bucket,
                    1, bs, is_confident, is_confident_wrong,
                ))
                if res.one().applied:
                    break
            else:
                d_count = d_row.prediction_count or 0
                res = session.execute(update_domain_stmt, (
                    d_count + 1,
                    (d_row.brier_sum or 0.0) + bs,
                    (d_row.confident_prediction_count or 0) + is_confident,
                    (d_row.confident_wrong_count      or 0) + is_confident_wrong,
                    op.domain, week_bucket,
                    d_count,
                ))
                if res.one().applied:
                    break
            if attempt == _MAX_CAS_RETRIES - 1:
                logger.warning("domain ledger CAS exhausted", domain=op.domain)

        # Step 4: Agent resilience ledger — only for shock periods
        if op.domain not in domain_events_cache:
            domain_events_cache[op.domain] = list(
                session.execute(FETCH_EVENTS_CQL, (op.domain,))
            )

        events = domain_events_cache[op.domain]
        if not events:
            continue

        conf_weight    = (2.0 * abs(p - 0.5)) ** 2
        weighted_error = conf_weight * bs

        wb_date = week_bucket.date() if hasattr(week_bucket, "date") and callable(week_bucket.date) else week_bucket

        for ev in events:
            ev_start = ev.attribution_start.date() if hasattr(ev.attribution_start, "date") and callable(ev.attribution_start.date) else ev.attribution_start
            ev_end   = ev.attribution_end.date()   if hasattr(ev.attribution_end,   "date") and callable(ev.attribution_end.date)   else ev.attribution_end

            if not (ev_start <= wb_date <= ev_end):
                continue

            ev_det = ev.detection_week.date() if hasattr(ev.detection_week, "date") and callable(ev.detection_week.date) else ev.detection_week

            for attempt in range(_MAX_CAS_RETRIES):
                r_row = session.execute(read_resilience_stmt, (op.agent_id, op.domain, ev_det)).one()
                if r_row is None:
                    res = session.execute(insert_resilience_stmt, (
                        op.agent_id, op.domain, ev_det,
                        weighted_error, conf_weight,
                    ))
                    if res.one().applied:
                        break
                else:
                    curr_conf = r_row.shock_weighted_confidence_sum or 0.0
                    res = session.execute(update_resilience_stmt, (
                        (r_row.shock_weighted_error_sum or 0.0) + weighted_error,
                        curr_conf + conf_weight,
                        op.agent_id, op.domain, ev_det,
                        curr_conf,
                    ))
                    if res.one().applied:
                        break
                if attempt == _MAX_CAS_RETRIES - 1:
                    logger.warning("resilience ledger CAS exhausted", agent_id=str(op.agent_id))
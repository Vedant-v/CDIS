import urllib.request
import urllib.error
import json
import uuid
import time
from app.db.cassandra import get_session

# Constants
API_URL = "http://localhost:8000"
DECISION_ID = uuid.uuid4()
D1_ID = uuid.uuid4() # Root Claim
C1_ID = uuid.uuid4() # Volunteer Retention
C2_ID = uuid.uuid4() # Idol Production Capacity
C3_ID = uuid.uuid4() # Loan Repayment
C4_ID = uuid.uuid4() # Supply Chain Stability

domain_map = {
    C1_ID: "retention",
    C2_ID: "production",
    C3_ID: "repayment",
    C4_ID: "supply",
    D1_ID: "root",
}

print(f"Decision ID: {DECISION_ID}")
print("--- Claim IDs ---")
print(f"D1 (Root Claim): {D1_ID}")
print(f"C1 (Retention): {C1_ID}")
print(f"C2 (Production): {C2_ID}")
print(f"C3 (Repayment): {C3_ID}")
print(f"C4 (Supply): {C4_ID}")
print("-----------------")

def setup_dag_edges():
    print("Setting up DAG Edges...")
    session = get_session()
    
    # Clean previous edges for this decision only
    rows = session.execute("""
    SELECT parent_claim_id, child_claim_id
    FROM decision_claim_edges
    WHERE decision_id = %s
    """, (DECISION_ID,))
    
    delete_stmt = session.prepare("""
    DELETE FROM decision_claim_edges
    WHERE decision_id = ?
    AND parent_claim_id = ?
    AND child_claim_id = ?
    """)
    
    for r in rows:
        session.execute(delete_stmt, (DECISION_ID, r.parent_claim_id, r.child_claim_id))

    edges = [
        {"child": C4_ID, "parent": C2_ID, "w": 0.5, "root": False},
        {"child": C2_ID, "parent": C3_ID, "w": 0.4, "root": False},
        {"child": C1_ID, "parent": C3_ID, "w": 0.6, "root": False},
        {"child": C3_ID, "parent": D1_ID, "w": 0.7, "root": True},
        {"child": C4_ID, "parent": D1_ID, "w": 0.3, "root": True},
    ]
    
    insert_stmt = session.prepare("""
        INSERT INTO decision_claim_edges (decision_id, child_claim_id, parent_claim_id, weight, is_decision_root)
        VALUES (?, ?, ?, ?, ?)
    """)
    
    for edge in edges:
        session.execute(insert_stmt, (DECISION_ID, edge["child"], edge["parent"], edge["w"], edge["root"]))
    print("DAG Edges inserted.\n")

def submit_opinion(claim_id, agent_id, belief, confidence, cluster_id, domain):
    payload = {
        "claim_id": str(claim_id),
        "agent_id": str(agent_id),
        "belief": belief,
        "confidence": confidence,
        "domain": domain,
        "stakeholder_type": "auto",
        "cluster_id": cluster_id
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(f"{API_URL}/submit", data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req) as res:
            print("Submit OK:", res.read().decode())
    except urllib.error.HTTPError as e:
        print("HTTP ERROR:", e.code)
        print(e.read().decode())
    except Exception as e:
        print("Failed:", e)

def evaluate_decision():
    payload = {
        "domain_map": {str(k): v for k, v in domain_map.items()}
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(f"{API_URL}/decision/{str(DECISION_ID)}/evaluate", data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode())

def resolve_outcome(claim_id, outcome):
    payload = {
        "outcome": outcome,
        "observed_value": 0.0
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(f"{API_URL}/outcome/{str(claim_id)}", data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req) as res:
            print("Outcome OK:", res.read().decode())
    except Exception as e:
        print("Outcome Failed:", e)

from app.services.expertise_temporal import fetch_temporal_expertise

def fetch_expertise(agent_id, domain):
    exp = fetch_temporal_expertise(agent_id, domain)
    if exp:
        return {"calibration": exp.calibration_score, "bias": exp.bias_score, "entropy": exp.entropy_score}
    return None

def run_test_scenario():
    setup_dag_edges()
    
    # --- PHASE 1: VENDOR DOMINANCE ---
    print("--- [PHASE 1] Injecting Vendor Consensus ---")
    
    # 10 vendors submitting optimistic C4 (Supply Chain)
    vendor_agent = uuid.uuid4()
    for i in range(10):
        agent = vendor_agent if i == 0 else uuid.uuid4()
        submit_opinion(C4_ID, agent, belief=0.85, confidence=0.9, cluster_id="vendor", domain=domain_map[C4_ID])
        
    # Give Celery a second to process entropy workers (if running)
    print("Waiting for entropy worker sync...")
    time.sleep(10)
    
    res1 = evaluate_decision()
    print("EVALUATION 1 (Vendor Dominance):")
    print(res1)
    print("\n")
    
    # --- PHASE 2: INDEPENDENT DISSENT ---
    print("--- [PHASE 2] Injecting Independent Dissent ---")
    
    # 1 logistics expert dissenting on C4
    dissent_agent = uuid.uuid4()
    submit_opinion(C4_ID, dissent_agent, belief=0.42, confidence=0.8, cluster_id="logistics", domain=domain_map[C4_ID])
    
    # 3 mentors dissenting on C3
    for i in range(3):
        submit_opinion(C3_ID, uuid.uuid4(), belief=0.55, confidence=0.7, cluster_id="mentor", domain=domain_map[C3_ID])

    # Let Celery workers compute pre-outcome entropy credit
    print("Waiting for entropy worker sync...")
    time.sleep(10)
    
    res2 = evaluate_decision()
    print("EVALUATION 2 (After Dissent):")
    print(res2)
    print("\n")

    # --- PHASE 3: GROUND TRUTH OUTCOME RESOLUTION ---
    print("--- [PHASE 3] Ground Truth Resolution & Learning ---")
    
    # Observe that the supply chain actually failed (outcome = 0)
    resolve_outcome(C4_ID, 0.0)
    
    print("Waiting for expertise worker to perform Closed-Loop Learning...")
    # Give celery enough time to process and write to Cassandra
    time.sleep(6)
    
    print("Target Domain:", domain_map[C4_ID])
    print("Evaluating Expertise for Vendor Agent (Predicted 0.85, True 0.0):")
    cal_1 = fetch_expertise(vendor_agent, domain_map[C4_ID])
    print(cal_1)
    
    print("\nEvaluating Expertise for Dissent Agent (Predicted 0.42, True 0.0):")
    print(fetch_expertise(dissent_agent, domain_map[C4_ID]))
    print("\n")

    # --- PHASE 4A IDEMPOTENCY CHECK ---
    print("--- [PHASE 4A] Idempotency Double-Resolution Check ---")
    resolve_outcome(C4_ID, 0.0)
    print("Waiting for expertise worker to perform redundant Closed-Loop Learning...")
    time.sleep(6)
    
    print("Re-Evaluating Expertise for Vendor Agent (Should Match Exactly):")
    cal_2 = fetch_expertise(vendor_agent, domain_map[C4_ID])
    print(cal_2)
    
    if cal_1 == cal_2:
        print("\n✅ SUCCESS: Idempotency Verified! Temporal Decay is stable.")
    else:
        print("\n❌ FAILED: Idempotency broken. Double-counting detected.")

    # --- PHASE 4B: DOMAIN-CONDITIONED DECAY ---
    print("--- [PHASE 4B] Domain-Conditioned Decay ---")
    
    test_agent = uuid.uuid4()
    from datetime import date, timedelta
    from app.db.cassandra import get_session
    
    session = get_session()
    
    # 12 weeks ago: Great performance (brier=0)
    past_date = date.today() - timedelta(weeks=12)
    # This week: Terrible performance (brier=10)
    recent_date = date.today()
    
    insert_hist = session.prepare("""
    INSERT INTO expertise_ledger_by_agent (agent_id, domain, week_bucket, prediction_count, brier_sum, bias_sum, overconfidence_count)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """)
    
    # Insert for 'supply' (fast decay, lam=0.15)
    session.execute(insert_hist, (test_agent, "supply", past_date, 10, 0.0, 0.0, 0))
    session.execute(insert_hist, (test_agent, "supply", recent_date, 10, 10.0, 10.0, 0))
    
    # Insert for 'retention' (slow decay, lam=0.04)
    session.execute(insert_hist, (test_agent, "retention", past_date, 10, 0.0, 0.0, 0))
    session.execute(insert_hist, (test_agent, "retention", recent_date, 10, 10.0, 10.0, 0))
    
    cal_supply = fetch_expertise(test_agent, "supply")["calibration"]
    cal_retention = fetch_expertise(test_agent, "retention")["calibration"]
    
    print(f"Calibration Supply (Fast Decay): {cal_supply:.4f}")
    print(f"Calibration Retention (Slow Decay): {cal_retention:.4f}")
    
    if cal_supply < cal_retention:
        print("\n✅ SUCCESS: Domain-Conditioned Decay Verified! (Supply forgotten faster than Retention)")
    else:
        print("\n❌ FAILED: Supply calibration should be lower than Retention calibration due to faster decay.")

    # --- PHASE 4C: ADAPTIVE DOMAIN VOLATILITY (CQRS) ---
    print("--- [PHASE 4C] Self-Tuning Lambda (Regime Drift via CQRS) ---")
    
    from app.services.domain_decay import auto_tune_lambda_task, get_domain_lambda, _cache
    
    test_domain = "repayment"
    if test_domain in _cache:
        del _cache[test_domain]
        
    initial_lam = get_domain_lambda(test_domain)
    print(f"Initial λ for {test_domain}: {initial_lam:.4f}")
    
    test_epoch = uuid.uuid1()
    
    # Simulate Monotonic Epoch in DB
    session.execute("INSERT INTO domain_projection_epoch (domain, projection_epoch) VALUES (%s, %s)", (test_domain, test_epoch))

    # Insert Mock Projections (Highly confident, but wrong = High ErrorRate = Shock)
    insert_proj = session.prepare("""
    INSERT INTO claim_fusion_projection_by_epoch 
    (domain, projection_epoch, claim_id, st_alpha, st_beta, lt_alpha, lt_beta, st_conf, lt_conf)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """)
    
    insert_outcome = session.prepare("INSERT INTO claim_outcomes (claim_id, outcome, observed_value, resolved_at) VALUES (?, ?, ?, ?)")
    
    for _ in range(20):
        mock_claim = uuid.uuid4()
        # ST is highly confident (alpha=90, beta=10 -> 90%) but Outcome is 0.0 (Total Shock)
        # LT is highly confident AND correct (alpha=10, beta=90 -> 10% -> pred 0.0).  Shock is difference in error rates. 
        session.execute(insert_proj, (test_domain, test_epoch, mock_claim, 90.0, 10.0, 10.0, 90.0, 0.9, 0.9))
        session.execute(insert_outcome, (mock_claim, 0.0, 0.0, date.today()))

    # Trigger Auto-Tune Task with the Epoch
    auto_tune_lambda_task(test_domain, str(test_epoch))
    
    adapted_lam = get_domain_lambda(test_domain)
    print(f"Adapted λ after Shock for {test_domain}: {adapted_lam:.4f}")
    
    if adapted_lam > initial_lam:
        print("✅ SUCCESS: Adaptive Volatility CQRS Verified! (λ increased)")
    else:
        print("❌ FAILED: λ did not adapt.")

    # --- PHASE 4D: SELECTIVE FORGETTING (VRS) ---
    print("\n--- [PHASE 4D] Selective Forgetting (Targeted VRS Preservation) ---")
    
    herd_agent = uuid.uuid4()
    dissenter_agent = uuid.uuid4()
    test_domain = "supply"
    
    # 1. 12 weeks ago (Outside attribution window)
    # Both agents were great, they built up old high credibility
    past_date_12w = date.today() - timedelta(weeks=12)
    session.execute(insert_hist, (herd_agent, test_domain, past_date_12w, 10, 0.0, 0.0, 0))
    session.execute(insert_hist, (dissenter_agent, test_domain, past_date_12w, 10, 0.0, 0.0, 0))
    
    # 2. 2 weeks ago (Inside attribution window - The Collapse)
    # Herd was confidently wrong (Brier=1.0 per pred)
    # Dissenter was confidently right (Brier=0.0 per pred)
    collapse_date = date.today() - timedelta(weeks=2)
    session.execute(insert_hist, (herd_agent, test_domain, collapse_date, 10, 10.0, 10.0, 10))
    session.execute(insert_hist, (dissenter_agent, test_domain, collapse_date, 10, 0.0, 0.0, 0))
    
    # Simulate the Domain Volatility Event for this collapse
    detection_week = date.today() - timedelta(weeks=1)
    
    insert_volatility_event = session.prepare("""
    INSERT INTO domain_volatility_events (domain, detection_week, attribution_start, attribution_end)
    VALUES (?, ?, ?, ?)
    """)
    session.execute(insert_volatility_event, (test_domain, detection_week, collapse_date, collapse_date))
    
    # Simulate Agent Resilience Projection (The O(1) Phase 5 read)
    insert_vrs_projection = session.prepare("""
    INSERT INTO agent_resilience_projection (agent_id, domain, projection_epoch, vrs)
    VALUES (?, ?, ?, ?)
    """)
    
    test_epoch_vrs = uuid.uuid1()
    
    # Herd was completely wrong, their VRS is 0.0
    session.execute(insert_vrs_projection, (herd_agent, test_domain, test_epoch_vrs, 0.0))
    
    # Dissenter was completely right, their VRS is 1.0 (100% immune to the lambda shock)
    session.execute(insert_vrs_projection, (dissenter_agent, test_domain, test_epoch_vrs, 1.0))

    # Evaluate both agents.
    herd_cal = fetch_expertise(herd_agent, test_domain)["calibration"]
    dissenter_cal = fetch_expertise(dissenter_agent, test_domain)["calibration"]
    
    print(f"Herd Calibration (VRS=0): {herd_cal:.4f}")
    print(f"Dissenter Calibration (VRS=1.0): {dissenter_cal:.4f}")
    
    if dissenter_cal > herd_cal + 0.3:
        print("✅ SUCCESS: Selective Forgetting Verified! (Dissenter's accurate performance preserved)")
    else:
        print("❌ FAILED: Dissenter did not retain enough advantage despite VRS preservation.")

if __name__ == "__main__":
    run_test_scenario()

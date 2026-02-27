"""
opinion_service.py
------------------
Converts human-supplied (belief, confidence) into Beta distribution parameters.
"""

# Hard cap on evidence strength per single opinion submission.
# Prevents one very confident agent from overwhelming the entire fusion pool.
# This is intentionally NOT domain-configurable at runtime — it is a
# mathematical safety invariant, not a tuning knob.
MAX_OPINION_STRENGTH: float = 50.0


def belief_to_beta(belief: float, confidence: float):
    """
    Map belief ∈ [0, 1] and confidence ∈ [0, 1] → (alpha, beta) > 0.

    alpha = belief      × strength
    beta  = (1-belief)  × strength
    where strength = confidence × MAX_OPINION_STRENGTH

    Clipping prevents degenerate distributions (alpha or beta → 0).
    """
    belief     = max(0.01, min(0.99, belief))
    confidence = max(0.01, min(1.00, confidence))

    strength = confidence * MAX_OPINION_STRENGTH
    alpha    = belief       * strength
    beta     = (1.0 - belief) * strength

    return alpha, beta
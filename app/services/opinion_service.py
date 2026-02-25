def belief_to_beta(belief: float, confidence: float):
    """
    Convert:
    belief + confidence -> Beta parameters (alpha, beta)
    
    Constraint: Beta parameters should strictly be > 0.
    """
    # Clip bounds to prevent degenerate distributions
    belief = max(0.01, min(0.99, belief))
    confidence = max(0.01, min(1.0, confidence))
    
    strength = confidence * 20

    alpha = belief * strength
    beta = (1 - belief) * strength

    return alpha, beta

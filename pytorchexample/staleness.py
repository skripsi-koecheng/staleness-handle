def polynomial_staleness_weight(tau: int, alpha: float) -> float:
    """Polynomial decay: (tau + 1)^(-alpha). tau=0 gives weight 1.0."""
    return (tau + 1) ** (-alpha)

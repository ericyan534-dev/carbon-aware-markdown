"""Behavioral gates composing the effective substitution coefficient delta_eff.

delta_eff(RSL, d) = g_fresh(RSL) * g_spoil(RSL) * g_reb(d)

The rebound gate is micro-founded: with constant-elasticity demand
q(d) = q0 (1-d)^(-eps), the share of discounted purchases that would also have
occurred at full price is q0 / q(d) = (1-d)^eps. Incremental purchases displace
no counterfactual production, except a cannibalized fraction kappa that does.
Its first-order expansion 1 - eps*d is the linear gate of the original model.
"""
import numpy as np


def _check_discount(d):
    d = np.asarray(d, dtype=float)
    if np.any((d < 0) | (d >= 1)):
        raise ValueError("discount must lie in [0, 1)")
    return d


def freshness_gate(rsl, lam):
    return 1.0 - np.exp(-lam * np.asarray(rsl, dtype=float))


def spoilage_gate(rsl, lam, p_base):
    p_waste = p_base + (1.0 - p_base) * np.exp(-lam * np.asarray(rsl, dtype=float))
    return 1.0 - p_waste


def rebound_gate(discount, eps, kappa=0.0):
    d = _check_discount(discount)
    retained = (1.0 - d) ** eps
    return retained + kappa * (1.0 - retained)


def rebound_gate_linear(discount, eps):
    d = _check_discount(discount)
    return np.clip(1.0 - eps * d, 0.0, 1.0)


def delta_eff(rsl, discount, lam, p_base, eps, kappa=0.0):
    return (freshness_gate(rsl, lam) * spoilage_gate(rsl, lam, p_base)
            * rebound_gate(discount, eps, kappa))

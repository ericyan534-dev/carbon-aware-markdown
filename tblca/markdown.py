"""Finite-horizon markdown DP for a near-expiry batch under TB-LCA accounting.

State: (r, x) = remaining shelf-life days, units on hand. Action: discount d.
Demand ~ Poisson(q0 * a(r) * (1-d)^(-eps)), with appeal a(r) = 1 - exp(-lam_a r).
Per unit sold, the share g(d) = (1-d)^eps + kappa (1 - (1-d)^eps) displaces a fresh
full-price purchase: it earns the production credit and costs the fresh margin.
Unsold units at r = 0 are discarded (the System-0 counterfactual), earning no credit.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import poisson

from tblca.substitution import freshness_gate, spoilage_gate, rebound_gate

OBJECTIVES = ("profit", "carbon", "waste", "weighted")
METRICS = ("profit", "carbon_kg", "waste_units", "sales", "revenue", "disc_units", "rsl_units")


@dataclass(frozen=True)
class MarkdownProblem:
    horizon: int
    x0: int
    q0: float
    lam_a: float
    lam_f: float
    p_base: float
    eps: float
    kappa: float
    price: float
    margin_fresh: float
    unit_kg: float
    ef_prod: float
    ef_eol: float
    ops_kg: float
    carbon_price: float = 0.0          # $ per kg CO2e, used by "weighted"
    waste_penalty: float = 0.0         # $ per unsold unit at expiry, used by "weighted"
    objective: str = "weighted"
    discounts: tuple = field(default=(0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6))

    def __post_init__(self):
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}")
        if self.horizon < 1 or self.x0 < 0:
            raise ValueError("horizon >= 1 and x0 >= 0 required")


@dataclass(frozen=True)
class Solution:
    value: np.ndarray       # (horizon+1, x0+1)
    policy: np.ndarray      # discount chosen at (r, x); row 0 unused
    metrics: dict = field(default_factory=dict)


def unit_economics(p: MarkdownProblem, r: int, d: float) -> dict:
    g = float(rebound_gate(d, p.eps, p.kappa))
    fresh = float(freshness_gate(r, p.lam_f))
    consumed = float(spoilage_gate(r, p.lam_f, p.p_base))
    carbon = p.unit_kg * (g * fresh * consumed * p.ef_prod + consumed * p.ef_eol - p.ops_kg)
    revenue = p.price * (1.0 - d)
    return {"profit": revenue - g * p.margin_fresh, "carbon_kg": carbon,
            "revenue": revenue, "sales": 1.0, "waste_units": 0.0,
            "disc_units": d, "rsl_units": float(r)}


def demand_rate(p: MarkdownProblem, r: int, d: float) -> float:
    return p.q0 * float(freshness_gate(r, p.lam_a)) * (1.0 - d) ** (-p.eps)


def _sales_pmf(rate, x):
    pmf = poisson.pmf(np.arange(x), rate)
    return np.append(pmf, max(0.0, 1.0 - pmf.sum()))


def stage_outcomes(p: MarkdownProblem, x: int, r: int, d: float) -> dict:
    pmf = _sales_pmf(demand_rate(p, r, d), x)
    sales = float(pmf @ np.arange(x + 1))
    unit = unit_economics(p, r, d)
    out = {k: sales * v for k, v in unit.items()}
    return {**out, "sales": sales, "pmf_sales": pmf}


def _reward(p: MarkdownProblem, out: dict) -> float:
    if p.objective == "profit":
        return out["profit"]
    if p.objective == "carbon":
        return out["carbon_kg"]
    if p.objective == "waste":
        return 0.0
    return out["profit"] + p.carbon_price * out["carbon_kg"]


def _terminal(p: MarkdownProblem) -> np.ndarray:
    x = np.arange(p.x0 + 1, dtype=float)
    if p.objective == "waste":
        return -x
    if p.objective == "weighted":
        return -p.waste_penalty * x
    return np.zeros_like(x)


def _continuation(v_next, x, pmf):
    return float(pmf @ v_next[x - np.arange(x + 1)])


def solve(p: MarkdownProblem) -> Solution:
    value = np.zeros((p.horizon + 1, p.x0 + 1))
    policy = np.zeros_like(value)
    value[0] = _terminal(p)
    for r in range(1, p.horizon + 1):
        for x in range(p.x0 + 1):
            best, best_d = -np.inf, p.discounts[0]
            for d in p.discounts:
                out = stage_outcomes(p, x, r, d)
                q = _reward(p, out) + _continuation(value[r - 1], x, out["pmf_sales"])
                if q > best + 1e-12:
                    best, best_d = q, d
            value[r, x], policy[r, x] = best, best_d
    return Solution(value=value, policy=policy,
                    metrics=evaluate_policy(p, policy).metrics)


def fixed_policy(p: MarkdownProblem, d) -> np.ndarray:
    """Constant discount d, or a callable d(r) giving a time-only schedule."""
    pol = np.zeros((p.horizon + 1, p.x0 + 1))
    for r in range(1, p.horizon + 1):
        pol[r] = d(r) if callable(d) else d
    return pol


def evaluate_policy(p: MarkdownProblem, policy: np.ndarray) -> Solution:
    shape = (p.horizon + 1, p.x0 + 1)
    vals = {m: np.zeros(shape) for m in METRICS}
    vals["waste_units"][0] = np.arange(p.x0 + 1)
    value = np.zeros(shape)
    value[0] = _terminal(p)
    for r in range(1, p.horizon + 1):
        for x in range(p.x0 + 1):
            out = stage_outcomes(p, x, r, float(policy[r, x]))
            pmf = out["pmf_sales"]
            value[r, x] = _reward(p, out) + _continuation(value[r - 1], x, pmf)
            for m in METRICS:
                stage = 0.0 if m == "waste_units" else out[m]
                vals[m][r, x] = stage + _continuation(vals[m][r - 1], x, pmf)
    metrics = {m: float(vals[m][p.horizon, p.x0]) for m in METRICS}
    return Solution(value=value, policy=policy, metrics=metrics)


def simulate(p: MarkdownProblem, policy: np.ndarray, n=10000, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    acc = {m: np.zeros(n) for m in METRICS}
    x = np.full(n, p.x0)
    for r in range(p.horizon, 0, -1):
        d = policy[r, x]
        rates = np.array([demand_rate(p, r, di) for di in d])
        sold = np.minimum(rng.poisson(rates), x)
        for i, di in enumerate(np.unique(d)):
            mask = d == di
            unit = unit_economics(p, r, float(di))
            for m in ("profit", "carbon_kg", "revenue", "sales", "disc_units", "rsl_units"):
                acc[m][mask] += sold[mask] * unit[m]
        x = x - sold
    acc["waste_units"] = x.astype(float)
    return {m: float(v.mean()) for m, v in acc.items()}

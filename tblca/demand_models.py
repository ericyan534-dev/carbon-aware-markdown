"""Alternative 'true' demand models for misspecification and robustness experiments.

The paper's planning model (tblca.markdown) uses CES-Poisson demand:
    rate = q0 * a(r) * (1-d)^(-eps),  sales ~ Poisson(rate) capped at stock,
and the displacement gate g(d) = s(d) + kappa (1 - s(d)) with retained share
s(d) = q(0)/q(d) = (1-d)^eps.

Here a DemandModel generalises the truth along three axes:
  * shape: "ces" (paper), "linear_chord" (linear in d, same q0 at d=0 and same
    demand -- hence the same log-arc elasticity -- at the anchor discount), or
    "linear_tangent" (linear in d, tangent to the CES curve at the anchor: same
    level and point elasticity there);
  * dispersion: variance/mean of daily demand (1 = Poisson, >1 = negative binomial);
  * eps_true: the true elasticity (defaults to the planning problem's eps).
The retained share used by the gate is always computed from the *true* demand
curve, s(d) = m(0)/m(d), so profit and carbon stay internally consistent.
With DemandModel() the functions reproduce tblca.markdown exactly.
"""
from dataclasses import dataclass

import numpy as np
from scipy.stats import nbinom, poisson

from tblca.markdown import METRICS, MarkdownProblem, _reward, _terminal
from tblca.substitution import freshness_gate, spoilage_gate

SHAPES = ("ces", "linear_chord", "linear_tangent")


@dataclass(frozen=True)
class DemandModel:
    shape: str = "ces"
    dispersion: float = 1.0
    eps_true: float | None = None
    anchor_d: float = 0.3

    def __post_init__(self):
        if self.shape not in SHAPES:
            raise ValueError(f"shape must be one of {SHAPES}")
        if self.dispersion < 1.0:
            raise ValueError("dispersion (variance/mean) must be >= 1")
        if not 0.0 < self.anchor_d < 1.0:
            raise ValueError("anchor_d must lie in (0, 1)")
        if self.eps_true is not None and self.eps_true <= 0:
            raise ValueError("eps_true must be positive")


def _eps(p: MarkdownProblem, m: DemandModel) -> float:
    return p.eps if m.eps_true is None else m.eps_true


def multiplier(p: MarkdownProblem, m: DemandModel, d: float) -> float:
    """Demand multiplier relative to the full-price CES level q0 * a(r)."""
    eps, a = _eps(p, m), m.anchor_d
    if m.shape == "ces":
        return (1.0 - d) ** (-eps)
    level = (1.0 - a) ** (-eps)
    if m.shape == "linear_chord":
        return 1.0 + (level - 1.0) / a * d
    slope = eps * (1.0 - a) ** (-eps - 1.0)
    return max(0.0, level + slope * (d - a))


def retained_share(p: MarkdownProblem, m: DemandModel, d: float) -> float:
    """Share of purchases at discount d that would also occur at full price."""
    md, m0 = multiplier(p, m, d), multiplier(p, m, 0.0)
    if md <= 0.0:
        return 1.0
    return float(min(1.0, max(0.0, m0 / md)))


def gate(p: MarkdownProblem, m: DemandModel, d: float) -> float:
    s = retained_share(p, m, d)
    return s + p.kappa * (1.0 - s)


def rate(p: MarkdownProblem, m: DemandModel, r: int, d: float) -> float:
    return p.q0 * float(freshness_gate(r, p.lam_a)) * multiplier(p, m, d)


def sales_pmf(lam: float, x: int, dispersion: float = 1.0) -> np.ndarray:
    """PMF of min(D, x) for D with mean lam and variance dispersion*lam."""
    k = np.arange(x)
    if lam <= 0.0:
        pmf = np.zeros(x)
        if x > 0:
            pmf[0] = 1.0
        return np.append(pmf, 0.0 if x > 0 else 1.0)
    if dispersion <= 1.0 + 1e-12:
        pmf = poisson.pmf(k, lam)
    else:
        pmf = nbinom.pmf(k, lam / (dispersion - 1.0), 1.0 / dispersion)
    return np.append(pmf, max(0.0, 1.0 - pmf.sum()))


def unit_economics(p: MarkdownProblem, m: DemandModel, r: int, d: float) -> dict:
    g = gate(p, m, d)
    fresh = float(freshness_gate(r, p.lam_f))
    consumed = float(spoilage_gate(r, p.lam_f, p.p_base))
    carbon = p.unit_kg * (g * fresh * consumed * p.ef_prod + consumed * p.ef_eol - p.ops_kg)
    revenue = p.price * (1.0 - d)
    return {"profit": revenue - g * p.margin_fresh, "carbon_kg": carbon, "revenue": revenue,
            "sales": 1.0, "waste_units": 0.0, "disc_units": d, "rsl_units": float(r)}


def stage_outcomes(p: MarkdownProblem, m: DemandModel, x: int, r: int, d: float) -> dict:
    pmf = sales_pmf(rate(p, m, r, d), x, m.dispersion)
    sales = float(pmf @ np.arange(x + 1))
    unit = unit_economics(p, m, r, d)
    return {**{k: sales * v for k, v in unit.items()}, "sales": sales, "pmf_sales": pmf}


def _continuation(v_next, x, pmf):
    return float(pmf @ v_next[x - np.arange(x + 1)])


def evaluate_under(p: MarkdownProblem, policy: np.ndarray, m: DemandModel) -> dict:
    """Exact expected metrics (and objective value) of `policy` when demand follows `m`."""
    shape = (p.horizon + 1, p.x0 + 1)
    vals = {k: np.zeros(shape) for k in METRICS}
    vals["waste_units"][0] = np.arange(p.x0 + 1)
    value = np.zeros(shape)
    value[0] = _terminal(p)
    for r in range(1, p.horizon + 1):
        for x in range(p.x0 + 1):
            out = stage_outcomes(p, m, x, r, float(policy[r, x]))
            pmf = out["pmf_sales"]
            value[r, x] = _reward(p, out) + _continuation(value[r - 1], x, pmf)
            for k in METRICS:
                stage = 0.0 if k == "waste_units" else out[k]
                vals[k][r, x] = stage + _continuation(vals[k][r - 1], x, pmf)
    metrics = {k: float(vals[k][p.horizon, p.x0]) for k in METRICS}
    return {**metrics, "value": float(value[p.horizon, p.x0])}


def solve_under(p: MarkdownProblem, m: DemandModel):
    """Optimal policy when demand truly follows `m` (oracle). Returns (policy, metrics)."""
    value = np.zeros((p.horizon + 1, p.x0 + 1))
    policy = np.zeros_like(value)
    value[0] = _terminal(p)
    for r in range(1, p.horizon + 1):
        for x in range(p.x0 + 1):
            best, best_d = -np.inf, p.discounts[0]
            for d in p.discounts:
                out = stage_outcomes(p, m, x, r, d)
                q = _reward(p, out) + _continuation(value[r - 1], x, out["pmf_sales"])
                if q > best + 1e-12:
                    best, best_d = q, d
            value[r, x], policy[r, x] = best, best_d
    return policy, evaluate_under(p, policy, m)

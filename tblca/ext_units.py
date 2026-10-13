"""Per-unit economics for two behavioural extensions (plug into `ext_dp`).

Re-spending rebound
-------------------
A buyer who displaces a full-price purchase (share g = g_disp(d)) pays p(1-d)
instead of p and saves p*d. A share `mpc` of the saving is re-spent on other
consumption with intensity I (kg CO2e per $), adding I*mpc*p*d*g kg per unit.
Incremental buyers (share 1-g) did not intend to buy and *spend* p(1-d):

* ``incremental="none"`` (headline, conservative): their outlay is treated as
  carbon-neutral (financed from saving or from food spending already in the
  counterfactual), so only displacing buyers' savings create rebound.
* ``incremental="diversion"`` (budget-neutral bound): under a fixed budget their
  outlay replaces other consumption of equal value, which is *avoided* at the
  same intensity, crediting I*mpc*p*(1-d)*(1-g) per unit.

Net re-spending emissions per unit sold:
    R(d) = I * mpc * p * [ g d - beta (1-g)(1-d) ],   beta in {0, 1}.

Strategic waiting
-----------------
A share s of full-price fresh buyers has learned the markdown pattern. A
strategic buyer with a wait cost uniform on [0, p] waits for a discount d iff
p*d exceeds the cost, i.e. with probability d. Part of the demand response that
the elasticity attributes to incremental buyers is therefore shifted fresh
demand: of the incremental share (1-g), a fraction s*d are in fact waiters who
would otherwise have bought fresh at full price. The displacing share becomes
    g_s(d) = g(d) + s * d * (1 - g(d)),
so each such unit costs the fresh margin c_f (cannibalization) and, because
the fresh unit is not bought, also earns the displaced-production credit
(``credit=True``). ``credit=False`` charges the cannibalization cost only.
"""
from typing import Callable

from tblca.markdown import MarkdownProblem
from tblca.substitution import freshness_gate, rebound_gate, spoilage_gate

INCREMENTAL_MODES = ("none", "diversion")


def _base_terms(p: MarkdownProblem, r: int, d: float):
    g = float(rebound_gate(d, p.eps, p.kappa))
    fresh = float(freshness_gate(r, p.lam_f))
    consumed = float(spoilage_gate(r, p.lam_f, p.p_base))
    return g, fresh, consumed


def _unit(p, r, d, g_profit, g_carbon, fresh, consumed, extra_kg=0.0) -> dict:
    carbon = p.unit_kg * (g_carbon * fresh * consumed * p.ef_prod
                          + consumed * p.ef_eol - p.ops_kg) + extra_kg
    revenue = p.price * (1.0 - d)
    return {"profit": revenue - g_profit * p.margin_fresh, "carbon_kg": carbon,
            "revenue": revenue, "sales": 1.0, "waste_units": 0.0,
            "disc_units": d, "rsl_units": float(r)}


def _validate_respend(intensity: float, mpc: float, incremental: str) -> None:
    if incremental not in INCREMENTAL_MODES:
        raise ValueError(f"incremental must be one of {INCREMENTAL_MODES}")
    if intensity < 0 or not 0.0 <= mpc <= 1.0:
        raise ValueError("intensity >= 0 and 0 <= mpc <= 1 required")


def respend_kg(p: MarkdownProblem, d: float, intensity: float, mpc: float = 1.0,
               incremental: str = "none") -> float:
    """Net re-spending emissions (kg CO2e) per unit sold at discount d."""
    _validate_respend(intensity, mpc, incremental)
    g = float(rebound_gate(d, p.eps, p.kappa))
    beta = 1.0 if incremental == "diversion" else 0.0
    return intensity * mpc * p.price * (g * d - beta * (1.0 - g) * (1.0 - d))


def respend_unit_fn(intensity: float, mpc: float = 1.0,
                    incremental: str = "none") -> Callable:
    _validate_respend(intensity, mpc, incremental)

    def unit_fn(p: MarkdownProblem, r: int, d: float) -> dict:
        g, fresh, consumed = _base_terms(p, r, d)
        extra = -respend_kg(p, d, intensity, mpc, incremental)
        return _unit(p, r, d, g, g, fresh, consumed, extra_kg=extra)
    return unit_fn


def waiting_share(g: float, d: float, s: float, stress: bool = False) -> float:
    """Displacing share with strategic waiters. Waiting propensity is d (uniform
    wait cost); stress=True makes every strategic buyer wait for any d > 0."""
    h = (1.0 if d > 0 else 0.0) if stress else d
    return g + s * h * (1.0 - g)


def waiting_unit_fn(s: float, credit: bool = True, stress: bool = False) -> Callable:
    if not 0.0 <= s <= 1.0:
        raise ValueError("strategic share s must lie in [0, 1]")

    def unit_fn(p: MarkdownProblem, r: int, d: float) -> dict:
        g, fresh, consumed = _base_terms(p, r, d)
        g_s = waiting_share(g, d, s, stress)
        return _unit(p, r, d, g_s, g_s if credit else g, fresh, consumed)
    return unit_fn

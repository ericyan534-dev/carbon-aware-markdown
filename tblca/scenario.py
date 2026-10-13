"""Build MarkdownProblem instances from archetype x region and define baseline policies."""
from dataclasses import replace

from tblca.markdown import MarkdownProblem, fixed_policy
from tblca.params import ARCHETYPES, REGIONS, ops_per_kg

DEFAULTS = dict(q0=8.0, stock_ratio=2.0, lam_a=0.7, lam_f=1.5, eps=1.94, kappa=0.67,
                margin_ratio=0.30)


def build(archetype: str, region: str, objective="weighted", carbon_price=0.0, **over):
    cfg = {**DEFAULTS, **over}
    a, g = ARCHETYPES[archetype], REGIONS[region]
    x0 = int(round(cfg["stock_ratio"] * cfg["q0"] * a.horizon))
    return MarkdownProblem(
        horizon=a.horizon, x0=x0, q0=cfg["q0"], lam_a=cfg["lam_a"], lam_f=cfg["lam_f"],
        p_base=cfg.get("p_base", a.p_base), eps=cfg["eps"], kappa=cfg["kappa"],
        price=a.price, margin_fresh=cfg["margin_ratio"] * a.price, unit_kg=a.unit_kg,
        ef_prod=cfg.get("ef_prod", a.ef_prod), ef_eol=cfg.get("ef_eol", g.eol(a)),
        ops_kg=ops_per_kg(g, a.horizon), carbon_price=carbon_price, objective=objective)


def heuristic_policies(p: MarkdownProblem) -> dict:
    return {
        "No markdown": fixed_policy(p, 0.0),
        "Last-day 35%": fixed_policy(p, lambda r: 0.35 if r == 1 else 0.0),
        "Flat 30%": fixed_policy(p, 0.3),
    }


def with_objective(p: MarkdownProblem, objective, carbon_price=None):
    cp = p.carbon_price if carbon_price is None else carbon_price
    return replace(p, objective=objective, carbon_price=cp)

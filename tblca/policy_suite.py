"""The paper's five-policy comparison on one problem, plus headline-ratio derivation.

Policies are optimised under the planning model (tblca.markdown, CES-Poisson) and
evaluated either under the same model or under a different true DemandModel.
"""
from dataclasses import replace

import numpy as np

from tblca.demand_models import DemandModel, evaluate_under
from tblca.markdown import MarkdownProblem, fixed_policy, solve

POLICIES = ("static", "waste", "profit", "carbon", "scc")
DOMINANCE_TOL = 1e-6   # relative tolerance for weak dominance


def static_rule(p: MarkdownProblem) -> np.ndarray:
    return fixed_policy(p, lambda r: 0.35 if r == 1 else 0.0)


def plan_policies(p: MarkdownProblem) -> dict:
    """Optimal policies under the planning model; p.carbon_price is the SCC."""
    pols = {"static": static_rule(p)}
    for name, obj in (("waste", "waste"), ("profit", "profit"), ("carbon", "carbon"),
                      ("scc", "weighted")):
        pols[name] = solve(replace(p, objective=obj)).policy
    return pols


def naive_carbon(p: MarkdownProblem, sales: float) -> float:
    """Claimed carbon if every sold unit fully displaced production."""
    return sales * p.unit_kg * (p.ef_prod + p.ef_eol - p.ops_kg)


def evaluate_all(p: MarkdownProblem, pols: dict, model: DemandModel | None = None) -> dict:
    m = model or DemandModel()
    q = replace(p, objective="weighted")
    return {k: evaluate_under(q, pol, m) for k, pol in pols.items()}


def headline(p: MarkdownProblem, ev: dict) -> dict:
    """Headline ratios of the paper from evaluated metrics of the five policies."""
    s, w, pm, c, scc = (ev[k] for k in POLICIES)
    tol_p = DOMINANCE_TOL * max(1.0, abs(w["profit"]))
    tol_c = DOMINANCE_TOL * max(1.0, abs(w["carbon_kg"]))
    return {
        "profit_gain_scc_vs_static": 100 * (scc["profit"] / s["profit"] - 1),
        "carbon_gain_scc_vs_static": 100 * (scc["carbon_kg"] / s["carbon_kg"] - 1),
        "scc_beats_static_both": bool(scc["profit"] > s["profit"] and scc["carbon_kg"] > s["carbon_kg"]),
        "carbon_dominates_waste": bool(c["profit"] >= w["profit"] - tol_p
                                       and c["carbon_kg"] >= w["carbon_kg"] - tol_c),
        "carbon_vs_waste_profit_pct": 100 * (c["profit"] / w["profit"] - 1),
        "carbon_vs_waste_carbon_pct": 100 * (c["carbon_kg"] / w["carbon_kg"] - 1),
        "scc_retention_pct": 100 * scc["carbon_kg"] / w["carbon_kg"],
        "scc_profit_gain_vs_waste_pct": 100 * (scc["profit"] / w["profit"] - 1),
        "profitmax_share_scc_carbon_pct": 100 * pm["carbon_kg"] / scc["carbon_kg"],
        "overstatement_static_pct": 100 * (naive_carbon(p, s["sales"]) / s["carbon_kg"] - 1),
    }


def run(p: MarkdownProblem, model: DemandModel | None = None) -> tuple[dict, dict]:
    ev = evaluate_all(p, plan_policies(p), model)
    return ev, headline(p, ev)

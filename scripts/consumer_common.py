"""Shared helpers for the consumer-behavior robustness checks (policy table under a unit_fn)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tblca.ext_dp import evaluate_ext, solve_ext  # noqa: E402
from tblca.markdown import unit_economics  # noqa: E402
from tblca.params import ARCHETYPES, REGIONS, SCC_USD_PER_KG  # noqa: E402
from tblca.scenario import build, heuristic_policies, with_objective  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs" / "consumer"
RULE, OURS = "Last-day 35%", "SCC-weighted (ours)"
SOLVED = (("waste", "Waste-optimal"), ("profit", "Profit-optimal"),
          ("carbon", "Carbon-optimal"), ("weighted", OURS))
CELLS = [(a, g) for a in ARCHETYPES for g in REGIONS]


def policy_table(arch, region, unit_fn=unit_economics, **over):
    """Same policy set as scripts/run_experiments.policy_table, under unit_fn."""
    base = build(arch, region, carbon_price=SCC_USD_PER_KG, **over)
    rows = {name: evaluate_ext(base, pol, unit_fn).metrics
            for name, pol in heuristic_policies(base).items()}
    for obj, label in SOLVED:
        rows[label] = solve_ext(with_objective(base, obj), unit_fn).metrics
    df = pd.DataFrame(rows).T
    df["waste_pct"] = 100 * df["waste_units"] / base.x0
    df["mean_disc"] = df["disc_units"] / df["sales"]
    df["naive_carbon_kg"] = df["sales"] * base.unit_kg * (base.ef_prod + base.ef_eol - base.ops_kg)
    return df.rename_axis("policy").reset_index().assign(archetype=arch, region=region)


def cell_summary(df: pd.DataFrame) -> dict:
    s = df.set_index("policy")
    w, c, o = s.loc["Waste-optimal"], s.loc["Carbon-optimal"], s.loc[OURS]
    pm, rule = s.loc["Profit-optimal"], s.loc[RULE]
    return dict(
        scc_profit_gain_vs_rule_pct=100 * (o.profit / rule.profit - 1),
        scc_carbon_gain_vs_rule_pct=100 * (o.carbon_kg / rule.carbon_kg - 1),
        carbonopt_dominates_wastemin=bool(c.profit >= w.profit - 1e-9
                                          and c.carbon_kg >= w.carbon_kg - 1e-9),
        carbonopt_profit_vs_waste_pct=100 * (c.profit / w.profit - 1),
        carbonopt_carbon_vs_waste_pct=100 * (c.carbon_kg / w.carbon_kg - 1),
        scc_carbon_retained_vs_waste_pct=100 * o.carbon_kg / w.carbon_kg,
        scc_profit_gain_vs_waste_pct=100 * (o.profit / w.profit - 1),
        profitmax_carbon_share_of_scc_pct=100 * pm.carbon_kg / o.carbon_kg,
        rule_overstatement_pct=100 * (rule.naive_carbon_kg / rule.carbon_kg - 1),
        mean_disc_scc=o.mean_disc, mean_disc_carbon=c.mean_disc, mean_disc_waste=w.mean_disc,
    )


def run_scenario(unit_fn, label: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = [policy_table(a, g, unit_fn) for a, g in CELLS]
    full = pd.concat(tables, ignore_index=True).assign(**label)
    summ = pd.DataFrame([{**label, "archetype": t.archetype.iloc[0],
                          "region": t.region.iloc[0], **cell_summary(t)} for t in tables])
    return full, summ


def describe(summ: pd.DataFrame, cols) -> pd.DataFrame:
    return summ[cols].agg(["min", "mean", "max"]).T.round(1)

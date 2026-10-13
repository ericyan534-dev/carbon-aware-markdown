"""Main experiments: five policies x 20 product-region cases (Table II, Fig. 2, Sec. III-B, V-B, V-D).

Writes to outputs/main/:
  policy_comparison.csv  every policy's metrics in every case (Table II is the Hong Kong slice)
  gains.csv              carbon-priced policy vs. static rule and vs. profit-maximizing policy
  summary.json           overstatement of cleared-unit crediting under the static rule
  headlines.json         every range quoted in Sec. III-B, V-B and V-D
  goals.csv              each policy's value at the SCC as % of the best attainable (Fig. 2)
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tblca.markdown import evaluate_policy, solve  # noqa: E402
from tblca.params import ARCHETYPES, REGIONS, SCC_USD_PER_KG  # noqa: E402
from tblca.scenario import build, heuristic_policies, with_objective  # noqa: E402

OUT = ROOT / "outputs" / "main"
PM, SCC, WM, CO, STATIC = ("Profit-optimal", "SCC-weighted (ours)", "Waste-optimal",
                           "Carbon-optimal", "Last-day 35%")


def policy_table(arch, region):
    base = build(arch, region, carbon_price=SCC_USD_PER_KG)
    rows = {name: evaluate_policy(base, pol).metrics
            for name, pol in heuristic_policies(base).items()}
    for obj, label in (("waste", WM), ("profit", PM), ("carbon", CO), ("weighted", SCC)):
        rows[label] = solve(with_objective(base, obj)).metrics
    df = pd.DataFrame(rows).T
    df["waste_pct"] = 100 * df["waste_units"] / base.x0
    df["mean_disc"] = df["disc_units"] / df["sales"]
    df["naive_carbon_kg"] = naive_carbon(base, df)
    df["ops_kg"] = df["sales"] * base.unit_kg * base.ops_kg
    return df.assign(archetype=arch, region=region)


def naive_carbon(p, df):
    """Claimed carbon if every sold unit fully displaced production (delta = 1)."""
    per_unit = p.unit_kg * (p.ef_prod + p.ef_eol - p.ops_kg)
    return df["sales"] * per_unit


GOALS = {PM: "profit", SCC: "weighted", CO: "carbon", WM: "waste", STATIC: "static"}


def goals_table(full):
    """Value at the SCC (profit + priced CO2e) of five policies, % of the SCC-optimal value."""
    rows = []
    for (arch, region), g in full.groupby(["archetype", "region"], sort=False):
        s = g.set_index("policy")
        value = s.profit + SCC_USD_PER_KG * s.carbon_kg
        for label, key in GOALS.items():
            rows.append(dict(archetype=arch, region=region, policy=key,
                             pct_best=100 * value[label] / value[SCC],
                             profit=s.profit[label], carbon=s.carbon_kg[label]))
    return pd.DataFrame(rows)


def _range(s):
    return {"min": round(float(s.min()), 2), "median": round(float(s.median()), 2),
            "max": round(float(s.max()), 2)}


def headlines(full):
    """Ranges over the 20 cases; every policy is scored by its value at the SCC."""
    mu = SCC_USD_PER_KG
    f = full.assign(value=full.profit + mu * full.carbon_kg)
    k = f.pivot_table(index=["archetype", "region"], columns="policy",
                      values=["profit", "carbon_kg", "value", "waste_pct", "ops_kg"])
    P, C, V = k["profit"], k["carbon_kg"], k["value"]
    best = V.max(axis=1)
    abatement = (P[SCC] - P[WM]) / ((C[WM] - C[SCC]) / 1000)        # USD per tonne CO2e
    hk = k["waste_pct"].xs("Hong Kong", level="region")
    over = 100 * (_naive(f, STATIC) / C[STATIC] - 1)
    return {
        "V-B profit-max share of best value at SCC (%)": _range(100 * V[PM] / best),
        "V-B waste-min share of best value at SCC (%)": _range(100 * V[WM] / best),
        "V-B carbon-priced profit below profit-max (%)": _range(100 * (1 - P[SCC] / P[PM])),
        "V-B profit-max share of carbon-priced avoided CO2e (%)": _range(100 * C[PM] / C[SCC]),
        "V-B share of carbon-priced improvement over static captured by profit-max (%)":
            _range(100 * (V[PM] - V[STATIC]) / (V[SCC] - V[STATIC])),
        "V-B profit-max shortfall at SCC (% of profit)": _range(100 * (V[SCC] - V[PM]) / P[PM]),
        "V-B waste-min cost per extra tonne vs carbon-priced (USD/t)": _range(abatement),
        "V-B waste-min cost per extra tonne (multiple of SCC)": _range(abatement / (1000 * mu)),
        "V-B carbon-only profit above waste-min (%)": _range(100 * (P[CO] / P[WM] - 1)),
        "V-B carbon-only avoids at least as much CO2e as waste-min (cases)":
            int((C[CO] >= C[WM] - 1e-9).sum()),
        "V-D profit-max profit gain vs static (%)": _range(100 * (P[PM] / P[STATIC] - 1)),
        "V-D profit-max CO2e gain vs static (%)": _range(100 * (C[PM] / C[STATIC] - 1)),
        "V-D Hong Kong unsold, static (%)": _range(hk[STATIC]),
        "V-D Hong Kong unsold, profit-max (%)": _range(hk[PM]),
        "III-B static-rule overstatement (%)": _range(over),
        "III-B static-rule overstatement mean (%)": round(float(over.mean()), 2),
        "IV-A operational emissions, max share of net avoided CO2e (%)":
            round(float((100 * k["ops_kg"] / C).max().max()), 2),
    }


def _naive(f, policy):
    return f[f.policy == policy].set_index(["archetype", "region"])["naive_carbon_kg"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tables = [policy_table(a, g) for a in ARCHETYPES for g in REGIONS]
    full = pd.concat(tables).rename_axis("policy").reset_index()
    full.to_csv(OUT / "policy_comparison.csv", index=False)
    hk = full[full.region == "Hong Kong"]
    cols = ["archetype", "policy", "profit", "carbon_kg", "naive_carbon_kg", "waste_pct", "mean_disc"]
    print(hk[cols].round(2).to_string(index=False))

    key = full.pivot_table(index=["archetype", "region"], columns="policy",
                           values=["profit", "carbon_kg"])
    gains = pd.DataFrame({
        "profit_gain_pct": 100 * (key["profit"][SCC] / key["profit"][STATIC] - 1),
        "carbon_gain_pct": 100 * (key["carbon_kg"][SCC] / key["carbon_kg"][STATIC] - 1),
        "carbon_vs_profitopt_pct": 100 * (key["carbon_kg"][SCC] / key["carbon_kg"][PM] - 1),
        "profit_cost_vs_profitopt_pct": 100 * (1 - key["profit"][SCC] / key["profit"][PM]),
    }).round(1)
    gains.to_csv(OUT / "gains.csv")
    print(gains.to_string())
    naive = full[full.policy == STATIC]
    over = 100 * (naive.naive_carbon_kg / naive.carbon_kg - 1)
    summary = {"naive_overstatement_pct_last_day_rule": dict(
        zip(naive.archetype + "|" + naive.region, over.round(1)))}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print("naive overstatement (last-day rule):", over.describe().round(1).to_dict())
    goals_table(full).to_csv(OUT / "goals.csv", index=False)
    head = headlines(full)
    (OUT / "headlines.json").write_text(json.dumps(head, indent=1))
    print(json.dumps(head, indent=1))


if __name__ == "__main__":
    main()

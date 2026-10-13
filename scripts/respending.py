"""Re-spending of consumer savings at 0.2-0.6 kgCO2e per dollar (Sec. V-F).

Two accounting modes: only displacing buyers' savings are re-spent ("none"), or a fixed budget
in which incremental purchases divert other spending ("diversion"). Writes outputs/consumer/respend_*.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.consumer_common import OURS, OUT, describe, run_scenario  # noqa: E402
from tblca.ext_units import respend_unit_fn  # noqa: E402

INTENSITIES = (0.0, 0.2, 0.4, 0.6)        # kg CO2e per $ re-spent
MODES = ("none", "diversion")
POLICIES = ["No markdown", "Last-day 35%", "Flat 30%", "Waste-optimal", "Profit-optimal",
            "Carbon-optimal", "SCC-weighted (ours)"]
PCT_COLS = ["scc_profit_gain_vs_rule_pct", "scc_carbon_gain_vs_rule_pct",
            "carbonopt_profit_vs_waste_pct", "carbonopt_carbon_vs_waste_pct",
            "scc_carbon_retained_vs_waste_pct", "scc_profit_gain_vs_waste_pct",
            "profitmax_carbon_share_of_scc_pct", "rule_overstatement_pct"]


def run_all():
    fulls, summs = [], []
    for mode in MODES:
        for i in INTENSITIES:
            if mode == "diversion" and i == 0.0:
                continue
            f, s = run_scenario(respend_unit_fn(i, 1.0, mode), {"intensity": i, "incremental": mode})
            fulls.append(f)
            summs.append(s)
    return pd.concat(fulls, ignore_index=True), pd.concat(summs, ignore_index=True)


def carbon_change(full):
    base = full[full.intensity == 0].set_index(["archetype", "region", "policy"]).carbon_kg
    rows = []
    for (mode, i), g in full.groupby(["incremental", "intensity"]):
        ch = 100 * (g.set_index(["archetype", "region", "policy"]).carbon_kg / base - 1)
        stats = ch.groupby(level="policy").agg(["min", "mean", "max"])
        for pol, r in stats.iterrows():
            rows.append(dict(incremental=mode, intensity=i, policy=pol, **r.round(1).to_dict()))
    return pd.DataFrame(rows)


def robust_summary(full):
    """Sign-robust comparisons (ratios explode when a policy's net carbon nears zero)."""
    key = ["archetype", "region"]
    base_w = (full[(full.intensity == 0) & (full.policy == "Waste-optimal")]
              .set_index(key).carbon_kg)
    rows = []
    for (mode, i), g in full.groupby(["incremental", "intensity"]):
        c = g.pivot_table(index=key, columns="policy", values="carbon_kg")
        pr = g.pivot_table(index=key, columns="policy", values="profit")
        rule_over = 100 * (g[g.policy == "Last-day 35%"].set_index(key).naive_carbon_kg
                           / c["Last-day 35%"] - 1)
        gap = 100 * (c["Carbon-optimal"] - c["Waste-optimal"]) / base_w
        rows.append(dict(
            incremental=mode, intensity=i,
            n_carbonopt_dominates=int(((c["Carbon-optimal"] >= c["Waste-optimal"] - 1e-9)
                                       & (pr["Carbon-optimal"] >= pr["Waste-optimal"] - 1e-9)).sum()),
            carbon_gap_pct_of_base_waste_min=gap.min(), carbon_gap_pct_mean=gap.mean(),
            carbon_gap_pct_max=gap.max(),
            n_scc_ge_waste_carbon=int((c[OURS] >= c["Waste-optimal"]).sum()),
            n_scc_gt_rule_carbon=int((c[OURS] > c["Last-day 35%"]).sum()),
            n_scc_gt_nomarkdown=int((c[OURS] > c["No markdown"]).sum()),
            n_scc_negative=int((c[OURS] < 0).sum()),
            n_waste_negative=int((c["Waste-optimal"] < 0).sum()),
            n_profitmax_ge96_scc=int((c["Profit-optimal"] >= 0.96 * c[OURS]).sum()),
            n_carbonopt_zero_markdown=int((g[g.policy == "Carbon-optimal"].mean_disc < 1e-9).sum()),
            rule_over_min=rule_over.min(), rule_over_mean=rule_over.mean(),
            rule_over_max=rule_over.max()))
    return pd.DataFrame(rows).round(1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    full, summ = run_all()
    full.to_csv(OUT / "respend_policy_comparison.csv", index=False)
    summ.to_csv(OUT / "respend_case_summary.csv", index=False)
    ch = carbon_change(full)
    ch.to_csv(OUT / "respend_carbon_change.csv", index=False)
    agg = {f"{m}|I={i}": {**describe(g, PCT_COLS).to_dict(orient="index"),
                          "n_dominate": int(g.carbonopt_dominates_wastemin.sum())}
           for (m, i), g in summ.groupby(["incremental", "intensity"])}
    (OUT / "respend_summary.json").write_text(json.dumps(agg, indent=1))
    for k, v in agg.items():
        print(k, "dominate", v["n_dominate"])
        print(pd.DataFrame({c: v[c] for c in PCT_COLS}).T.to_string())
    print(ch[ch.policy.isin(POLICIES)].to_string(index=False))
    rob = robust_summary(full)
    rob.to_csv(OUT / "respend_robust_summary.csv", index=False)
    print(rob.T.to_string())


if __name__ == "__main__":
    main()

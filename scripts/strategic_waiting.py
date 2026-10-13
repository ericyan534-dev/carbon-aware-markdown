"""Strategic waiting: a share s of full-price buyers waits for markdowns (Sec. V-F).

s in {0.1, 0.2, 0.3}, with and without the displaced-production credit, plus a stress case in
which every strategic buyer waits for any discount. Writes outputs/consumer/waiting_*.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.consumer_common import OUT, describe, run_scenario  # noqa: E402
from tblca.ext_units import waiting_unit_fn  # noqa: E402

SHARES = (0.0, 0.1, 0.2, 0.3)
PCT_COLS = ["scc_profit_gain_vs_rule_pct", "scc_carbon_gain_vs_rule_pct",
            "carbonopt_profit_vs_waste_pct", "carbonopt_carbon_vs_waste_pct",
            "scc_carbon_retained_vs_waste_pct", "scc_profit_gain_vs_waste_pct",
            "profitmax_carbon_share_of_scc_pct"]
KEY = ["archetype", "region", "policy"]


def run_all():
    fulls, summs = [], []
    for credit in (True, False):
        for s in SHARES:
            if not credit and s == 0.0:
                continue
            f, sm = run_scenario(waiting_unit_fn(s, credit), {"s": s, "credit": credit,
                                                               "stress": False})
            fulls.append(f)
            summs.append(sm)
    for s in SHARES[1:]:   # stress: every strategic buyer waits for any discount
        f, sm = run_scenario(waiting_unit_fn(s, True, stress=True),
                             {"s": s, "credit": True, "stress": True})
        fulls.append(f)
        summs.append(sm)
    return pd.concat(fulls, ignore_index=True), pd.concat(summs, ignore_index=True)


def changes(full):
    base = full[full.s == 0].set_index(KEY)
    rows = []
    for (stress, credit, s), g in full.groupby(["stress", "credit", "s"]):
        g = g.set_index(KEY)
        d = pd.DataFrame({"profit_chg_pct": 100 * (g.profit / base.profit - 1),
                          "carbon_chg_pct": 100 * (g.carbon_kg / base.carbon_kg - 1),
                          "mean_disc": g.mean_disc, "waste_pct": g.waste_pct})
        st = d.groupby(level="policy").mean().round(2)
        for pol, r in st.iterrows():
            rows.append(dict(stress=stress, credit=credit, s=s, policy=pol, **r.to_dict()))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    full, summ = run_all()
    full.to_csv(OUT / "waiting_policy_comparison.csv", index=False)
    summ.to_csv(OUT / "waiting_case_summary.csv", index=False)
    ch = changes(full)
    ch.to_csv(OUT / "waiting_changes.csv", index=False)
    agg = {f"stress={st}|credit={c}|s={s}": {**describe(g, PCT_COLS).to_dict(orient="index"),
                                           "n_dominate": int(g.carbonopt_dominates_wastemin.sum())}
           for (st, c, s), g in summ.groupby(["stress", "credit", "s"])}
    (OUT / "waiting_summary.json").write_text(json.dumps(agg, indent=1))
    for k, v in agg.items():
        print(k, "dominate", v["n_dominate"])
        print(pd.DataFrame({c: v[c] for c in PCT_COLS}).T.to_string())
    print(ch.to_string(index=False))


if __name__ == "__main__":
    main()

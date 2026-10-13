"""Household freshness decay x home-waste grid: 4 products x 21 x 15 = 1,260 cases (Sec. V-E).

Hong Kong, lam_f in [0.5, 2.5], p_base in [0.10, 0.45]. Writes outputs/household/.
"""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tblca.params import ARCHETYPES, SCC_USD_PER_KG  # noqa: E402
from tblca.policy_suite import run  # noqa: E402
from tblca.scenario import build  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs" / "household"
LAM_F = np.round(np.linspace(0.5, 2.5, 21), 3)
P_BASE = np.round(np.linspace(0.10, 0.45, 15), 4)
KEEP = ("carbon_dominates_waste", "scc_retention_pct", "profitmax_share_scc_carbon_pct",
        "overstatement_static_pct", "carbon_gain_scc_vs_static", "profit_gain_scc_vs_static",
        "scc_beats_static_both", "carbon_vs_waste_profit_pct", "carbon_vs_waste_carbon_pct")


def run_cell(job):
    arch, lam_f, p_base = job
    p = build(arch, "Hong Kong", carbon_price=SCC_USD_PER_KG, lam_f=lam_f, p_base=p_base)
    _, h = run(p)
    return dict(archetype=arch, lam_f=lam_f, p_base=p_base, **{k: h[k] for k in KEEP})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(a, lf, pb) for a in ARCHETYPES for lf in LAM_F for pb in P_BASE]
    with Pool(max(1, (os.cpu_count() or 2) - 2)) as pool:
        res = pd.DataFrame(pool.map(run_cell, jobs, chunksize=8))
    res.to_csv(OUT / "household_grid.csv", index=False)
    agg = res.groupby("archetype").agg(
        n=("lam_f", "size"),
        dominance_share=("carbon_dominates_waste", "mean"),
        scc_beats_static_share=("scc_beats_static_both", "mean"),
        retention_min=("scc_retention_pct", "min"), retention_max=("scc_retention_pct", "max"),
        pmshare_min=("profitmax_share_scc_carbon_pct", "min"),
        pmshare_max=("profitmax_share_scc_carbon_pct", "max"),
        over_min=("overstatement_static_pct", "min"), over_max=("overstatement_static_pct", "max"),
        cgain_min=("carbon_gain_scc_vs_static", "min"), cgain_max=("carbon_gain_scc_vs_static", "max"))
    agg.to_csv(OUT / "summary.csv")
    print(agg.round(2).to_string())
    print(f"cases: {len(res)}; carbon-only dominates waste-min: {int(res.carbon_dominates_waste.sum())}; "
          f"carbon-priced beats static on profit and CO2e: {int(res.scc_beats_static_both.sum())}")


if __name__ == "__main__":
    main()

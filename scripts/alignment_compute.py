"""Alignment index theta and monotonicity of the optimal discount in the carbon price (Sec. IV-C).

Outputs (outputs/alignment/):
  s1_cells.csv      20 cases at the SCC: theta_bar, carbon gap vs. profit-max
  s1_sweep.csv      20 cases x 71 carbon prices (0-5 USD/kg): same quantities (Fig. 4c)
  s1_monotone.csv   per case: stage-problem (2,080 states) and full-DP (145,600 comparisons) checks
Run scripts/alignment_analyze.py afterwards for the summary statistics.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tblca.alignment import (  # noqa: E402
    monotonicity_violations, myopic_argmax, stage_carbon_nondecreasing, theta_bar,
)
from tblca.markdown import solve  # noqa: E402
from tblca.params import ARCHETYPES, REGIONS, SCC_USD_PER_KG  # noqa: E402
from tblca.scenario import build  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs" / "alignment"
MU_GRID = np.round(np.concatenate([np.linspace(0, 0.5, 26), np.linspace(0.6, 5.0, 45)]), 4)
MYOPIC_MU = np.linspace(0, 5, 51)


def cell_sweep(args):
    arch, region = args
    base = build(arch, region)
    pm = solve(replace(base, objective="profit")).metrics
    co = solve(replace(base, objective="carbon")).metrics
    sols = [solve(replace(base, objective="weighted", carbon_price=float(m))) for m in MU_GRID]
    rows = []
    for m, s in zip(MU_GRID, sols):
        gap = s.metrics["carbon_kg"] - pm["carbon_kg"]
        rows.append(dict(archetype=arch, region=region, mu=m, theta_bar=theta_bar(base, m),
                         carbon=s.metrics["carbon_kg"], profit=s.metrics["profit"],
                         carbon_pm=pm["carbon_kg"], profit_pm=pm["profit"],
                         carbon_opt=co["carbon_kg"], gap_kg=gap,
                         gap_pct=100 * gap / pm["carbon_kg"],
                         gap_closed=gap / (co["carbon_kg"] - pm["carbon_kg"]),
                         gap_kg_per_unit=gap / base.x0))
    mono = monotone_checks(base, sols)
    return rows, dict(archetype=arch, region=region, **mono)


def monotone_checks(base, sols):
    states = [(r, x) for r in range(1, base.horizon + 1) for x in range(1, base.x0 + 1)]
    cond = [stage_carbon_nondecreasing(base, x, r) for r, x in states]
    myo_viol = myo_viol_cond = 0
    for (r, x), ok in zip(states, cond):
        ds = np.array([myopic_argmax(base, x, r, m) for m in MYOPIC_MU])
        bad = bool(np.any(np.diff(ds) < -1e-12))
        myo_viol += bad
        myo_viol_cond += bad and ok
    dp = monotonicity_violations(sols)
    ids = [i for i, m in enumerate(MU_GRID) if m <= 1.0 + 1e-9]
    dp1 = monotonicity_violations([sols[i] for i in ids])
    return dict(states=len(states), id_condition_share=float(np.mean(cond)),
                myopic_states_violating=myo_viol, myopic_viol_where_condition=myo_viol_cond,
                dp_pairs=dp["pairs"], dp_violations=dp["violations"],
                dp_states_ever_violated=dp["states_ever_violated"],
                dp_violations_mu_le_1=dp1["violations"],
                dp_states_ever_violated_mu_le_1=dp1["states_ever_violated"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cells = [(a, g) for a in ARCHETYPES for g in REGIONS]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(cell_sweep, cells))
    sweep = pd.DataFrame([r for rows, _ in res for r in rows])
    sweep.to_csv(OUT / "s1_sweep.csv", index=False)
    pd.DataFrame([m for _, m in res]).to_csv(OUT / "s1_monotone.csv", index=False)
    scc = []
    for a, g in cells:
        base = build(a, g)
        pm = solve(replace(base, objective="profit")).metrics
        co = solve(replace(base, objective="carbon")).metrics
        s = solve(replace(base, carbon_price=SCC_USD_PER_KG)).metrics
        gap = s["carbon_kg"] - pm["carbon_kg"]
        scc.append(dict(archetype=a, region=g, theta_bar=theta_bar(base, SCC_USD_PER_KG),
                        gap_kg=gap, gap_pct=100 * gap / pm["carbon_kg"],
                        gap_closed=gap / (co["carbon_kg"] - pm["carbon_kg"]),
                        profit_loss_pct=100 * (pm["profit"] - s["profit"]) / pm["profit"]))
    pd.DataFrame(scc).to_csv(OUT / "s1_cells.csv", index=False)


if __name__ == "__main__":
    main()

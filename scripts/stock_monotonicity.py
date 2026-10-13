"""Optimal discounts deepen as stock rises (Sec. V-D).

Checks that the optimal policy is nondecreasing in inventory x on every day:
  * paper cases: 20 product-region cases x 6 goals (profit, carbon-only, waste-min, carbon
    priced at the SCC with and without a waste penalty, carbon at 1 USD/kg) = 120 cases;
  * randomized solves: 2,000 parameter draws from the Sobol box of scripts/run_sobol.py
    (product and region rotated) x 5 goals = 10,000 solves on the paper's discount grid.

Writes outputs/monotonicity/{paper_cases.csv, randomized.csv, summary.json}.
"""
import json
import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_sobol import PROBLEM  # noqa: E402
from tblca.monotone_theory import (  # noqa: E402
    from_markdown, is_monotone_in_x, pairwise_single_crossing, solve_generic,
)
from tblca.params import ARCHETYPES, REGIONS, SCC_USD_PER_KG  # noqa: E402
from tblca.scenario import build  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs" / "monotonicity"
N_RANDOM = 2000
# (objective, carbon price USD/kg, waste penalty as a share of price)
SETTINGS = [("profit", 0.0, 0.0), ("carbon", 0.0, 0.0), ("weighted", SCC_USD_PER_KG, 0.0),
            ("weighted", SCC_USD_PER_KG, 0.5), ("weighted", 1.0, 0.0), ("waste", 0.0, 0.0)]


def analyse(p):
    g = from_markdown(p)
    _, pol, qs = solve_generic(g)
    return dict(mono=is_monotone_in_x(pol),
                pairwise_sc=all(pairwise_single_crossing(qs[r]) for r in range(1, g.horizon + 1)))


def paper_cases():
    rows = []
    for a in ARCHETYPES:
        for reg in REGIONS:
            for obj, cp, wp in SETTINGS:
                p = build(a, reg, objective=obj, carbon_price=cp)
                p = replace(p, waste_penalty=wp * p.price)
                rows.append(dict(archetype=a, region=reg, objective=obj, carbon_price=cp,
                                 waste_penalty_ratio=wp, **analyse(p)))
    return pd.DataFrame(rows)


def random_draw(seed):
    rng = np.random.default_rng(seed)
    lo, hi = np.array(PROBLEM["bounds"]).T
    kw = dict(zip(PROBLEM["names"], lo + rng.random(len(lo)) * (hi - lo)))
    arch, reg = list(ARCHETYPES)[seed % 4], list(REGIONS)[(seed // 4) % 5]
    kw["ef_prod"] *= ARCHETYPES[arch].ef_prod / ARCHETYPES["dairy"].ef_prod
    rows = []
    for obj, cp, wp in SETTINGS[:5]:
        p = build(arch, reg, objective=obj, carbon_price=cp, **kw)
        p = replace(p, waste_penalty=wp * p.price)
        rows.append(dict(seed=seed, archetype=arch, region=reg, objective=obj, carbon_price=cp,
                         waste_penalty_ratio=wp, **analyse(p)))
    return rows


def summarize(df, keys):
    agg = df.groupby(keys).agg(n=("mono", "size"), monotone=("mono", "mean"),
                               pairwise_single_crossing=("pairwise_sc", "mean"))
    return agg.round(4).reset_index()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = paper_cases()
    cases.to_csv(OUT / "paper_cases.csv", index=False)
    with Pool() as pool:
        rnd = pd.DataFrame(sum(pool.map(random_draw, range(N_RANDOM), chunksize=10), []))
    rnd.to_csv(OUT / "randomized.csv", index=False)
    keys = ["objective", "carbon_price", "waste_penalty_ratio"]
    summary = {"paper_cases": dict(n=len(cases), monotone=int(cases.mono.sum())),
               "randomized": dict(n=len(rnd), monotone=int(rnd.mono.sum())),
               "paper_cases_by_goal": summarize(cases, keys).to_dict("records"),
               "randomized_by_goal": summarize(rnd, keys).to_dict("records")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({k: summary[k] for k in ("paper_cases", "randomized")}, indent=1))


if __name__ == "__main__":
    main()

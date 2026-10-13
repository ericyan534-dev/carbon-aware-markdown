"""Sobol analysis over ten inputs of the carbon-priced policy's gain over the static rule (Sec. V-F).

Dairy, Hong Kong; 512 x (10 + 2) = 6,144 evaluations. Writes outputs/sobol/.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from SALib.analyze import sobol
from SALib.sample import sobol as sobol_sample

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tblca.markdown import solve, evaluate_policy  # noqa: E402
from tblca.params import ARCHETYPES, SCC_USD_PER_KG  # noqa: E402
from tblca.scenario import build, heuristic_policies  # noqa: E402

DAIRY = ARCHETYPES["dairy"]
PROBLEM = {
    "num_vars": 10,
    "names": ["eps", "kappa", "lam_a", "lam_f", "p_base", "ef_prod", "ef_eol",
              "stock_ratio", "margin_ratio", "q0"],
    "bounds": [[0.7, 4.0], [0.4, 0.9], [0.3, 1.5], [0.5, 2.5], [0.10, 0.45],
               [0.5 * DAIRY.ef_prod, 1.5 * DAIRY.ef_prod], [-0.15, 1.2],
               [0.75, 3.0], [0.15, 0.45], [4.0, 16.0]],
}
OUT = Path(__file__).resolve().parents[1] / "outputs" / "sobol"


def monotone(pol):
    body = pol[1:, 1:]
    return bool(np.all(np.diff(body, axis=1) >= -1e-9) and np.all(np.diff(body, axis=0) <= 1e-9))


def evaluate(x):
    kw = dict(zip(PROBLEM["names"], x))
    p = build("dairy", "Hong Kong", carbon_price=SCC_USD_PER_KG, **kw)
    ours = solve(p)
    rule = evaluate_policy(p, heuristic_policies(p)["Last-day 35%"]).metrics
    m = ours.metrics
    naive = rule["sales"] * p.unit_kg * (p.ef_prod + p.ef_eol - p.ops_kg)
    return (100 * (m["carbon_kg"] / rule["carbon_kg"] - 1),
            100 * (m["profit"] / rule["profit"] - 1),
            100 * (naive / rule["carbon_kg"] - 1),
            monotone(ours.policy))


def main(n=512):
    OUT.mkdir(parents=True, exist_ok=True)
    X = sobol_sample.sample(PROBLEM, n, calc_second_order=False, seed=11)
    with Pool() as pool:
        res = np.array(pool.map(evaluate, X, chunksize=64))
    names = ["carbon_gain_pct", "profit_gain_pct", "naive_overstatement_pct"]
    out = {"n_evals": len(X), "monotone_share": float(res[:, 3].mean())}
    rows = []
    for j, name in enumerate(names):
        y = res[:, j]
        si = sobol.analyze(PROBLEM, y, calc_second_order=False, seed=11)
        out[name] = {"mean": float(y.mean()), "p5": float(np.percentile(y, 5)),
                     "p50": float(np.median(y)), "p95": float(np.percentile(y, 95)),
                     "prob_positive": float((y > 0).mean())}
        for k, v in enumerate(PROBLEM["names"]):
            rows.append(dict(output=name, param=v, S1=si["S1"][k], ST=si["ST"][k],
                             ST_conf=si["ST_conf"][k]))
    pd.DataFrame(rows).to_csv(OUT / "sobol.csv", index=False)
    pd.DataFrame(res, columns=names + ["monotone"]).to_csv(OUT / "sobol_samples.csv", index=False)
    (OUT / "sobol_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(pd.DataFrame(rows).pivot(index="param", columns="output", values="ST").round(3))


if __name__ == "__main__":
    main()

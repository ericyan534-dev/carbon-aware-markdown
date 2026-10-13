"""Where profit and climate can disagree: elasticity thresholds and stylized curves (Sec. V-C).

* eps*        = 1 / (1 - kappa c_f/p)                 discounts start to pay (Proposition 1)
* eps*_theta  = 1 / (1 - kappa (c_f/p)(1 - theta_bar)) the same with the carbon-priced margin,
                one value per product type (theta_bar at the SCC, Hong Kong)
* ratio(eps)  value at the SCC of the profit-maximizing policy as % of the SCC-optimal value,
              per product type at the stylized q0 = 8 units/day, eps in [0.5, 7] step 0.01

Writes outputs/threshold/theory_curves.csv and theory_meta.json (Fig. 3a).
"""
import json
import os
import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.profiles_core import tables_for  # noqa: E402
from tblca.alignment import theta_bar  # noqa: E402
from tblca.fast_dp import build_tables, fast_evaluate, fast_solve  # noqa: E402
from tblca.params import SCC_USD_PER_KG  # noqa: E402
from tblca.scenario import DEFAULTS, build  # noqa: E402

OUT = ROOT / "outputs" / "threshold"
REGION = "Hong Kong"
ARCHETYPES = ("dairy", "bakery", "produce", "poultry")
KAPPA, MARGIN = DEFAULTS["kappa"], DEFAULTS["margin_ratio"]
EPS_STAR = 1.0 / (1.0 - KAPPA * MARGIN)
EPS_GRID = np.round(np.arange(0.50, 7.0001, 0.01), 2)
Q0 = 8.0
N_WORKERS = os.cpu_count() or 1


def eps_star_theta(theta: float) -> float:
    """Proposition 1 with the carbon-priced margin c_f (1 - theta)."""
    return 1.0 / (1.0 - KAPPA * MARGIN * (1.0 - theta))


def ratio_for(archetype: str, q0: float, eps: float) -> tuple:
    """(profit-max value at the SCC as % of the SCC-optimal value, profit-max day-1 discount)."""
    p = build(archetype, REGION, carbon_price=SCC_USD_PER_KG, q0=float(q0), eps=float(eps))
    t = build_tables(p)
    tp, tw = tables_for(t, "profit"), tables_for(t, "weighted")
    _, pol_p = fast_solve(replace(p, objective="profit"), tp)
    _, pol_s = fast_solve(p, tw)
    vp, vs = fast_evaluate(p, pol_p, tw)["value"], fast_evaluate(p, pol_s, tw)["value"]
    return (100.0 * vp / vs if vs > 0 else np.nan), float(pol_p[p.horizon, p.x0])


def _job(row: dict) -> dict:
    ratio, first = ratio_for(row["archetype"], row["q0"], row["eps"])
    return {**row, "ratio": ratio, "profit_first_discount": first}


def solve_profiles(df: pd.DataFrame, workers: int = N_WORKERS) -> pd.DataFrame:
    """Ratio for every (archetype, q0, eps) row, solved in parallel (largest q0 first)."""
    rows = df.sort_values("q0", ascending=False).to_dict("records")
    with Pool(workers) as pool:
        return pd.DataFrame(pool.imap_unordered(_job, rows, chunksize=4))


def curve_predict(curves: pd.DataFrame, archetype, eps) -> np.ndarray:
    """Stylized ratio(eps) curve of each profile's product type, linearly interpolated."""
    a_arr, e_arr = np.asarray(archetype), np.asarray(eps, float)
    out = np.empty(len(e_arr))
    for a in np.unique(a_arr):
        c = curves[curves.archetype == a].sort_values("eps")
        m = a_arr == a
        out[m] = np.interp(e_arr[m], c.eps.to_numpy(), c.ratio.to_numpy())
    return out


def curve_meta(cur: pd.DataFrame) -> dict:
    th = {a: theta_bar(build(a, REGION, carbon_price=SCC_USD_PER_KG)) for a in ARCHETYPES}
    est = {a: eps_star_theta(t) for a, t in th.items()}
    meta = dict(eps_star=EPS_STAR, theta_bar=th, eps_star_theta=est,
                eps_star_theta_range=[round(min(est.values()), 2), round(max(est.values()), 2)])
    for a in ARCHETYPES:
        c = cur[cur.archetype == a]
        disc = c[c.profit_first_discount > 0]
        meta[f"{a}_profit_starts_discount_eps"] = float(disc.eps.min()) if len(disc) else None
        meta[f"{a}_curve_min"] = dict(ratio=float(c.ratio.min()), eps=float(c.eps[c.ratio.idxmin()]))
        below = c[c.ratio < 95]
        meta[f"{a}_eps_range_ratio_lt_95"] = (
            [float(below.eps.min()), float(below.eps.max())] if len(below) else None)
    return meta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = pd.DataFrame([dict(archetype=a, q0=Q0, eps=float(e)) for a in ARCHETYPES for e in EPS_GRID])
    cur = solve_profiles(jobs).sort_values(["archetype", "eps"]).reset_index(drop=True)
    cur.to_csv(OUT / "theory_curves.csv", index=False)
    meta = curve_meta(cur)
    (OUT / "theory_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()

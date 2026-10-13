"""Propagate the sampling uncertainty of the elasticity (IV 1.94, s.e. 0.017; 500 draws) (Sec. V-E).

Writes outputs/elasticity_uncertainty/{draws.csv, headline_bands.csv, per_case_bands.csv, meta.json}.
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tblca.fast_dp import build_tables, fast_evaluate, fast_solve  # noqa: E402
from tblca.markdown import fixed_policy  # noqa: E402
from tblca.params import ARCHETYPES, REGIONS, SCC_USD_PER_KG  # noqa: E402
from tblca.scenario import build, with_objective  # noqa: E402

OUT = ROOT / "outputs" / "elasticity_uncertainty"
EPS_HAT, EPS_SE, N_DRAWS, SEED = 1.94, 0.017, 500, 20261005
CELLS = [(a, g) for a in ARCHETYPES for g in REGIONS]


def optimal_metrics(p):
    t = build_tables(p)
    _, pol = fast_solve(p, t)
    return fast_evaluate(p, pol, t)


def cell_quantities(arch, region, eps):
    base = build(arch, region, carbon_price=SCC_USD_PER_KG, eps=eps)
    scc = optimal_metrics(base)
    prof = optimal_metrics(with_objective(base, "profit"))
    waste = optimal_metrics(with_objective(base, "waste"))
    static = fast_evaluate(base, fixed_policy(base, lambda r: 0.35 if r == 1 else 0.0))
    naive_static = static["sales"] * base.unit_kg * (base.ef_prod + base.ef_eol - base.ops_kg)
    return dict(
        profit_gain_pct=100 * (scc["profit"] / static["profit"] - 1),
        carbon_gain_pct=100 * (scc["carbon_kg"] / static["carbon_kg"] - 1),
        overstatement_pct=100 * (naive_static / static["carbon_kg"] - 1),
        retention_waste_carbon_pct=100 * scc["carbon_kg"] / waste["carbon_kg"],
        profit_gain_vs_waste_pct=100 * (scc["profit"] / waste["profit"] - 1),
        profitmax_share_scc_carbon_pct=100 * prof["carbon_kg"] / scc["carbon_kg"],
    )


def draw_rows(args):
    k, eps = args
    return [dict(draw=k, eps=eps, archetype=a, region=g, **cell_quantities(a, g, eps))
            for a, g in CELLS]


def summarise_draw(df):
    g = df.groupby("draw")
    return pd.DataFrame({
        "profit_gain_min": g.profit_gain_pct.min(), "profit_gain_max": g.profit_gain_pct.max(),
        "carbon_gain_min": g.carbon_gain_pct.min(), "carbon_gain_max": g.carbon_gain_pct.max(),
        "overstatement_min": g.overstatement_pct.min(), "overstatement_max": g.overstatement_pct.max(),
        "overstatement_mean": g.overstatement_pct.mean(),
        "retention_min": g.retention_waste_carbon_pct.min(),
        "retention_max": g.retention_waste_carbon_pct.max(),
        "profit_vs_waste_min": g.profit_gain_vs_waste_pct.min(),
        "profit_vs_waste_max": g.profit_gain_vs_waste_pct.max(),
        "profitmax_share_min": g.profitmax_share_scc_carbon_pct.min(),
        "profitmax_share_max": g.profitmax_share_scc_carbon_pct.max(),
    })


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    eps = rng.normal(EPS_HAT, EPS_SE, N_DRAWS)
    t0 = time.perf_counter()
    with Pool() as pool:
        rows = [r for chunk in pool.map(draw_rows, list(enumerate(eps))) for r in chunk]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "draws.csv", index=False)
    point = pd.DataFrame(draw_rows((-1, EPS_HAT)))
    summ = summarise_draw(df)
    point_summ = summarise_draw(point).iloc[0]
    band = pd.DataFrame({"point_eps_1.94": point_summ,
                         "q025": summ.quantile(0.025), "median": summ.median(),
                         "q975": summ.quantile(0.975)})
    band.to_csv(OUT / "headline_bands.csv")
    q = [c for c in df.columns if c.endswith("_pct")]
    per_cell = df.groupby(["archetype", "region"])[q].quantile([0.025, 0.975]).unstack()
    per_cell.columns = [f"{a}_{'lo' if b < 0.5 else 'hi'}" for a, b in per_cell.columns]
    per_cell = per_cell.join(point.set_index(["archetype", "region"])[q])
    per_cell.to_csv(OUT / "per_case_bands.csv")
    meta = dict(eps_hat=EPS_HAT, eps_se=EPS_SE, n_draws=N_DRAWS, seed=SEED,
                eps_draw_min=float(eps.min()), eps_draw_max=float(eps.max()),
                elapsed_s=time.perf_counter() - t0)
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(band.round(2).to_string())
    print(meta)


if __name__ == "__main__":
    main()

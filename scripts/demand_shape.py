"""Displaced share under flexible demand curves (Sec. II-C).

Ten FE-2SLS specifications let the elasticity vary with discount depth: five curve shapes
(quadratic and cubic in log price, two linear splines, and a spline with extra instruments)
times two instruments (leave-one-store-out and leave-city-out mean log price of the same
product on the same day). All use store-product and day fixed effects, the weather and
holiday controls, and standard errors clustered by store.

For each curve, at discount d:
  share_new = 1 - q(0)/q(d)            units that would not have been bought at full price
  g_disp    = (1 - share_new) + kappa * share_new
and the same quantities under the constant elasticity 1.94.

Writes outputs/demand_shape/displaced_share.csv and summary.json.
Requires data/raw/train.parquet (see README).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.iv_robustness import leave_city_out  # noqa: E402
from tblca import nonlinear_iv as nl  # noqa: E402
from tblca.elasticity import CONTROLS, prepare_panel  # noqa: E402

RAW = ROOT / "data" / "raw" / "train.parquet"
OUT = ROOT / "outputs" / "demand_shape"
COLS = ["city_id", "store_id", "first_category_id", "product_id", "dt", "sale_amount", "discount",
        "stock_hour6_22_cnt", *CONTROLS]
DEPTHS = (0.3, 0.5)
EPS_HAT = 1.94
KAPPA = 0.67
FE = ("series", "dt")
INSTRUMENTS = {"z": "leave-one-store-out", "z_city": "leave-city-out"}


def _extra_spline_z(z: np.ndarray) -> np.ndarray:
    return nl.spline_basis(z, (0.95, 0.85, 0.75, 0.6))[:, 1:]


SHAPES = {  # name: (basis kind, spec, extra instruments)
    "quadratic": ("poly", 2, None),
    "cubic": ("poly", 3, None),
    "spline, knots 0.9/0.8/0.7": ("spline", (0.9, 0.8, 0.7), None),
    "spline, knots 0.95/0.85/0.75/0.65/0.55": ("spline", (0.95, 0.85, 0.75, 0.65, 0.55), None),
    "spline 0.9/0.8/0.7, extra instruments": ("spline", (0.9, 0.8, 0.7), _extra_spline_z),
}


def basis(kind: str, spec, v: np.ndarray) -> np.ndarray:
    return nl.spline_basis(v, spec) if kind == "spline" else nl.poly_basis(v, spec)


def fit_curve(panel: pd.DataFrame, kind: str, spec, extra_z, zcol: str) -> dict:
    """FE-2SLS of log q on basis(log p) instrumented by basis(z) (+ extra instruments)."""
    ctrl = [c for c in CONTROLS if panel[c].std() > 0]
    sub = panel.dropna(subset=["log_q", "log_p", zcol, *ctrl])
    x = basis(kind, spec, sub["log_p"].to_numpy())
    zr = sub[zcol].to_numpy()
    z = basis(kind, spec, zr) if extra_z is None else np.column_stack([basis(kind, spec, zr),
                                                                      extra_z(zr)])
    k, ell = x.shape[1], z.shape[1]
    mat = np.column_stack([sub["log_q"].to_numpy(), x, z, sub[ctrl].to_numpy(float)])
    m, _ = nl.demean(mat, nl.fe_codes(sub, FE))
    w = m[:, 1 + k + ell:]
    m = nl.partial_out(m[:, :1 + k + ell], w[:, w.std(axis=0) > 1e-8])
    y, xm, zm = m[:, 0], m[:, 1:1 + k], m[:, 1 + k:]
    cl = sub["store_id"].to_numpy()
    fit = nl.tsls(y, xm, zm, cl)
    fs = nl.first_stage_stats(xm, zm, cl)
    return {"beta": fit["beta"], "vcov": fit["vcov"], "n": int(len(sub)),
            "min_F_SW": float(min(fs["F_SW"]))}


def _slope_grad(kind: str, spec, x0: float) -> np.ndarray:
    return nl.spline_slope_grad(x0, spec) if kind == "spline" else nl.poly_slope_grad(x0, spec)


def displaced_rows(kind: str, spec, fit: dict) -> list:
    rows = []
    for d in DEPTHS:
        x0 = float(np.log(1 - d))
        a = basis(kind, spec, np.array([x0]))[0] - basis(kind, spec, np.array([0.0]))[0]
        lift, se_lift = nl.lincomb(a, fit["beta"], fit["vcov"])      # log q(d) - log q(0)
        local_eps = -nl.lincomb(_slope_grad(kind, spec, x0), fit["beta"], fit["vcov"])[0]
        share_new = 1 - np.exp(-lift)
        share_const = 1 - (1 - d) ** EPS_HAT
        g, g_const = 1 - (1 - KAPPA) * share_new, 1 - (1 - KAPPA) * share_const
        rows.append(dict(d=d, local_eps=local_eps, share_new=share_new,
                         se_share_new=np.exp(-lift) * se_lift, share_new_const=share_const,
                         g_disp=g, g_disp_const=g_const, g_disp_diff=g - g_const))
    return rows


def main():
    if not RAW.exists():
        raise SystemExit(f"missing {RAW.relative_to(ROOT)}; see README for the download step")
    OUT.mkdir(parents=True, exist_ok=True)
    panel = prepare_panel(pd.read_parquet(RAW, columns=COLS))
    panel["z_city"] = leave_city_out(panel)
    rows = []
    for zcol, zname in INSTRUMENTS.items():
        for shape, (kind, spec, extra) in SHAPES.items():
            fit = fit_curve(panel, kind, spec, extra, zcol)
            rows += [dict(shape=shape, instrument=zname, n=fit["n"], min_F_SW=fit["min_F_SW"], **r)
                     for r in displaced_rows(kind, spec, fit)]
            print(f"{zname:20s} {shape:40s} F_SW>={fit['min_F_SW']:.0f}")
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "displaced_share.csv", index=False)
    summary = {"n_specifications": int(tab.groupby(["shape", "instrument"]).ngroups),
               "max_abs_g_disp_diff": round(float(tab.g_disp_diff.abs().max()), 4)}
    for d, g in tab.groupby("d"):
        summary[f"d={d}"] = {
            "g_disp_const": round(float(g.g_disp_const.iloc[0]), 3),
            "g_disp_range": [round(float(g.g_disp.min()), 3), round(float(g.g_disp.max()), 3)],
            "share_new_const": round(float(g.share_new_const.iloc[0]), 3),
            "share_new_range": [round(float(g.share_new.min()), 3), round(float(g.share_new.max()), 3)]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(tab.round(3).to_string(index=False))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

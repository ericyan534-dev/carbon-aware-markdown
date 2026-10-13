"""Multi-endogenous FE-2SLS for flexible (spline / polynomial) demand curves.

Model (after absorbing high-dimensional FEs and partialling out exogenous
controls):  y = X b + u,  X = f(log p) (k columns), instruments Z = g(z).
Standard errors are cluster-robust (CR0, same convention as tblca.elasticity).
Weak-IV diagnostics: per-equation cluster-robust first-stage F and the
Sanderson-Windmeijer (2016) conditional F (cluster-robust Wald version).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- basis functions
def spline_basis(x: np.ndarray, knots: tuple[float, ...]) -> np.ndarray:
    """Linear spline in x = log price ratio; hinges open below each knot (price ratio)."""
    cols = [x] + [np.maximum(0.0, np.log(k) - x) for k in knots]
    return np.column_stack(cols)


def spline_slope_grad(x0: float, knots: tuple[float, ...]) -> np.ndarray:
    """d basis / dx at x0 (left/deeper side at a knot is never hit for interior x0)."""
    return np.array([1.0] + [-1.0 if x0 < np.log(k) else 0.0 for k in knots])


def poly_basis(x: np.ndarray, degree: int) -> np.ndarray:
    return np.column_stack([x ** p for p in range(1, degree + 1)])


def poly_slope_grad(x0: float, degree: int) -> np.ndarray:
    return np.array([p * x0 ** (p - 1) for p in range(1, degree + 1)])


# ---------------------------------------------------------------- FE absorption
def fe_codes(frame: pd.DataFrame, fe_cols) -> list[np.ndarray]:
    return [pd.factorize(frame[c])[0] for c in fe_cols]


def demean(mat: np.ndarray, codes: list[np.ndarray], tol: float = 1e-10,
           max_iter: int = 500) -> tuple[np.ndarray, int]:
    """Alternating projections (method of alternating projections / Gauss-Seidel)."""
    out = np.array(mat, dtype=float, copy=True)
    counts = [np.bincount(c).astype(float) for c in codes]
    scale = np.maximum(out.std(axis=0), 1e-12)
    for it in range(1, max_iter + 1):
        delta = 0.0
        for c, cnt in zip(codes, counts):
            for j in range(out.shape[1]):
                means = np.bincount(c, weights=out[:, j], minlength=len(cnt)) / cnt
                shift = means[c]
                out[:, j] -= shift
                delta = max(delta, float(np.abs(shift).max() / scale[j]))
        if delta < tol:
            return out, it
    return out, max_iter


def partial_out(mat: np.ndarray, w: np.ndarray | None) -> np.ndarray:
    if w is None or w.shape[1] == 0:
        return mat
    coef = np.linalg.lstsq(w, mat, rcond=None)[0]
    return mat - w @ coef


# ---------------------------------------------------------------- estimation
def cluster_vcov(xu: np.ndarray, resid: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """Sandwich (X'X)^-1 meat (X'X)^-1 with meat from cluster-summed scores (CR0)."""
    bread = np.linalg.pinv(xu.T @ xu)
    codes = pd.factorize(clusters)[0]
    scores = np.column_stack([np.bincount(codes, weights=xu[:, j] * resid)
                              for j in range(xu.shape[1])])
    return bread @ (scores.T @ scores) @ bread


def tsls(y: np.ndarray, x: np.ndarray, z: np.ndarray, clusters: np.ndarray) -> dict:
    pi = np.linalg.lstsq(z, x, rcond=None)[0]
    x_hat = z @ pi
    beta = np.linalg.lstsq(x_hat, y, rcond=None)[0]
    resid = y - x @ beta
    return {"beta": beta, "vcov": cluster_vcov(x_hat, resid, clusters), "resid": resid}


def ols(y: np.ndarray, x: np.ndarray, clusters: np.ndarray) -> dict:
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ beta
    return {"beta": beta, "vcov": cluster_vcov(x, resid, clusters), "resid": resid}


def _wald_f(target: np.ndarray, z: np.ndarray, clusters: np.ndarray, df: int) -> float:
    fit = ols(target, z, clusters)
    b, v = fit["beta"], fit["vcov"]
    return float(b @ np.linalg.solve(v, b) / df)


def first_stage_stats(x: np.ndarray, z: np.ndarray, clusters: np.ndarray) -> dict:
    """Per-equation and Sanderson-Windmeijer conditional F (cluster-robust Wald / df)."""
    k, ell = x.shape[1], z.shape[1]
    per_eq = [_wald_f(x[:, j], z, clusters, ell) for j in range(k)]
    sw = []
    for j in range(k):
        if k == 1:
            sw.append(per_eq[0])
            continue
        others = np.delete(x, j, axis=1)
        delta = tsls(x[:, j], others, z, clusters)["beta"]
        e = x[:, j] - others @ delta
        sw.append(_wald_f(e, z, clusters, ell - k + 1))
    return {"F_per_equation": per_eq, "F_SW": sw}


def lincomb(a: np.ndarray, beta: np.ndarray, vcov: np.ndarray) -> tuple[float, float]:
    return float(a @ beta), float(np.sqrt(max(a @ vcov @ a, 0.0)))

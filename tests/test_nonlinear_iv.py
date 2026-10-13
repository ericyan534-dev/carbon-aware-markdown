"""Flexible-curve FE-2SLS: absorption, estimation and slope gradients."""
import numpy as np
import pandas as pd

from tblca.nonlinear_iv import (demean, fe_codes, first_stage_stats, lincomb, poly_basis,
                                poly_slope_grad, spline_basis, spline_slope_grad, tsls)


def _synthetic(n=60000, seed=0, curvature=0.0):
    rng = np.random.default_rng(seed)
    g1, g2 = rng.integers(0, 300, n), rng.integers(0, 50, n)
    a1, a2 = rng.normal(size=300), rng.normal(size=50)
    z = np.log(rng.uniform(0.45, 1.0, n))
    v = rng.normal(scale=0.05, size=n)
    x = np.clip(0.8 * z + v, np.log(0.32), 0.0)
    u = rng.normal(scale=0.3, size=n) + 2.0 * v          # endogeneity
    y = a1[g1] + a2[g2] - 2.0 * x + curvature * x ** 2 + u
    return pd.DataFrame(dict(g1=g1, g2=g2, x=x, z=z, y=y, cl=g1 % 40))


def test_demean_removes_group_means():
    df = _synthetic()
    m, _ = demean(df[["y"]].to_numpy(), fe_codes(df, ["g1", "g2"]))
    assert abs(pd.Series(m[:, 0]).groupby(df.g1).mean()).max() < 1e-6
    assert abs(pd.Series(m[:, 0]).groupby(df.g2).mean()).max() < 1e-6


def test_tsls_recovers_quadratic_curve():
    df = _synthetic(curvature=1.5)
    codes = fe_codes(df, ["g1", "g2"])
    X, Z = poly_basis(df.x.to_numpy(), 2), poly_basis(df.z.to_numpy(), 2)
    m, _ = demean(np.column_stack([df.y, X, Z]), codes)
    fit = tsls(m[:, 0], m[:, 1:3], m[:, 3:5], df.cl.to_numpy())
    assert np.allclose(fit["beta"], [-2.0, 1.5], atol=0.15)
    slope, se = lincomb(poly_slope_grad(np.log(0.6), 2), fit["beta"], fit["vcov"])
    assert abs(slope - (-2.0 + 3.0 * np.log(0.6))) < 0.1 and se > 0
    fs = first_stage_stats(m[:, 1:3], m[:, 3:5], df.cl.to_numpy())
    assert min(fs["F_SW"]) > 10


def test_spline_slope_grad_matches_numeric():
    knots = (0.9, 0.8, 0.7)
    beta = np.array([-1.5, 0.3, 0.5, -0.2])
    for x0 in np.log([0.95, 0.85, 0.75, 0.6]):
        h = 1e-6
        num = (spline_basis(np.array([x0 + h]), knots) - spline_basis(np.array([x0 - h]), knots)) / (2 * h)
        assert np.allclose(num[0] @ beta, spline_slope_grad(x0, knots) @ beta)

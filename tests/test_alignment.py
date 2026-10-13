from dataclasses import replace

import numpy as np
import pytest

from tblca.alignment import (
    carbon_coefficients, id_margin, matched_ratio, monotonicity_violations, myopic_argmax,
    policy_path, stage_carbon_nondecreasing, theta, theta_bar, upper_hull,
)
from tblca.markdown import MarkdownProblem, solve, stage_outcomes, unit_economics


def problem(**kw):
    base = dict(horizon=3, x0=20, q0=6.0, lam_a=0.6, lam_f=1.5, p_base=0.2, eps=1.9,
                kappa=0.6, price=3.0, margin_fresh=1.0, unit_kg=1.0, ef_prod=3.0,
                ef_eol=0.5, ops_kg=0.01, objective="weighted")
    base.update(kw)
    return MarkdownProblem(**base)


def test_carbon_coefficients_reproduce_unit_carbon():
    p = problem()
    for r in (1, 2, 3):
        a, b = carbon_coefficients(p, r)
        for d in p.discounts:
            g = (1 - d) ** p.eps + p.kappa * (1 - (1 - d) ** p.eps)
            assert a * g + b == pytest.approx(unit_economics(p, r, d)["carbon_kg"])


def test_theta_linear_in_mu_and_zero_at_zero():
    p = problem()
    assert theta(p, 2, 0.0) == 0.0
    assert theta(p, 2, 0.4) == pytest.approx(2 * theta(p, 2, 0.2))
    assert theta_bar(p, 0.2) == pytest.approx(np.mean([theta(p, r, 0.2) for r in (1, 2, 3)]))


def test_theta_rejects_nonpositive_margin():
    with pytest.raises(ValueError):
        theta(problem(margin_fresh=0.0), 1, 0.1)


def test_uncapped_stage_carbon_increasing_iff_kappa_a_plus_b_nonnegative():
    p = problem(x0=400)
    a, b = carbon_coefficients(p, 1)
    assert a * p.kappa + b > 0
    assert stage_carbon_nondecreasing(p, 400, 1)
    assert id_margin(p, 400, 1, 0.3) == pytest.approx(a * p.kappa + b, rel=1e-6)
    bad = problem(x0=400, ef_eol=-20.0)
    a2, b2 = carbon_coefficients(bad, 1)
    assert a2 * bad.kappa + b2 < 0
    assert not stage_carbon_nondecreasing(bad, 400, 1)


def test_myopic_argmax_monotone_in_mu_when_condition_holds():
    p = problem(x0=60, price=6.0, margin_fresh=3.0)
    x, r = 20, 2
    assert stage_carbon_nondecreasing(p, x, r)
    ds = [myopic_argmax(p, x, r, m) for m in np.linspace(0, 3, 31)]
    assert np.all(np.diff(ds) >= 0)
    assert ds[-1] > ds[0]


def test_policy_path_profit_falls_and_carbon_rises_with_mu():
    p = problem()
    sols = policy_path(p, [0.0, 0.2, 0.5, 1.0, 3.0])
    prof = [s.metrics["profit"] for s in sols]
    carb = [s.metrics["carbon_kg"] for s in sols]
    assert np.all(np.diff(prof) <= 1e-9) and np.all(np.diff(carb) >= -1e-9)
    v = monotonicity_violations(sols)
    assert v["pairs"] == 4 * p.horizon * p.x0 and v["violations"] >= 0


def test_large_mu_recovers_carbon_optimum_and_zero_mu_profit_optimum():
    p = problem()
    big = solve(replace(p, carbon_price=1e4)).metrics["carbon_kg"]
    assert big == pytest.approx(solve(replace(p, objective="carbon")).metrics["carbon_kg"])
    assert solve(replace(p, carbon_price=0.0)).metrics["profit"] == pytest.approx(
        solve(replace(p, objective="profit")).metrics["profit"])


def test_upper_hull_drops_interior_points():
    h = upper_hull(np.array([0, 1, 2, 1]), np.array([0, 2, 3, 1]))
    assert h.tolist() == [[0, 0], [1, 2], [2, 3]]


def test_matched_ratio_interpolates_on_hull():
    lw, cp, gw, ratio = matched_ratio([0, 1, 2], [0, 2, 3], [0.5, 1.5, 5.0], [0.5, 1.0, 9.0])
    assert lw.tolist() == [0.5, 1.5]
    assert cp.tolist() == [1.0, 2.5]
    assert ratio.tolist() == [2.0, 2.5]


def test_scalarisation_dominance_on_random_policies():
    p = problem(x0=12)
    mu = 0.3
    opt = solve(replace(p, carbon_price=mu)).metrics
    rng = np.random.default_rng(0)
    from tblca.markdown import evaluate_policy
    for _ in range(30):
        pol = rng.choice(p.discounts, size=(p.horizon + 1, p.x0 + 1))
        m = evaluate_policy(p, pol).metrics
        assert m["profit"] + mu * m["carbon_kg"] <= opt["profit"] + mu * opt["carbon_kg"] + 1e-9
        assert not (m["profit"] >= opt["profit"] and m["carbon_kg"] > opt["carbon_kg"] + 1e-9)


def test_stage_outcomes_carbon_matches_coefficients():
    p = problem()
    a, b = carbon_coefficients(p, 2)
    out = stage_outcomes(p, 10, 2, 0.3)
    g = 0.7 ** p.eps + p.kappa * (1 - 0.7 ** p.eps)
    assert out["carbon_kg"] == pytest.approx(out["sales"] * (a * g + b))


def test_efficient_condition_weaker_than_global():
    from tblca.alignment import carbon_nondecreasing_on_efficient, efficient_discounts
    p = problem(x0=60, price=6.0, margin_fresh=3.0)
    for x in (3, 8, 20, 60):
        for r in (1, 2, 3):
            eff = efficient_discounts(p, x, r)
            assert len(eff) >= 1
            if stage_carbon_nondecreasing(p, x, r):
                assert carbon_nondecreasing_on_efficient(p, x, r)
            if carbon_nondecreasing_on_efficient(p, x, r):
                ds = [myopic_argmax(p, x, r, m) for m in np.linspace(0.01, 3, 30)]
                assert np.all(np.diff(ds) >= 0)


def test_supported_frontier_dominates_any_policy_and_waste_penalty_points():
    from tblca.alignment import supported_frontier
    from tblca.markdown import evaluate_policy
    p = problem(x0=15)
    verts = supported_frontier(p, 5.0)
    prof = np.array([v[1] for v in verts])
    carb = np.array([v[2] for v in verts])
    assert np.all(np.diff(prof) < 0) and np.all(np.diff(carb) > 0)
    pm = prof[0]
    hl, hg = pm - prof, carb - carb[0]
    pts = [solve(replace(p, waste_penalty=w)).metrics for w in np.linspace(0, 6, 25)]
    rng = np.random.default_rng(1)
    pts += [evaluate_policy(p, rng.choice(p.discounts, size=(p.horizon + 1, p.x0 + 1))).metrics
            for _ in range(20)]
    for m in pts:
        loss = pm - m["profit"]
        if 0 <= loss <= hl[-1]:
            assert np.interp(loss, hl, hg) >= m["carbon_kg"] - carb[0] - 1e-7

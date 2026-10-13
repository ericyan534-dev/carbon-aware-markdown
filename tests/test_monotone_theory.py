"""Tests for tblca.monotone_theory generic DP and monotonicity diagnostics."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import binom

from tblca.markdown import solve
from tblca.monotone_theory import (
    GenericProblem, concavity_gap, expected_sales, from_markdown, is_monotone_in_x,
    pair_single_crossing, pairwise_single_crossing, q_table, q_table_pmf, solve_bernoulli,
    solve_generic,
)
from tblca.scenario import build


@pytest.mark.parametrize("obj,cp,wp", [("profit", 0, 0), ("carbon", 0, 0), ("waste", 0, 0),
                                       ("weighted", 0.19, 0), ("weighted", 0.19, 1.0)])
def test_generic_solver_reproduces_markdown_solve(obj, cp, wp):
    p = replace(build("dairy", "UK", objective=obj, carbon_price=cp), waste_penalty=wp)
    ref = solve(p)
    v, pol, qs = solve_generic(from_markdown(p))
    assert np.allclose(v, ref.value, atol=1e-9)
    idx = {d: j for j, d in enumerate(p.discounts)}
    for r in range(1, p.horizon + 1):       # reference action is optimal in our Q (ties allowed)
        ref_j = np.array([idx[d] for d in ref.policy[r]])
        assert np.allclose(qs[r][np.arange(p.x0 + 1), ref_j], v[r], atol=1e-9)


def test_q_table_pmf_matches_poisson_q_table():
    rng = np.random.default_rng(0)
    lam, u, V = np.array([0.5, 2.0, 7.0]), np.array([1.0, 0.6, -0.2]), rng.normal(size=25)
    from scipy.stats import poisson
    pmfs = [poisson.pmf(np.arange(25), t) for t in lam]
    assert np.allclose(q_table(lam, u, V), q_table_pmf(pmfs, u, V), atol=1e-12)


def test_expected_sales_tp2_lemma2():
    rng = np.random.default_rng(1)
    xs = np.arange(1, 200)
    for _ in range(200):
        l1, l2 = np.sort(rng.uniform(1e-3, 80, 2))
        ratio = expected_sales(l2, xs) / expected_sales(l1, xs)
        assert np.all(np.diff(ratio) >= -1e-12 * ratio[1:])


def test_theorem1_final_day_monotone_when_some_margin_positive():
    rng = np.random.default_rng(2)
    for _ in range(300):
        J, X = int(rng.integers(2, 8)), int(rng.integers(1, 80))
        lam = np.cumsum(rng.exponential(3, J)) + 0.01
        w, u = rng.uniform(-1, 1), rng.normal(0.3, 1.5, J)
        if (u + w).max() <= 0:
            continue
        g = GenericProblem(np.vstack([np.zeros(J), lam]), np.vstack([np.zeros(J), u]), X, w)
        _, pol, _ = solve_generic(g)
        assert is_monotone_in_x(pol)
        assert np.all((u + w)[pol[1, 1:]] > 0)


def test_theorem2_bernoulli_concave_and_monotone_under_R():
    rng = np.random.default_rng(3)
    for _ in range(300):
        H, J, X = int(rng.integers(1, 7)), int(rng.integers(2, 7)), int(rng.integers(1, 40))
        pi = np.vstack([np.zeros(J)] + [np.sort(rng.uniform(0, 1, J)) for _ in range(H)])
        base = rng.normal(0.5, 1, J)
        u = np.vstack([np.zeros(J)] + [base] * H)     # time-invariant rewards => (R)
        w = max(rng.uniform(-0.5, 1.0), -base.max())
        v, pol = solve_bernoulli(pi, u, X, w)
        assert concavity_gap(v) <= 1e-9
        assert is_monotone_in_x(pol)


def test_theorem3_single_crossing_with_concave_continuation_binomial():
    rng = np.random.default_rng(4)
    for _ in range(300):
        J, X = int(rng.integers(2, 5)), int(rng.integers(2, 40))
        n, ps = int(rng.integers(1, X + 5)), np.sort(rng.uniform(0.01, 1, J))
        pmfs = [binom.pmf(np.arange(X + 1), n, t) for t in ps]
        inc = np.sort(rng.normal(0, 1.5, X))[::-1]
        V = np.concatenate(([0.0], np.cumsum(inc)))
        u = rng.normal(inc[0], 1.5, J)
        q = q_table_pmf(pmfs, u, V)
        for i in range(J):
            for j in range(i + 1, J):
                if max(u[i], u[j]) >= inc[0]:
                    assert pair_single_crossing(q[:, j] - q[:, i], 1e-10)


def test_single_crossing_helpers():
    assert pair_single_crossing(np.array([0.0, -1.0, -0.5, 0.2, 1.0]))
    assert not pair_single_crossing(np.array([0.0, 0.3, -0.1]))
    q = np.array([[0.0, 0.0], [1.0, 0.5], [1.0, 2.0]])
    assert pairwise_single_crossing(q)
    assert not pairwise_single_crossing(np.array([[0.0, 0.0], [1.0, 2.0], [1.0, 0.5]]))


def test_counterexample_poisson_batch_is_non_monotone():
    rho, u = np.array([1.0, 2.0, 6.5]), np.array([1.0, 0.67, 0.43])
    lam = np.vstack([np.zeros(3), 4.0 * rho, 4.9 * rho])
    g = GenericProblem(lam, np.vstack([np.zeros(3), u, u]), 23, 0.0)
    v, pol, qs = solve_generic(g)
    assert is_monotone_in_x(pol[:2])                      # last day: Theorem 1
    assert list(pol[2, 20:24]) == [1, 1, 0, 0]            # r = 2: drops back at x = 22
    assert qs[2][21, 1] - qs[2][21, 0] > 0.04
    assert qs[2][22, 0] - qs[2][22, 1] > 0.05
    assert concavity_gap(v[1:2]) > 0.1                    # V_1 not concave


def test_paper_default_cell_monotone_all_objectives():
    for obj, cp in (("profit", 0.0), ("carbon", 0.0), ("weighted", 0.19)):
        _, pol, qs = solve_generic(from_markdown(build("poultry", "US", objective=obj,
                                                       carbon_price=cp)))
        assert is_monotone_in_x(pol)
        assert all(pairwise_single_crossing(q) for q in qs[1:])

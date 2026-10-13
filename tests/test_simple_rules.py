import numpy as np
import pytest

from tblca.fast_dp import build_tables, fast_evaluate, fast_solve, sales_matrix
from tblca.markdown import MarkdownProblem, evaluate_policy, fixed_policy, solve
from tblca.simple_rules import (best_scaled_rule, best_threshold_rule, best_time_only_rule,
                                scaled_thresholds, threshold_policy)


def problem(**kw):
    base = dict(horizon=3, x0=20, q0=5.0, lam_a=0.6, lam_f=0.5, p_base=0.2, eps=1.9,
                kappa=0.6, price=3.0, margin_fresh=1.0, unit_kg=0.5, ef_prod=3.0,
                ef_eol=0.6, ops_kg=0.05, carbon_price=0.19, objective="weighted")
    base.update(kw)
    return MarkdownProblem(**base)


def test_sales_matrix_rows_are_distributions():
    s = sales_matrix(4.0, 15)
    assert s.sum(1) == pytest.approx(np.ones(16))
    assert np.allclose(np.triu(s, 1), 0.0)


@pytest.mark.parametrize("obj", ["profit", "carbon", "waste", "weighted"])
def test_fast_solve_matches_reference(obj):
    p = problem(objective=obj)
    ref = solve(p)
    value, policy = fast_solve(p)
    assert value == pytest.approx(ref.value, abs=1e-9)
    assert np.array_equal(policy, ref.policy)


def test_fast_evaluate_matches_reference():
    p = problem()
    pol = fixed_policy(p, lambda r: 0.35 if r == 1 else 0.0)
    ref = evaluate_policy(p, pol)
    fast = fast_evaluate(p, pol)
    assert fast["value"] == pytest.approx(ref.value[p.horizon, p.x0])
    for m, v in ref.metrics.items():
        assert fast[m] == pytest.approx(v, abs=1e-9)


def test_fast_evaluate_rejects_off_grid_discount():
    p = problem()
    with pytest.raises(ValueError):
        fast_evaluate(p, fixed_policy(p, 0.33))


def test_threshold_policy_shape_and_levels():
    p = problem()
    pol = threshold_policy(p, 0.0, 0.5, (5, 10, 15))
    assert pol[1, 5] == 0.0 and pol[1, 6] == 0.5 and pol[3, 20] == 0.5
    with pytest.raises(ValueError):
        threshold_policy(p, 0.0, 0.5, (5,))


def test_best_threshold_rule_bounded_by_optimum_and_consistent():
    p = problem()
    opt = solve(p).value[p.horizon, p.x0]
    t = build_tables(p)
    rule = best_threshold_rule(p, t)
    assert rule.value <= opt + 1e-9
    pol = threshold_policy(p, rule.lo, rule.hi, rule.thresholds)
    assert evaluate_policy(p, pol).value[p.horizon, p.x0] == pytest.approx(rule.value)
    sched, time_val = best_time_only_rule(p, t)
    assert len(sched) == p.horizon and time_val <= opt + 1e-9
    scaled, theta = best_scaled_rule(p, tables=t)
    assert scaled.value <= rule.value + 1e-9
    assert scaled.thresholds == pytest.approx(scaled_thresholds(p, theta))

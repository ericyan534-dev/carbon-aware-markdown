import numpy as np
import pytest

from tblca.markdown import (
    MarkdownProblem, stage_outcomes, solve, evaluate_policy, fixed_policy, simulate,
)


def problem(**kw):
    base = dict(horizon=4, x0=30, q0=6.0, lam_a=0.6, lam_f=0.5, p_base=0.2, eps=1.9,
                kappa=0.6, price=3.0, margin_fresh=1.0, unit_kg=0.5, ef_prod=3.0,
                ef_eol=0.6, ops_kg=0.05, carbon_price=0.0, objective="weighted")
    base.update(kw)
    return MarkdownProblem(**base)


def test_zero_inventory_has_zero_value():
    sol = solve(problem())
    assert sol.value[:, 0] == pytest.approx(0.0)


def test_stage_outcomes_expected_sales_capped_by_inventory():
    out = stage_outcomes(problem(), x=2, r=3, d=0.5)
    assert out["sales"] <= 2 + 1e-12
    assert out["sales"] == pytest.approx(sum(out["pmf_sales"] * np.arange(3)))


@pytest.mark.parametrize("obj", ["profit", "carbon", "waste", "weighted"])
def test_optimal_dominates_fixed_policies(obj):
    prob = problem(objective=obj, carbon_price=0.2)
    opt = solve(prob)
    for d in prob.discounts:
        v = evaluate_policy(prob, fixed_policy(prob, d)).value[prob.horizon, prob.x0]
        assert opt.value[prob.horizon, prob.x0] >= v - 1e-9


def test_policy_evaluation_matches_dp_value():
    prob = problem(carbon_price=0.1)
    opt = solve(prob)
    ev = evaluate_policy(prob, opt.policy)
    assert ev.value[prob.horizon, prob.x0] == pytest.approx(opt.value[prob.horizon, prob.x0])


def test_waste_objective_discounts_deepest_with_ample_stock():
    prob = problem(objective="waste", x0=200)
    opt = solve(prob)
    assert opt.policy[prob.horizon, prob.x0] == max(prob.discounts)


def test_simulation_agrees_with_exact_metrics():
    prob = problem(carbon_price=0.1)
    opt = solve(prob)
    ev = evaluate_policy(prob, opt.policy)
    sim = simulate(prob, opt.policy, n=20000, seed=1)
    for k in ("profit", "carbon_kg", "waste_units"):
        assert sim[k] == pytest.approx(ev.metrics[k], rel=0.03, abs=0.05)


def test_invalid_objective():
    with pytest.raises(ValueError):
        problem(objective="nonsense")


def test_zero_waste_penalty_leaves_solution_unchanged():
    a = solve(problem(carbon_price=0.1))
    b = solve(problem(carbon_price=0.1, waste_penalty=0.0))
    assert np.array_equal(a.policy, b.policy)


def test_waste_penalty_reduces_expected_waste():
    lo = solve(problem(waste_penalty=0.0)).metrics["waste_units"]
    hi = solve(problem(waste_penalty=5.0)).metrics["waste_units"]
    assert hi < lo

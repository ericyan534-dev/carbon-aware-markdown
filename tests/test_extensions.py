import numpy as np
import pytest

from tblca.ext_dp import evaluate_ext, solve_ext
from tblca.ext_units import respend_kg, respend_unit_fn, waiting_share, waiting_unit_fn
from tblca.markdown import (
    MarkdownProblem, evaluate_policy, fixed_policy, solve, unit_economics,
)


def problem(**kw):
    base = dict(horizon=3, x0=20, q0=5.0, lam_a=0.6, lam_f=0.5, p_base=0.2, eps=1.9,
                kappa=0.6, price=3.0, margin_fresh=1.0, unit_kg=0.5, ef_prod=3.0,
                ef_eol=0.6, ops_kg=0.05, carbon_price=0.2, objective="weighted")
    base.update(kw)
    return MarkdownProblem(**base)


@pytest.mark.parametrize("obj", ["profit", "carbon", "waste", "weighted"])
def test_generalized_dp_reproduces_base(obj):
    p = problem(objective=obj)
    a, b = solve(p), solve_ext(p)
    assert np.allclose(a.value, b.value, atol=1e-10)
    assert np.array_equal(a.policy, b.policy)
    for m, v in a.metrics.items():
        assert b.metrics[m] == pytest.approx(v, abs=1e-10)


def test_by_inventory_matches_evaluate_at_each_x():
    p = problem()
    pol = fixed_policy(p, lambda r: 0.35 if r == 1 else 0.0)
    full = evaluate_ext(p, pol).by_inventory
    from dataclasses import replace
    for x in (0, 5, 13):
        px = replace(p, x0=x)
        m = evaluate_policy(px, fixed_policy(px, lambda r: 0.35 if r == 1 else 0.0)).metrics
        assert full["carbon_kg"][x] == pytest.approx(m["carbon_kg"])
        assert full["waste_units"][x] == pytest.approx(m["waste_units"])


def test_zero_intensity_and_zero_waiting_are_identity():
    p = problem()
    for fn in (respend_unit_fn(0.0), waiting_unit_fn(0.0)):
        for d in p.discounts:
            assert fn(p, 2, d) == pytest.approx(unit_economics(p, 2, d))


def test_respend_formula_and_signs():
    p = problem()
    assert respend_kg(p, 0.0, 0.4) == pytest.approx(0.0)
    d = 0.3
    g = (1 - d) ** p.eps + p.kappa * (1 - (1 - d) ** p.eps)
    assert respend_kg(p, d, 0.4) == pytest.approx(0.4 * p.price * g * d)
    div = respend_kg(p, d, 0.4, incremental="diversion")
    assert div == pytest.approx(0.4 * p.price * (g * d - (1 - g) * (1 - d)))
    assert respend_kg(p, d, 0.4, mpc=0.5) == pytest.approx(0.5 * respend_kg(p, d, 0.4))
    u0, u1 = unit_economics(p, 1, d), respend_unit_fn(0.4)(p, 1, d)
    assert u1["carbon_kg"] == pytest.approx(u0["carbon_kg"] - respend_kg(p, d, 0.4))
    assert u1["profit"] == pytest.approx(u0["profit"])


def test_respend_validation():
    with pytest.raises(ValueError):
        respend_unit_fn(0.4, incremental="bogus")
    with pytest.raises(ValueError):
        respend_unit_fn(-1.0)
    with pytest.raises(ValueError):
        waiting_unit_fn(1.5)


def test_waiting_raises_cannibalization():
    p = problem()
    assert waiting_share(0.7, 0.0, 0.3) == pytest.approx(0.7)
    assert waiting_share(0.7, 0.5, 0.2) == pytest.approx(0.7 + 0.2 * 0.5 * 0.3)
    assert waiting_share(0.7, 0.1, 0.2, stress=True) == pytest.approx(0.7 + 0.2 * 0.3)
    assert waiting_share(0.7, 0.0, 0.2, stress=True) == pytest.approx(0.7)
    base, w = unit_economics(p, 1, 0.4), waiting_unit_fn(0.3)(p, 1, 0.4)
    assert w["profit"] < base["profit"] and w["carbon_kg"] > base["carbon_kg"]
    cost_only = waiting_unit_fn(0.3, credit=False)(p, 1, 0.4)
    assert cost_only["carbon_kg"] == pytest.approx(base["carbon_kg"])
    assert cost_only["profit"] == pytest.approx(w["profit"])


def test_respend_makes_carbon_policy_shallower_or_equal():
    p = problem(objective="carbon")
    base = solve_ext(p).metrics
    ext = solve_ext(p, respend_unit_fn(0.6)).metrics
    assert ext["disc_units"] / ext["sales"] <= base["disc_units"] / base["sales"] + 1e-9

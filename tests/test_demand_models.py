import numpy as np
import pytest

from tblca import demand_models as dm
from tblca.markdown import evaluate_policy, solve
from tblca.params import SCC_USD_PER_KG
from tblca.policy_suite import headline, plan_policies, evaluate_all, run
from tblca.scenario import build
from scripts.run_experiments import policy_table


@pytest.fixture(scope="module")
def prob():
    return build("dairy", "Hong Kong", carbon_price=SCC_USD_PER_KG)


def test_default_model_reproduces_planning_evaluation(prob):
    pol = solve(prob).policy
    ref = evaluate_policy(prob, pol)
    got = dm.evaluate_under(prob, pol, dm.DemandModel())
    for k in ("profit", "carbon_kg", "waste_units", "sales"):
        assert got[k] == pytest.approx(ref.metrics[k], rel=1e-12)
    assert got["value"] == pytest.approx(ref.value[prob.horizon, prob.x0], rel=1e-12)


def test_default_model_oracle_matches_solve(prob):
    pol, _ = dm.solve_under(prob, dm.DemandModel())
    assert np.array_equal(pol, solve(prob).policy)


def test_ces_gate_matches_rebound_gate(prob):
    from tblca.substitution import rebound_gate
    for d in (0.0, 0.2, 0.5):
        assert dm.gate(prob, dm.DemandModel(), d) == pytest.approx(
            float(rebound_gate(d, prob.eps, prob.kappa)))


def test_linear_chord_matches_level_at_anchor_and_full_price(prob):
    m = dm.DemandModel(shape="linear_chord", anchor_d=0.3)
    assert dm.multiplier(prob, m, 0.0) == pytest.approx(1.0)
    assert dm.multiplier(prob, m, 0.3) == pytest.approx(0.7 ** -prob.eps)


def test_linear_tangent_matches_point_elasticity_at_anchor(prob):
    m = dm.DemandModel(shape="linear_tangent", anchor_d=0.3)
    h = 1e-6
    lvl = dm.multiplier(prob, m, 0.3)
    slope = (dm.multiplier(prob, m, 0.3 + h) - dm.multiplier(prob, m, 0.3 - h)) / (2 * h)
    elasticity = slope * 0.7 / lvl          # -dlnq/dln(1-d)
    assert lvl == pytest.approx(0.7 ** -prob.eps)
    assert elasticity == pytest.approx(prob.eps, rel=1e-5)


def test_retained_share_bounds(prob):
    for shape in dm.SHAPES:
        m = dm.DemandModel(shape=shape)
        for d in prob.discounts:
            assert 0.0 <= dm.retained_share(prob, m, d) <= 1.0


@pytest.mark.parametrize("disp", [1.0, 2.5])
def test_sales_pmf_mean_and_mass(disp):
    pmf = dm.sales_pmf(6.0, 200, disp)
    assert pmf.sum() == pytest.approx(1.0)
    assert pmf @ np.arange(201) == pytest.approx(6.0, rel=1e-6)
    var = pmf @ np.arange(201) ** 2 - 36.0
    assert var == pytest.approx(6.0 * disp, rel=1e-4)


def test_sales_pmf_edge_cases():
    assert dm.sales_pmf(0.0, 3).tolist() == [1.0, 0.0, 0.0, 0.0]
    assert dm.sales_pmf(5.0, 0).tolist() == [1.0]


def test_invalid_models_rejected():
    with pytest.raises(ValueError):
        dm.DemandModel(shape="cubic")
    with pytest.raises(ValueError):
        dm.DemandModel(dispersion=0.5)
    with pytest.raises(ValueError):
        dm.DemandModel(eps_true=-1.0)


def test_oracle_beats_planning_policy_under_misspecification(prob):
    m = dm.DemandModel(dispersion=3.0, eps_true=1.3)
    plan = solve(prob).policy
    _, oracle = dm.solve_under(prob, m)
    assert oracle["value"] >= dm.evaluate_under(prob, plan, m)["value"] - 1e-9


def test_policy_suite_reproduces_paper_dairy_hk(prob):
    ref = policy_table("dairy", "Hong Kong")
    _, h = run(prob)
    rule, scc = ref.loc["Last-day 35%"], ref.loc["SCC-weighted (ours)"]
    assert h["profit_gain_scc_vs_static"] == pytest.approx(100 * (scc.profit / rule.profit - 1), rel=1e-9)
    assert h["carbon_gain_scc_vs_static"] == pytest.approx(100 * (scc.carbon_kg / rule.carbon_kg - 1), rel=1e-9)
    assert h["overstatement_static_pct"] == pytest.approx(
        100 * (rule.naive_carbon_kg / rule.carbon_kg - 1), rel=1e-9)
    assert h["carbon_dominates_waste"]


def test_policy_suite_keys(prob):
    ev = evaluate_all(prob, plan_policies(prob))
    assert set(ev) == {"static", "waste", "profit", "carbon", "scc"}
    assert "scc_retention_pct" in headline(prob, ev)

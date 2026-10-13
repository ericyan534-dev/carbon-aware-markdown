import numpy as np
import pytest

from tblca.substitution import (
    freshness_gate, spoilage_gate, rebound_gate, rebound_gate_linear, delta_eff,
)


def test_freshness_gate_zero_at_expiry_and_saturates():
    assert freshness_gate(0.0, lam=0.5) == pytest.approx(0.0)
    assert freshness_gate(100.0, lam=0.5) == pytest.approx(1.0)


def test_freshness_gate_monotone_in_rsl():
    rsl = np.linspace(0, 10, 50)
    assert np.all(np.diff(freshness_gate(rsl, lam=0.3)) > 0)


def test_spoilage_gate_limits():
    # at RSL=0 consumption is certain to be wasted; long RSL -> 1 - p_base
    assert spoilage_gate(0.0, lam=0.5, p_base=0.2) == pytest.approx(0.0)
    assert spoilage_gate(1e3, lam=0.5, p_base=0.2) == pytest.approx(0.8)


def test_rebound_gate_no_discount_is_one():
    assert rebound_gate(0.0, eps=1.5) == pytest.approx(1.0)


def test_rebound_gate_equals_share_of_non_incremental_demand():
    # demand multiplier m = (1-d)^-eps; non-incremental share = 1/m
    d, eps = 0.3, 1.5
    m = (1 - d) ** (-eps)
    assert rebound_gate(d, eps) == pytest.approx(1 / m)


def test_rebound_gate_first_order_matches_linear_gate():
    d, eps = 1e-4, 1.7
    assert rebound_gate(d, eps) == pytest.approx(rebound_gate_linear(d, eps), abs=1e-7)


def test_cannibalization_raises_gate():
    assert rebound_gate(0.4, 1.5, kappa=0.5) > rebound_gate(0.4, 1.5, kappa=0.0)
    assert rebound_gate(0.4, 1.5, kappa=1.0) == pytest.approx(1.0)


def test_linear_gate_is_clipped():
    assert rebound_gate_linear(0.9, 3.0) == 0.0


def test_delta_eff_bounded_and_product():
    d = delta_eff(rsl=3.0, discount=0.3, lam=0.4, p_base=0.25, eps=1.2)
    expected = (freshness_gate(3.0, 0.4) * spoilage_gate(3.0, 0.4, 0.25)
                * rebound_gate(0.3, 1.2))
    assert d == pytest.approx(expected)
    assert 0.0 <= d <= 1.0


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_invalid_discount_raises(bad):
    with pytest.raises(ValueError):
        rebound_gate(bad, 1.0)

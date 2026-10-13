import pytest

from tblca.params import ARCHETYPES, REGIONS, ops_per_kg
from tblca.scenario import build, heuristic_policies


def test_pathway_shares_sum_to_one():
    for g in REGIONS.values():
        assert g.landfill + g.combustion + g.ad + g.compost == pytest.approx(1.0)


def test_incineration_region_has_lower_eol_credit():
    a = ARCHETYPES["dairy"]
    assert REGIONS["Guangdong"].eol(a) < REGIONS["Hong Kong"].eol(a)


def test_ops_small_relative_to_production():
    for g in REGIONS.values():
        assert 0 < ops_per_kg(g, 3) < 0.05


def test_build_scales_inventory_with_horizon():
    p = build("bakery", "UK", stock_ratio=2.0, q0=8.0)
    assert p.x0 == 32 and p.horizon == 2


def test_heuristics_include_last_day_rule():
    p = build("dairy", "Hong Kong")
    pol = heuristic_policies(p)["Last-day 35%"]
    assert pol[1, 5] == 0.35 and pol[2, 5] == 0.0

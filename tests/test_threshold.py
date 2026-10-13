"""Elasticity thresholds, the stylized curve predictor and the AUC used out of sample."""
import numpy as np
import pandas as pd

from scripts.threshold_curves import EPS_STAR, curve_predict, eps_star_theta, ratio_for
from scripts.threshold_prediction import auc


def test_eps_star_theta_reduces_to_eps_star_without_carbon_value():
    assert np.isclose(eps_star_theta(0.0), EPS_STAR)
    assert np.isclose(EPS_STAR, 1.2516, atol=1e-4)
    assert eps_star_theta(0.5) < eps_star_theta(0.1) < EPS_STAR
    assert np.isclose(eps_star_theta(1.0), 1.0)


def test_curve_predict_interpolates_per_type():
    curves = pd.DataFrame({"archetype": ["dairy"] * 3 + ["bakery"] * 3,
                           "eps": [1.0, 2.0, 3.0] * 2, "ratio": [90, 100, 100, 100, 96, 100]})
    pred = curve_predict(curves, np.array(["dairy", "bakery", "dairy"]), np.array([1.5, 2.5, 9.0]))
    assert np.allclose(pred, [95.0, 98.0, 100.0])


def test_auc_extremes_and_ties():
    y = np.array([True, True, False, False])
    assert auc([4, 3, 2, 1], y) == 1.0
    assert auc([1, 2, 3, 4], y) == 0.0
    assert auc([1, 1, 1, 1], y) == 0.5
    assert np.isnan(auc([1, 2], np.array([True, True])))


def test_profit_policy_is_optimal_far_from_threshold():
    ratio, _ = ratio_for("dairy", 8.0, 0.6)          # discounts never pay: both policies hold price
    assert np.isclose(ratio, 100.0)
    ratio, _ = ratio_for("dairy", 8.0, 1.31)      # inside the weak zone
    assert ratio < 95.0

import numpy as np
import pandas as pd
import pytest

from tblca.elasticity import prepare_panel, estimate_fe_ols, estimate_fe_iv


def synthetic(eps=1.5, n_store=40, n_prod=6, n_days=60, confound=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_prod):
        chain = rng.uniform(0.6, 1.0, n_days)          # chain-wide promo (exogenous)
        for s in range(n_store):
            base = rng.normal(0, 0.5)
            shock = rng.normal(0, 0.3, n_days)          # local demand shock
            disc = np.clip(chain + rng.normal(0, 0.03, n_days) - confound * shock, 0.35, 1.0)
            sales = np.exp(base - eps * np.log(disc) + shock)
            for t in range(n_days):
                rows.append(dict(store_id=s, product_id=p, first_category_id=p % 2,
                                 dt=f"2024-01-{t+1:02d}" if t < 31 else f"2024-02-{t-30:02d}",
                                 sale_amount=sales[t], discount=disc[t],
                                 stock_hour6_22_cnt=0, holiday_flag=0, activity_flag=0,
                                 avg_temperature=20.0, precpt=0.0))
    return pd.DataFrame(rows)


def test_prepare_panel_drops_stockouts_and_bad_discounts():
    df = synthetic(n_store=3, n_prod=1, n_days=10)
    df.loc[0, "stock_hour6_22_cnt"] = 5
    df.loc[1, "discount"] = 0.0
    out = prepare_panel(df)
    assert len(out) == len(df) - 2
    assert {"log_q", "log_p", "series"}.issubset(out.columns)


def test_fe_ols_recovers_elasticity_without_confounding():
    res = estimate_fe_ols(prepare_panel(synthetic(eps=1.5)))
    assert res["eps"] == pytest.approx(1.5, abs=0.08)


def test_fe_iv_recovers_elasticity_under_confounding():
    df = prepare_panel(synthetic(eps=1.5, confound=0.4))
    ols = estimate_fe_ols(df)["eps"]
    iv = estimate_fe_iv(df)["eps"]
    assert abs(iv - 1.5) < abs(ols - 1.5)
    assert iv == pytest.approx(1.5, abs=0.15)


def test_estimates_by_category_returned():
    res = estimate_fe_ols(prepare_panel(synthetic()), by="first_category_id")
    assert set(res) == {0, 1}


def test_standard_error_positive_and_small_for_large_sample():
    res = estimate_fe_ols(prepare_panel(synthetic(eps=1.5)))
    assert 0 < res["se"] < 0.1

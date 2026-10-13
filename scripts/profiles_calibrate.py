"""Calibrate 4,121 store-product demand profiles from FreshRetailNet-50K (Sec. V-D).

Each profile has its own full-price demand level q0 (scaled to a median of 8 units/day),
variance/mean ratio, and its category's IV elasticity (empirical-Bayes shrunk toward the
pooled estimate). Requires data/raw/train.parquet (see README).

Outputs (outputs/profiles/):
  category_elasticity.csv  per-category FE-IV elasticity and shrinkage weight
  series_profiles.csv      sampled series with q0, dispersion, eps, product type
  calibration.json         scale factor, sampling and shrinkage choices
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tblca.elasticity import estimate_fe_iv, prepare_panel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "train.parquet"
OUT = ROOT / "outputs" / "profiles"
SEED = 20261005
N_TARGET = 4000
MIN_PER_CAT = 20
MIN_STABLE_DAYS = 10
MIN_CAT_OBS_IV = 1000
POOLED_EPS = 1.943193886417333      # pooled IV estimate (scripts/estimate_elasticity.py)
TARGET_MEDIAN_Q0 = 8.0
FULL_PRICE = (0.99, 1.01)
ARCHETYPES = ("dairy", "bakery", "produce", "poultry")
COLS = ["store_id", "product_id", "first_category_id", "dt", "sale_amount", "discount",
        "stock_hour6_22_cnt", "holiday_flag", "activity_flag", "avg_temperature", "precpt"]


def series_stats(df):
    stable = df[(df.stock_hour6_22_cnt == 0) & df.discount.between(*FULL_PRICE)]
    g = stable.groupby(["store_id", "product_id"])
    stats = g.sale_amount.agg(n_stable="count", level="mean", var="var")
    cat = df.groupby(["store_id", "product_id"]).first_category_id.first()
    stats = stats.join(cat).reset_index()
    return stats[(stats.n_stable >= MIN_STABLE_DAYS) & (stats.level > 0)]


def stratified_sample(stats, rng):
    counts = stats.first_category_id.value_counts()
    alloc = np.maximum(MIN_PER_CAT, np.round(N_TARGET * counts / counts.sum())).astype(int)
    alloc = np.minimum(alloc, counts)
    parts = [stats[stats.first_category_id == c].sample(n=int(k), random_state=rng.integers(1 << 31))
             for c, k in alloc.items()]
    return pd.concat(parts, ignore_index=True)


def category_elasticity(df):
    panel = prepare_panel(df)
    rows = []
    for cat, g in panel.groupby("first_category_id"):
        if len(g) < MIN_CAT_OBS_IV:
            rows.append(dict(category=cat, n=len(g), eps_iv=np.nan, se_iv=np.nan))
            continue
        r = estimate_fe_iv(g)
        rows.append(dict(category=cat, n=r["n"], eps_iv=r["eps"], se_iv=r["se"],
                         first_stage_r=r["first_stage_r"]))
    return pd.DataFrame(rows)


def shrink(tab):
    """Empirical-Bayes shrinkage toward the pooled IV estimate.

    tau^2 = max(0, n-weighted var of category estimates around pooled - mean se^2);
    w_c = tau^2 / (tau^2 + se_c^2). Categories with no estimate get the pooled value.
    """
    ok = tab.eps_iv.notna() & np.isfinite(tab.se_iv)
    e, s, n = tab.eps_iv[ok], tab.se_iv[ok], tab.n[ok]
    tau2 = max(0.0, float(np.average((e - POOLED_EPS) ** 2, weights=n) - np.average(s ** 2, weights=n)))
    w = pd.Series(0.0, index=tab.index)
    w[ok] = tau2 / (tau2 + tab.se_iv[ok] ** 2)
    eps = w * tab.eps_iv.fillna(POOLED_EPS) + (1 - w) * POOLED_EPS
    return tab.assign(weight=w, eps_shrunk=eps), tau2


def assign_archetypes(sample, rng):
    """Within each category, balanced random assignment of series to the four archetypes."""
    out = []
    for _, g in sample.groupby("first_category_id"):
        labels = np.resize(np.array(ARCHETYPES), len(g))
        out.append(g.assign(archetype=rng.permutation(labels)))
    return pd.concat(out, ignore_index=True)


def main():
    if not RAW.exists():
        raise SystemExit(f"missing {RAW.relative_to(ROOT)}; see README for the download step")
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    df = pd.read_parquet(RAW, columns=COLS)
    stats = series_stats(df)
    sample = stratified_sample(stats, rng)
    scale = TARGET_MEDIAN_Q0 / float(sample.level.median())
    cat_tab, tau2 = shrink(category_elasticity(df))
    cat_tab.to_csv(OUT / "category_elasticity.csv", index=False)
    eps_map = cat_tab.set_index("category").eps_shrunk
    prof = sample.assign(
        q0=sample.level * scale,
        dispersion=np.maximum(1.0, scale * sample["var"] / sample.level),
        eps=sample.first_category_id.map(eps_map).fillna(POOLED_EPS))
    prof = assign_archetypes(prof, rng).sort_values(["first_category_id", "store_id", "product_id"])
    prof.to_csv(OUT / "series_profiles.csv", index=False)
    meta = dict(seed=SEED, n_eligible=int(len(stats)), n_sampled=int(len(prof)),
                n_categories=int(prof.first_category_id.nunique()), scale=scale,
                stable_day_rule="stock_hour6_22_cnt==0 and 0.99<=discount<=1.01",
                min_stable_days=MIN_STABLE_DAYS, min_per_category=MIN_PER_CAT,
                tau2=tau2, pooled_eps=POOLED_EPS,
                q0_quantiles=prof.q0.quantile([.1, .25, .5, .75, .9]).round(3).to_dict(),
                dispersion_quantiles=prof.dispersion.quantile([.1, .25, .5, .75, .9]).round(3).to_dict(),
                eps_quantiles=prof.eps.quantile([.1, .25, .5, .75, .9]).round(3).to_dict())
    (OUT / "calibration.json").write_text(json.dumps(meta, indent=1, default=float))
    print(json.dumps(meta, indent=1, default=float))
    print(cat_tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

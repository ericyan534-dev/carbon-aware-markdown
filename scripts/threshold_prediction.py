"""Out-of-sample test of the elasticity prediction on store halves (Sec. V-C).

For each of 10 seeds the stores are split at random into halves A and B. Category IV
elasticities are estimated separately on each half and shrunk toward that half's pooled IV
estimate (categories with s.e. above 2 take the pooled value). Each calibrated profile
(outputs/profiles/series_profiles.csv) belongs to one store and so to one half:
  realized   the profit-maximizing policy's value at the SCC as % of the SCC-optimal value,
             solved with the elasticity of the profile's own half
  predicted  the stylized curve ratio(eps) of the profile's product type
             (outputs/threshold/theory_curves.csv) at the elasticity of the other half
The AUC measures how well the prediction ranks the profiles whose realized value falls below
95% of the best; both directions (A->B, B->A) of every seed give 20 splits.

Writes outputs/threshold/store_splits.csv and prediction_summary.json.
Requires data/raw/train.parquet, scripts/profiles_calibrate.py and scripts/threshold_curves.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.threshold_curves import curve_predict, solve_profiles  # noqa: E402
from tblca.elasticity import estimate_fe_iv, prepare_panel  # noqa: E402

RAW = ROOT / "data" / "raw" / "train.parquet"
PROFILES = ROOT / "outputs" / "profiles" / "series_profiles.csv"
CURVES = ROOT / "outputs" / "threshold" / "theory_curves.csv"
OUT = ROOT / "outputs" / "threshold"
COLS = ["store_id", "product_id", "first_category_id", "dt", "sale_amount", "discount",
        "stock_hour6_22_cnt", "holiday_flag", "activity_flag", "avg_temperature", "precpt"]
SEEDS = (11, 22, 33, 44, 55, 66, 77, 88, 99, 110)
MIN_CAT_OBS_IV = 1000
MAX_SE = 2.0
THRESHOLD = 95.0


def category_eps(df: pd.DataFrame) -> pd.DataFrame:
    """Per-category FE-IV elasticity, empirical-Bayes shrunk toward this sample's pooled IV."""
    panel = prepare_panel(df)
    pooled = estimate_fe_iv(panel)["eps"]
    rows = []
    for cat, g in panel.groupby("first_category_id"):
        est = dict(category=cat, n=len(g), eps_iv=np.nan, se_iv=np.nan)
        if len(g) >= MIN_CAT_OBS_IV:
            try:
                r = estimate_fe_iv(g)
                est.update(n=r["n"], eps_iv=r["eps"], se_iv=r["se"])
            except (np.linalg.LinAlgError, ValueError) as exc:
                print(f"category {cat}: IV failed ({exc}); using the pooled estimate", file=sys.stderr)
        rows.append(est)
    tab = pd.DataFrame(rows)
    ok = tab.eps_iv.notna() & np.isfinite(tab.se_iv) & (tab.se_iv <= MAX_SE)
    e, s, n = tab.eps_iv[ok], tab.se_iv[ok], tab.n[ok]
    tau2 = max(0.0, float(np.average((e - pooled) ** 2, weights=n) - np.average(s ** 2, weights=n)))
    w = pd.Series(0.0, index=tab.index)
    w[ok] = tau2 / (tau2 + tab.se_iv[ok] ** 2)
    eps = w * tab.eps_iv.where(ok, pooled) + (1 - w) * pooled
    return tab.assign(pooled_eps=pooled, eps=eps)


def store_split(df: pd.DataFrame, prof: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Both test directions of one random store halving: realized eps and training eps."""
    stores = np.sort(df.store_id.unique())
    a_set = set(np.random.default_rng(seed).choice(stores, size=len(stores) // 2,
                                                   replace=False).tolist())
    in_a = df.store_id.isin(a_set)
    tabs = {"A": category_eps(df[in_a]), "B": category_eps(df[~in_a])}
    prof_a = prof.store_id.isin(a_set)
    out = []
    for train, test in (("A", "B"), ("B", "A")):
        t = prof[prof_a if test == "A" else ~prof_a]
        eps_test = tabs[test].set_index("category").eps
        eps_train = tabs[train].set_index("category").eps
        out.append(t.assign(seed=seed, direction=f"{train}->{test}",
                            eps=t.first_category_id.map(eps_test),
                            eps_train=t.first_category_id.map(eps_train).fillna(
                                tabs[train].pooled_eps.iloc[0])))
    return pd.concat(out, ignore_index=True)


def auc(score, label) -> float:
    """ROC AUC via the Mann-Whitney statistic (ties count one half)."""
    s, y = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def score(test: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, direction), g in test.groupby(["seed", "direction"]):
        pred = curve_predict(curves, g.archetype.to_numpy(), g.eps_train.to_numpy())
        below = g.ratio.to_numpy() < THRESHOLD
        rows.append(dict(seed=seed, direction=direction, n=len(g), n_below=int(below.sum()),
                         auc=auc(-pred, below)))
    return pd.DataFrame(rows)


def main():
    for f in (RAW, PROFILES, CURVES):
        if not f.exists():
            raise SystemExit(f"missing {f.relative_to(ROOT)}; see README for the script to run")
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(RAW, columns=COLS)
    prof = pd.read_csv(PROFILES)[["store_id", "product_id", "first_category_id", "archetype", "q0"]]
    test = pd.concat([store_split(df, prof, s) for s in SEEDS], ignore_index=True)
    key = ["archetype", "q0", "eps"]
    solved = solve_profiles(test[key].drop_duplicates())       # identical jobs solved once
    test = test.merge(solved[key + ["ratio"]], on=key, how="left")
    curves = pd.read_csv(CURVES)
    splits = score(test, curves)
    splits.to_csv(OUT / "store_splits.csv", index=False)
    summary = {"n_splits": len(splits), "threshold_pct": THRESHOLD,
               "auc_mean": round(float(splits.auc.mean()), 4),
               "auc_min": round(float(splits.auc.min()), 4),
               "auc_max": round(float(splits.auc.max()), 4),
               "profiles_below_threshold_mean": round(float(splits.n_below.mean()), 1)}
    (OUT / "prediction_summary.json").write_text(json.dumps(summary, indent=1))
    print(splits.round(3).to_string(index=False))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

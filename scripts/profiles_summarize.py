"""Summarise profile shares, overstatement and timings (outputs/profiles); run after profiles_run.py.

Writes profile_distributions.csv (Sec. V-D/V-F shares), profile_by_eps_band.csv, timing_stages.csv,
timing_extrapolation.json and headlines.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tblca.params import SCC_USD_PER_KG  # noqa: E402
from tblca.scenario import build  # noqa: E402

OUT = ROOT / "outputs" / "profiles"
EPS_CUT = 1.5
CHAIN_SKUS = 50_000

SHARES = {  # name: (numerator column, denominator column)
    "rule_profit_share_of_opt_profit": ("rule_profit_profit", "profit_profit"),
    "rule_profit_share_of_scc_opt_value": ("rule_profit_scc_value", "scc_scc_value"),
    "profitopt_share_of_scc_opt_value": ("profit_scc_value", "scc_scc_value"),
    "wastemin_share_of_scc_opt_value": ("waste_scc_value", "scc_scc_value"),
    "static_share_of_scc_opt_value": ("static_scc_value", "scc_scc_value"),
    "rule_scc_share_of_scc_opt_value": ("rule_scc_scc_value", "scc_scc_value"),
    "profitopt_share_of_scc_opt_carbon": ("profit_carbon_kg", "scc_carbon_kg"),
}


def add_shares(df):
    out = df.copy()
    for tag in ("", "nb_"):
        for k, (a, b) in SHARES.items():
            out[f"{tag}{k}"] = 100 * df[tag + a] / df[tag + b]
    out["eps_band"] = np.where(out.eps < EPS_CUT, "eps<1.5", "eps>=1.5")
    return out


def dist(s):
    s = s.astype(float)
    return dict(n=len(s), min=s.min(), p5=s.quantile(.05), p25=s.quantile(.25), median=s.median(),
                p75=s.quantile(.75), p95=s.quantile(.95), max=s.max(),
                ge99_pct=100 * (s >= 99).mean(), ge95_pct=100 * (s >= 95).mean())


def table(df, metrics, by=None):
    rows = []
    groups = [("ALL", df)] if by is None else list(df.groupby(by))
    for key, g in groups:
        for m in metrics:
            rows.append(dict(group=key, metric=m, **dist(g[m])))
    return pd.DataFrame(rows)


def timing(df, prefix):
    cols = [c for c in df.columns if c.startswith("t_")]
    return pd.DataFrame([dict(source=prefix, stage=c, median_ms=1e3 * df[c].median(),
                              p95_ms=1e3 * df[c].quantile(.95), max_ms=1e3 * df[c].max(),
                              mean_ms=1e3 * df[c].mean(), sum_s=df[c].sum()) for c in cols])


def main():
    df = add_shares(pd.read_csv(OUT / "profile_results.csv"))
    df.to_csv(OUT / "profile_results_with_shares.csv", index=False)
    pois = list(SHARES)
    nb = [f"nb_{k}" for k in SHARES]
    overall = pd.concat([table(df, pois), table(df, nb)])
    overall.to_csv(OUT / "profile_distributions.csv", index=False)
    by_band = table(df, pois + nb, "eps_band")
    by_band.to_csv(OUT / "profile_by_eps_band.csv", index=False)
    by_cat = table(df, pois, "first_category_id")
    cat_eps = df.groupby("first_category_id").eps.first()
    by_cat["eps"] = by_cat.group.map(cat_eps)
    by_cat.to_csv(OUT / "profile_by_category.csv", index=False)
    by_arch = table(df, pois, "archetype")
    by_arch.to_csv(OUT / "profile_by_archetype.csv", index=False)
    pd.set_option("display.width", 250)
    print(overall.round(2).to_string(index=False))
    print(by_band.round(2).to_string(index=False))
    print(by_arch.round(2).to_string(index=False))
    # profiles below share thresholds
    keep = ["store_id", "product_id", "first_category_id", "archetype", "q0", "eps", "dispersion", "x0"]
    fails = {
        "rule_profit_lt99": df[df.rule_profit_share_of_opt_profit < 99],
        "profitopt_scc_lt99": df[df.profitopt_share_of_scc_opt_value < 99],
        "profitopt_scc_lt95": df[df.profitopt_share_of_scc_opt_value < 95],
        "rule_profit_scc_lt95": df[df.rule_profit_share_of_scc_opt_value < 95],
        "wastemin_ge_95": df[df.wastemin_share_of_scc_opt_value >= 95],
        "nonpositive_scc_opt": df[df.scc_scc_value <= 0],
    }
    fsum = {}
    for k, g in fails.items():
        g[keep + pois].to_csv(OUT / f"profile_subset_{k}.csv", index=False)
        fsum[k] = dict(n=len(g), eps_lt_1p5=int((g.eps < EPS_CUT).sum()),
                       x0_median=float(g.x0.median()) if len(g) else None,
                       q0_median=float(g.q0.median()) if len(g) else None,
                       categories=g.first_category_id.value_counts().head(8).to_dict(),
                       archetypes=g.archetype.value_counts().to_dict())
    print(json.dumps(fsum, indent=1, default=str))
    corr = df[["eps", "q0", "x0", "dispersion"] + pois].corr(method="spearman").loc[
        ["eps", "q0", "x0", "dispersion"], pois]
    print(corr.round(2).to_string())
    # timings
    tt = [timing(df, "profile_parallel_full")]
    runs = {}
    for m in ("all", "bench_serial", "bench_parallel", "tracemalloc"):
        f = OUT / f"run_{m}.json"
        if f.exists():
            runs[m] = json.loads(f.read_text())
    for m in ("bench_serial", "bench_parallel"):
        f = OUT / f"timing_{m}.csv"
        if f.exists():
            tt.append(timing(pd.read_csv(f), m))
    tt = pd.concat(tt)
    tt.to_csv(OUT / "timing_stages.csv", index=False)
    print(tt.round(3).to_string(index=False))
    n = len(df)
    extrap = {m: dict(wall_s=r["wall_s"], workers=r["workers"], per_sku_wall_ms=1e3 * r["wall_s"] / n,
                      chain_50k_wall_s=r["wall_s"] * CHAIN_SKUS / n)
              for m, r in runs.items() if "wall_s" in r}
    (OUT / "timing_extrapolation.json").write_text(json.dumps(dict(runs=runs, extrap=extrap,
                                                                fails=fsum), indent=1, default=str))
    print(json.dumps(extrap, indent=1))
    head = headlines(df, OUT / "timing_bench_serial.csv")
    (OUT / "headlines.json").write_text(json.dumps(head, indent=1))
    print(json.dumps(head, indent=1))


def static_overstatement(df):
    """Overstatement of cleared-unit crediting under the static rule, per profile (%)."""
    per_unit = {a: (lambda p: p.unit_kg * (p.ef_prod + p.ef_eol - p.ops_kg))(
        build(a, "Hong Kong", carbon_price=SCC_USD_PER_KG)) for a in df.archetype.unique()}
    return 100 * (df.static_sales * df.archetype.map(per_unit) / df.static_carbon_kg - 1)


def headlines(df, serial_csv):
    over = static_overstatement(df)
    high = df[df.eps >= EPS_CUT]
    out = {
        "n_profiles": len(df),
        "rule (profit) share of optimal profit, min / median (%)":
            [round(df.rule_profit_share_of_opt_profit.min(), 2),
             round(df.rule_profit_share_of_opt_profit.median(), 2)],
        "rule (carbon-priced) share of best value at SCC, min (%)":
            round(df.rule_scc_share_of_scc_opt_value.min(), 2),
        "profit-max share of best value at SCC, median (%)":
            round(df.profitopt_share_of_scc_opt_value.median(), 2),
        "profit-max share of best value at SCC, chain total (%)":
            round(100 * df.profit_scc_value.sum() / df.scc_scc_value.sum(), 2),
        "profit-max share of best value at SCC, min over eps >= 1.5 (%)":
            round(high.profitopt_share_of_scc_opt_value.min(), 2),
        "static-rule overstatement, median (%)": round(float(over.median()), 2),
        "static-rule overstatement positive (share of profiles)": float((over > 0).mean()),
    }
    if serial_csv.exists():
        out["single-core pipeline time per profile, median (ms)"] = round(
            1e3 * float(pd.read_csv(serial_csv).t_core.median()), 2)
    return out


if __name__ == "__main__":
    main()

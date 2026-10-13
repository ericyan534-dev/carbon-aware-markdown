"""Estimate discount elasticities on the FreshRetailNet-50K train split (pooled and by category).

Writes outputs/elasticity/{elasticity_pooled.json, elasticity_by_category.csv}.
Requires data/raw/train.parquet (see README).
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tblca.elasticity import estimate_fe_iv, estimate_fe_ols, prepare_panel  # noqa: E402

RAW = ROOT / "data" / "raw" / "train.parquet"
OUT = ROOT / "outputs" / "elasticity"
COLS = ["store_id", "product_id", "first_category_id", "dt", "sale_amount", "discount",
        "stock_hour6_22_cnt", "holiday_flag", "activity_flag", "avg_temperature", "precpt"]
MIN_CATEGORY_OBS = 5000


def load_panel():
    if not RAW.exists():
        raise SystemExit(f"missing {RAW.relative_to(ROOT)}; see README for the download step")
    return prepare_panel(pd.read_parquet(RAW, columns=COLS))


def main():
    panel = load_panel()
    pooled = {"ols": estimate_fe_ols(panel), "iv": estimate_fe_iv(panel)}
    print(json.dumps(pooled, indent=1))
    rows = []
    for cat, g in panel.groupby("first_category_id"):
        if len(g) < MIN_CATEGORY_OBS:
            continue
        o, i = estimate_fe_ols(g), estimate_fe_iv(g)
        rows.append(dict(category=cat, n=o["n"], eps_ols=o["eps"], se_ols=o["se"],
                         eps_iv=i["eps"], se_iv=i["se"], first_stage_r=i["first_stage_r"]))
    table = pd.DataFrame(rows).sort_values("n", ascending=False)
    print(table.round(3).to_string(index=False))
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "elasticity_by_category.csv", index=False)
    (OUT / "elasticity_pooled.json").write_text(json.dumps(pooled, indent=1))


if __name__ == "__main__":
    main()

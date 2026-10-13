"""Robustness of the IV elasticity to common demand shocks and to the sample restriction.

Designs (all with store-product fixed effects, standard errors clustered by store):
  baseline                      day FE, leave-one-store-out instrument
  city x day FE                 absorbs local demand shocks
  subcategory x day FE          absorbs product-group demand shocks (third_category_id)
  leave-city-out instrument     instrument uses only prices in other cities (city x day FE)
  including stockout days       keeps days with a stockout between 06:00 and 22:00

Writes outputs/elasticity/iv_robustness.json. Requires data/raw/train.parquet (see README).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tblca.elasticity import MIN_PRICE_RATIO, _fit, _hausman_instrument  # noqa: E402

RAW = ROOT / "data" / "raw" / "train.parquet"
OUT = ROOT / "outputs" / "elasticity" / "iv_robustness.json"
COLS = ["city_id", "store_id", "first_category_id", "third_category_id", "product_id", "dt",
        "sale_amount", "discount", "stock_hour6_22_cnt", "holiday_flag", "activity_flag",
        "avg_temperature", "precpt"]


def leave_city_out(df: pd.DataFrame) -> pd.Series:
    """Mean log price of the same product on the same day in other cities."""
    g = df.groupby(["product_id", "dt"])["log_p"]
    tot, cnt = g.transform("sum"), g.transform("count")
    c = df.groupby(["product_id", "dt", "city_id"])["log_p"]
    ctot, ccnt = c.transform("sum"), c.transform("count")
    return (tot - ctot) / (cnt - ccnt).where(cnt > ccnt)


def panel(raw: pd.DataFrame, keep: pd.Series) -> pd.DataFrame:
    out = raw.loc[keep].copy()
    out["log_q"] = np.log(out["sale_amount"])
    out["log_p"] = np.log(out["discount"])
    out["series"] = out["store_id"].astype(str) + "_" + out["product_id"].astype(str)
    out["city_day"] = out["city_id"].astype(str) + "_" + out["dt"].astype(str)
    out["cat_day"] = out["third_category_id"].astype(str) + "_" + out["dt"].astype(str)
    return out.assign(z=_hausman_instrument(out))


def fit(frame: pd.DataFrame, fe: tuple, instrument: str = "z") -> dict:
    if instrument != "z":
        frame = frame.drop(columns="z").rename(columns={instrument: "z"})
    return _fit(frame, iv=True, fe=fe)


def main():
    if not RAW.exists():
        raise SystemExit(f"missing {RAW.relative_to(ROOT)}; see README for the download step")
    raw = pd.read_parquet(RAW, columns=COLS)
    valid = (raw["discount"] > MIN_PRICE_RATIO) & (raw["discount"] <= 1.0) & (raw["sale_amount"] > 0)
    base = panel(raw, valid & (raw["stock_hour6_22_cnt"] == 0))
    base["z_city"] = leave_city_out(base)
    res = {
        "baseline (series + day FE)": fit(base, ("series", "dt")),
        "series + city x day FE": fit(base, ("series", "city_day")),
        "series + subcategory x day FE": fit(base, ("series", "cat_day")),
        "leave-city-out instrument, series + city x day FE":
            fit(base, ("series", "city_day"), instrument="z_city"),
        "including stockout days (series + day FE)": fit(panel(raw, valid), ("series", "dt")),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()

"""Discount elasticity estimation on store x SKU x day panels (FreshRetailNet schema).

log q_{sit} = a_{si} + b_{t} - eps * log p_{sit} + controls + u_{sit}, with p the price
ratio (discount column). Fixed effects are removed by alternating demeaning.
FE-IV instruments log p with the leave-one-store-out mean log price of the same
product on the same day (Hausman instrument).
"""
import numpy as np
import pandas as pd

CONTROLS = ("holiday_flag", "activity_flag", "avg_temperature", "precpt")
MIN_PRICE_RATIO = 0.3


def prepare_panel(df: pd.DataFrame) -> pd.DataFrame:
    keep = ((df["stock_hour6_22_cnt"] == 0) & (df["sale_amount"] > 0)
            & (df["discount"] > MIN_PRICE_RATIO) & (df["discount"] <= 1.0))
    out = df.loc[keep].copy()
    out["log_q"] = np.log(out["sale_amount"])
    out["log_p"] = np.log(out["discount"])
    out["series"] = out["store_id"].astype(str) + "_" + out["product_id"].astype(str)
    return out.assign(z=_hausman_instrument(out))


def _hausman_instrument(df):
    grp = df.groupby(["product_id", "dt"])["log_p"]
    total, count = grp.transform("sum"), grp.transform("count")
    loo = (total - df["log_p"]) / (count - 1).where(count > 1)
    return loo


def _demean(frame, cols, fe=("series", "dt"), iters=8):
    x = frame[list(cols)].astype(float).copy()
    for _ in range(iters):
        for f in fe:
            x = x - x.groupby(frame[f]).transform("mean")
    return x


def _controls(frame):
    return [c for c in CONTROLS if c in frame and frame[c].std() > 0]


def _fit(frame, iv, fe=("series", "dt")):
    ctrl = _controls(frame)
    cols = ["log_q", "log_p", *ctrl] + (["z"] if iv else [])
    sub = frame.dropna(subset=cols)
    m = _demean(sub, cols, fe=fe)
    ctrl = [c for c in ctrl if m[c].std() > 1e-8]   # drop controls absorbed by the FEs
    y = m["log_q"].to_numpy()
    X = m[["log_p", *ctrl]].to_numpy()
    if iv:
        Z = m[["z", *ctrl]].to_numpy()
        X_hat = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
        beta = np.linalg.lstsq(X_hat, y, rcond=None)[0]
        first_stage_r = np.corrcoef(m["z"], m["log_p"])[0, 1]
    else:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        first_stage_r = None
    X_used = X_hat if iv else X
    resid = y - X @ beta
    se = _cluster_se(X_used, resid, sub["store_id"].to_numpy())
    return {"eps": float(-beta[0]), "se": float(se[0]), "n": int(len(sub)),
            "resid_sd": float(resid.std()), "first_stage_r": first_stage_r}


def _cluster_se(X, resid, clusters):
    """Cluster-robust (CR0) standard errors, clustered by store."""
    bread = np.linalg.pinv(X.T @ X)
    scores = pd.DataFrame(X * resid[:, None]).groupby(clusters).sum().to_numpy()
    meat = scores.T @ scores
    return np.sqrt(np.diag(bread @ meat @ bread))


def _dispatch(df, iv, by):
    if by is None:
        return _fit(df, iv)
    return {k: _fit(g, iv) for k, g in df.groupby(by)}


def estimate_fe_ols(df, by=None):
    return _dispatch(df, iv=False, by=by)


def estimate_fe_iv(df, by=None):
    return _dispatch(df, iv=True, by=by)

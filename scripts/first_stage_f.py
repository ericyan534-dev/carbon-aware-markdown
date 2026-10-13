"""Cluster-robust (by store) first-stage F of the leave-one-store-out price instrument.

With a single instrument F = t^2. Writes outputs/elasticity/first_stage_f.json.
Requires data/raw/train.parquet (see README).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.estimate_elasticity import OUT, load_panel  # noqa: E402
from tblca.elasticity import _cluster_se, _controls, _demean  # noqa: E402


def main():
    panel = load_panel()
    ctrl = _controls(panel)
    use = ["log_p", "z", *ctrl]
    sub = panel.dropna(subset=use)
    m = _demean(sub, use)
    ctrl = [c for c in ctrl if m[c].std() > 1e-8]
    Z = m[["z", *ctrl]].to_numpy()
    y = m["log_p"].to_numpy()
    b = np.linalg.lstsq(Z, y, rcond=None)[0]
    se = _cluster_se(Z, y - Z @ b, sub["store_id"].to_numpy())
    out = {"coef": float(b[0]), "se": float(se[0]), "F": float((b[0] / se[0]) ** 2),
           "n": int(len(sub)), "clusters": int(sub["store_id"].nunique())}
    print(json.dumps(out, indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "first_stage_f.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

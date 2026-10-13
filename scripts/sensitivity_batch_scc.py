"""Profit-max share of the best value at the SCC over batch size, SCC level, kappa and eps (Sec. V-E).

Grid: 4 products x kappa {0.4, 0.67, 0.9} x eps {1.5, 1.94, 3.0} x SCC {120, 190, 340 USD/t}
x batch {1.25, 2, 3} times full-price demand (Hong Kong). Writes outputs/sensitivity/.
"""
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tblca.markdown import solve  # noqa: E402
from tblca.scenario import build  # noqa: E402

ARCHES = ("dairy", "bakery", "produce", "poultry")
KAPPAS = (0.4, 0.67, 0.9)
EPSS = (1.5, 1.94, 3.0)
SCCS = (0.120, 0.190, 0.340)
RATIOS = (1.25, 2.0, 3.0)
OUT = Path(__file__).resolve().parents[1] / "outputs" / "sensitivity"


def shares(arch: str, kappa: float, eps: float, mu: float, ratio: float) -> dict:
    over = dict(kappa=kappa, eps=eps, stock_ratio=ratio)
    prof = solve(build(arch, "Hong Kong", objective="profit", **over)).metrics
    best = solve(build(arch, "Hong Kong", carbon_price=mu, **over)).metrics
    value = lambda m: m["profit"] + mu * m["carbon_kg"]  # noqa: E731
    return dict(arch=arch, kappa=kappa, eps=eps, scc=mu, ratio=ratio,
                value_share=100 * value(prof) / value(best),
                carbon_share=100 * prof["carbon_kg"] / best["carbon_kg"])


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [shares(*c) for c in itertools.product(ARCHES, KAPPAS, EPSS, SCCS, RATIOS)]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sensitivity.csv", index=False)
    summ = {k: df.groupby(k)[["value_share", "carbon_share"]].min().round(2).to_dict()
            for k in ("kappa", "eps", "scc", "ratio", "arch")}
    summ["overall_min"] = df[["value_share", "carbon_share"]].min().round(2).to_dict()
    summ["share_ge99"] = float((df.value_share >= 99).mean())
    summ["share_ge98"] = float((df.value_share >= 98).mean())
    est = df[(df.eps == 1.94) & (df.kappa == 0.67)]
    summ["at_estimate_batch_range_at_scc_min"] = round(float(est[est.scc == 0.190].value_share.min()), 2)
    summ["at_estimate_scc_range_min"] = round(float(est.value_share.min()), 2)
    (OUT / "sensitivity_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))

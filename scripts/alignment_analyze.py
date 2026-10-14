"""Alignment-index statistics (Sec. IV-C, V-C); run after scripts/alignment_compute.py.

Writes outputs/alignment/summary.json with
  * the batch-average alignment index at the SCC per case and the carbon price at which it
    reaches one (theta_bar is linear in the carbon price);
  * Spearman correlations and the Hill fit of the universal curve (Fig. 4c);
  * monotonicity counts: stage problem over 2,080 states and full DP over 145,600 comparisons.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tblca.alignment import (  # noqa: E402
    carbon_nondecreasing_on_efficient, stage_carbon_nondecreasing, theta_bar,
)
from tblca.params import ARCHETYPES, REGIONS, SCC_USD_PER_KG  # noqa: E402
from tblca.scenario import build  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs" / "alignment"


def hill(t, t50, k):
    return t ** k / (t50 ** k + t ** k)


def theta_stats(cells):
    """theta_bar at the SCC and the carbon price (USD/t) at which theta_bar = 1."""
    mu_one = {}
    for a in ARCHETYPES:
        vals = [1000.0 / theta_bar(build(a, g), 1.0) for g in REGIONS]
        mu_one[a] = [round(min(vals), 1), round(max(vals), 1)]
    return {"theta_bar_scc_min": float(cells.theta_bar.min()),
            "theta_bar_scc_max": float(cells.theta_bar.max()),
            "theta_bar_scc_by_archetype": cells.groupby("archetype").theta_bar.mean().round(3).to_dict(),
            "carbon_price_usd_per_t_at_theta_bar_1": mu_one,
            "scc_usd_per_t": 1000 * SCC_USD_PER_KG}


def collapse_stats(cells, sweep):
    sw = sweep[sweep.mu > 0]
    out = {"cells_n": len(cells)}
    for k in ("gap_pct", "gap_kg", "gap_closed", "profit_loss_pct"):
        out[f"cells_spearman_{k}"] = float(spearmanr(cells.theta_bar, cells[k])[0])
        out[f"cells_pearson_{k}"] = float(pearsonr(cells.theta_bar, cells[k])[0])
    for k in ("gap_pct", "gap_closed"):
        out[f"sweep_spearman_{k}"] = float(spearmanr(sw.theta_bar, sw[k])[0])
    (t50, k), _ = curve_fit(hill, sw.theta_bar, sw.gap_closed, p0=(1.0, 1.5), maxfev=20000)
    pred = hill(sw.theta_bar, t50, k)
    ss = ((sw.gap_closed - pred) ** 2).sum() / ((sw.gap_closed - sw.gap_closed.mean()) ** 2).sum()
    out.update(sweep_n=len(sw), hill_t50=float(t50), hill_k=float(k), hill_r2=float(1 - ss),
               hill_rmse=float(np.sqrt(((sw.gap_closed - pred) ** 2).mean())))
    return out


def monotone_stats():
    mono = pd.read_csv(OUT / "s1_monotone.csv")
    out = dict(dp_pairs=int(mono.dp_pairs.sum()), dp_violations=int(mono.dp_violations.sum()),
               myopic_states=int(mono.states.sum()),
               myopic_violations=int(mono.myopic_states_violating.sum()))
    glob = eff = tot = 0
    for a in ARCHETYPES:
        for g in REGIONS:
            p = build(a, g)
            for r in range(1, p.horizon + 1):
                for x in range(1, p.x0 + 1):
                    tot += 1
                    glob += stage_carbon_nondecreasing(p, x, r)
                    eff += carbon_nondecreasing_on_efficient(p, x, r)
    out.update(stage_states=tot, id_global_holds=glob, id_efficient_holds=eff)
    return out


def main():
    cells = pd.read_csv(OUT / "s1_cells.csv")
    sweep = pd.read_csv(OUT / "s1_sweep.csv")
    stats = {**theta_stats(cells), **collapse_stats(cells, sweep), **monotone_stats()}
    (OUT / "summary.json").write_text(json.dumps(stats, indent=1, default=float))
    print(json.dumps(stats, indent=1, default=float))


if __name__ == "__main__":
    main()

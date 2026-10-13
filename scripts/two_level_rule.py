"""Two-level stock rule vs. the optimal policy in the 20 product-region cases (Sec. V-D).

For each case the best per-day two-level threshold rule is found by exhaustive search, once
for the profit goal and once for the carbon-priced goal. Writes outputs/rule/{rules.csv,
summary.json}.
"""
import json
import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tblca.fast_dp import build_tables, fast_evaluate, fast_solve  # noqa: E402
from tblca.markdown import fixed_policy  # noqa: E402
from tblca.params import ARCHETYPES, REGIONS, SCC_USD_PER_KG  # noqa: E402
from tblca.scenario import build  # noqa: E402
from tblca.simple_rules import best_threshold_rule, threshold_policy  # noqa: E402

OUT = ROOT / "outputs" / "rule"


def static_rule(r):
    return 0.35 if r == 1 else 0.0


def fit(p):
    t = build_tables(p)
    _, opt_pol = fast_solve(p, t)
    opt = fast_evaluate(p, opt_pol, t)
    rule = best_threshold_rule(p, t)
    m = fast_evaluate(p, threshold_policy(p, rule.lo, rule.hi, rule.thresholds), t)
    static = fast_evaluate(p, fixed_policy(p, static_rule), t)
    return opt, rule, m, static


def case(args):
    arch, region = args
    p = build(arch, region, carbon_price=SCC_USD_PER_KG)
    row = dict(archetype=arch, region=region, horizon=p.horizon, x0=p.x0)
    for goal, q in (("profit", replace(p, objective="profit")), ("scc", p)):
        opt, rule, m, static = fit(q)
        row.update({f"{goal}_rule_lo": rule.lo, f"{goal}_rule_hi": rule.hi,
                    f"{goal}_rule_thresholds": "/".join(str(c) for c in rule.thresholds),
                    f"{goal}_rule_profit_pct_of_opt": 100 * m["profit"] / opt["profit"],
                    f"{goal}_rule_value_pct_of_opt": 100 * m["value"] / opt["value"],
                    f"{goal}_rule_gap_closed_pct":
                        100 * (m["value"] - static["value"]) / (opt["value"] - static["value"])})
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool() as pool:
        df = pd.DataFrame(pool.map(case, [(a, g) for a in ARCHETYPES for g in REGIONS]))
    df.to_csv(OUT / "rules.csv", index=False)
    pct = df.filter(like="_pct")
    summary = pct.agg(["min", "median", "max"]).round(2).to_dict()
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(pct.agg(["min", "median", "max"]).T.round(2).to_string())


if __name__ == "__main__":
    main()

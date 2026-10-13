"""Decoupled switching shares, split decay rates and the accounting ablation (Sec. III-B, V-E).

* metrics:    profit-max share of the best value at the SCC and related ratios, 20 cases
* decoupled:  the switching share in the retailer's margin (kappa_R) and in displaced production
              (kappa_C) vary independently over {0.15, 0.33, 0.5, 0.67, 0.85, 1}: 36 pairs
* split_lambda: separate decay rates for freshness (substitutability) and household consumption,
              each in {0.75, 1.5, 3} per day
* ablation:   static last-day 35% rule; overstatement of cleared-unit crediting with the full
              ledger and with displacement switched off (carbon-side switching share = 1, so
              only the freshness and household-consumption corrections remain); the difference
              is the part added by displacement (Sec. III-B)

Writes outputs/switching/switching.json and outputs/switching/ablation_by_case.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tblca.fast_dp import build_tables, fast_evaluate, fast_solve  # noqa: E402
from tblca.markdown import unit_economics  # noqa: E402
from tblca.params import ARCHETYPES, REGIONS, SCC_USD_PER_KG as MU  # noqa: E402
from tblca.scenario import build, fixed_policy, with_objective  # noqa: E402
from tblca.substitution import freshness_gate, rebound_gate, spoilage_gate  # noqa: E402

OUT = ROOT / "outputs" / "switching" / "switching.json"
ABLATION_CSV = OUT.parent / "ablation_by_case.csv"
CASES = [(a, g) for a in ARCHETYPES for g in REGIONS]
KAPPA_GRID = (0.15, 0.33, 0.5, 0.67, 0.85, 1.0)
LAMBDA_GRID = (0.75, 1.5, 3.0)


def carbon_unit_fn(kappa_c=None, lam_fresh=None, lam_home=None):
    """Per-unit economics with the carbon side's switching share and decay rates overridden."""
    def unit(p, r, d):
        u = unit_economics(p, r, d)
        g = float(rebound_gate(d, p.eps, p.kappa if kappa_c is None else kappa_c))
        fresh = float(freshness_gate(r, p.lam_f if lam_fresh is None else lam_fresh))
        home = float(spoilage_gate(r, p.lam_f if lam_home is None else lam_home, p.p_base))
        carbon = p.unit_kg * (g * fresh * home * p.ef_prod + home * p.ef_eol - p.ops_kg)
        return {**u, "carbon_kg": carbon}
    return unit


def scorecard(p, unit_fn=unit_economics):
    def solve(obj):
        q = with_objective(p, obj, MU if obj == "weighted" else 0.0)
        return fast_solve(q, build_tables(q, unit_fn))[1]
    pols = {o: solve(o) for o in ("profit", "weighted", "waste")}
    pols["static"] = fixed_policy(p, lambda r: 0.35 if r == 1 else 0.0)
    q = with_objective(p, "weighted", MU)
    tables = build_tables(q, unit_fn)
    ev = {k: fast_evaluate(q, pol, tables) for k, pol in pols.items()}
    w = {k: m["profit"] + MU * m["carbon_kg"] for k, m in ev.items()}
    naive = ev["static"]["sales"] * p.unit_kg * (p.ef_prod + p.ef_eol - p.ops_kg)
    return {"ratio": 100 * w["profit"] / w["weighted"],
            "regret_pct_profit": 100 * (w["weighted"] - w["profit"]) / ev["profit"]["profit"],
            "closure_vs_static": 100 * (w["profit"] - w["static"]) / (w["weighted"] - w["static"]),
            "waste_ratio": 100 * w["waste"] / w["weighted"],
            "carbon_share": 100 * ev["profit"]["carbon_kg"] / ev["weighted"]["carbon_kg"],
            "overstatement": 100 * (naive / ev["static"]["carbon_kg"] - 1)}


def summarize(vals):
    a = np.asarray(vals, dtype=float)
    return {"min": round(float(a.min()), 2), "median": round(float(np.median(a)), 2),
            "max": round(float(a.max()), 2)}


def exp_metrics():
    rows = [scorecard(build(a, g)) for a, g in CASES]
    return {m: summarize([r[m] for r in rows]) for m in rows[0]}


def exp_decoupled():
    out = {}
    for kr in KAPPA_GRID:
        for kc in KAPPA_GRID:
            rows = [scorecard(build(a, g, kappa=kr), carbon_unit_fn(kappa_c=kc)) for a, g in CASES]
            out[f"kR={kr}|kC={kc}"] = {m: summarize([r[m] for r in rows])
                                       for m in ("ratio", "closure_vs_static", "waste_ratio")}
    return out


def exp_split_lambda():
    out = {}
    for lf in LAMBDA_GRID:
        for lh in LAMBDA_GRID:
            fn = carbon_unit_fn(lam_fresh=lf, lam_home=lh)
            rows = [scorecard(build(a, g), fn) for a, g in CASES]
            out[f"lam_fresh={lf}|lam_home={lh}"] = {m: summarize([r[m] for r in rows])
                                                    for m in ("ratio", "overstatement")}
    return out


def static_overstatement(p, unit_fn=unit_economics):
    """Overstatement (%) of crediting every unit the static rule clears as avoided production."""
    pol = fixed_policy(p, lambda r: 0.35 if r == 1 else 0.0)
    ev = fast_evaluate(p, pol, build_tables(p, unit_fn))
    naive = ev["sales"] * p.unit_kg * (p.ef_prod + p.ef_eol - p.ops_kg)
    return 100 * (naive / ev["carbon_kg"] - 1)


def exp_ablation():
    rows = []
    for a, g in CASES:
        p = build(a, g)
        full = static_overstatement(p)
        gates = static_overstatement(p, carbon_unit_fn(kappa_c=1.0))
        rows.append(dict(archetype=a, region=g, full_pct=full, gates_only_pct=gates,
                         displacement_pp=full - gates, displacement_share_pct=100 * (full - gates) / full))
    tab = pd.DataFrame(rows)
    ABLATION_CSV.parent.mkdir(parents=True, exist_ok=True)
    tab.round(2).to_csv(ABLATION_CSV, index=False)
    out = {m: summarize(tab[m]) for m in ("full_pct", "gates_only_pct", "displacement_pp",
                                          "displacement_share_pct")}
    return {m: {**v, "mean": round(float(tab[m].mean()), 2)} for m, v in out.items()}


def main():
    res = {"metrics": exp_metrics(), "decoupled": exp_decoupled(), "split_lambda": exp_split_lambda(),
           "ablation": exp_ablation()}
    res["decoupled_min_ratio"] = min(v["ratio"]["min"] for v in res["decoupled"].values())
    res["split_lambda_min_ratio"] = min(v["ratio"]["min"] for v in res["split_lambda"].values())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    keys = ("metrics", "decoupled_min_ratio", "split_lambda_min_ratio", "ablation")
    print(json.dumps({k: res[k] for k in keys}, indent=1))


if __name__ == "__main__":
    main()

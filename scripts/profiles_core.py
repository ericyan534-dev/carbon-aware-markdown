"""Per-profile work for the 4,121 calibrated store-product profiles (Sec. V-D, V-F).

Each profile is solved as
    build(product_type, "Hong Kong", carbon_price=SCC_USD_PER_KG, q0=row.q0, eps=row.eps)
with tblca.fast_dp (same model and tie-breaking as tblca.markdown.solve). The two-level
threshold rule is the rule family and exhaustive search of
tblca.simple_rules.best_threshold_rule; the last two decision days are searched with a
prefix-sum identity instead of enumerating every threshold vector, which keeps the exact
search fast for large batches.
"""
import itertools
import resource
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import nbinom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tblca.fast_dp import StageTables, build_tables, fast_evaluate, fast_solve  # noqa: E402
from tblca.markdown import METRICS, MarkdownProblem, _reward, _terminal, demand_rate, unit_economics  # noqa: E402
from tblca.params import SCC_USD_PER_KG  # noqa: E402
from tblca.scenario import build  # noqa: E402
from tblca.simple_rules import ThresholdRule, threshold_policy  # noqa: E402

REGION = "Hong Kong"
STATIC_LAST_DAY = 0.35
MAX_FRONTIER = 5_000_000   # guard on materialised threshold combinations (H >= 4)


# ---------------------------------------------------------------- exact rule search
def _pair_search(p: MarkdownProblem, t: StageTables, j_lo: int, j_hi: int):
    """Best per-day thresholds for one (lo, hi) pair; same enumeration order as simple_rules."""
    n = p.x0 + 1
    x = np.arange(n)
    grid = np.arange(-1, n)                      # T in {-1, 0, ..., x0}
    V = _terminal(p).astype(float)[:, None]      # (X, N) frontier of value vectors
    combos = [()]
    for r in range(1, p.horizon - 1):            # days materialised explicitly
        ql = t.reward[r, j_lo][:, None] + t.trans[r, j_lo] @ V
        qh = t.reward[r, j_hi][:, None] + t.trans[r, j_hi] @ V
        if V.shape[1] * len(grid) > MAX_FRONTIER:
            raise MemoryError("threshold frontier too large")
        V = np.where(x[:, None, None] <= grid[None, None, :], ql[:, :, None], qh[:, :, None])
        V = V.reshape(n, -1)
        combos = [c + (int(g),) for c in combos for g in grid]
    H = p.horizon
    if H == 1:
        vals = np.stack([t.reward[1, j][p.x0] + t.trans[1, j, p.x0] @ V for j in (j_hi, j_lo)], -1)
        vals = vals.reshape(1, 2)
        k = int(vals.argmax())
        return float(vals.flat[k]), ((-1, p.x0)[k],)
    # day H-1: thresholds over grid, then day H binary (all hi: T=-1 / all lo: T=x0)
    r = H - 1
    ql = t.reward[r, j_lo][:, None] + t.trans[r, j_lo] @ V          # (X, N)
    qh = t.reward[r, j_hi][:, None] + t.trans[r, j_hi] @ V
    out = []
    for j in (j_hi, j_lo):                                           # c = -1 then c = x0
        w = t.trans[H, j, p.x0][:, None]                             # (X, 1)
        cl = np.vstack([np.zeros((1, V.shape[1])), np.cumsum(w * ql, 0)])   # (G, N)
        ch = np.vstack([np.zeros((1, V.shape[1])), np.cumsum(w * qh, 0)])
        out.append(t.reward[H, j, p.x0] + cl + (ch[-1:] - ch))       # (G, N)
    vals = np.stack(out, -1).transpose(1, 0, 2)                      # (N, G, 2)
    k = int(vals.argmax())
    i_n, i_g, i_c = np.unravel_index(k, vals.shape)
    ts = combos[i_n] + (int(grid[i_g]), (-1, p.x0)[i_c])
    return float(vals[i_n, i_g, i_c]), ts


def fast_threshold_rule(p: MarkdownProblem, t: StageTables) -> ThresholdRule:
    best = ThresholdRule(0.0, 0.0, (), -np.inf)
    for j_lo, j_hi in itertools.combinations(range(len(p.discounts)), 2):
        val, ts = _pair_search(p, t, j_lo, j_hi)
        if val > best.value + 1e-12:
            best = ThresholdRule(p.discounts[j_lo], p.discounts[j_hi], ts, val)
    return best


# ---------------------------------------------------------------- tables per objective
def tables_for(t: StageTables, objective: str, mu: float = SCC_USD_PER_KG) -> StageTables:
    if objective == "profit":
        rew = t.metrics["profit"]
    elif objective == "waste":
        rew = np.zeros_like(t.reward)
    elif objective == "weighted":
        rew = t.metrics["profit"] + mu * t.metrics["carbon_kg"]
    else:
        raise ValueError(objective)
    return replace(t, reward=rew)


def nb_tables(p: MarkdownProblem, dispersion: float) -> StageTables:
    """Stage tables with negative-binomial demand (variance/mean = dispersion), CES gate."""
    n = p.x0 + 1
    s = np.arange(n)
    shape = (p.horizon + 1, len(p.discounts), n)
    reward = np.zeros(shape)
    mets = {m: np.zeros(shape) for m in METRICS}
    trans = np.zeros(shape + (n,))
    for r in range(1, p.horizon + 1):
        for j, d in enumerate(p.discounts):
            lam = demand_rate(p, r, d)
            if dispersion <= 1 + 1e-12:
                from scipy.stats import poisson
                pk = poisson.pmf(s, lam)
            else:
                pk = nbinom.pmf(s, lam / (dispersion - 1.0), 1.0 / dispersion)
            # S[x, k] = P(min(D, x) = k)
            tail = np.clip(1.0 - np.concatenate([[0.0], np.cumsum(pk)[:-1]]), 0.0, None)
            S = np.where(s[None, :] < s[:, None], pk[None, :], 0.0) + np.diag(tail)
            exp_sales = S @ s
            unit = unit_economics(p, r, d)
            out = {k: exp_sales * v for k, v in unit.items()}
            out["sales"] = exp_sales
            reward[r, j] = _reward(p, out)
            for m in METRICS:
                mets[m][r, j] = 0.0 if m == "waste_units" else out[m]
            # next inventory x - k
            T = np.zeros((n, n))
            for xx in range(n):
                T[xx, xx - np.arange(xx + 1)] = S[xx, :xx + 1]
            trans[r, j] = T
    return StageTables(reward=reward, metrics=mets, trans=trans)


# ---------------------------------------------------------------- per-profile work
def problem_for(row) -> MarkdownProblem:
    return build(row["archetype"], REGION, carbon_price=SCC_USD_PER_KG,
                 q0=float(row["q0"]), eps=float(row["eps"]))


def core(row):
    """Deployment pipeline (timing benchmark): tables, profit DP, SCC DP, profit rule fit."""
    p = problem_for(row)
    t0 = time.perf_counter()
    t = build_tables(p)
    t1 = time.perf_counter()
    fast_solve(replace(p, objective="profit"), tables_for(t, "profit"))
    t2 = time.perf_counter()
    fast_solve(p, tables_for(t, "weighted"))
    t3 = time.perf_counter()
    fast_threshold_rule(replace(p, objective="profit"), tables_for(t, "profit"))
    t4 = time.perf_counter()
    return dict(store_id=row["store_id"], product_id=row["product_id"], x0=p.x0, horizon=p.horizon,
                t_tables=t1 - t0, t_profit_dp=t2 - t1, t_scc_dp=t3 - t2, t_rule=t4 - t3,
                t_core=t4 - t0)


def _ev(p, pol, tw):
    m = fast_evaluate(p, pol, tw)
    return {"scc_value": m["value"], "profit": m["profit"], "carbon_kg": m["carbon_kg"],
            "waste_units": m["waste_units"], "sales": m["sales"]}


def full(row):
    """All policies, rule fits, Poisson and negative-binomial evaluation, with timings."""
    p = problem_for(row)
    pp, pw = replace(p, objective="profit"), replace(p, objective="waste")
    tm = {}
    t0 = time.perf_counter()
    t = build_tables(p)
    tp, tw, twst = tables_for(t, "profit"), tables_for(t, "weighted"), tables_for(t, "waste")
    tm["t_tables"] = time.perf_counter() - t0
    pols = {}
    for name, prob, tab in (("profit", pp, tp), ("scc", p, tw), ("waste", pw, twst)):
        s = time.perf_counter()
        pols[name] = fast_solve(prob, tab)[1]
        tm[f"t_{name}_dp"] = time.perf_counter() - s
    pols["static"] = np.zeros((p.horizon + 1, p.x0 + 1))
    pols["static"][1] = STATIC_LAST_DAY
    rules = {}
    for name, prob, tab in (("rule_profit", pp, tp), ("rule_scc", p, tw)):
        s = time.perf_counter()
        ru = fast_threshold_rule(prob, tab)
        tm[f"t_{name}"] = time.perf_counter() - s
        rules[name] = ru
        pols[name] = threshold_policy(p, ru.lo, ru.hi, ru.thresholds)
    tm["t_core"] = tm["t_tables"] + tm["t_profit_dp"] + tm["t_scc_dp"] + tm["t_rule_profit"]
    s = time.perf_counter()
    ev = {k: _ev(p, pol, tw) for k, pol in pols.items()}
    tm["t_eval_poisson"] = time.perf_counter() - s
    s = time.perf_counter()
    tnb = tables_for(nb_tables(p, float(row["dispersion"])), "weighted")
    ev_nb = {k: _ev(p, pol, tnb) for k, pol in pols.items()}
    tm["t_eval_negbin"] = time.perf_counter() - s
    out = dict(store_id=row["store_id"], product_id=row["product_id"],
               first_category_id=row["first_category_id"], archetype=row["archetype"],
               q0=row["q0"], eps=row["eps"], dispersion=row["dispersion"],
               x0=p.x0, horizon=p.horizon)
    for name, ru in rules.items():
        out[f"{name}_lo"], out[f"{name}_hi"] = ru.lo, ru.hi
        out[f"{name}_thresholds"] = "/".join(str(c) for c in ru.thresholds)
        out[f"{name}_fit_value"] = ru.value
    for tag, e in (("", ev), ("nb_", ev_nb)):
        for k, m in e.items():
            for metric, v in m.items():
                out[f"{tag}{k}_{metric}"] = v
    out.update(tm)
    out["maxrss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
    return out

"""Deployable two-level threshold markdown rules and exhaustive search over them.

Rule family (per-day thresholds): on day r (r = remaining shelf-life days) charge
discount `lo` if inventory x <= T_r, else `hi`. On the first day of the window only
x = x0 is reachable, so T_H reduces to a binary choice (lo or hi at the start).
Scaled family: T_r = theta * x0 * r / H (one parameter scaled by days remaining).
"""
import itertools
from dataclasses import dataclass

import numpy as np

from tblca.fast_dp import StageTables, build_tables
from tblca.markdown import MarkdownProblem, _terminal


@dataclass(frozen=True)
class ThresholdRule:
    lo: float
    hi: float
    thresholds: tuple       # T_1..T_H (T_r = -1 means always hi, T_r >= x0 always lo)
    value: float


def threshold_policy(p: MarkdownProblem, lo: float, hi: float, thresholds) -> np.ndarray:
    if len(thresholds) != p.horizon:
        raise ValueError("need one threshold per day")
    x = np.arange(p.x0 + 1)
    pol = np.zeros((p.horizon + 1, p.x0 + 1))
    for r, t in enumerate(thresholds, start=1):
        pol[r] = np.where(x <= t, lo, hi)
    return pol


def scaled_thresholds(p: MarkdownProblem, theta: float) -> tuple:
    return tuple(theta * p.x0 * r / p.horizon for r in range(1, p.horizon + 1))


def _stage_options(t: StageTables, r: int, j_lo: int, j_hi: int, v_next: np.ndarray):
    q_lo = t.reward[r, j_lo] + t.trans[r, j_lo] @ v_next
    q_hi = t.reward[r, j_hi] + t.trans[r, j_hi] @ v_next
    return q_lo, q_hi


def _search_pair(p, t, j_lo, j_hi):
    """Exhaustive search over per-day thresholds for one (lo, hi) pair."""
    x = np.arange(p.x0 + 1)
    grid = np.arange(-1, p.x0 + 1)
    # frontier: list of (value vector, thresholds so far for days 1..r)
    frontier = [(_terminal(p).astype(float), ())]
    for r in range(1, p.horizon + 1):
        last = r == p.horizon
        new = []
        for v_next, ts in frontier:
            q_lo, q_hi = _stage_options(t, r, j_lo, j_hi, v_next)
            cands = (-1, p.x0) if last else grid
            new.extend((np.where(x <= c, q_lo, q_hi), ts + (int(c),)) for c in cands)
        frontier = new
    vals = np.array([v[p.x0] for v, _ in frontier])
    k = int(vals.argmax())
    return float(vals[k]), frontier[k][1]


def best_threshold_rule(p: MarkdownProblem, tables: StageTables = None) -> ThresholdRule:
    t = tables or build_tables(p)
    best = ThresholdRule(0.0, 0.0, (), -np.inf)
    for j_lo, j_hi in itertools.combinations(range(len(p.discounts)), 2):
        val, ts = _search_pair(p, t, j_lo, j_hi)
        if val > best.value + 1e-12:
            best = ThresholdRule(p.discounts[j_lo], p.discounts[j_hi], ts, val)
    return best


def _value_of(p, t, lo_j, hi_j, thresholds):
    x = np.arange(p.x0 + 1)
    v = _terminal(p).astype(float)
    for r, c in enumerate(thresholds, start=1):
        q_lo, q_hi = _stage_options(t, r, lo_j, hi_j, v)
        v = np.where(x <= c, q_lo, q_hi)
    return float(v[p.x0])


def best_scaled_rule(p: MarkdownProblem, thetas=np.round(np.arange(0.0, 2.001, 0.01), 2),
                     tables: StageTables = None) -> tuple:
    """Best (lo, hi, theta) with T_r = theta * x0 * r / H. Returns (ThresholdRule, theta)."""
    t = tables or build_tables(p)
    best, best_theta = ThresholdRule(0.0, 0.0, (), -np.inf), None
    for j_lo, j_hi in itertools.combinations(range(len(p.discounts)), 2):
        for th in thetas:
            ts = scaled_thresholds(p, th)
            val = _value_of(p, t, j_lo, j_hi, ts)
            if val > best.value + 1e-12:
                best = ThresholdRule(p.discounts[j_lo], p.discounts[j_hi], ts, val)
                best_theta = float(th)
    return best, best_theta


def best_time_only_rule(p: MarkdownProblem, tables: StageTables = None) -> tuple:
    """Best inventory-independent schedule d(r) over the discount grid: (schedule, value)."""
    t = tables or build_tables(p)
    best, best_s = -np.inf, None
    for sched in itertools.product(range(len(p.discounts)), repeat=p.horizon):
        v = _terminal(p).astype(float)
        for r, j in enumerate(sched, start=1):
            v = t.reward[r, j] + t.trans[r, j] @ v
        if v[p.x0] > best + 1e-12:
            best, best_s = float(v[p.x0]), tuple(p.discounts[j] for j in sched)
    return best_s, best

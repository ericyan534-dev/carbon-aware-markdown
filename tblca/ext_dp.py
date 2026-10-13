"""Generalized markdown DP with pluggable per-unit economics (extension module).

`tblca.markdown` hard-wires the per-unit profit and carbon of a sold unit. The
extensions (re-spending rebound, strategic waiting) only change
what a sold unit is worth, so this module re-implements the same backward
induction with a `unit_fn(p, r, d) -> dict` hook. With the default hook
(`markdown.unit_economics`) it reproduces `markdown.solve` / `evaluate_policy`
exactly (same Poisson truncation, same tie-breaking), which the tests check.

It also returns metric arrays for every starting inventory x.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.stats import poisson

from tblca.markdown import (
    METRICS, MarkdownProblem, _reward, _terminal, demand_rate, unit_economics,
)

UnitFn = Callable[[MarkdownProblem, int, float], dict]


@dataclass(frozen=True)
class ExtSolution:
    value: np.ndarray            # (horizon+1, x0+1)
    policy: np.ndarray           # discount at (r, x); row 0 unused
    metrics: dict = field(default_factory=dict)       # at (horizon, x0)
    by_inventory: dict = field(default_factory=dict)  # metric -> array over x at r=horizon


def _stage_table(p: MarkdownProblem, r: int, d: float):
    """Poisson pmf over k = 0..x0-1 for (r, d); state x uses the first x entries + tail."""
    return poisson.pmf(np.arange(p.x0), demand_rate(p, r, d))


def _pmf_for(base_pmf: np.ndarray, x: int) -> np.ndarray:
    head = base_pmf[:x]
    return np.append(head, max(0.0, 1.0 - head.sum()))


def _stage(p, unit_fn, r, d, x, base_pmf):
    pmf = _pmf_for(base_pmf, x)
    sales = float(pmf @ np.arange(x + 1))
    unit = unit_fn(p, r, d)
    out = {k: sales * v for k, v in unit.items()}
    return {**out, "sales": sales}, pmf


def _cont(v_next, x, pmf):
    return float(pmf @ v_next[x - np.arange(x + 1)])


def solve_ext(p: MarkdownProblem, unit_fn: UnitFn = unit_economics) -> ExtSolution:
    value = np.zeros((p.horizon + 1, p.x0 + 1))
    policy = np.zeros_like(value)
    value[0] = _terminal(p)
    for r in range(1, p.horizon + 1):
        tables = {d: _stage_table(p, r, d) for d in p.discounts}
        for x in range(p.x0 + 1):
            best, best_d = -np.inf, p.discounts[0]
            for d in p.discounts:
                out, pmf = _stage(p, unit_fn, r, d, x, tables[d])
                q = _reward(p, out) + _cont(value[r - 1], x, pmf)
                if q > best + 1e-12:
                    best, best_d = q, d
            value[r, x], policy[r, x] = best, best_d
    ev = evaluate_ext(p, policy, unit_fn)
    return ExtSolution(value=value, policy=policy, metrics=ev.metrics,
                       by_inventory=ev.by_inventory)


def evaluate_ext(p: MarkdownProblem, policy: np.ndarray,
                 unit_fn: UnitFn = unit_economics) -> ExtSolution:
    shape = (p.horizon + 1, p.x0 + 1)
    vals = {m: np.zeros(shape) for m in METRICS}
    vals["waste_units"][0] = np.arange(p.x0 + 1)
    value = np.zeros(shape)
    value[0] = _terminal(p)
    for r in range(1, p.horizon + 1):
        tables = {}
        for x in range(p.x0 + 1):
            d = float(policy[r, x])
            if d not in tables:
                tables[d] = _stage_table(p, r, d)
            out, pmf = _stage(p, unit_fn, r, d, x, tables[d])
            value[r, x] = _reward(p, out) + _cont(value[r - 1], x, pmf)
            for m in METRICS:
                stage = 0.0 if m == "waste_units" else out[m]
                vals[m][r, x] = stage + _cont(vals[m][r - 1], x, pmf)
    by_inv = {m: vals[m][p.horizon].copy() for m in METRICS}
    metrics = {m: float(by_inv[m][p.x0]) for m in METRICS}
    return ExtSolution(value=value, policy=policy, metrics=metrics, by_inventory=by_inv)

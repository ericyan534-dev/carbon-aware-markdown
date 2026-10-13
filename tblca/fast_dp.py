"""Vectorized backward induction for MarkdownProblem (same model as tblca.markdown).

Per (r, d) the stage is summarised by a reward vector over inventory x, a matrix of
metric vectors, and a transition matrix M[x, x'] = P(next inventory = x' | x).
Results match markdown.solve / evaluate_policy to floating-point precision but run
one to two orders of magnitude faster, which makes exhaustive rule searches feasible.
"""
from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson

from tblca.markdown import (METRICS, MarkdownProblem, _reward, _terminal, demand_rate,
                            unit_economics)

TIE_TOL = 1e-12


@dataclass(frozen=True)
class StageTables:
    reward: np.ndarray      # (H+1, D, X) expected stage objective
    metrics: dict           # metric -> (H+1, D, X) expected stage metric
    trans: np.ndarray       # (H+1, D, X, X) transition matrices


def sales_matrix(rate: float, x0: int) -> np.ndarray:
    """S[x, s] = P(sales = s | inventory x) with sales = min(Poisson(rate), x)."""
    s = np.arange(x0 + 1)
    pk = poisson.pmf(s, rate)
    tail = np.clip(1.0 - np.concatenate([[0.0], np.cumsum(pk)[:-1]]), 0.0, None)
    x = s[:, None]
    return np.where(s[None, :] < x, pk[None, :], 0.0) + np.where(s[None, :] == x, tail[:, None], 0.0)


def _transition(sales: np.ndarray) -> np.ndarray:
    x0 = sales.shape[0] - 1
    x = np.arange(x0 + 1)[:, None]
    nxt = x - np.arange(x0 + 1)[None, :]
    m = np.zeros_like(sales)
    rows = np.broadcast_to(x, sales.shape)
    valid = nxt >= 0
    np.add.at(m, (rows[valid], nxt[valid]), sales[valid])
    return m


def build_tables(p: MarkdownProblem, unit_fn=unit_economics) -> StageTables:
    """Stage tables; `unit_fn(p, r, d)` gives per-unit profit/carbon (default: the paper model)."""
    shape = (p.horizon + 1, len(p.discounts), p.x0 + 1)
    reward = np.zeros(shape)
    mets = {m: np.zeros(shape) for m in METRICS}
    trans = np.zeros(shape + (p.x0 + 1,))
    s = np.arange(p.x0 + 1)
    for r in range(1, p.horizon + 1):
        for j, d in enumerate(p.discounts):
            sm = sales_matrix(demand_rate(p, r, d), p.x0)
            exp_sales = sm @ s
            unit = unit_fn(p, r, d)
            out = {k: exp_sales * v for k, v in unit.items()}
            out["sales"] = exp_sales
            reward[r, j] = _reward(p, out)
            for m in METRICS:
                mets[m][r, j] = 0.0 if m == "waste_units" else out[m]
            trans[r, j] = _transition(sm)
    return StageTables(reward=reward, metrics=mets, trans=trans)


def fast_solve(p: MarkdownProblem, tables: StageTables = None):
    """Return (value, policy) with the same first-best tie-breaking as markdown.solve."""
    t = tables or build_tables(p)
    value = np.zeros((p.horizon + 1, p.x0 + 1))
    policy = np.zeros_like(value)
    value[0] = _terminal(p)
    disc = np.asarray(p.discounts)
    for r in range(1, p.horizon + 1):
        q = t.reward[r] + t.trans[r] @ value[r - 1]          # (D, X)
        best = np.full(p.x0 + 1, -np.inf)
        idx = np.zeros(p.x0 + 1, dtype=int)
        for j in range(len(disc)):
            better = q[j] > best + TIE_TOL
            best = np.where(better, q[j], best)
            idx = np.where(better, j, idx)
        value[r], policy[r] = best, disc[idx]
    return value, policy


def policy_index(p: MarkdownProblem, policy: np.ndarray) -> np.ndarray:
    disc = np.asarray(p.discounts)
    idx = np.abs(policy[..., None] - disc).argmin(-1)
    if not np.allclose(disc[idx[1:]], policy[1:]):
        raise ValueError("policy uses discounts outside p.discounts")
    return idx


def fast_evaluate(p: MarkdownProblem, policy: np.ndarray, tables: StageTables = None) -> dict:
    """Expected objective value and metrics from (horizon, x0) under a policy array."""
    t = tables or build_tables(p)
    idx = policy_index(p, policy)
    x = np.arange(p.x0 + 1)
    value = _terminal(p).astype(float)
    acc = {m: np.zeros(p.x0 + 1) for m in METRICS}
    acc["waste_units"] = x.astype(float)
    for r in range(1, p.horizon + 1):
        j = idx[r]
        m_r = t.trans[r, j, x]                                  # (X, X)
        value = t.reward[r, j, x] + m_r @ value
        acc = {m: t.metrics[m][r, j, x] + m_r @ acc[m] for m in METRICS}
    return {"value": float(value[p.x0]), **{m: float(acc[m][p.x0]) for m in METRICS}}

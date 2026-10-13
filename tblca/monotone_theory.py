"""Generic Poisson markdown DP and structural diagnostics.

Abstract model: stages r = H..1, inventory x in {0..X}, action index j over an
ordered discount grid. Demand D ~ Poisson(lam[r, j]) with lam increasing in j,
sales S = min(D, x), stage reward u[r, j] * S, terminal V_0(x) = -w * x.

    Q_r(x, j) = E[u_rj S + V_{r-1}(x - S)]
              = V_{r-1}(x) + sum_{k=1}^{x} P(D >= k) (u_rj - dV_{r-1}(x - k))

Used to verify numerically that optimal discounts are nondecreasing in stock.
"""
from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson

from tblca.markdown import MarkdownProblem, demand_rate, unit_economics


@dataclass(frozen=True)
class GenericProblem:
    lam: np.ndarray     # (H+1, J) Poisson rates; row 0 unused
    u: np.ndarray       # (H+1, J) per-unit rewards; row 0 unused
    x0: int
    w: float = 0.0      # terminal penalty per unsold unit

    @property
    def horizon(self) -> int:
        return self.lam.shape[0] - 1


def from_markdown(p: MarkdownProblem) -> GenericProblem:
    """Map a MarkdownProblem to the generic (lam, u, w) form, objective-aware."""
    H, J = p.horizon, len(p.discounts)
    lam, u = np.zeros((H + 1, J)), np.zeros((H + 1, J))
    for r in range(1, H + 1):
        for j, d in enumerate(p.discounts):
            lam[r, j] = demand_rate(p, r, d)
            e = unit_economics(p, r, d)
            u[r, j] = {"profit": e["profit"], "carbon": e["carbon_kg"], "waste": 0.0,
                       "weighted": e["profit"] + p.carbon_price * e["carbon_kg"]}[p.objective]
    w = {"waste": 1.0, "weighted": p.waste_penalty}.get(p.objective, 0.0)
    return GenericProblem(lam=lam, u=u, x0=p.x0, w=w)


def q_table(lam_row, u_row, v_next):
    """Q(x, j) for all x in 0..X and actions j, given V_{r-1} (length X+1)."""
    X = len(v_next) - 1
    xs = np.arange(X + 1)
    q = np.empty((X + 1, len(lam_row)))
    for j, (lam, u) in enumerate(zip(lam_row, u_row)):
        pmf = poisson.pmf(xs, lam)
        surv = poisson.sf(xs - 1, lam)               # P(D >= k), k = 0..X
        mean_sales = np.concatenate(([0.0], np.cumsum(surv[1:])))  # E min(D, x)
        conv = np.convolve(pmf, v_next)[: X + 1]     # sum_{k<=x} pmf_k V(x-k)
        cont = conv - pmf * v_next[0] + surv * v_next[0]
        q[:, j] = u * mean_sales + cont
    return q


def solve_generic(g: GenericProblem, tol=1e-12):
    """Backward induction. Ties broken toward the smallest action index."""
    H, X = g.horizon, g.x0
    value = np.zeros((H + 1, X + 1))
    policy = np.zeros((H + 1, X + 1), dtype=int)
    qs = [None]
    value[0] = -g.w * np.arange(X + 1)
    for r in range(1, H + 1):
        q = q_table(g.lam[r], g.u[r], value[r - 1])
        best = q.max(axis=1, keepdims=True)
        policy[r] = np.argmax(q >= best - tol * (1 + np.abs(best)), axis=1)
        value[r] = best[:, 0]
        qs.append(q)
    return value, policy, qs


def is_monotone_in_x(policy, tol=0):
    return bool(np.all(np.diff(policy[1:], axis=1) >= -tol))


def concavity_gap(value):
    """max over r>=0, x of second difference (<= 0 iff each V_r concave)."""
    if value.shape[1] < 3:
        return -np.inf
    return float(np.diff(value, n=2, axis=1).max())


def expected_sales(lam: float, xs: np.ndarray) -> np.ndarray:
    """M_lam(x) = E min(D, x) for D ~ Poisson(lam), x = 0..max(xs)."""
    X = int(np.max(xs))
    surv = poisson.sf(np.arange(X + 1) - 1, lam)
    m = np.concatenate(([0.0], np.cumsum(surv[1:])))
    return m[np.asarray(xs, dtype=int)]


def pairwise_single_crossing(q: np.ndarray, tol=1e-12) -> bool:
    """True iff for every action pair i<j, x -> Q(x,j)-Q(x,i) never goes from >0 back to <0.

    This is the Milgrom-Shannon single-crossing condition for every pair, which makes the
    argmax monotone for every sub-grid of actions (not just the full grid).
    """
    J = q.shape[1]
    for i in range(J):
        for j in range(i + 1, J):
            diff = q[:, j] - q[:, i]
            pos = np.where(diff > tol)[0]
            if pos.size and np.any(diff[pos[0]:] < -tol):
                return False
    return True


def solve_bernoulli(pi: np.ndarray, u: np.ndarray, x0: int, w: float = 0.0, tol=1e-12):
    """One-arrival-per-period model: at most one sale per period, w.p. pi[r, j]."""
    H = pi.shape[0] - 1
    value = np.zeros((H + 1, x0 + 1))
    policy = np.zeros((H + 1, x0 + 1), dtype=int)
    value[0] = -w * np.arange(x0 + 1)
    for r in range(1, H + 1):
        v = value[r - 1]
        c = np.diff(v)                                    # c[x-1] = dV(x-1), x >= 1
        q = np.repeat(v[:, None], len(pi[r]), axis=1)
        q[1:] += pi[r][None, :] * (u[r][None, :] - c[:, None])
        best = q.max(axis=1, keepdims=True)
        policy[r] = np.argmax(q >= best - tol * (1 + np.abs(best)), axis=1)
        value[r] = best[:, 0]
    return value, policy


def q_table_pmf(pmfs, u_row, v_next):
    """Like q_table but for arbitrary demand pmfs (each of length >= X+1, on 0..X)."""
    X = len(v_next) - 1
    q = np.empty((X + 1, len(u_row)))
    for j, (pmf, u) in enumerate(zip(pmfs, u_row)):
        pmf = np.asarray(pmf[: X + 1], dtype=float)
        surv = np.clip(1.0 - np.concatenate(([0.0], np.cumsum(pmf)))[: X + 1], 0.0, 1.0)
        mean_sales = np.concatenate(([0.0], np.cumsum(surv[1:])))
        cont = np.convolve(pmf, v_next)[: X + 1] - pmf * v_next[0] + surv * v_next[0]
        q[:, j] = u * mean_sales + cont
    return q


def pair_single_crossing(diff: np.ndarray, tol=1e-12) -> bool:
    """diff(x) never returns below -tol after exceeding +tol, and never returns <= 0 after > tol."""
    pos = np.where(diff > tol)[0]
    return not (pos.size and np.any(diff[pos[0]:] < -tol))

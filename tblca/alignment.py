"""Alignment index and carbon-price vs waste-penalty frontiers.

Per unit sold at (r, d), the weighted objective pays
    p(1-d) - c_f g(d) + mu [a(r) g(d) + b(r)],
    a(r) = m_u F(r) EF_prod,  F = g_fresh g_spoil,  b(r) = m_u (g_spoil EF_eol - ops),
so mu a(r) is a per-unit subsidy that offsets the cannibalisation cost c_f.  The
alignment index theta(r) = mu a(r) / c_f measures how much of it is offset.
"""
from dataclasses import replace

import numpy as np
from scipy.stats import poisson

from tblca.markdown import MarkdownProblem, demand_rate, solve, stage_outcomes
from tblca.substitution import freshness_gate, rebound_gate, spoilage_gate


def carbon_coefficients(p: MarkdownProblem, r: int) -> tuple:
    """(a, b): per-unit carbon = a * g(d) + b at remaining life r."""
    fresh = float(freshness_gate(r, p.lam_f))
    consumed = float(spoilage_gate(r, p.lam_f, p.p_base))
    a = p.unit_kg * fresh * consumed * p.ef_prod
    b = p.unit_kg * (consumed * p.ef_eol - p.ops_kg)
    return a, b


def theta(p: MarkdownProblem, r: int, mu: float = None) -> float:
    """Alignment index theta(r) = mu m_u F(r) EF_prod / c_f."""
    mu = p.carbon_price if mu is None else mu
    if p.margin_fresh <= 0:
        raise ValueError("margin_fresh (c_f) must be positive")
    return mu * carbon_coefficients(p, r)[0] / p.margin_fresh


def theta_bar(p: MarkdownProblem, mu: float = None) -> float:
    """Batch-level index: mean of theta(r) over r = 1..H."""
    return float(np.mean([theta(p, r, mu) for r in range(1, p.horizon + 1)]))


def sales_elasticity(p: MarkdownProblem, x: int, r: int, d: float) -> float:
    """eta_S = dlnE[min(D,x)]/dln(lambda) = lambda P(D <= x-1) / E[min(D,x)]."""
    if x <= 0:
        return 0.0
    lam = demand_rate(p, r, d)
    sales = stage_outcomes(p, x, r, d)["sales"]
    return float(lam * poisson.cdf(x - 1, lam) / sales) if sales > 0 else 1.0


def id_margin(p: MarkdownProblem, x: int, r: int, d: float) -> float:
    """eta_S (a g + b) - a (1-kappa)(1-d)^eps; >= 0 iff stage carbon rises with d."""
    a, b = carbon_coefficients(p, r)
    g = float(rebound_gate(d, p.eps, p.kappa))
    return sales_elasticity(p, x, r, d) * (a * g + b) - a * (1 - p.kappa) * (1 - d) ** p.eps


def stage_carbon_nondecreasing(p: MarkdownProblem, x: int, r: int, tol=1e-12) -> bool:
    """Increasing-differences condition on the discount grid: C(d) nondecreasing."""
    c = [stage_outcomes(p, x, r, d)["carbon_kg"] for d in sorted(p.discounts)]
    return bool(np.all(np.diff(c) >= -tol))


def myopic_argmax(p: MarkdownProblem, x: int, r: int, mu: float) -> float:
    """Largest maximiser of the one-stage weighted reward (terminal waste penalty included)."""
    best, best_d = -np.inf, None
    for d in sorted(p.discounts):
        out = stage_outcomes(p, x, r, d)
        q = out["profit"] + mu * out["carbon_kg"] - p.waste_penalty * (x - out["sales"])
        if q >= best - 1e-12:
            best, best_d = max(q, best), d
    return best_d


def policy_path(p: MarkdownProblem, mus) -> list:
    """Optimal weighted policies along a carbon-price path (waste penalty kept)."""
    return [solve(replace(p, objective="weighted", carbon_price=float(m))) for m in mus]


def monotonicity_violations(policies, tol=1e-12) -> dict:
    """Count states where the optimal discount falls as mu rises (consecutive mus)."""
    stack = np.stack([s.policy[1:, 1:] for s in policies])
    drops = np.diff(stack, axis=0) < -tol
    return {"pairs": int(drops.size), "violations": int(drops.sum()),
            "states_ever_violated": int(drops.any(axis=0).sum()),
            "states": int(drops[0].size)}


def upper_hull(loss, gain):
    """Upper concave envelope of points (loss, gain): vertices sorted by loss."""
    pts = sorted(set(zip(np.round(loss, 12), np.round(gain, 12))))
    hull = []
    for pt in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (pt[1] - y1) - (y2 - y1) * (pt[0] - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append(pt)
    # keep the nondecreasing (efficient) part
    best, out = -np.inf, []
    for x, y in hull:
        if y > best:
            out.append((x, y))
            best = y
    return np.array(out)


def matched_ratio(loss_cp, gain_cp, loss_wp, gain_wp, min_loss=1e-6):
    """Carbon gain of the carbon-price hull over the waste-penalty curve at matched loss.

    Carbon-price points are interpolated along their upper hull (attainable by
    randomising between adjacent mu-optimal policies); waste-penalty points are
    evaluated at their own losses, restricted to the common loss range.
    """
    hull = upper_hull(np.asarray(loss_cp), np.asarray(gain_cp))
    lw, gw = np.asarray(loss_wp, float), np.asarray(gain_wp, float)
    keep = (lw > min_loss) & (lw <= hull[-1, 0] + 1e-12) & (gw > 0)
    interp = np.interp(lw[keep], hull[:, 0], hull[:, 1])
    return lw[keep], interp, gw[keep], interp / gw[keep]


def efficient_discounts(p: MarkdownProblem, x: int, r: int) -> list:
    """Discounts not Pareto-dominated in (stage profit incl. waste term, stage carbon)."""
    pts = []
    for d in sorted(p.discounts):
        out = stage_outcomes(p, x, r, d)
        pts.append((d, out["profit"] - p.waste_penalty * (x - out["sales"]), out["carbon_kg"]))
    def dominated(i):
        _, pi, ci = pts[i]
        return any(pj >= pi - 1e-12 and cj >= ci - 1e-12 and (pj > pi + 1e-12 or cj > ci + 1e-12)
                   for j, (_, pj, cj) in enumerate(pts) if j != i)
    return [(d, pi, ci) for i, (d, pi, ci) in enumerate(pts) if not dominated(i)]


def carbon_nondecreasing_on_efficient(p: MarkdownProblem, x: int, r: int, tol=1e-12) -> bool:
    """Weaker sufficient condition: C(d) nondecreasing in d over the efficient discounts."""
    c = [ci for _, _, ci in efficient_discounts(p, x, r)]
    return bool(np.all(np.diff(c) >= -tol))


def supported_frontier(p: MarkdownProblem, mu_max: float, tol=1e-9, max_iter=500) -> list:
    """Exact vertices of the (profit, carbon) upper hull for mu in [0, mu_max].

    Dichotomic (Aneja-Nair) search: between two vertices with profits P1 > P2 and
    carbon C1 < C2, solve at the breakpoint mu* = (P1 - P2) / (C2 - C1); a new vertex
    exists iff the mu*-optimal value beats both.  Returns [(mu, profit, carbon)] sorted
    by mu.  Ties at mu = 0 are broken toward higher carbon via a tiny mu.
    """
    def point(mu):
        m = solve(replace(p, objective="weighted", carbon_price=float(mu), waste_penalty=0.0)).metrics
        return (float(mu), m["profit"], m["carbon_kg"])

    pts = {0.0: point(0.0), mu_max: point(mu_max)}
    stack, it = [(0.0, mu_max)], 0
    while stack and it < max_iter:
        it += 1
        lo, hi = stack.pop()
        (_, p1, c1), (_, p2, c2) = pts[lo], pts[hi]
        if c2 - c1 <= tol:
            continue
        mu_star = (p1 - p2) / (c2 - c1)
        if not (lo < mu_star < hi):
            continue
        _, pm, cm = pts.setdefault(mu_star, point(mu_star))
        if pm + mu_star * cm > p1 + mu_star * c1 + tol:
            stack += [(lo, mu_star), (mu_star, hi)]
        else:
            # mu_star sits on the segment; probe just above it to fetch the right vertex
            pts.pop(mu_star)
    verts = sorted(pts.values())
    out = []
    for v in verts:
        if not out or abs(v[1] - out[-1][1]) > tol or abs(v[2] - out[-1][2]) > tol:
            out.append(v)
    return out

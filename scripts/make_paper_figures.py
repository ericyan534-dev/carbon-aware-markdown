"""Figures of the paper, written to outputs/figures/{name}.pdf (+ .png preview).

    teaser          Fig. 1  buyers, overstatement map, dairy landscape <- outputs/main/landscape_dairy_hk.npz
    elasticity_col  Fig. 2  category elasticities   <- outputs/elasticity/elasticity_by_category.csv
    goals2          Fig. 3  value at the SCC, carbon-price paths
                            <- outputs/main/{goals.csv,pareto.csv,policy_comparison.csv}
    why4            Fig. 4  unsold stock, profile alignment, alignment index
                            <- outputs/threshold/{theory_meta.json,theory_curves.csv},
                               outputs/profiles/profile_results_with_shares.csv,
                               outputs/alignment/{s1_sweep.csv,summary.json}
    policy_maps     Fig. 5  profit-maximizing policy maps (no inputs needed)

Usage: python scripts/make_paper_figures.py [name ...]   (default: all five)
"""
import json
import sys
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.plot_style import COL_W, INK, MUTED, OUTPUTS, SERIES, save  # noqa: E402
from tblca.markdown import demand_rate, solve  # noqa: E402
from tblca.params import ARCHETYPES, REGIONS, SCC_USD_PER_KG  # noqa: E402
from tblca.scenario import DEFAULTS, build  # noqa: E402
from tblca.substitution import freshness_gate, rebound_gate, spoilage_gate  # noqa: E402

EPS, KAPPA = DEFAULTS["eps"], DEFAULTS["kappa"]
EPS_HAT = 1.94
MIN_OBS = 30000          # categories shown in the elasticity figure
EPS_GRID = np.round(np.arange(0.8, 3.01, 0.05), 3)
TITLE = {"fontsize": 8, "color": INK, "loc": "left"}

MAIN_DIR = OUTPUTS / "main"
LANDSCAPE_NPZ = MAIN_DIR / "landscape_dairy_hk.npz"
GOALS_CSV = MAIN_DIR / "goals.csv"
PARETO_CSV = MAIN_DIR / "pareto.csv"
POLICY_CSV = MAIN_DIR / "policy_comparison.csv"
ELASTICITY_CSV = OUTPUTS / "elasticity" / "elasticity_by_category.csv"
THRESHOLD_DIR = OUTPUTS / "threshold"
PROFILES_CSV = OUTPUTS / "profiles" / "profile_results_with_shares.csv"
ALIGN_DIR = OUTPUTS / "alignment"

LANDSCAPE_MARK = {"waste": ("D", "Waste-min.", (8, 66)), "profit": ("o", "Profit-max.", (8, 22)),
                  "scc": ("*", "Carbon-priced", (8, 30)), "carbon": ("^", "Carbon-opt.", (8, 58))}
GOALS = [("profit", "Profit-maximizing"), ("weighted", "Carbon priced at SCC"),
         ("carbon", "Carbon-only"), ("static", "Static last-day 35%"), ("waste", "Waste-minimizing")]
STATIC, CARBON_PRICED = "Last-day 35%", "SCC-weighted (ours)"


# ---------------------------------------------------------------- Fig. 1 (teaser)
def panel_composition(ax):
    d = np.linspace(0, 0.6, 200)
    inc = (1 - d) ** (-EPS) - 1
    layers = [np.ones_like(d), KAPPA * inc, (1 - KAPPA) * inc]
    cols = [SERIES["dairy"], SERIES["produce"], SERIES["bakery"]]
    ax.stackplot(100 * d, *layers, colors=cols, alpha=0.9, edgecolor="white", linewidth=0.6)
    ax.text(4, 0.5, "regular buyers", color="white", fontsize=6.3, va="center")
    ax.annotate("switchers, share $\\kappa$\n(replace a fresh purchase)", (40, 1.9), xytext=(3, 3.0),
                fontsize=6.3, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.annotate("new shoppers (replace nothing)", (56, 4.3), xytext=(3, 5.2), fontsize=6.3,
                color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set_xlabel("Discount depth $d$ (%)")
    ax.set_ylabel("Sales / full-price demand")
    ax.set_xlim(0, 60)
    ax.set_title("(a) Who buys at a discount", **TITLE)


def panel_overstatement(ax):
    a, g = ARCHETYPES["dairy"], REGIONS["Hong Kong"]
    lam, eol = DEFAULTS["lam_f"], g.eol(a)
    R, D = np.meshgrid(np.linspace(0.25, 3, 120), np.linspace(0, 0.6, 120))
    spoil = spoilage_gate(R, lam, a.p_base)
    delta = freshness_gate(R, lam) * spoil * rebound_gate(D, EPS, KAPPA)
    omega = (a.ef_prod + eol) / (delta * a.ef_prod + spoil * eol)
    pcm = ax.contourf(R, 100 * D, omega, levels=np.linspace(1, 6, 21), cmap="Oranges", extend="max")
    cs = ax.contour(R, 100 * D, omega, levels=[1.5, 2, 3, 4], colors=INK, linewidths=0.6)
    ax.clabel(cs, fmt=lambda v: f"{v:g}×", fontsize=6)
    ax.plot(1, 35, marker="X", ms=7, color=INK, mec="white")
    ax.annotate("static rule", (1, 35), xytext=(5, 6), textcoords="offset points", fontsize=6.5,
                color=INK)
    ax.set_xlabel("Remaining shelf life at sale (days)")
    ax.set_ylabel("Discount depth $d$ (%)")
    ax.grid(False)
    ax.set_title("(b) Overstatement $\\Omega$", **TITLE)
    return pcm


def panel_landscape(ax):
    z = np.load(LANDSCAPE_NPZ)
    g, C, P, W = 100 * z["g"], z["C"], z["P"], z["W"]
    E, L = np.meshgrid(g, g, indexing="ij")
    pcm = ax.contourf(L, E, C, levels=14, cmap="Blues")
    ax.contour(L, E, P, levels=[30, 38], colors=INK, linewidths=0.6, linestyles="--")
    best = {"waste": np.argmin(W + 1e-6 * (C.max() - C)), "carbon": np.argmax(C),
            "scc": np.argmax(P + SCC_USD_PER_KG * C), "profit": np.argmax(P)}
    for key, (marker, label, offset) in LANDSCAPE_MARK.items():
        i, j = np.unravel_index(best[key], P.shape)
        ax.plot(g[j], g[i], marker=marker, ms=8 if key == "scc" else 6, color="white", mec=INK,
                mew=0.8, zorder=5)
        ax.annotate(label, (g[j], g[i]), xytext=offset, fontsize=6.3, color=INK, va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.5))
    ax.plot(35, 0, marker="X", ms=7, color=INK, mec="white", clip_on=False, zorder=6)
    ax.annotate("static rule", (35, 0), xytext=(-22, 7), textcoords="offset points", fontsize=6.3,
                color=INK)
    ax.set_xlim(-2, 63)
    ax.set_ylim(-3, 70)
    ax.set_xlabel("Last-day discount (%)")
    ax.set_ylabel("Earlier-day discount (%)")
    ax.grid(False)
    ax.set_title("(c) Dairy: CO$_2$e (fill), profit (lines)", **TITLE)
    return pcm


def fig_teaser():
    fig, axes = plt.subplots(1, 3, figsize=(COL_W * 2, 2.3), gridspec_kw={"wspace": 0.55})
    panel_composition(axes[0])
    p1 = panel_overstatement(axes[1])
    fig.colorbar(p1, ax=axes[1], pad=0.02, ticks=[1, 2, 3, 4, 5, 6]).set_label("$\\Omega$", fontsize=7)
    p2 = panel_landscape(axes[2])
    fig.colorbar(p2, ax=axes[2], pad=0.02).set_label("Net CO$_2$e avoided (kg)", fontsize=7)
    save(fig, "teaser")


# ---------------------------------------------------------------- Fig. 2 (elasticity_col)
def fig_elasticity_col():
    t = pd.read_csv(ELASTICITY_CSV).sort_values("eps_iv")
    t = t[t.n >= MIN_OBS]
    fig, ax = plt.subplots(figsize=(COL_W, 2.9))
    y = np.arange(len(t))
    ax.errorbar(t.eps_ols, y + 0.17, xerr=1.96 * t.se_ols, fmt="o", ms=3, color=MUTED, lw=1,
                label="Fixed effects")
    ax.errorbar(t.eps_iv, y - 0.17, xerr=1.96 * t.se_iv, fmt="s", ms=3, color=SERIES["dairy"],
                lw=1, label="Instrumental variable")
    ax.set_yticks(y, [f"C{c}" for c in t.category], fontsize=6.5)
    ax.set_ylim(-0.7, len(t) - 0.3)
    ax.set_xlabel(r"Discount elasticity $\varepsilon$ (95% CI)")
    ax.set_ylabel("Product category")
    ax.axvline(EPS_HAT, color=SERIES["dairy"], lw=0.8, ls="--")
    ax.text(EPS_HAT + 0.08, len(t) - 1.4, f"pooled estimate {EPS_HAT}", color=INK, fontsize=6.5)
    ax.legend(loc="lower right", fontsize=6.5)
    fig.tight_layout()
    save(fig, "elasticity_col")


# ---------------------------------------------------------------- Fig. 3 (goals2)
def panel_goals(ax):
    d = pd.read_csv(GOALS_CSV)
    rng = np.random.default_rng(1)
    for i, (key, label) in enumerate(GOALS):
        g = d[d.policy == key]
        ax.axhspan(-i - 0.4, -i + 0.4, color="#f4f2ed" if key == "profit" else "none", zorder=0)
        y = -i + rng.uniform(-0.2, 0.2, len(g))
        ax.scatter(g.pct_best, y, s=11, c=[SERIES[a] for a in g.archetype], alpha=0.9, lw=0.3,
                   edgecolors="white", zorder=3)
        lo, hi = g.pct_best.min(), g.pct_best.max()
        ax.plot([lo, hi], [-i, -i], color=MUTED, lw=0.7, zorder=2)
        txt = "100% (by definition)" if key == "weighted" else f"{lo:.1f}–{hi:.1f}%"
        ax.text(49 if hi > 99 else hi + 1.2, -i, txt, fontsize=6.3, va="center", ha="left",
                color=INK if key == "profit" else MUTED, weight="bold" if key == "profit" else "normal")
    ax.set_yticks([-i for i in range(len(GOALS))])
    ax.set_yticklabels([lab for _, lab in GOALS], fontsize=6.8)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xlim(47.5, 101.5)
    ax.set_ylim(-len(GOALS) + 0.5, 0.5)
    ax.set_xlabel("Value at the SCC, % of best attainable")
    ax.grid(axis="y", visible=False)
    h = [plt.Line2D([], [], ls="", marker="o", ms=3.5, color=c, label=a) for a, c in SERIES.items()]
    ax.legend(handles=h, fontsize=6, loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=4,
              handletextpad=0.1, columnspacing=0.7)
    ax.set_title("(a) Each goal at the SCC, 20 cases", **TITLE)


def _vs_static(row, base):
    return 100 * (row.profit / base.profit - 1), 100 * (row.carbon_kg / base.carbon_kg - 1)


def panel_paths(ax):
    par = pd.read_csv(PARETO_CSV)
    pc = pd.read_csv(POLICY_CSV).query("region == 'Hong Kong'").set_index(["archetype", "policy"])
    for arch, col in SERIES.items():
        base = pc.loc[(arch, STATIC)]
        x, y = _vs_static(par[par.archetype == arch], base)
        ax.plot(x, y, color=col, lw=1.6, marker="o", ms=2, zorder=3)
        ax.plot(*_vs_static(pc.loc[(arch, CARBON_PRICED)], base), marker="*", ms=9, color=col,
                mec="white", mew=0.8, zorder=5)
        i = int(np.argmin(x.to_numpy()))
        ax.annotate(arch, (x.iloc[i], y.iloc[i]), xytext=(0, -10 if arch == "poultry" else 3),
                    textcoords="offset points", fontsize=6.5, color=INK)
    ax.plot(0, 0, marker="X", ms=7, color=MUTED, zorder=5)
    ax.annotate("static last-day 35%", (0, 0), xytext=(5, 3), textcoords="offset points",
                fontsize=6.5, color=MUTED)
    ax.text(0.03, 0.47, "lines: carbon price 0 → \\$5/kg\nstars: carbon priced at SCC\n"
            "right end: profit-max.", transform=ax.transAxes, fontsize=6.2, va="top", color=MUTED)
    ax.set_ylim(-8, 130)
    ax.set_xlabel("Profit change vs. static rule (%)")
    ax.set_ylabel("Net CO$_2$e change vs. static rule (%)")
    ax.set_title("(b) Carbon-price paths, Hong Kong", **TITLE)


def fig_goals2():
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W * 2, 2.3),
                               gridspec_kw={"wspace": 0.42, "width_ratios": [1.1, 1]})
    panel_goals(a)
    panel_paths(b)
    save(fig, "goals2")


# ---------------------------------------------------------------- Fig. 4 (why4)
def unsold_share(arch: str, eps: float) -> float:
    p = build(arch, "Hong Kong", objective="profit", eps=float(eps))
    return 100.0 * solve(p).metrics["waste_units"] / p.x0


def panel_threshold(ax, eps_star):
    # The unsold share depends on the clearance window only, so archetypes pair up.
    for arch, label in (("dairy", "3-day window"), ("poultry", "2-day window")):
        ax.plot(EPS_GRID, [unsold_share(arch, e) for e in EPS_GRID], color=SERIES[arch], lw=1.3,
                label=label)
    ax.axvline(eps_star, color=MUTED, lw=0.8, ls="--")
    ax.axvline(EPS_HAT, color=INK, lw=0.8, ls=":")
    ax.text(1.22, 97, rf"$\varepsilon^*$={eps_star:.2f}" "\n(start to pay)", fontsize=5.8,
            color=MUTED, va="top", ha="right")
    ax.text(1.97, 97, rf"$\hat\varepsilon$={EPS_HAT}" "\n(measured)", fontsize=5.8, color=INK,
            va="top")
    ax.set_xlim(EPS_GRID[0], EPS_GRID[-1])
    ax.set_ylim(0, 100)
    ax.set_xlabel(r"Discount elasticity of demand $\varepsilon$")
    ax.set_ylabel("Stock left unsold (%)")
    ax.legend(fontsize=5.8, loc="center right", bbox_to_anchor=(1.0, 0.52), handlelength=1.2)
    ax.set_title("(a) When markdowns pay", **TITLE)


def panel_alignment(ax, meta):
    eps_star, theta_lo = meta["eps_star"], min(meta["eps_star_theta"].values())
    prof = pd.read_csv(PROFILES_CSV).sort_values(["store_id", "product_id"])
    rng = np.random.default_rng(0)
    xj = prof.eps * np.exp(rng.uniform(-0.02, 0.02, len(prof)))
    ax.axvspan(theta_lo, 1.2 * eps_star, color="#f2efe8", zorder=0)
    ax.scatter(xj, prof.profitopt_share_of_scc_opt_value.clip(lower=86), s=3,
               c=[SERIES[a] for a in prof.archetype], alpha=0.25, lw=0, zorder=2)
    cur = pd.read_csv(THRESHOLD_DIR / "theory_curves.csv")
    for arch, col in SERIES.items():
        g = cur[cur.archetype == arch].sort_values("eps")
        ax.plot(g.eps, g.ratio, color=col, lw=1.1, zorder=3, label=arch)
    ax.axvline(eps_star, color=MUTED, lw=0.8, ls="--", zorder=1)
    ax.axvline(EPS_HAT, color=INK, lw=0.8, ls=":", zorder=1)
    ax.text(eps_star * 0.97, 100.9, rf"$\varepsilon^*$={eps_star:.2f}", fontsize=6.3, color=MUTED,
            ha="right")
    ax.text(EPS_HAT * 1.03, 87.3, rf"$\hat\varepsilon$={EPS_HAT}", fontsize=6.3, color=INK)
    ax.text(theta_lo * 0.95, 94.0, "predicted\nweak zone", fontsize=5.8, color=MUTED,
            ha="right", va="center")
    cats = prof.groupby("first_category_id").agg(eps=("eps", "first"), w=("q0", "sum"))
    ax.scatter(cats.eps, np.full(len(cats), 85.6), marker="|",
               s=40 * np.sqrt(cats.w / cats.w.max()) + 8, color=INK, lw=0.8, zorder=4, clip_on=False)
    ax.set_xscale("log")
    ax.set_xlim(0.25, 7.5)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([0.3, 0.5, 1, 2, 4, 7])
    ax.set_xticklabels(["0.3", "0.5", "1", "2", "4", "7"])
    ax.set_ylim(85.3, 101.6)
    ax.set_xlabel(r"Discount elasticity $\varepsilon$")
    ax.set_ylabel("Profit-max., % of best value")
    ax.legend(fontsize=6, loc="lower left", bbox_to_anchor=(0.0, 0.08), handlelength=1.2, ncol=1)
    ax.set_title("(b) Where profit and climate agree", **TITLE)


def panel_collapse(ax):
    sw = pd.read_csv(ALIGN_DIR / "s1_sweep.csv")
    sw = sw[sw.mu > 0]
    for (arch, _), g in sw.groupby(["archetype", "region"]):
        g = g.sort_values("theta_bar")
        ax.plot(g.theta_bar, g.gap_closed, color=SERIES[arch], lw=0.9, alpha=0.55)
    meta = json.loads((ALIGN_DIR / "summary.json").read_text())
    t = np.geomspace(5e-3, 20, 300)
    k, t50 = meta["hill_k"], meta["hill_t50"]
    ax.plot(t, t ** k / (t50 ** k + t ** k), color=INK, lw=1.4, ls="--")
    ax.axvline(1, color=MUTED, lw=0.7, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(5e-3, 20)
    ax.set_xlabel(r"Alignment index $\bar\theta$")
    ax.set_ylabel("Share of carbon-opt. gain")
    ax.text(0.03, 0.95, f"Hill fit: $\\bar\\theta_{{50}}$={t50:.2f}, $R^2$={meta['hill_r2']:.2f}\n"
            f"Spearman {meta['sweep_spearman_gap_closed']:.2f}, n={meta['sweep_n']}",
            transform=ax.transAxes, va="top", fontsize=6.3, color=MUTED)
    ax.set_title("(c) How far a carbon price moves them", **TITLE)


def fig_why4():
    meta = json.loads((THRESHOLD_DIR / "theory_meta.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(COL_W * 2, 2.45), gridspec_kw={"wspace": 0.5})
    panel_threshold(axes[0], meta["eps_star"])
    panel_alignment(axes[1], meta)
    panel_collapse(axes[2])
    save(fig, "why4")


# ---------------------------------------------------------------- Fig. 5 (policy_maps)
def simulate_paths(p, policy, n=40, seed=3):
    rng = np.random.default_rng(seed)
    paths = np.zeros((n, p.horizon + 1))
    paths[:, 0] = p.x0
    for t, r in enumerate(range(p.horizon, 0, -1)):
        x = paths[:, t].astype(int)
        rates = np.array([demand_rate(p, r, policy[r, xi]) for xi in x])
        paths[:, t + 1] = x - np.minimum(rng.poisson(rates), x)
    return paths


def fig_policy_maps():
    fig, axes = plt.subplots(2, 2, figsize=(COL_W, 2.9), sharey=False)
    for ax, arch in zip(axes.ravel(), SERIES):
        p = build(arch, "Hong Kong", objective="profit")
        pol = solve(p).policy
        im = ax.imshow(100 * pol[1:], aspect="auto", origin="lower", cmap="Blues", vmin=0, vmax=60,
                       extent=[-0.5, p.x0 + 0.5, 0.5, p.horizon + 0.5])
        paths = simulate_paths(p, pol)
        days = np.arange(p.horizon, -1, -1) + 0.0
        days[-1] = 0.55
        for path in paths:
            ax.plot(path, days, color="white", lw=0.5, alpha=0.35)
        ax.plot(np.median(paths, axis=0), days, color="white", lw=1.8,
                path_effects=[pe.Stroke(linewidth=3, foreground=INK), pe.Normal()])
        ax.set_yticks(range(1, p.horizon + 1))
        ax.set_title(arch, fontsize=7.5, color=INK, pad=2)
        ax.grid(False)
        ax.tick_params(labelsize=6.5)
    for ax in axes[1]:
        ax.set_xlabel("Units on hand", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("Days to expiry", fontsize=7)
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    cb = fig.colorbar(im, ax=axes, pad=0.02, shrink=0.9)
    cb.set_label("Discount (%)", fontsize=7)
    save(fig, "policy_maps")


FIGURES = {"teaser": (fig_teaser, [LANDSCAPE_NPZ]),
           "elasticity_col": (fig_elasticity_col, [ELASTICITY_CSV]),
           "goals2": (fig_goals2, [GOALS_CSV, PARETO_CSV, POLICY_CSV]),
           "why4": (fig_why4, [THRESHOLD_DIR / "theory_meta.json", THRESHOLD_DIR / "theory_curves.csv",
                               PROFILES_CSV, ALIGN_DIR / "s1_sweep.csv", ALIGN_DIR / "summary.json"]),
           "policy_maps": (fig_policy_maps, [])}


def main(names):
    for name in names or FIGURES:
        if name not in FIGURES:
            raise SystemExit(f"unknown figure {name!r}; choose from {sorted(FIGURES)}")
        fn, needs = FIGURES[name]
        missing = [str(f.relative_to(OUTPUTS.parent)) for f in needs if not f.exists()]
        if missing:
            print(f"skip {name}: missing {', '.join(missing)} (see README for the script to run)")
            continue
        fn()
        print(f"wrote outputs/figures/{name}.pdf")


if __name__ == "__main__":
    main(sys.argv[1:])

"""Figures of the paper, written to outputs/figures/{name}.pdf (+ .png preview).

    teaser3  Fig. 1  what a sale replaces, who buys, overstatement map  (no inputs needed)
    goals    Fig. 2  five policies scored at the SCC, 20 cases   <- outputs/main/goals.csv
    why3     Fig. 3  where profit and climate agree; carbon price vs. alignment index
                     <- outputs/threshold/{theory_meta.json,theory_curves.csv},
                        outputs/profiles/profile_results_with_shares.csv,
                        outputs/alignment/{s1_sweep.csv,summary.json}

Usage: python scripts/make_paper_figures.py [name ...]   (default: all three)
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.plot_style import COL_W, INK, MUTED, OUTPUTS, SERIES, save  # noqa: E402
from tblca.params import ARCHETYPES, REGIONS  # noqa: E402
from tblca.scenario import DEFAULTS  # noqa: E402
from tblca.substitution import freshness_gate, rebound_gate, spoilage_gate  # noqa: E402

EPS, KAPPA = DEFAULTS["eps"], DEFAULTS["kappa"]
EPS_HAT = 1.94
GOALS_CSV = OUTPUTS / "main" / "goals.csv"
THRESHOLD_DIR = OUTPUTS / "threshold"
PROFILES_CSV = OUTPUTS / "profiles" / "profile_results_with_shares.csv"
ALIGN_DIR = OUTPUTS / "alignment"
BUYERS = [  # (label, explanation, colour, replaces a purchase?)
    ("Regular buyer", "would pay full price", SERIES["dairy"], True),
    ("Switcher", "else buys a fresh unit", SERIES["produce"], True),
    ("New shopper", "buys only for the discount", SERIES["bakery"], False),
]
GOALS = [("profit", "Profit-maximizing"), ("weighted", "Carbon priced at SCC"),
         ("carbon", "Carbon-only"), ("static", "Static last-day 35%"), ("waste", "Waste-minimizing")]


# ---------------------------------------------------------------- Fig. 1 (teaser3)
def panel_schematic(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.15, 4.0), 2.3, 2.0, boxstyle="round,pad=0.12", fc="#ffe14d",
                                ec=INK, lw=0.8))
    ax.text(1.3, 5.45, "REDUCED", ha="center", va="center", fontsize=5.6, weight="bold", color=INK)
    ax.text(1.3, 4.6, "1 day left\n35% off", ha="center", va="center", fontsize=5.8, color=INK)
    for i, (who, why, col, displaces) in enumerate(BUYERS):
        y = 8.4 - 3.4 * i
        ax.add_patch(FancyArrowPatch((2.65, 5.0), (3.45, y), arrowstyle="-|>", mutation_scale=7,
                                     color=MUTED, lw=0.7))
        ax.add_patch(FancyBboxPatch((3.55, y - 1.4), 6.4, 2.8, boxstyle="round,pad=0.06",
                                    fc=col, ec="none", alpha=0.2))
        ax.text(3.75, y + 0.75, who, fontsize=6.6, weight="bold", color=INK, va="center")
        ax.text(3.75, y - 0.05, why, fontsize=5.6, color=MUTED, va="center")
        verdict = "→ replaces a purchase" if displaces else "→ replaces nothing"
        ax.text(3.75, y - 0.85, verdict, fontsize=5.8, va="center", weight="bold",
                color=INK if displaces else SERIES["bakery"])
    ax.set_title("(a) What a sale replaces", fontsize=8, color=INK, loc="left")


def panel_buyers(ax):
    d = np.linspace(0, 0.6, 200)
    inc = (1 - d) ** (-EPS) - 1
    layers = [np.ones_like(d), KAPPA * inc, (1 - KAPPA) * inc]
    cols = [SERIES["dairy"], SERIES["produce"], SERIES["bakery"]]
    ax.stackplot(100 * d, *layers, colors=cols, alpha=0.9, edgecolor="white", linewidth=0.6)
    ax.text(3, 0.5, "regular buyers", color="white", fontsize=6.3, va="center")
    ax.text(3, 3.4, "switchers ($\\kappa$)", color=SERIES["produce"], fontsize=6.3, weight="bold")
    ax.text(3, 5.2, "new shoppers", color=SERIES["bakery"], fontsize=6.3, weight="bold")
    ax.set_xlabel("Discount depth $d$ (%)")
    ax.set_ylabel("Sales / full-price demand")
    ax.set_xlim(0, 60)
    ax.set_title("(b) Who buys at a discount", fontsize=8, color=INK, loc="left")


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
    ax.set_title("(c) Overstatement $\\Omega$ of crediting cleared units", fontsize=8, color=INK,
                 loc="left")
    return pcm


def fig_teaser3():
    fig, axes = plt.subplots(1, 3, figsize=(COL_W * 2, 2.15),
                             gridspec_kw={"width_ratios": [1.05, 1, 1.1], "wspace": 0.38})
    panel_schematic(axes[0])
    panel_buyers(axes[1])
    pcm = panel_overstatement(axes[2])
    cb = fig.colorbar(pcm, ax=axes[2], fraction=0.05, pad=0.03)
    cb.set_label(r"$\Omega$", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    save(fig, "teaser3")


# ---------------------------------------------------------------- Fig. 2 (goals)
def fig_goals():
    d = pd.read_csv(GOALS_CSV)
    fig, ax = plt.subplots(figsize=(COL_W, 1.85))
    rng = np.random.default_rng(1)
    for i, (key, label) in enumerate(GOALS):
        g = d[d.policy == key]
        y = -i + rng.uniform(-0.17, 0.17, len(g))
        ax.scatter(g.pct_best, y, s=9, c=[SERIES[a] for a in g.archetype], alpha=0.85, lw=0, zorder=3)
        lo, hi = g.pct_best.min(), g.pct_best.max()
        ax.plot([lo, hi], [-i, -i], color=MUTED, lw=0.6, zorder=2)
        txt = "100% (by definition)" if key == "weighted" else f"{lo:.1f}–{hi:.1f}%"
        ax.text(48.5, -i, txt, fontsize=6.3, va="center", ha="left",
                color=INK if key == "profit" else MUTED, weight="bold" if key == "profit" else "normal")
    ax.set_yticks([-i for i in range(len(GOALS))])
    ax.set_yticklabels([lab for _, lab in GOALS], fontsize=7)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xlim(47, 101.5)
    ax.set_ylim(-len(GOALS) + 0.5, 0.5)
    ax.set_xlabel("Value at the SCC, % of the best attainable (profit + priced CO$_2$e)")
    ax.grid(axis="y", visible=False)
    h = [plt.Line2D([], [], ls="", marker="o", ms=3.5, color=c, label=a) for a, c in SERIES.items()]
    ax.legend(handles=h, fontsize=6, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4,
              handletextpad=0.1, columnspacing=0.8)
    save(fig, "goals")


# ---------------------------------------------------------------- Fig. 3 (why3)
def panel_alignment(ax):
    meta = json.loads((THRESHOLD_DIR / "theory_meta.json").read_text())
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
    ax.text(EPS_HAT * 1.03, 87.3, rf"$\hat\varepsilon$={EPS_HAT} (measured)", fontsize=6.3, color=INK)
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
    ax.set_xlabel(r"Discount elasticity $\varepsilon$ (ticks: product categories)")
    ax.set_ylabel("Profit-max. policy, % of best SCC value")
    ax.legend(fontsize=6, loc="lower left", bbox_to_anchor=(0.0, 0.08), handlelength=1.2, ncol=1)
    ax.set_title("(a) Where profit and climate agree", fontsize=8, color=INK, loc="left")


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
    ax.set_ylabel("Share of carbon-opt. gain captured")
    ax.text(0.03, 0.95, f"Hill fit: $\\bar\\theta_{{50}}$={t50:.2f}, $R^2$={meta['hill_r2']:.2f}\n"
            f"Spearman {meta['sweep_spearman_gap_closed']:.2f}, n={meta['sweep_n']}",
            transform=ax.transAxes, va="top", fontsize=6.3, color=MUTED)
    ax.set_title("(b) How far a carbon price moves the discounts", fontsize=8, color=INK, loc="left")


def fig_why3():
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W * 2, 2.45), gridspec_kw={"wspace": 0.3})
    panel_alignment(a)
    panel_collapse(b)
    save(fig, "why3")


FIGURES = {"teaser3": (fig_teaser3, []),
           "goals": (fig_goals, [GOALS_CSV]),
           "why3": (fig_why3, [THRESHOLD_DIR / "theory_meta.json", THRESHOLD_DIR / "theory_curves.csv",
                               PROFILES_CSV, ALIGN_DIR / "s1_sweep.csv", ALIGN_DIR / "summary.json"])}


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

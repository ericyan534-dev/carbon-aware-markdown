"""Figures of the paper, written to outputs/figures/{name}.pdf (+ .png preview).

    teaser2         Fig. 1  buyer composition and overstatement map (no inputs needed)
    elasticity_col  Fig. 2  category elasticities   <- outputs/elasticity/elasticity_by_category.csv
    frontier        Fig. 3  20 product-region cases <- outputs/main/policy_comparison.csv
    why             Fig. 4  threshold + collapse    <- outputs/alignment/{s1_sweep.csv,summary.json}
    policy_maps     Fig. 5  policy maps (no inputs needed)

Usage: python scripts/make_paper_figures.py [name ...]   (default: all five)
"""
import json
import sys
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.plot_style import COL_W, INK, MUTED, OUTPUTS, SERIES, save  # noqa: E402
from tblca.markdown import demand_rate, solve  # noqa: E402
from tblca.params import ARCHETYPES, REGIONS  # noqa: E402
from tblca.scenario import DEFAULTS, build  # noqa: E402
from tblca.substitution import freshness_gate, rebound_gate, spoilage_gate  # noqa: E402

EPS, KAPPA = DEFAULTS["eps"], DEFAULTS["kappa"]
EPS_HAT = 1.94
MIN_OBS = 30000          # categories shown in the elasticity figure
EPS_GRID = np.round(np.arange(0.8, 3.01, 0.05), 3)
REGION_MARK = {"Hong Kong": "o", "Guangdong": "s", "UK": "^", "US": "D", "EU": "v"}
ELASTICITY_CSV = OUTPUTS / "elasticity" / "elasticity_by_category.csv"
POLICY_CSV = OUTPUTS / "main" / "policy_comparison.csv"
ALIGN_DIR = OUTPUTS / "alignment"


# ---------------------------------------------------------------- Fig. 1 (teaser2)
def panel_composition(ax):
    d = np.linspace(0, 0.6, 200)
    inc = (1 - d) ** (-EPS) - 1
    layers = [np.ones_like(d), KAPPA * inc, (1 - KAPPA) * inc]
    cols = [SERIES["dairy"], SERIES["produce"], SERIES["bakery"]]
    ax.stackplot(100 * d, *layers, colors=cols, alpha=0.9, edgecolor="white", linewidth=0.6)
    ax.text(4, 0.5, "regular buyers (replace a purchase)", color="white", fontsize=6.3, va="center")
    ax.annotate("switchers, share $\\kappa$\n(replace a fresh purchase)", (40, 1.9), xytext=(3, 3.0),
                fontsize=6.3, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.annotate("new shoppers (replace nothing)", (56, 4.3), xytext=(3, 5.2), fontsize=6.3,
                color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set_xlabel("Discount depth $d$ (%)")
    ax.set_ylabel("Sales / full-price demand")
    ax.set_xlim(0, 60)
    ax.set_title("(a) Buyers of a discounted unit", fontsize=8, color=INK, loc="left")


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
    ax.set_title("(b) Overstatement of cleared-unit crediting $\\Omega$", fontsize=8, color=INK,
                 loc="left")
    return pcm


def fig_teaser2():
    fig, axes = plt.subplots(1, 2, figsize=(COL_W * 2, 2.35), gridspec_kw={"wspace": 0.35})
    panel_composition(axes[0])
    p1 = panel_overstatement(axes[1])
    fig.colorbar(p1, ax=axes[1], pad=0.02, ticks=[1, 2, 3, 4, 5, 6]).set_label("$\\Omega$", fontsize=7)
    save(fig, "teaser2")


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


# ---------------------------------------------------------------- Fig. 3 (frontier)
def fig_frontier():
    pc = pd.read_csv(POLICY_CSV)
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    for (arch, reg), sub in pc.groupby(["archetype", "region"]):
        s = sub.set_index("policy")
        base = s.loc["Waste-optimal"]
        for pol, filled in (("SCC-weighted (ours)", True), ("Profit-optimal", False)):
            x = 100 * s.loc[pol, "carbon_kg"] / base["carbon_kg"]
            y = 100 * (s.loc[pol, "profit"] / base["profit"] - 1)
            ax.plot(x, y, marker=REGION_MARK[reg], ls="", ms=5, mew=0.9,
                    color=SERIES[arch] if filled else "none", mec=SERIES[arch])
    ax.axvline(100, color=MUTED, lw=0.6)
    ax.set_xlabel("CO$_2$e retained vs. waste-min. (%)")
    ax.set_ylabel("Profit gain vs. waste-min. (%)")
    reg_h = [plt.Line2D([], [], marker=m, ls="", color=MUTED, label=r, ms=4.5)
             for r, m in REGION_MARK.items()]
    arch_h = [plt.Line2D([], [], marker="s", ls="", color=c, label=a, ms=4.5)
              for a, c in SERIES.items()]
    ax.legend(handles=arch_h + reg_h, fontsize=5.8, ncol=2, loc="center left",
              bbox_to_anchor=(0.30, 0.45), handletextpad=0.2, columnspacing=0.5, labelspacing=0.25)
    ax.text(0.98, 0.97, "filled: carbon-priced\nhollow: profit-optimal", transform=ax.transAxes,
            ha="right", va="top", fontsize=6, color=MUTED)
    save(fig, "frontier")


# ---------------------------------------------------------------- Fig. 4 (why)
def eps_star(kappa: float, margin_ratio: float) -> float:
    """Elasticity above which a small discount raises the expected margin (Proposition 1)."""
    return 1.0 / (1.0 - kappa * margin_ratio)


def unsold_share(arch: str, eps: float) -> float:
    p = build(arch, "Hong Kong", objective="profit", eps=float(eps))
    return 100.0 * solve(p).metrics["waste_units"] / p.x0


def panel_threshold(ax):
    # The unsold share depends on the clearance window only, so archetypes pair up.
    for arch, label in (("dairy", "3-day window (dairy, produce)"),
                        ("poultry", "2-day window (bakery, poultry)")):
        ax.plot(EPS_GRID, [unsold_share(arch, e) for e in EPS_GRID], color=SERIES[arch], lw=1.3,
                label=label)
    e_star = eps_star(DEFAULTS["kappa"], DEFAULTS["margin_ratio"])
    ax.axvline(e_star, color=MUTED, lw=0.8, ls="--")
    ax.axvline(EPS_HAT, color=INK, lw=0.8, ls=":")
    ax.text(e_star + 0.04, 92, rf"$\varepsilon^*$={e_star:.2f}" "\n(discounts\nstart to pay)",
            fontsize=6.3, color=MUTED, va="top")
    ax.text(EPS_HAT + 0.04, 92, rf"$\hat\varepsilon$={EPS_HAT}" "\n(measured)", fontsize=6.3,
            color=INK, va="top")
    ax.set_xlim(EPS_GRID[0], EPS_GRID[-1])
    ax.set_ylim(0, 100)
    ax.set_xlabel(r"Discount elasticity of demand $\varepsilon$")
    ax.set_ylabel("Stock left unsold (%)")
    ax.legend(fontsize=6, loc="center right", bbox_to_anchor=(1.0, 0.45), handlelength=1.4)
    ax.set_title("(a) When markdowns pay", fontsize=8, color=INK, loc="left")


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
    ax.set_title("(b) How far a carbon price moves them", fontsize=8, color=INK, loc="left")


def discount_onset(arch: str) -> float:
    """Smallest elasticity (step 0.01) at which the profit-maximizing policy discounts at all."""
    for eps in np.round(np.arange(1.0, 2.0, 0.01), 2):
        if solve(build(arch, "Hong Kong", objective="profit", eps=float(eps))).metrics["disc_units"] > 0:
            return float(eps)
    return float("nan")


def fig_why():
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W * 2, 2.4), gridspec_kw={"wspace": 0.3})
    panel_threshold(a)
    panel_collapse(b)
    save(fig, "why")
    onset = {arch: discount_onset(arch) for arch in ARCHETYPES}
    stats = {"eps_star": eps_star(DEFAULTS["kappa"], DEFAULTS["margin_ratio"]),
             "profit_policy_discount_onset_eps": onset}
    (OUTPUTS / "figures" / "why_threshold.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


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


FIGURES = {"teaser2": (fig_teaser2, []),
           "elasticity_col": (fig_elasticity_col, [ELASTICITY_CSV]),
           "frontier": (fig_frontier, [POLICY_CSV]),
           "why": (fig_why, [ALIGN_DIR / "s1_sweep.csv", ALIGN_DIR / "summary.json"]),
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

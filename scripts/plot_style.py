"""Shared matplotlib style (IEEE column width) and output locations for figures."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
FIG_DIR = OUTPUTS / "figures"

COL_W = 3.5
SERIES = {"dairy": "#2a78d6", "bakery": "#eb6834", "produce": "#1baf7a", "poultry": "#eda100"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "pdf.fonttype": 42})


def save(fig, name: str, out_dir: Path = FIG_DIR) -> None:
    """Write `name`.pdf and a 200-dpi `name`.png preview to `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

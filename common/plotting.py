"""One consistent chart style for every figure (validated categorical palette, recessive axes)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.labelsize": 10, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2.0, "font.family": "DejaVu Sans",
        "axes.prop_cycle": matplotlib.cycler(color=SERIES), "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def threshold_line(ax, x, label, color=INK2, y_text=0.97, vertical=True):
    if vertical:
        ax.axvline(x, color=color, lw=1.2, ls="--")
        ax.text(x, y_text, f" {label}", transform=ax.get_xaxis_transform(), color=INK2, fontsize=8.5,
                va="top", ha="left")
    else:
        ax.axhline(x, color=color, lw=1.2, ls="--")

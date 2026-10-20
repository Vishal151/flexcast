"""Shared chart style for the notebooks (presentation only; not part of the library).

Palette: the dataviz reference instance, categorical slots in fixed order. Light surface, thin
recessive grid, 2 px lines, one y-axis per chart.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SERIES = [BLUE, ORANGE, AQUA, YELLOW]
SEQUENTIAL = "Blues"  # one hue, light -> dark, for magnitude


def use() -> None:
    mpl.rcParams.update(
        {
            "figure.figsize": (9, 4),
            "figure.dpi": 110,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_2,
            "axes.titlecolor": TEXT,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.titlelocation": "left",
            "axes.grid": True,
            "axes.axisbelow": True,  # grid behind the data
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "lines.linewidth": 2,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "legend.frameon": False,
            "legend.labelcolor": TEXT,
            "savefig.facecolor": SURFACE,
        }
    )


def label_end(ax: plt.Axes, x: object, y: float, text: str, color: str) -> None:
    """Direct label at a line's end, in text ink with a colour swatch beside it."""
    ax.annotate(
        f"■ {text}",
        (x, y),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        color=TEXT,
        fontsize=9,
    )
    ax.annotate(
        "■", (x, y), xytext=(6, 0), textcoords="offset points", va="center", color=color, fontsize=9
    )

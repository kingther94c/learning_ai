"""One palette, shared by the notebooks and the interactive pages.

The hex values here are identical to the ones in `interactive/shared/learnai.css`,
so a matplotlib figure in a notebook and an HTML page in the browser are visibly
the same system. If you change a color, change it in both places.

The categorical order is fixed and validated for colorblind separation — assign
slots in order (SERIES[0], SERIES[1], ...) and never cycle past 8. Sequential
magnitude (attention weights, similarity matrices) uses the one-hue blue ramp.
"""

from __future__ import annotations

# Categorical — identity. Fixed order; the ordering IS the accessibility mechanism.
SERIES = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

# Sequential — magnitude. One hue, light to dark. This is the attention-map ramp.
BLUE_RAMP = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]

# Diverging — polarity (e.g. logit differences, weight deltas). Neutral gray middle.
DIVERGING = ("#2a78d6", "#f0efec", "#d03b3b")  # blue <- neutral -> red

# Chrome
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}


def sequential_cmap(name: str = "learnai_blue"):
    """The blue ramp as a matplotlib colormap, for attention/similarity heatmaps."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, BLUE_RAMP)


def diverging_cmap(name: str = "learnai_div"):
    """Blue-neutral-red, for anything signed. Always center the norm at zero."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, list(DIVERGING))


def use_style() -> None:
    """Apply the house matplotlib style. Call once per notebook, after imports.

    Recessive chrome, no top/right spines, hairline grid — so the data is the
    only thing with contrast.
    """
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "600",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "text.color": INK,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "figure.dpi": 110,
            "savefig.bbox": "tight",
        }
    )

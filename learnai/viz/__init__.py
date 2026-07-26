"""Plotting helpers and the shared palette."""

from learnai.viz.plots import (  # noqa: F401
    attention_grid,
    attention_heatmap,
    plot_decision_boundary,
    plot_embeddings,
    plot_pe_matrix,
    plot_similarity,
)
from learnai.viz.theme import (  # noqa: F401
    BLUE_RAMP,
    DIVERGING,
    SERIES,
    diverging_cmap,
    sequential_cmap,
    use_style,
)

__all__ = [
    "BLUE_RAMP",
    "DIVERGING",
    "SERIES",
    "attention_grid",
    "attention_heatmap",
    "diverging_cmap",
    "plot_decision_boundary",
    "plot_embeddings",
    "plot_pe_matrix",
    "plot_similarity",
    "sequential_cmap",
    "use_style",
]

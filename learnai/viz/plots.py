"""Plots you will want over and over.

Each one answers a specific question:

    attention_heatmap    "what is this head looking at?"
    attention_grid       "what are ALL the heads looking at?" — the fastest way
                         to spot a head with a recognisable job
    plot_decision_boundary  "what function did this classifier actually learn?"
    plot_embeddings      "did similar things end up near each other?"
    plot_pe_matrix       "what does a position encoding look like?"
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from learnai.viz.theme import INK_MUTED, SERIES, diverging_cmap, sequential_cmap


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def attention_heatmap(
    attn,
    tokens: Sequence[str] | None = None,
    ax=None,
    title: str | None = None,
    show_values: bool = False,
    cbar: bool = True,
):
    """One (T, T) attention map. Row i = where token i sent its attention.

    Read it row-wise: each row is a probability distribution summing to 1. A
    causal model gives a lower-triangular map. Bright columns mean "everyone
    looks at this token" — usually a delimiter or the first token, which acts as
    an attention sink where heads park when they have nothing to do.
    """
    import matplotlib.pyplot as plt

    a = _np(attn)
    if a.ndim != 2:
        raise ValueError(f"expected a (T, T) map, got {a.shape}; index a head first")

    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5))

    im = ax.imshow(a, cmap=sequential_cmap(), vmin=0, vmax=max(a.max(), 1e-8), aspect="equal")

    if tokens is not None:
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=7)
        ax.set_yticklabels(tokens, fontsize=7)
    ax.set_xlabel("attending to (key)")
    ax.set_ylabel("attending from (query)")
    if title:
        ax.set_title(title)
    ax.grid(False)

    if show_values:
        # Only readable for short sequences; flip text color on dark cells.
        thresh = a.max() * 0.55
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if a[i, j] > 0.01:
                    ax.text(
                        j,
                        i,
                        f"{a[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if a[i, j] > thresh else "#0b0b0b",
                    )
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="weight")
    return ax


def attention_grid(
    attn,
    tokens: Sequence[str] | None = None,
    max_heads: int = 12,
    title: str | None = None,
):
    """All heads of one layer as small multiples: attn is (H, T, T).

    Scanning this grid is how you find interesting heads. Recurring patterns:
        diagonal        attends to self
        sub-diagonal    attends to the previous token ("previous-token head")
        first column    attention sink / no-op
        vertical stripe fixates on one content token
        anti-diagonal   reversal (shows up on the reverse toy task)
    """
    import matplotlib.pyplot as plt

    a = _np(attn)
    if a.ndim != 3:
        raise ValueError(f"expected (H, T, T), got {a.shape}")

    n = min(a.shape[0], max_heads)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.7 * rows), squeeze=False)

    for h in range(rows * cols):
        ax = axes[h // cols][h % cols]
        if h >= n:
            ax.axis("off")
            continue
        ax.imshow(a[h], cmap=sequential_cmap(), vmin=0, vmax=a[h].max() or 1, aspect="equal")
        ax.set_title(f"head {h}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    if title:
        fig.suptitle(title, fontsize=11, x=0.02, ha="left", weight="600")
    fig.tight_layout()
    return fig


def plot_decision_boundary(model_fn, X, y, ax=None, resolution: int = 200, title=None):
    """Shade the predicted class over a 2D grid, then scatter the data on top.

    `model_fn` takes (N, 2) and returns (N,) predicted labels. This works with
    sklearn (`clf.predict`) or a torch model wrapped in a lambda.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    X = _np(X)
    y = _np(y)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    pad = 0.25
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
    )
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    Z = _np(model_fn(grid)).reshape(xx.shape)

    n_classes = int(max(y.max(), Z.max())) + 1
    cmap = ListedColormap(SERIES[:n_classes])
    ax.contourf(xx, yy, Z, alpha=0.18, cmap=cmap, levels=np.arange(n_classes + 1) - 0.5)
    ax.contour(xx, yy, Z, colors="white", linewidths=1.2, levels=np.arange(n_classes + 1) - 0.5)

    for c in range(n_classes):
        m = y == c
        if m.any():
            # 2px surface ring on each point keeps overlapping dots readable.
            ax.scatter(
                X[m, 0],
                X[m, 1],
                c=SERIES[c],
                s=22,
                label=f"class {c}",
                edgecolors="#fcfcfb",
                linewidths=1.0,
                zorder=3,
            )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc="upper right")
    ax.grid(False)
    if title:
        ax.set_title(title)
    return ax


def plot_embeddings(coords, labels=None, ax=None, title=None, annotate: bool = True):
    """Scatter 2D-projected vectors, optionally labelled with their token text."""
    import matplotlib.pyplot as plt

    c = _np(coords)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(c[:, 0], c[:, 1], s=26, c=SERIES[0], edgecolors="#fcfcfb", linewidths=1.0, zorder=3)
    if labels is not None and annotate:
        for (x, y_), lab in zip(c, labels, strict=True):
            ax.annotate(
                str(lab),
                (x, y_),
                fontsize=7,
                color=INK_MUTED,
                xytext=(3, 3),
                textcoords="offset points",
            )
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    if title:
        ax.set_title(title)
    return ax


def plot_pe_matrix(pe, ax=None, title="position encoding", max_len: int = 100):
    """Heatmap of a (L, D) position-encoding matrix — position down, dimension across.

    For sinusoidal encodings you get the famous banded pattern: fast-oscillating
    low dimensions on the left, slow ones on the right. For a *learned* table
    trained on real text, you instead see smooth structure emerge only in the
    positions the model actually saw often.
    """
    import matplotlib.pyplot as plt

    p = _np(pe)
    p = p.squeeze()[:max_len]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    lim = np.abs(p).max()
    im = ax.imshow(p, cmap=diverging_cmap(), aspect="auto", vmin=-lim, vmax=lim)
    ax.set_xlabel("embedding dimension")
    ax.set_ylabel("position")
    ax.set_title(title)
    ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    return ax


def plot_similarity(vectors, labels=None, ax=None, title="cosine similarity"):
    """Pairwise cosine similarity of (N, D) vectors as a diverging heatmap.

    Useful on position embeddings (a trained model shows a bright diagonal band —
    nearby positions are similar) and on token embeddings (semantic clusters).
    """
    import matplotlib.pyplot as plt

    v = _np(vectors).astype(np.float64)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    sim = v @ v.T

    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(sim, cmap=diverging_cmap(), vmin=-1, vmax=1)
    if labels is not None:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title)
    ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax

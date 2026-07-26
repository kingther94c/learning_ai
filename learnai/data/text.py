"""Text corpora, downloaded once and cached under `data/cache/`.

Everything here is small on purpose. The point is a training run that finishes
while you are still looking at it — you learn far more from twenty 60-second
runs than from one overnight run.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path

import torch

from learnai.env import CACHE_DIR

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

# Enough text to exercise the whole pipeline when the network is unavailable.
_FALLBACK = (
    """First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.
"""
    * 40
)


def download(url: str, filename: str, force: bool = False) -> Path:
    """Fetch `url` into the cache, returning the local path. No-op if present."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / filename
    offline = os.environ.get("LEARNAI_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}
    if offline:
        if path.exists():
            return path
        raise OSError(
            f"LEARNAI_OFFLINE is set and {path.name} is not cached; "
            "the caller should use its built-in fallback"
        )
    if path.exists() and not force:
        return path
    print(f"downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)
    return path


def tiny_shakespeare(force: bool = False) -> str:
    """~1.1MB of Shakespeare — the standard character-level benchmark.

    Small enough for a laptop, structured enough that you can *hear* the model
    improve: first spaces, then word shapes, then speaker names in ALL CAPS
    followed by a colon, then something like iambic rhythm.
    """
    try:
        path = download(TINY_SHAKESPEARE_URL, "tinyshakespeare.txt", force=force)
        return path.read_text(encoding="utf-8")
    except (urllib.error.URLError, OSError) as e:
        print(f"download failed ({e}); using the small built-in fallback corpus")
        return _FALLBACK


def train_val_split(ids: list[int] | torch.Tensor, val_frac: float = 0.1):
    """Split a token stream by position — NOT at random.

    Random splitting leaks: adjacent windows would land on both sides and the
    model would effectively see its validation set. For sequence data the split
    must be contiguous.
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be between 0 and 1, got {val_frac}")
    data = torch.tensor(ids, dtype=torch.long) if not torch.is_tensor(ids) else ids
    if data.ndim != 1:
        raise ValueError(f"expected a 1-D token stream, got shape {tuple(data.shape)}")
    if len(data) < 2:
        raise ValueError("need at least two tokens to make non-empty train and validation splits")
    n = int(len(data) * (1 - val_frac))
    n = min(max(n, 1), len(data) - 1)
    return data[:n], data[n:]


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device=None, generator=None):
    """Sample `batch_size` random windows of length `block_size`.

    Returns (x, y) each (B, T), where y is x shifted one position left: the
    target at position t is the token that actually followed. Sampling windows at
    random offsets — rather than iterating an epoch — is what nanoGPT does, and
    it means every window boundary is different on every pass.
    """
    if data.ndim != 1:
        raise ValueError(f"expected a 1-D token stream, got shape {tuple(data.shape)}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    # A window of T inputs also needs the following token for its T targets, so
    # a stream of exactly T+1 tokens has one valid start (zero). `randint`'s high
    # bound is exclusive: high = len(data)-T includes the final valid window.
    n_starts = len(data) - block_size
    if n_starts <= 0:
        raise ValueError(
            f"data has {len(data)} tokens, but block_size={block_size} needs at least "
            f"{block_size + 1}"
        )
    ix = torch.randint(n_starts, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])  # (B, T)
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])  # (B, T) shifted by one
    if device is not None:
        x, y = x.to(device), y.to(device)
    return x, y

"""Synthetic tasks with known solutions.

The reason to train on toy tasks: **you know what the right algorithm is**, so
when the model succeeds you can go looking for that algorithm in the weights.
Real corpora never give you that. Nearly every mechanistic-interpretability
result started on a task from this family.

    copy       output = input. Trivial for attention, impossible for a bag of words.
    reverse    needs position-dependent routing, not just content matching.
    sort       needs comparison between arbitrary pairs.
    induction  "...[A][B]...[A]" -> predict [B]. THE canonical circuit: two heads
               composing, one to find the earlier A, one to copy what followed.
    parity     XOR over a sequence — provably hard for a fixed-depth transformer,
               a useful demonstration of an architectural limit.
"""

from __future__ import annotations

import numpy as np
import torch


def copy_task(n_samples: int, seq_len: int = 10, vocab_size: int = 20, seed: int | None = None):
    """x: (N, T) random tokens.  y: (N, T) the same tokens.

    Solvable by a single attention head that attends to its own position — the
    "identity head". Good first sanity check that a training loop works at all.
    """
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    x = torch.randint(0, vocab_size, (n_samples, seq_len), generator=g)
    return x, x.clone()


def reverse_task(n_samples: int, seq_len: int = 10, vocab_size: int = 20, seed: int | None = None):
    """y = x reversed. Requires attending to position T-1-i, so the model must use
    position embeddings rather than content. Watch the attention map: a trained
    model shows a clean anti-diagonal."""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    x = torch.randint(0, vocab_size, (n_samples, seq_len), generator=g)
    return x, x.flip(dims=[1])


def sort_task(n_samples: int, seq_len: int = 10, vocab_size: int = 20, seed: int | None = None):
    """y = x sorted ascending. Content-based routing: each output position must
    find the k-th smallest element, which no fixed positional pattern gives you."""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    x = torch.randint(0, vocab_size, (n_samples, seq_len), generator=g)
    return x, x.sort(dim=1).values


def induction_task(
    n_samples: int,
    seq_len: int = 32,
    vocab_size: int = 48,
    seed: int | None = None,
):
    """Random tokens with one planted repeat: ... A B ... A -> the answer is B.

    Returns (x, y, positions) where `positions` marks the index whose *next*
    token is the predictable one, so you can measure accuracy on exactly the
    position that requires the circuit rather than on the random filler.

    Why this task matters: induction heads are the first non-trivial circuit ever
    fully reverse-engineered (Olsson et al., "In-context Learning and Induction
    Heads"), and their formation coincides with a visible bump in the training
    loss curve. It takes two layers — one head marks "the token after the
    previous A", a second head copies it forward. A one-layer model cannot do it,
    which makes it a sharp architectural experiment.
    """
    if vocab_size <= seq_len:
        raise ValueError(
            f"vocab_size ({vocab_size}) must exceed seq_len ({seq_len}) so that a "
            "token absent from the filler always exists to use as the cue"
        )

    rng = np.random.default_rng(seed)
    xs, ys, positions = [], [], []
    for _ in range(n_samples):
        seq = rng.integers(0, vocab_size, size=seq_len)
        a_pos = int(rng.integers(1, seq_len // 2 - 1))
        repeat_pos = int(rng.integers(seq_len // 2 + 1, seq_len - 1))

        # The cue A must occur ONLY at a_pos and repeat_pos. If it also turned up
        # in the random filler with some other token after it, the task would be
        # ambiguous — two different "correct" answers — and accuracy at
        # `repeat_pos` would stop measuring the induction circuit.
        unused = np.setdiff1d(np.arange(vocab_size), seq)
        a_tok = int(rng.choice(unused))

        b_tok = int(seq[a_pos + 1])  # whatever happened to follow that slot
        seq[a_pos] = a_tok
        seq[repeat_pos] = a_tok
        seq[repeat_pos + 1] = b_tok

        xs.append(seq[:-1])
        ys.append(seq[1:])
        positions.append(repeat_pos)  # predicting from here requires the circuit
    return (
        torch.tensor(np.array(xs), dtype=torch.long),
        torch.tensor(np.array(ys), dtype=torch.long),
        torch.tensor(positions, dtype=torch.long),
    )


def parity_task(n_samples: int, seq_len: int = 16, seed: int | None = None):
    """x: random bits. y: running XOR (parity of the prefix).

    A known hard case. Parity needs a serial scan, and a transformer's depth is
    fixed regardless of sequence length, so it cannot implement one in general —
    it memorises instead, and generalisation to longer sequences fails. Train it,
    then test on 2x the length to see the wall.
    """
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    x = torch.randint(0, 2, (n_samples, seq_len), generator=g)
    return x, x.cumsum(dim=1) % 2


# --------------------------------------------------------------------------------------
# Classic-ML datasets: 2D so every decision boundary can simply be drawn
# --------------------------------------------------------------------------------------


def make_spirals(n_samples: int = 300, n_classes: int = 3, noise: float = 0.2, seed: int = 0):
    """Interleaved spirals — the standard "linear models cannot do this" demo.

    Returns X (N, 2), y (N,). A logistic regression gets ~33%; one hidden layer
    with enough units gets ~99%. Plot the boundary and the reason is obvious.
    """
    rng = np.random.default_rng(seed)
    per = n_samples // n_classes
    X = np.zeros((per * n_classes, 2), dtype=np.float32)
    y = np.zeros(per * n_classes, dtype=np.int64)
    for c in range(n_classes):
        idx = slice(per * c, per * (c + 1))
        r = np.linspace(0.0, 1.0, per)
        t = np.linspace(c * 4, (c + 1) * 4, per) + rng.normal(0, noise, per)
        X[idx] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[idx] = c
    return X, y


def make_xor(n_samples: int = 400, noise: float = 0.15, seed: int = 0):
    """The four-quadrant XOR. Not linearly separable — the 1969 objection to
    perceptrons, and the reason hidden layers exist."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n_samples, 2)).astype(np.float32)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.int64)
    X += rng.normal(0, noise, X.shape).astype(np.float32)
    return X, y

"""The Skip-gram with Negative Sampling (SGNS) core of word2vec.

The original 2013 word2vec work is easiest to understand as two pieces:

1. Skip-gram turns a token sequence into ``center -> nearby context`` pairs.
2. Negative sampling replaces a vocabulary-sized softmax with a small binary
   classification problem for every positive pair.

This module keeps those pieces separate so a lesson can inspect the data and
the objective independently.  It deliberately does not hide training behind a
high-level embedding package.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def skipgram_pairs(
    token_ids: Sequence[int] | torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return center/context IDs for one token sequence.

    For each position ``i``, every token within ``window_size`` positions is a
    context target except the token at ``i`` itself.  Call the function once per
    sentence when sentence boundaries should not create training pairs.
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    ids = torch.as_tensor(token_ids, dtype=torch.long)
    if ids.ndim != 1:
        raise ValueError(f"token_ids must be one-dimensional, got shape {tuple(ids.shape)}")
    if ids.numel() and ids.min() < 0:
        raise ValueError("token_ids cannot contain negative IDs")

    centers: list[int] = []
    contexts: list[int] = []
    for center_position in range(ids.numel()):
        left = max(0, center_position - window_size)
        right = min(ids.numel(), center_position + window_size + 1)
        for context_position in range(left, right):
            if context_position == center_position:
                continue
            centers.append(int(ids[center_position]))
            contexts.append(int(ids[context_position]))

    return torch.tensor(centers, dtype=torch.long), torch.tensor(contexts, dtype=torch.long)


def unigram_noise_distribution(
    token_ids: Sequence[int] | torch.Tensor,
    *,
    vocab_size: int | None = None,
    power: float = 0.75,
) -> torch.Tensor:
    """Build ``P_n(w) ∝ count(w) ** power`` for negative sampling.

    Mikolov et al. found ``power=3/4`` better than the raw unigram and uniform
    distributions in their experiments.  Unseen vocabulary entries receive
    zero probability.
    """
    if power <= 0:
        raise ValueError(f"power must be positive, got {power}")
    ids = torch.as_tensor(token_ids, dtype=torch.long)
    if ids.ndim != 1 or ids.numel() == 0:
        raise ValueError("token_ids must be a non-empty one-dimensional sequence")
    if ids.min() < 0:
        raise ValueError("token_ids cannot contain negative IDs")

    inferred_size = int(ids.max()) + 1
    if vocab_size is None:
        vocab_size = inferred_size
    if vocab_size < inferred_size:
        raise ValueError(
            f"vocab_size={vocab_size} cannot represent maximum token ID {inferred_size - 1}"
        )

    counts = torch.bincount(ids, minlength=vocab_size).to(dtype=torch.float32)
    weights = counts.pow(power)
    return weights / weights.sum()


def subsampling_keep_probability(
    frequencies: torch.Tensor,
    *,
    threshold: float = 1e-5,
    variant: str = "paper",
) -> torch.Tensor:
    """Return frequent-token keep probabilities from the 2013 word2vec work.

    ``frequencies`` are corpus fractions, not raw counts.  The NeurIPS paper's
    Eq. (5) implies ``sqrt(t / f)``.  The authors' released ``word2vec.c`` uses
    ``sqrt(t / f) + t / f`` instead.  Both are clipped at one; exposing the
    variant prevents an implementation detail from being silently attributed to
    the equation in the paper.
    """
    if frequencies.ndim == 0 or not torch.is_floating_point(frequencies):
        raise ValueError("frequencies must be a floating-point tensor")
    if not torch.isfinite(frequencies).all() or (frequencies < 0).any():
        raise ValueError("frequencies must be finite and non-negative")
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    if variant not in {"paper", "reference_code"}:
        raise ValueError("variant must be 'paper' or 'reference_code'")

    safe_frequencies = frequencies.clamp_min(torch.finfo(frequencies.dtype).tiny)
    ratio = threshold / safe_frequencies
    keep = torch.sqrt(ratio)
    if variant == "reference_code":
        keep = keep + ratio
    keep = keep.clamp(max=1.0)
    return torch.where(frequencies == 0, torch.ones_like(keep), keep)


def sample_negative_ids(
    distribution: torch.Tensor,
    batch_size: int,
    num_negatives: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw independent negative token IDs with shape ``(B, K)``."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_negatives <= 0:
        raise ValueError(f"num_negatives must be positive, got {num_negatives}")
    if distribution.ndim != 1 or distribution.numel() == 0:
        raise ValueError("distribution must be a non-empty one-dimensional tensor")
    if not torch.isfinite(distribution).all() or (distribution < 0).any():
        raise ValueError("distribution must contain finite, non-negative weights")
    if distribution.sum() <= 0:
        raise ValueError("distribution must assign positive mass to at least one token")

    draws = torch.multinomial(
        distribution,
        batch_size * num_negatives,
        replacement=True,
        generator=generator,
    )
    return draws.view(batch_size, num_negatives)


def negative_sampling_loss(
    center_vectors: torch.Tensor,
    positive_vectors: torch.Tensor,
    negative_vectors: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute the negative SGNS objective as a loss to minimise.

    ``center_vectors`` and ``positive_vectors`` have shape ``(B, D)``;
    ``negative_vectors`` has shape ``(B, K, D)``.  The paper maximises

    ``log σ(v_pos · v_center) + Σ log σ(-v_neg · v_center)``.

    We return its negative.  The sum over K is part of each example; ``reduction``
    then combines examples in the batch.
    """
    if center_vectors.ndim != 2 or positive_vectors.shape != center_vectors.shape:
        raise ValueError(
            "center_vectors and positive_vectors must share shape (B, D), got "
            f"{tuple(center_vectors.shape)} and {tuple(positive_vectors.shape)}"
        )
    if negative_vectors.ndim != 3:
        raise ValueError(
            f"negative_vectors must have shape (B, K, D), got {tuple(negative_vectors.shape)}"
        )
    if negative_vectors.size(0) != center_vectors.size(0) or negative_vectors.size(
        2
    ) != center_vectors.size(1):
        raise ValueError("negative_vectors must match the batch and embedding dimensions")
    if negative_vectors.size(1) == 0:
        raise ValueError("negative_vectors must contain at least one negative sample")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got {reduction!r}")

    positive_scores = (center_vectors * positive_vectors).sum(dim=-1)  # (B,)
    negative_scores = torch.einsum("bd,bkd->bk", center_vectors, negative_vectors)  # (B, K)
    per_example = -F.logsigmoid(positive_scores) - F.logsigmoid(-negative_scores).sum(dim=-1)

    if reduction == "none":
        return per_example
    if reduction == "sum":
        return per_example.sum()
    return per_example.mean()


class SkipGramNegativeSampling(nn.Module):
    """Two embedding tables trained with the SGNS objective.

    Word2vec gives each vocabulary item an input/center vector and a separate
    output/context vector.  Downstream use normally keeps the center table.
    Context vectors start at zero, matching the simple reference implementation;
    the initial loss is therefore exactly ``(K + 1) * log(2)``.
    """

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.center = nn.Embedding(vocab_size, embedding_dim)
        self.context = nn.Embedding(vocab_size, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 0.5 / self.embedding_dim
        nn.init.uniform_(self.center.weight, -bound, bound)
        nn.init.zeros_(self.context.weight)

    def forward(
        self,
        center_ids: torch.Tensor,
        positive_ids: torch.Tensor,
        negative_ids: torch.Tensor,
        *,
        reduction: str = "mean",
    ) -> torch.Tensor:
        if center_ids.ndim != 1 or positive_ids.shape != center_ids.shape:
            raise ValueError("center_ids and positive_ids must share shape (B,)")
        if negative_ids.ndim != 2 or negative_ids.size(0) != center_ids.size(0):
            raise ValueError("negative_ids must have shape (B, K)")

        center_vectors = self.center(center_ids)  # (B, D)
        positive_vectors = self.context(positive_ids)  # (B, D)
        negative_vectors = self.context(negative_ids)  # (B, K, D)
        return negative_sampling_loss(
            center_vectors,
            positive_vectors,
            negative_vectors,
            reduction=reduction,
        )

    def normalized_embeddings(self) -> torch.Tensor:
        """Return row-normalised center embeddings for cosine comparisons."""
        return F.normalize(self.center.weight, dim=-1)

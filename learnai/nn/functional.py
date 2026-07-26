"""The primitive functions, written out longhand.

`torch.nn.functional` already has all of these. They are re-derived here because
the *numerical* details (why we subtract the max, why we add epsilon, why GELU
has a tanh variant) are where most of the real understanding lives, and those
details are invisible when you call the fused kernel.

Every function here is checked against its torch equivalent in
`tests/test_nn.py`, so you can trust them as drop-in replacements.
"""

from __future__ import annotations

import math

import torch


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """softmax(x)_i = exp(x_i) / sum_j exp(x_j)

    The naive formula overflows: exp(1000) == inf, and inf/inf == nan. Since
    softmax is invariant to a constant shift (the constant cancels top and
    bottom), we subtract the max first. Then the largest exponent is exp(0) == 1
    and nothing can overflow. Underflow to 0 is harmless — those terms were
    negligible anyway.
    """
    x_max = x.amax(dim=dim, keepdim=True)  # (..., 1)
    e = torch.exp(x - x_max)  # (...,) all entries <= 1
    return e / e.sum(dim=dim, keepdim=True)


def log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """log(softmax(x)) computed without ever forming softmax(x).

    Taking log(softmax(x)) directly loses precision: softmax can underflow to
    exactly 0, and log(0) == -inf. The log-sum-exp identity keeps it in log
    space the whole way.
    """
    x_max = x.amax(dim=dim, keepdim=True)
    shifted = x - x_max
    return shifted - torch.log(torch.exp(shifted).sum(dim=dim, keepdim=True))


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean negative log-likelihood of the correct class.

    logits:  (N, C) unnormalised scores
    targets: (N,)   integer class indices in [0, C)

    Note we never compute probabilities — we index into log_softmax directly.
    This is why frameworks tell you to pass *logits*, not softmax output.
    """
    logp = log_softmax(logits, dim=-1)  # (N, C)
    picked = logp.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # (N,)
    return -picked.mean()


def relu(x: torch.Tensor) -> torch.Tensor:
    """max(0, x). Dead simple, and the gradient is exactly 0 or 1 — which is
    both why it fixed vanishing gradients and why units can die permanently."""
    return torch.clamp(x, min=0.0)


def gelu(x: torch.Tensor, approximate: bool = False) -> torch.Tensor:
    """Gaussian Error Linear Unit: x * P(X <= x) for X ~ N(0, 1).

    Read it as a *soft* ReLU gate: instead of the hard 0/1 switch, each input
    is scaled by the probability that a standard normal falls below it. Small
    negatives survive slightly, which keeps gradients alive near zero.

    `approximate=True` is the tanh form from the GPT-2 paper — it predates fast
    erf kernels, and GPT-2's published weights were trained with it, so you need
    it for exact reproduction.
    """
    if approximate:
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * torch.pow(x, 3.0))))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def silu(x: torch.Tensor) -> torch.Tensor:
    """x * sigmoid(x), a.k.a. Swish. Used in LLaMA's feed-forward (as SwiGLU)."""
    return x * torch.sigmoid(x)


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Normalise each token's feature vector to mean 0, variance 1, then rescale.

    x: (..., D) — statistics are taken over the LAST dim only.

    The thing to internalise: this is per-token and per-example. It does not mix
    information across the batch (unlike BatchNorm) or across the sequence, which
    is exactly why it works with variable-length sequences and batch size 1.

    Uses the *biased* variance (divide by D, not D-1) to match torch.
    """
    mean = x.mean(dim=-1, keepdim=True)  # (..., 1)
    var = x.var(dim=-1, keepdim=True, unbiased=False)  # (..., 1)
    normed = (x - mean) / torch.sqrt(var + eps)  # eps guards against a constant feature vector
    if weight is not None:
        normed = normed * weight
    if bias is not None:
        normed = normed + bias
    return normed


def rms_norm(
    x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6
) -> torch.Tensor:
    """LayerNorm minus the mean-centering: just divide by the root-mean-square.

    LLaMA/T5 use this. The observation was that re-centering contributes little
    while costing a pass over the data, so dropping it is nearly free accuracy.
    """
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)  # (..., 1)
    normed = x / rms
    return normed * weight if weight is not None else normed


def dropout(x: torch.Tensor, p: float = 0.5, training: bool = True) -> torch.Tensor:
    """Zero each activation with probability p, then scale survivors by 1/(1-p).

    That rescale ("inverted dropout") is what lets you leave the network
    untouched at eval time: the expected value of each unit already matches.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"dropout probability must be in [0, 1], got {p}")
    if not training or p == 0.0:
        return x
    if p == 1.0:
        return torch.zeros_like(x)
    keep = 1.0 - p
    mask = (torch.rand_like(x) < keep).to(x.dtype)
    return x * mask / keep

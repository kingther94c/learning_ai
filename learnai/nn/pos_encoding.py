"""Position encodings — four answers to the same problem.

Attention is a set operation. `softmax(QK^T)V` is completely permutation-
equivariant: shuffle the input tokens and the outputs shuffle with them,
unchanged. "the dog bit the man" and "the man bit the dog" are literally the
same input. Position has to be injected by hand.

The four strategies here are the ones worth knowing, in historical order:

    SinusoidalPE   (2017, Vaswani)  fixed waves added to the embedding
    LearnedPE      (2018, GPT/BERT) a lookup table, learned like any embedding
    RotaryPE       (2021, Su)       rotate Q and K by an angle set by position
    ALiBi          (2021, Press)    subtract a linear distance penalty from scores

The trend is unmistakable: from *adding to the input* toward *modifying the
attention score directly*, because that is what generalises past the training
length.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPE(nn.Module):
    """The original: PE[pos, 2i] = sin(pos / 10000^(2i/d)), odd dims use cos.

    Each pair of dimensions is a clock hand rotating at its own frequency —
    dimension 0 ticks every few tokens, the last dimension takes ~10000 tokens to
    complete a cycle. Together they form a binary-ish positional fingerprint.

    The property Vaswani highlighted: PE[pos+k] is a fixed linear function of
    PE[pos] for any offset k (it's a rotation matrix), so *relative* offsets are
    linearly recoverable. In practice the model exploits this only weakly, which
    is why RoPE — which enforces it structurally — took over.
    """

    def __init__(self, d_model: int, max_len: int = 5000, base: float = 10000.0):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if max_len <= 0:
            raise ValueError(f"max_len must be positive, got {max_len}")
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)  # (L, D)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)  # (L, 1)

        # exp(-log(base) * 2i/d) == 1 / base^(2i/d), computed in log space for stability
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(base) / d_model))

        angles = pos * div  # (L, ceil(D/2)) angle of every clock hand at every position
        pe[:, 0::2] = torch.sin(angles)
        # With an odd d_model there is one more sin column than cos column, so the
        # cos side uses one fewer frequency.
        pe[:, 1::2] = torch.cos(angles[:, : d_model // 2])
        # `register_buffer` -> moves with .to(device) and is saved, but is not a parameter
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> same shape, with position added in."""
        T = x.size(1)
        if T > self.max_len:
            raise ValueError(f"sequence length {T} exceeds max_len {self.max_len}")
        return x + self.pe[:, :T].to(dtype=x.dtype)


class LearnedPE(nn.Module):
    """A plain embedding table indexed by position. What GPT-2 and BERT use.

    Simpler and slightly better in-distribution than sinusoids, with one hard
    limit: position `max_len` has no row, so the model *cannot* run on a longer
    sequence. Every "context length" number you have seen quoted for a GPT-2-era
    model is the height of this table.
    """

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        self.max_len = max_len
        self.emb = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.emb.weight, std=0.02)  # GPT-2's init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        if T > self.max_len:
            raise ValueError(f"sequence length {T} exceeds max_len {self.max_len}")
        pos = torch.arange(T, device=x.device)  # (T,)
        return x + self.emb(pos).unsqueeze(0)  # (1, T, D) broadcasts over batch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(..., D) -> (..., D), treating the two halves as (real, imaginary).

    Equivalent to multiplying each complex number (x1 + i*x2) by i.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryPE(nn.Module):
    """RoPE: rotate each 2D slice of Q and K by an angle proportional to position.

    The trick is beautiful. Treat the head dim as D/2 complex numbers. Rotating
    the query at position m by angle m*theta and the key at position n by n*theta
    makes their inner product depend on the rotation *difference* (m-n)*theta:

        <R_m q, R_n k> = <q, R_{n-m} k>

    So the score is a function of relative distance, exactly, by construction —
    no learned parameters and nothing added to the residual stream. This is why
    RoPE extrapolates better and why every open-weight LLM since LLaMA uses it.

    Note: this implements the *half-split* convention (LLaMA / HuggingFace).
    The original paper interleaves even/odd dims. They are equivalent up to a
    fixed permutation of the head dimension, but weights are NOT interchangeable
    between conventions — a classic source of silent garbage when porting models.
    """

    def __init__(self, d_head: int, max_len: int = 4096, base: float = 10000.0):
        super().__init__()
        if d_head <= 0:
            raise ValueError(f"d_head must be positive, got {d_head}")
        if d_head % 2 != 0:
            raise ValueError(f"RoPE needs an even head dim, got {d_head}")
        if max_len <= 0:
            raise ValueError(f"max_len must be positive, got {max_len}")
        self.d_head = d_head
        self.max_len = max_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))  # (D/2,)
        t = torch.arange(max_len).float()  # (L,)
        freqs = torch.outer(t, inv_freq)  # (L, D/2) angle for each (position, pair)
        emb = torch.cat((freqs, freqs), dim=-1)  # (L, D) duplicated to match _rotate_half

        self.register_buffer("cos", emb.cos()[None, None, :, :], persistent=False)  # (1,1,L,D)
        self.register_buffer("sin", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0):
        """q, k: (B, H, T, Dh) -> rotated (B, H, T, Dh).

        `offset` is the position of the first token — needed during incremental
        decoding, where you feed one token at a time but it sits at position N.
        """
        T = q.size(-2)
        if q.shape[-2:] != k.shape[-2:]:
            raise ValueError(
                "q and k must have the same (sequence, head) dimensions, got "
                f"{tuple(q.shape[-2:])} and {tuple(k.shape[-2:])}"
            )
        if q.size(-1) != self.d_head:
            raise ValueError(f"expected head dim {self.d_head}, got {q.size(-1)}")
        if offset < 0:
            raise ValueError(f"offset must be non-negative, got {offset}")
        if offset + T > self.max_len:
            raise ValueError(f"positions [{offset}, {offset + T}) exceed max_len {self.max_len}")
        cos = self.cos[:, :, offset : offset + T, :]
        sin = self.sin[:, :, offset : offset + T, :]
        # Buffers are built in fp32 for accurate angles. Cast at the operation
        # boundary so a half-precision model stays half precision.
        q_cos, q_sin = cos.to(dtype=q.dtype), sin.to(dtype=q.dtype)
        k_cos, k_sin = cos.to(dtype=k.dtype), sin.to(dtype=k.dtype)
        q_rot = q * q_cos + _rotate_half(q) * q_sin
        k_rot = k * k_cos + _rotate_half(k) * k_sin
        return q_rot, k_rot


def alibi_slopes(n_heads: int) -> torch.Tensor:
    """The paper's geometric set of per-head distance penalties.

    For a power-of-two head count this is ``2^-8/n, 2^-16/n, ...``. The paper
    defines a small interleaving step for other head counts; reproducing it
    matters when comparing against published ALiBi implementations.
    """
    if n_heads <= 0:
        raise ValueError(f"n_heads must be positive, got {n_heads}")

    def power_of_two_slopes(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start ** (i + 1) for i in range(n)]

    if math.log2(n_heads).is_integer():
        values = power_of_two_slopes(n_heads)
    else:
        lower = 2 ** math.floor(math.log2(n_heads))
        extra = power_of_two_slopes(2 * lower)[::2][: n_heads - lower]
        values = [*power_of_two_slopes(lower), *extra]
    return torch.tensor(values)  # (H,)


class ALiBi(nn.Module):
    """Attention with Linear Biases: score[i,j] -= m_h * (i - j).

    No position information enters the embeddings at all. Instead each head
    subtracts a penalty proportional to how far back it is looking, applied
    *inside* the softmax. Because the penalty is defined for any distance, the
    model runs on sequences far longer than it trained on — that was the paper's
    headline result and it holds up.
    """

    def __init__(self, n_heads: int, max_len: int = 4096):
        super().__init__()
        if max_len <= 0:
            raise ValueError(f"max_len must be positive, got {max_len}")
        self.n_heads = n_heads
        self.max_len = max_len
        slopes = alibi_slopes(n_heads)  # (H,)
        pos = torch.arange(max_len)
        # distance[i, j] = j - i, which is <= 0 in the causal region
        distance = pos.view(1, -1) - pos.view(-1, 1)  # (L, L)
        bias = slopes.view(-1, 1, 1) * distance.unsqueeze(0)  # (H, L, L), negative going back
        self.register_buffer("bias", bias.unsqueeze(0), persistent=False)  # (1, H, L, L)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """scores: (B, H, T, T) pre-softmax -> biased scores, same shape."""
        if scores.ndim != 4 or scores.size(1) != self.n_heads:
            raise ValueError(
                f"expected scores shaped (B, {self.n_heads}, T, T), got {tuple(scores.shape)}"
            )
        Tq, Tk = scores.shape[-2:]
        if Tq != Tk:
            raise ValueError(f"ALiBi self-attention scores must be square, got ({Tq}, {Tk})")
        if Tk > self.max_len:
            raise ValueError(f"sequence length {Tk} exceeds max_len {self.max_len}")
        return scores + self.bias[:, :, :Tk, :Tk].to(dtype=scores.dtype)

"""Attention, from the dot product up.

The whole mechanism in one sentence: **every token proposes a query, every token
advertises a key, we score all query-key pairs, softmax the scores into weights,
and each token walks away with a weighted average of everyone's values.**

Read the file in this order:
    scaled_dot_product_attention  -- the mechanism itself, ~10 lines
    MultiHeadAttention            -- run the mechanism H times in parallel subspaces
    causal_mask / padding_mask    -- what a token is allowed to look at

Everything returns its attention weights when asked, because the weights are the
part you actually want to *see* (see `learnai.interp` and the interactive pages).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from learnai.nn.functional import softmax


def causal_mask(seq_len: int, device=None) -> torch.Tensor:
    """(1, 1, T, T) boolean mask where True = "this position may be attended to".

    Lower-triangular including the diagonal: token i sees tokens 0..i. This one
    line is the entire difference between BERT and GPT.
    """
    m = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
    return m.view(1, 1, seq_len, seq_len)


def padding_mask(lengths: torch.Tensor, seq_len: int) -> torch.Tensor:
    """(B, 1, 1, T) mask that hides padding. `lengths` is (B,) real token counts.

    Broadcasts over heads and over query positions: padding is invisible to
    everyone, regardless of who is asking.
    """
    device = lengths.device
    positions = torch.arange(seq_len, device=device).view(1, seq_len)  # (1, T)
    keep = positions < lengths.view(-1, 1)  # (B, T)
    return keep.view(-1, 1, 1, seq_len)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Shapes (H is heads; the function is agnostic to leading dims):
        q    (B, H, Tq, Dh)
        k    (B, H, Tk, Dh)
        v    (B, H, Tk, Dv)
        mask broadcastable to (B, H, Tq, Tk); True = keep, False = forbid
    Returns:
        out  (B, H, Tq, Dv)
        attn (B, H, Tq, Tk)  -- valid rows sum to 1; fully masked rows are all zero

    Why divide by sqrt(d_k)? If q and k have independent unit-variance entries,
    their dot product over d_k dims has variance d_k, so scores grow like
    sqrt(d_k). Feed those into softmax and it saturates — one weight goes to 1,
    the rest to 0, and the gradient dies. Dividing by sqrt(d_k) holds the score
    variance at ~1 no matter how wide the head is. Try it yourself: set
    `scale=1.0` and watch the entropy of `attn` collapse as Dh grows.
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)  # (B, H, Tq, Tk)

    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError(f"attention mask must be boolean, got {mask.dtype}")
        # masked_fill wants True where we WRITE the value, so invert.
        # `finfo.min` works in fp16 as well as fp32; a hard-coded -1e9 cannot be
        # represented in fp16 and raises an overflow error before softmax runs.
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    attn = softmax(scores, dim=-1)  # (B, H, Tq, Tk) each row is a distribution
    if mask is not None:
        # A fully masked row has no probability distribution. Softmax alone
        # would turn its equal sentinel scores into a uniform distribution,
        # silently letting the query read every forbidden value. Multiplying the
        # mask back in makes forbidden entries exact zero and an empty row all
        # zero, matching torch's scaled_dot_product_attention semantics.
        attn = attn * mask.to(dtype=attn.dtype)

    if dropout_p > 0.0 and training:
        # Dropping attention weights means a token randomly loses sight of some
        # positions — a surprisingly strong regulariser. Rows no longer sum to 1.
        attn = torch.nn.functional.dropout(attn, p=dropout_p)

    out = attn @ v  # (B, H, Tq, Dv) weighted average of value vectors
    return out, attn


class MultiHeadAttention(nn.Module):
    """H attention heads running in parallel over disjoint slices of the model dim.

    Why more than one head? A single softmax produces *one* distribution per
    token — it can only average one thing at a time. Splitting d_model into H
    subspaces of size d_model/H lets one head track syntax, another track
    coreference, and so on, at no extra parameter or FLOP cost (the head
    dimension shrinks to compensate).

    Set `store_attn=True` (or use `learnai.interp.capture_attention`) to keep the
    weights from the last forward pass in `self.attn_weights`.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool = False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        self.dropout_p = dropout

        # One fused projection for Q, K, V is what real implementations do — it is
        # exactly three separate Linear(d_model, d_model) layers stacked, but a
        # single matmul. We slice it apart in forward().
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)

        self.store_attn: bool = False
        self.attn_weights: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_attn: bool = False,
    ):
        """x: (B, T, d_model) -> (B, T, d_model), optionally with (B, H, T, T) weights."""
        B, T, _ = x.shape

        qkv = self.qkv_proj(x)  # (B, T, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)  # 3 x (B, T, d_model)

        # (B, T, d_model) -> (B, H, T, d_head): split the feature axis into heads,
        # then move heads next to batch so the matmul below treats each head as an
        # independent problem.
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        if self.causal:
            cm = causal_mask(T, device=x.device)
            mask = cm if mask is None else (mask & cm)

        out, attn = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout_p=self.dropout_p, training=self.training
        )

        # (B, H, T, Dh) -> (B, T, d_model): undo the split. `contiguous` is needed
        # because transpose only changed the stride metadata, not the memory layout.
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.resid_dropout(self.out_proj(out))

        if self.store_attn:
            self.attn_weights = attn.detach()
        if return_attn:
            return out, attn
        return out


class CrossAttention(nn.Module):
    """Queries come from one sequence, keys/values from another.

    This is the only structural difference between the encoder-decoder
    Transformer of "Attention Is All You Need" and a decoder-only GPT: the
    decoder gets a second attention block whose K and V come from the encoder.
    It is also exactly how modern retrieval and multimodal adapters graft one
    stream onto another.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout_p = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)

        self.store_attn: bool = False
        self.attn_weights: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_attn: bool = False,
    ):
        """x: (B, Tq, D) queries. context: (B, Tk, D) the thing being read from."""
        B, Tq, _ = x.shape
        Tk = context.size(1)

        q = self.q_proj(x).view(B, Tq, self.n_heads, self.d_head).transpose(1, 2)
        k, v = self.kv_proj(context).chunk(2, dim=-1)
        k = k.view(B, Tk, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, Tk, self.n_heads, self.d_head).transpose(1, 2)

        out, attn = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout_p=self.dropout_p, training=self.training
        )
        out = out.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        out = self.resid_dropout(self.out_proj(out))

        if self.store_attn:
            self.attn_weights = attn.detach()
        if return_attn:
            return out, attn
        return out


def attention_entropy(attn: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (nats) of each attention row: (B, H, Tq, Tk) -> (B, H, Tq).

    A diagnostic worth reaching for constantly. Entropy near 0 means the head has
    locked onto a single position (induction heads look like this); entropy near
    log(Tk) means it is just averaging everything and probably doing nothing.
    """
    p = attn.clamp_min(1e-12)
    return -(p * p.log()).sum(dim=-1)

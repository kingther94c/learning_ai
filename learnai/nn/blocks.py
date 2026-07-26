"""The Transformer block: attention + MLP, wrapped in residuals and norms.

A block does two things in sequence:
    1. **Attention** — tokens exchange information. This is the only place where
       information moves *between* positions.
    2. **MLP** — each token thinks privately. Same weights applied to every
       position independently; no mixing at all.

Communicate, then compute. Every Transformer variant is a rearrangement of those
two steps plus a choice of where to put the normalisation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from learnai.nn.attention import CrossAttention, MultiHeadAttention
from learnai.nn.functional import gelu


class FeedForward(nn.Module):
    """Position-wise MLP: d_model -> 4*d_model -> d_model.

    Two thirds of a Transformer's parameters live here, and the widening factor
    of 4 has survived from 2017 to today essentially unquestioned.

    Interpretability work (Geva et al., "Transformer Feed-Forward Layers Are
    Key-Value Memories") reads this as an associative store: the first matrix's
    rows are *keys* that detect input patterns, the second matrix's columns are
    *values* written into the residual stream when a key fires. That framing
    explains a lot of what circuit-level analysis later found.
    """

    def __init__(self, d_model: int, d_ff: int | None = None, dropout: float = 0.0):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.fc_in = nn.Linear(d_model, d_ff)
        self.fc_out = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, T, D). Note nothing here looks at other positions."""
        h = gelu(self.fc_in(x), approximate=True)  # (B, T, d_ff)
        return self.dropout(self.fc_out(h))  # (B, T, D)


class SwiGLU(nn.Module):
    """LLaMA's feed-forward: (silu(W1 x) * W3 x) W2 — a gated MLP.

    The elementwise product means one branch *gates* the other, letting the layer
    express multiplicative interactions that a plain MLP cannot. d_ff is usually
    set to 8/3 * d_model (not 4x) so the three matrices land at the same total
    parameter count as the two-matrix version.
    """

    def __init__(self, d_model: int, d_ff: int | None = None, dropout: float = 0.0):
        super().__init__()
        d_ff = d_ff or int(8 * d_model / 3)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.nn.functional.silu(self.w_gate(x))  # (B, T, d_ff)
        return self.dropout(self.w_down(gate * self.w_up(x)))


class TransformerBlock(nn.Module):
    """One decoder block.

    `pre_norm=True` (the default, and what every modern model does):

        x = x + Attn(LN(x))
        x = x + MLP(LN(x))

    `pre_norm=False` is the original 2017 arrangement:

        x = LN(x + Attn(x))
        x = LN(x + MLP(x))

    The difference is not cosmetic. With post-norm, the residual path passes
    through a LayerNorm at every layer, so gradients get rescaled repeatedly and
    deep models need a learning-rate warmup to train at all. With pre-norm the
    residual stream is a clean identity path from input to output — gradients
    flow straight down it. That single change is what made 100-layer
    Transformers routine.

    Read the residual stream as a shared workspace: every block *reads* a
    normalised copy, and *adds* its contribution back. Nothing is ever
    overwritten, which is why you can ablate an individual layer and often see
    graceful degradation instead of collapse.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int | None = None,
        dropout: float = 0.0,
        causal: bool = True,
        pre_norm: bool = True,
        cross_attention: bool = False,
    ):
        super().__init__()
        self.pre_norm = pre_norm

        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout, causal=causal)

        self.cross_attn = None
        if cross_attention:
            self.ln_x = nn.LayerNorm(d_model)
            self.cross_attn = CrossAttention(d_model, n_heads, dropout=dropout)

        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, d_ff, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.pre_norm:
            x = x + self.attn(self.ln_1(x), mask=mask)
            if self.cross_attn is not None and context is not None:
                x = x + self.cross_attn(self.ln_x(x), context, mask=context_mask)
            x = x + self.mlp(self.ln_2(x))
        else:
            x = self.ln_1(x + self.attn(x, mask=mask))
            if self.cross_attn is not None and context is not None:
                x = self.ln_x(x + self.cross_attn(x, context, mask=context_mask))
            x = self.ln_2(x + self.mlp(x))
        return x

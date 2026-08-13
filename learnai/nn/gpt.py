"""A minimal GPT you can read end to end in one sitting.

Architecturally this *is* GPT-2 — same decoder-only stack, same learned position
embeddings, same weight tying. What is missing is only scale and engineering
(no KV cache, no flash attention, no sharding). Nothing conceptual is left out.

    tokens ──embed──> residual stream ──[block] x N──> LN ──unembed──> logits

The residual stream is the thing to hold in your head. It is a (B, T, d_model)
tensor that every block reads from and adds to. Attention moves information
*across* positions; the MLP transforms it *within* a position; the stream itself
is never overwritten, only accumulated into.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from learnai.nn.blocks import TransformerBlock
from learnai.nn.functional import cross_entropy


@dataclass
class GPTConfig:
    """Sizes for a GPT. The defaults are a small model that trains on a laptop.

    For reference, the shapes that matter historically:
        this default   n_layer=4,  n_head=4,  d_model=128    ~0.8M params
        GPT-2 small    n_layer=12, n_head=12, d_model=768     124M
        GPT-2 XL       n_layer=48, n_head=25, d_model=1600    1.5B
        GPT-3          n_layer=96, n_head=96, d_model=12288   175B

    Notice that depth, width and head count all scale together — that ratio is
    not arbitrary, it falls out of the compute-optimal scaling laws.
    """

    vocab_size: int = 65
    block_size: int = 128  # max context length
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    d_ff: int | None = None  # defaults to 4 * d_model
    dropout: float = 0.0
    tie_weights: bool = True  # share the embedding matrix with the output head

    def __post_init__(self) -> None:
        for name in ("vocab_size", "block_size", "n_layer", "n_head", "d_model"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.d_ff is not None and self.d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {self.d_ff}")
        if self.d_model % self.n_head != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by n_head={self.n_head}")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")


class GPT(nn.Module):
    """Decoder-only Transformer language model."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.block_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_head,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                    causal=True,
                    pre_norm=True,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            # The embedding matrix maps token -> vector; the head maps vector ->
            # token scores. Making them the same matrix saves vocab*d_model
            # parameters (~30% of GPT-2 small) and reliably improves perplexity:
            # tokens that mean similar things get similar rows in *both* roles.
            self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

        # GPT-2's trick: scale down the init of anything writing to the residual
        # stream by 1/sqrt(2*n_layer). Each of the 2*n_layer sub-layers adds
        # variance to the stream; without this the stream's scale grows with
        # depth and the final LayerNorm has to fight it.
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("fc_out.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        """Parameter count. Papers usually quote the non-embedding number."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()
            if not self.config.tie_weights:
                n -= self.token_emb.weight.numel()
        return n

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """idx: (B, T) token ids -> logits (B, T, vocab), loss or None.

        When `targets` is given we predict token t+1 from position t at *every*
        position simultaneously. That density is the whole reason language
        modelling scales: one sequence of length T yields T training signals, and
        the causal mask is what makes it legal.
        """
        x = self.encode(idx)
        logits = self.lm_head(x)  # (B, T, vocab)

        loss = None
        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError(
                    f"targets must have shape {tuple(idx.shape)}, got {tuple(targets.shape)}"
                )
            loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def encode(self, idx: torch.Tensor) -> torch.Tensor:
        """Return the final hidden state before unembedding.

        GPT-1 reused this representation for supervised transfer by attaching a
        small task head to the last token.  Keeping the operation public also
        lets lessons inspect the residual stream without a forward hook.

        Args:
            idx: Integer token ids with shape ``(batch, time)``.

        Returns:
            Final layer-normalised hidden states with shape
            ``(batch, time, d_model)``.
        """
        if idx.ndim != 2:
            raise ValueError(f"idx must have shape (B, T), got {tuple(idx.shape)}")
        if idx.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"idx must contain integer token ids, got {idx.dtype}")

        _, T = idx.shape
        if T == 0:
            raise ValueError("idx must contain at least one token")
        if T > self.config.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.config.block_size}")

        pos = torch.arange(T, device=idx.device)  # (T,)
        x = self.token_emb(idx) + self.pos_emb(pos)  # (B, T, D) — the stream is born
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)  # (B, T, D) each block adds its contribution

        return self.ln_f(x)  # (B, T, D)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling. idx: (B, T_prompt) -> (B, T_prompt + max_new_tokens).

        The three knobs, which are worth understanding as *three different edits
        to the same distribution*:

        temperature  divides the logits. <1 sharpens (safer, repetitive), >1
                     flattens (wilder). At 0+ it becomes greedy argmax.
        top_k        keep only the k highest-probability tokens, renormalise.
                     Blunt: k=40 is too many after "The capital of France is",
                     too few mid-sentence.
        top_p        (nucleus) keep the smallest set whose probability sums to p.
                     Adapts its size to how confident the model is — usually the
                     better default.

        Deliberately has no KV cache: every step re-runs the full forward pass
        over the whole prefix, making this O(T^2) per token. That is the honest
        baseline, and it is what makes the cache feel like a revelation when you
        add it.
        """
        if idx.ndim != 2 or idx.size(1) == 0:
            raise ValueError(f"idx must be a non-empty (B, T) tensor, got {tuple(idx.shape)}")
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"temperature must be finite and positive, got {temperature}")
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if top_p is not None and not 0.0 < top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")

        # Sampling needs dropout off, but a convenience method should not
        # silently leave a model in a different mode after it returns.
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                # Crop to the context window — the model has no position embedding
                # beyond block_size.
                idx_cond = idx[:, -self.config.block_size :]
                logits, _ = self(idx_cond)  # (B, T, vocab)
                logits = logits[:, -1, :] / temperature  # (B, vocab) last position only

                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    kth = logits.topk(k, dim=-1).values[:, [-1]]  # (B, 1)
                    logits = logits.masked_fill(logits < kth, float("-inf"))

                if top_p is not None:
                    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
                    probs = sorted_logits.softmax(dim=-1)
                    cumulative = probs.cumsum(dim=-1)
                    # Drop everything past the point where the cumulative mass first
                    # exceeds p, but always keep the top-1 so we can never mask all.
                    remove = cumulative - probs > top_p
                    remove[:, 0] = False
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    # Undo the sort: put each kept logit back at its original vocab
                    # index. sorted_idx is a permutation, so every slot gets written.
                    logits = torch.full_like(logits, float("-inf")).scatter(
                        1, sorted_idx, sorted_logits
                    )

                probs = logits.softmax(dim=-1)  # (B, vocab)
                next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
                idx = torch.cat([idx, next_token], dim=1)
            return idx
        finally:
            self.train(was_training)

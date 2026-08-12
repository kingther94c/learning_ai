"""A small, faithful BERT encoder and its two original pre-training heads.

This is BERT with scale removed, not mechanisms removed:

    token + position + segment embeddings
        -> post-norm bidirectional Transformer encoder blocks
        -> masked-language-model head at every token
        -> next-sentence head at [CLS]

The defaults fit on a laptop.  ``causal=True`` exists only as an ablation: it
turns the visibility pattern into GPT's left-to-right triangle so a lesson can
measure what bidirectionality buys.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from learnai.nn.attention import MultiHeadAttention


@dataclass
class BERTConfig:
    """Dimensions for a compact BERT.

    Paper reference:
        BERT-base   layers=12, heads=12, d_model=768,  d_ff=3072, ~110M params
        BERT-large  layers=24, heads=16, d_model=1024, d_ff=4096, ~340M params
    """

    vocab_size: int
    max_seq_len: int = 128
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    d_ff: int | None = None
    dropout: float = 0.1
    pad_token_id: int = 0
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    tie_weights: bool = True
    causal: bool = False  # deliberately non-BERT; useful for the GPT ablation

    def __post_init__(self) -> None:
        for name in ("vocab_size", "max_seq_len", "n_layer", "n_head", "d_model"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.d_ff is not None and self.d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {self.d_ff}")
        if self.d_model % self.n_head != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by n_head={self.n_head}")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")
        if not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError(f"pad_token_id={self.pad_token_id} is outside the vocabulary")
        if self.type_vocab_size <= 0:
            raise ValueError(f"type_vocab_size must be positive, got {self.type_vocab_size}")
        if self.layer_norm_eps <= 0:
            raise ValueError(f"layer_norm_eps must be positive, got {self.layer_norm_eps}")
        if self.initializer_range <= 0:
            raise ValueError(f"initializer_range must be positive, got {self.initializer_range}")


@dataclass
class BERTOutput:
    last_hidden_state: torch.Tensor  # (B, T, D)
    pooler_output: torch.Tensor  # (B, D), derived from [CLS]
    attentions: tuple[torch.Tensor, ...] | None = None  # L x (B, H, T, T)


@dataclass
class BERTPreTrainingOutput:
    mlm_logits: torch.Tensor  # (B, T, V)
    nsp_logits: torch.Tensor  # (B, 2)
    hidden_states: torch.Tensor  # (B, T, D)
    attentions: tuple[torch.Tensor, ...] | None = None


class BERTEmbeddings(nn.Module):
    """Add token, learned position, and sentence-segment embeddings."""

    def __init__(self, config: BERTConfig):
        super().__init__()
        self.max_seq_len = config.max_seq_len
        self.token = nn.Embedding(config.vocab_size, config.d_model, config.pad_token_id)
        self.position = nn.Embedding(config.max_seq_len, config.d_model)
        self.type_vocab_size = config.type_vocab_size
        self.segment = nn.Embedding(config.type_vocab_size, config.d_model)
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, input_ids: torch.Tensor, token_type_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``(B, T) -> (B, T, D)``; all three tables are summed elementwise."""
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape (B, T), got {tuple(input_ids.shape)}")
        B, T = input_ids.shape
        if T > self.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len={self.max_seq_len}")

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if token_type_ids.shape != input_ids.shape:
            raise ValueError(
                f"token_type_ids must match input_ids {tuple(input_ids.shape)}, "
                f"got {tuple(token_type_ids.shape)}"
            )
        if token_type_ids.numel() and (
            token_type_ids.min() < 0 or token_type_ids.max() >= self.type_vocab_size
        ):
            raise ValueError(
                f"token_type_ids must be in [0, {self.type_vocab_size}), "
                f"got min={int(token_type_ids.min())}, max={int(token_type_ids.max())}"
            )

        positions = torch.arange(T, device=input_ids.device).view(1, T).expand(B, T)
        x = self.token(input_ids) + self.position(positions) + self.segment(token_type_ids)
        return self.dropout(self.norm(x))


class BERTEncoderLayer(nn.Module):
    """One original BERT block: bidirectional attention, MLP, post-LayerNorm.

    BERT inherited the 2017 Transformer's ``LN(x + sublayer(x))`` order.  This
    differs from the pre-norm arrangement used by most later large models.
    """

    def __init__(self, config: BERTConfig):
        super().__init__()
        d_ff = config.d_ff or 4 * config.d_model
        self.attn = MultiHeadAttention(
            config.d_model,
            config.n_head,
            dropout=config.dropout,
            causal=config.causal,
        )
        self.attn_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ff_in = nn.Linear(config.d_model, d_ff)
        self.ff_out = nn.Linear(d_ff, config.d_model)
        self.ff_dropout = nn.Dropout(config.dropout)
        self.output_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_out, weights = self.attn(x, mask=attention_mask, return_attn=True)
        x = self.attn_norm(x + attn_out)

        # The public BERT code uses the tanh GELU approximation and a 4x-wide
        # feed-forward layer.  Keeping the approximation matters for numerical
        # comparisons even though its difference from exact GELU is tiny.
        ff = self.ff_out(F.gelu(self.ff_in(x), approximate="tanh"))
        x = self.output_norm(x + self.ff_dropout(ff))
        return x, weights if return_attn else None


class BERT(nn.Module):
    """The encoder shared by pre-training and every downstream BERT task."""

    def __init__(self, config: BERTConfig):
        super().__init__()
        self.config = config
        self.embeddings = BERTEmbeddings(config)
        self.layers = nn.ModuleList([BERTEncoderLayer(config) for _ in range(config.n_layer)])
        self.pooler = nn.Linear(config.d_model, config.d_model)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        # The public BERT implementation used a normal distribution truncated
        # at two standard deviations, with zero biases and identity LayerNorm.
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @staticmethod
    def _expand_attention_mask(
        attention_mask: torch.Tensor | None,
        input_ids: torch.Tensor,
        pad_token_id: int,
    ) -> torch.Tensor:
        """Turn ``(B,T)`` keep flags into ``(B,1,1,T)`` attention visibility."""
        if attention_mask is None:
            attention_mask = input_ids != pad_token_id
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"attention_mask must match input_ids {tuple(input_ids.shape)}, "
                f"got {tuple(attention_mask.shape)}"
            )
        return attention_mask.to(dtype=torch.bool).view(input_ids.size(0), 1, 1, input_ids.size(1))

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        *,
        return_attn: bool = False,
    ) -> BERTOutput:
        mask = self._expand_attention_mask(attention_mask, input_ids, self.config.pad_token_id)
        x = self.embeddings(input_ids, token_type_ids)  # (B, T, D)

        saved: list[torch.Tensor] = []
        for layer in self.layers:
            x, weights = layer(x, attention_mask=mask, return_attn=return_attn)
            if weights is not None:
                saved.append(weights)

        pooled = torch.tanh(self.pooler(x[:, 0]))  # [CLS] -> (B, D)
        return BERTOutput(x, pooled, tuple(saved) if return_attn else None)


class BERTForPreTraining(nn.Module):
    """BERT plus the masked-token and next-sentence prediction heads."""

    def __init__(self, config: BERTConfig):
        super().__init__()
        self.config = config
        self.bert = BERT(config)

        # Before vocabulary projection BERT applies one more dense -> GELU -> LN.
        self.mlm_dense = nn.Linear(config.d_model, config.d_model)
        self.mlm_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.mlm_decoder = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.mlm_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.nsp_classifier = nn.Linear(config.d_model, 2)

        self.mlm_dense.apply(self.bert._init_weights)
        self.mlm_norm.apply(self.bert._init_weights)
        self.mlm_decoder.apply(self.bert._init_weights)
        self.nsp_classifier.apply(self.bert._init_weights)
        if config.tie_weights:
            self.mlm_decoder.weight = self.bert.embeddings.token.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        *,
        return_attn: bool = False,
    ) -> BERTPreTrainingOutput:
        encoded = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            return_attn=return_attn,
        )
        h = self.mlm_norm(F.gelu(self.mlm_dense(encoded.last_hidden_state), approximate="tanh"))
        mlm_logits = self.mlm_decoder(h) + self.mlm_bias
        nsp_logits = self.nsp_classifier(encoded.pooler_output)
        return BERTPreTrainingOutput(
            mlm_logits,
            nsp_logits,
            encoded.last_hidden_state,
            encoded.attentions,
        )

    @staticmethod
    def loss(
        output: BERTPreTrainingOutput,
        mlm_labels: torch.Tensor,
        nsp_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``total, mlm, nsp`` exactly as the paper's joint objective."""
        selected = mlm_labels != -100
        if selected.any():
            mlm = F.cross_entropy(output.mlm_logits[selected], mlm_labels[selected])
        else:
            # Short toy batches can select zero tokens.  Returning a connected
            # zero avoids NaN while correctly contributing no MLM gradient.
            mlm = output.mlm_logits.sum() * 0.0
        nsp = F.cross_entropy(output.nsp_logits, nsp_labels)
        return mlm + nsp, mlm, nsp

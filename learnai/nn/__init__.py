"""From-scratch neural network building blocks.

functional     softmax, gelu, layer_norm, ... written longhand
attention      scaled dot-product, multi-head, cross-attention, masks
pos_encoding   sinusoidal, learned, RoPE, ALiBi
blocks         FeedForward, SwiGLU, TransformerBlock
bert           BERTConfig, BERT, BERTForPreTraining — bidirectional encoder
gpt            GPTConfig, GPT — the full decoder-only model
"""

from learnai.nn.attention import (  # noqa: F401
    CrossAttention,
    MultiHeadAttention,
    attention_entropy,
    causal_mask,
    padding_mask,
    scaled_dot_product_attention,
)
from learnai.nn.bert import (  # noqa: F401
    BERT,
    BERTConfig,
    BERTEmbeddings,
    BERTEncoderLayer,
    BERTForPreTraining,
)
from learnai.nn.blocks import FeedForward, SwiGLU, TransformerBlock  # noqa: F401
from learnai.nn.functional import (  # noqa: F401
    cross_entropy,
    dropout,
    gelu,
    layer_norm,
    log_softmax,
    relu,
    rms_norm,
    silu,
    softmax,
)
from learnai.nn.gpt import GPT, GPTConfig  # noqa: F401
from learnai.nn.pos_encoding import ALiBi, LearnedPE, RotaryPE, SinusoidalPE  # noqa: F401

__all__ = [
    "ALiBi",
    "BERT",
    "BERTConfig",
    "BERTEmbeddings",
    "BERTEncoderLayer",
    "BERTForPreTraining",
    "CrossAttention",
    "FeedForward",
    "GPT",
    "GPTConfig",
    "LearnedPE",
    "MultiHeadAttention",
    "RotaryPE",
    "SinusoidalPE",
    "SwiGLU",
    "TransformerBlock",
    "attention_entropy",
    "causal_mask",
    "cross_entropy",
    "dropout",
    "gelu",
    "layer_norm",
    "log_softmax",
    "padding_mask",
    "relu",
    "rms_norm",
    "scaled_dot_product_attention",
    "silu",
    "softmax",
]

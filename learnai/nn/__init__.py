"""From-scratch neural network building blocks.

functional     softmax, gelu, layer_norm, ... written longhand
attention      scaled dot-product, multi-head, cross-attention, masks
pos_encoding   sinusoidal, learned, RoPE, ALiBi
blocks         FeedForward, SwiGLU, TransformerBlock
bert           BERTConfig, BERT, BERTForPreTraining — bidirectional encoder
gpt            GPTConfig, GPT — the full decoder-only model
lora           LoRALinear — low-rank adaptation and SVD diagnostics
resnet         residual stacks, original basic block, tiny image ResNet
word2vec       Skip-gram pairs and negative sampling
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
from learnai.nn.gpt_papers import (  # noqa: F401
    GPTSequenceClassifier,
    GPTTransferOutput,
    continuation_logprob,
    joint_transfer_loss,
    rank_completions,
)
from learnai.nn.lora import (  # noqa: F401
    LoRALinear,
    low_rank_approximation,
    relative_frobenius_error,
)
from learnai.nn.pos_encoding import ALiBi, LearnedPE, RotaryPE, SinusoidalPE  # noqa: F401
from learnai.nn.resnet import (  # noqa: F401
    BasicResidualBlock,
    PlainLinearStack,
    ResidualLinearStack,
    TinyResNet,
)
from learnai.nn.word2vec import (  # noqa: F401
    SkipGramNegativeSampling,
    negative_sampling_loss,
    sample_negative_ids,
    skipgram_pairs,
    subsampling_keep_probability,
    unigram_noise_distribution,
)

__all__ = [
    "ALiBi",
    "BERT",
    "BERTConfig",
    "BERTEmbeddings",
    "BERTEncoderLayer",
    "BERTForPreTraining",
    "BasicResidualBlock",
    "CrossAttention",
    "FeedForward",
    "GPT",
    "GPTConfig",
    "GPTSequenceClassifier",
    "GPTTransferOutput",
    "LearnedPE",
    "LoRALinear",
    "MultiHeadAttention",
    "PlainLinearStack",
    "ResidualLinearStack",
    "RotaryPE",
    "SinusoidalPE",
    "SwiGLU",
    "SkipGramNegativeSampling",
    "TinyResNet",
    "TransformerBlock",
    "attention_entropy",
    "causal_mask",
    "cross_entropy",
    "continuation_logprob",
    "dropout",
    "gelu",
    "layer_norm",
    "joint_transfer_loss",
    "low_rank_approximation",
    "log_softmax",
    "padding_mask",
    "negative_sampling_loss",
    "relu",
    "rms_norm",
    "relative_frobenius_error",
    "rank_completions",
    "sample_negative_ids",
    "scaled_dot_product_attention",
    "silu",
    "skipgram_pairs",
    "softmax",
    "subsampling_keep_probability",
    "unigram_noise_distribution",
]

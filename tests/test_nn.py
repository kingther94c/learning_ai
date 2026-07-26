"""The from-scratch implementations must match torch, exactly.

That is the whole contract of `learnai.nn`: it is written for reading, but it is
not a toy — if `learnai.nn.functional.gelu` disagreed with `F.gelu`, every
explanation built on it would be quietly wrong. These tests are what make the
annotated code trustworthy enough to learn from.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from learnai.nn import (
    GPT,
    ALiBi,
    GPTConfig,
    LearnedPE,
    MultiHeadAttention,
    RotaryPE,
    SinusoidalPE,
    attention_entropy,
    causal_mask,
    cross_entropy,
    dropout,
    gelu,
    layer_norm,
    log_softmax,
    padding_mask,
    rms_norm,
    scaled_dot_product_attention,
    silu,
    softmax,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# ----------------------------------------------------------------------- functional


def test_softmax_matches_torch():
    x = torch.randn(4, 8, 16)
    assert torch.allclose(softmax(x), F.softmax(x, dim=-1), atol=1e-6)


def test_softmax_is_stable_on_extreme_inputs():
    """The max-subtraction is the whole point; naive exp() would overflow here."""
    x = torch.tensor([[1000.0, 999.0, 998.0]])
    p = softmax(x)
    assert torch.isfinite(p).all()
    assert torch.allclose(p.sum(-1), torch.ones(1), atol=1e-6)


def test_log_softmax_matches_torch():
    x = torch.randn(4, 16)
    assert torch.allclose(log_softmax(x), F.log_softmax(x, dim=-1), atol=1e-6)


def test_cross_entropy_matches_torch():
    logits = torch.randn(32, 10)
    targets = torch.randint(0, 10, (32,))
    assert torch.allclose(
        cross_entropy(logits, targets), F.cross_entropy(logits, targets), atol=1e-6
    )


def test_cross_entropy_of_a_uniform_model_is_log_vocab():
    """The sanity check every LM training run should start with."""
    logits = torch.zeros(64, 50)
    targets = torch.randint(0, 50, (64,))
    assert cross_entropy(logits, targets).item() == pytest.approx(math.log(50), abs=1e-5)


@pytest.mark.parametrize("approximate", [False, True])
def test_gelu_matches_torch(approximate):
    x = torch.randn(4, 32)
    ref = F.gelu(x, approximate="tanh" if approximate else "none")
    assert torch.allclose(gelu(x, approximate=approximate), ref, atol=1e-6)


def test_silu_matches_torch():
    x = torch.randn(4, 32)
    assert torch.allclose(silu(x), F.silu(x), atol=1e-6)


def test_layer_norm_matches_torch():
    x = torch.randn(4, 8, 16)
    w, b = torch.randn(16), torch.randn(16)
    assert torch.allclose(layer_norm(x), F.layer_norm(x, (16,)), atol=1e-5)
    assert torch.allclose(layer_norm(x, w, b), F.layer_norm(x, (16,), w, b), atol=1e-5)


def test_layer_norm_normalises_per_token():
    """Statistics are taken over the feature dim only — never across batch or time."""
    x = torch.randn(4, 8, 16) * 5 + 3
    y = layer_norm(x)
    assert torch.allclose(y.mean(-1), torch.zeros(4, 8), atol=1e-5)
    assert torch.allclose(y.var(-1, unbiased=False), torch.ones(4, 8), atol=1e-4)


def test_rms_norm_leaves_the_mean_alone():
    """RMSNorm differs from LayerNorm precisely by not re-centering."""
    x = torch.randn(4, 16) + 10.0
    y = rms_norm(x)
    assert y.mean(-1).abs().min() > 0.1, "RMSNorm must not zero the mean"
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(4), atol=1e-3)


def test_dropout_handles_the_probability_boundaries():
    x = torch.ones(8)
    assert dropout(x, p=0.0, training=True) is x
    assert dropout(x, p=0.7, training=False) is x
    assert torch.equal(dropout(x, p=1.0, training=True), torch.zeros_like(x))
    with pytest.raises(ValueError, match="probability"):
        dropout(x, p=1.1)


# ----------------------------------------------------------------------- attention


def test_sdpa_matches_torch():
    q, k, v = (torch.randn(2, 3, 7, 8) for _ in range(3))
    out, attn = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(out, F.scaled_dot_product_attention(q, k, v), atol=1e-5)
    assert torch.allclose(attn.sum(-1), torch.ones(2, 3, 7), atol=1e-6)


def test_sdpa_causal_matches_torch():
    q, k, v = (torch.randn(2, 3, 7, 8) for _ in range(3))
    out, attn = scaled_dot_product_attention(q, k, v, mask=causal_mask(7))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(out, ref, atol=1e-5)
    assert (attn.triu(diagonal=1) == 0).all(), "a causal head must not see the future"


def test_sdpa_fully_masked_rows_are_zero_even_in_float16():
    """An empty row means 'read nothing', not 'average every forbidden value'."""
    q = torch.ones(1, 1, 2, 4, dtype=torch.float16)
    k = torch.ones_like(q)
    v = torch.arange(8, dtype=torch.float16).view(1, 1, 2, 4)
    mask = torch.tensor([[[[True, False], [False, False]]]])

    out, attn = scaled_dot_product_attention(q, k, v, mask=mask)
    assert torch.equal(attn[0, 0, 0], torch.tensor([1.0, 0.0], dtype=torch.float16))
    assert torch.equal(attn[0, 0, 1], torch.zeros(2, dtype=torch.float16))
    assert torch.equal(out[0, 0, 0], v[0, 0, 0])
    assert torch.equal(out[0, 0, 1], torch.zeros(4, dtype=torch.float16))


def test_sdpa_requires_a_boolean_mask():
    q, k, v = (torch.randn(1, 1, 2, 4) for _ in range(3))
    with pytest.raises(TypeError, match="boolean"):
        scaled_dot_product_attention(q, k, v, mask=torch.ones(2, 2))


def test_attention_output_stays_inside_the_value_hull():
    """Attention output is a convex combination — it cannot invent new extremes."""
    q, k = torch.randn(1, 1, 5, 4), torch.randn(1, 1, 5, 4)
    v = torch.randn(1, 1, 5, 4)
    out, _ = scaled_dot_product_attention(q, k, v)
    assert (out <= v.amax(dim=2, keepdim=True) + 1e-5).all()
    assert (out >= v.amin(dim=2, keepdim=True) - 1e-5).all()


def test_padding_mask_hides_padding_from_everyone():
    lengths = torch.tensor([3, 5])
    mask = padding_mask(lengths, seq_len=5)
    assert mask.shape == (2, 1, 1, 5)
    assert mask[0, 0, 0].tolist() == [True, True, True, False, False]
    assert mask[1, 0, 0].all()


def test_multihead_attention_shapes_and_normalisation():
    mha = MultiHeadAttention(d_model=32, n_heads=4, causal=True)
    x = torch.randn(2, 9, 32)
    y, attn = mha(x, return_attn=True)
    assert y.shape == x.shape
    assert attn.shape == (2, 4, 9, 9)
    assert torch.allclose(attn.sum(-1), torch.ones(2, 4, 9), atol=1e-6)
    assert (attn.triu(diagonal=1) == 0).all()


def test_multihead_attention_rejects_indivisible_dims():
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(d_model=30, n_heads=4)


def test_attention_entropy_bounds():
    """0 for a one-hot row, log(T) for a uniform one."""
    T = 8
    spike = torch.zeros(1, 1, 1, T)
    spike[..., 0] = 1.0
    uniform = torch.full((1, 1, 1, T), 1.0 / T)
    assert attention_entropy(spike).item() == pytest.approx(0.0, abs=1e-6)
    assert attention_entropy(uniform).item() == pytest.approx(math.log(T), abs=1e-6)


def test_scaling_prevents_softmax_saturation():
    """The measured justification for the /sqrt(d_k) in the formula."""

    def mean_entropy(d, scale):
        q, k = torch.randn(1, 1, 16, d), torch.randn(1, 1, 16, d)
        w = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        return float(attention_entropy(w).mean())

    torch.manual_seed(0)
    assert mean_entropy(1024, 1.0) < 0.5 * mean_entropy(4, 1.0)
    torch.manual_seed(0)
    scaled_wide = mean_entropy(1024, 1024**-0.5)
    torch.manual_seed(0)
    scaled_narrow = mean_entropy(4, 4**-0.5)
    assert abs(scaled_wide - scaled_narrow) < 0.5


# ----------------------------------------------------------------- position encodings


def test_sinusoidal_shapes_including_odd_d_model():
    for d in (16, 15):
        pe = SinusoidalPE(d_model=d, max_len=64)
        x = torch.randn(2, 10, d)
        assert pe(x).shape == x.shape


def test_position_encodings_preserve_dtype_and_report_table_limits():
    x = torch.zeros(1, 4, 8, dtype=torch.float16)
    pe = SinusoidalPE(d_model=8, max_len=4)
    assert pe(x).dtype == torch.float16
    with pytest.raises(ValueError, match="exceeds max_len"):
        pe(torch.zeros(1, 5, 8))

    rope = RotaryPE(d_head=8, max_len=4)
    q = torch.randn(1, 2, 2, 8, dtype=torch.float16)
    q_rot, k_rot = rope(q, q, offset=2)
    assert q_rot.dtype == k_rot.dtype == torch.float16
    with pytest.raises(ValueError, match="exceed max_len"):
        rope(q, q, offset=3)

    alibi = ALiBi(n_heads=2, max_len=4)
    assert alibi(torch.zeros(1, 2, 4, 4, dtype=torch.float16)).dtype == torch.float16


def test_learned_pe_rejects_sequences_past_its_table():
    pe = LearnedPE(d_model=8, max_len=16)
    assert pe(torch.randn(1, 16, 8)).shape == (1, 16, 8)
    with pytest.raises(ValueError, match="exceeds max_len"):
        pe(torch.randn(1, 17, 8))


def test_rope_score_depends_only_on_relative_position():
    """The identity that makes RoPE work: <R_m q, R_n k> = f(m - n)."""
    rope = RotaryPE(d_head=16)
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)

    def score(m, n):
        qq, _ = rope(q, q, offset=m)
        _, kk = rope(k, k, offset=n)
        return float((qq * kk).sum())

    assert score(5, 3) == pytest.approx(score(9, 7), abs=1e-4)
    assert score(20, 4) == pytest.approx(score(30, 14), abs=1e-4)
    assert abs(score(5, 3) - score(5, 4)) > 1e-4, "different distances must differ"


def test_rope_rejects_odd_head_dim():
    with pytest.raises(ValueError, match="even head dim"):
        RotaryPE(d_head=15)


def test_alibi_penalty_grows_with_distance_and_varies_by_head():
    alibi = ALiBi(n_heads=4, max_len=32)
    biased = alibi(torch.zeros(1, 4, 10, 10))
    # Looking further back is penalised more.
    assert biased[0, 0, 9, 0] < biased[0, 0, 9, 5] < biased[0, 0, 9, 9]
    # Head 0 has the steepest slope, so it is the most local.
    assert biased[0, 0, 9, 0] < biased[0, 3, 9, 0]


# ----------------------------------------------------------------------- GPT


def test_gpt_forward_shapes_and_loss():
    cfg = GPTConfig(vocab_size=65, block_size=32, n_layer=2, n_head=4, d_model=64)
    model = GPT(cfg)
    idx = torch.randint(0, 65, (2, 16))
    logits, loss = model(idx, targets=idx)
    assert logits.shape == (2, 16, 65)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_gpt_initial_loss_is_log_vocab():
    """A fresh model must be exactly as good as guessing uniformly."""
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=65, block_size=64, n_layer=2, n_head=4, d_model=64)
    model = GPT(cfg)
    idx = torch.randint(0, 65, (16, 33))
    _, loss = model(idx[:, :-1], targets=idx[:, 1:])
    assert loss.item() == pytest.approx(math.log(65), abs=0.15)


def test_gpt_is_causal_end_to_end():
    """Changing a future token must not change the prediction at position t.

    This catches the class of bug that produces a suspiciously good loss curve.
    """
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=20, block_size=16, n_layer=2, n_head=2, d_model=32))
    model.eval()
    a = torch.randint(0, 20, (1, 10))
    b = a.clone()
    b[0, -1] = (b[0, -1] + 7) % 20  # perturb only the LAST token

    with torch.no_grad():
        la, _ = model(a)
        lb, _ = model(b)
    assert torch.allclose(la[:, :-1], lb[:, :-1], atol=1e-5)
    assert not torch.allclose(la[:, -1], lb[:, -1], atol=1e-5)


def test_weight_tying_shares_one_matrix():
    tied = GPT(GPTConfig(vocab_size=100, d_model=64, tie_weights=True))
    untied = GPT(GPTConfig(vocab_size=100, d_model=64, tie_weights=False))
    assert tied.lm_head.weight is tied.token_emb.weight
    assert untied.lm_head.weight is not untied.token_emb.weight
    assert untied.num_params() - tied.num_params() == 100 * 64


def test_gpt_rejects_sequences_past_block_size():
    model = GPT(GPTConfig(vocab_size=20, block_size=8))
    with pytest.raises(ValueError, match="exceeds block_size"):
        model(torch.randint(0, 20, (1, 9)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"temperature": 0.5},
        {"top_k": 5},
        {"top_p": 0.9},
        {"temperature": 0.8, "top_k": 10, "top_p": 0.95},
    ],
)
def test_generate_produces_valid_tokens(kwargs):
    model = GPT(GPTConfig(vocab_size=20, block_size=16, n_layer=1, n_head=2, d_model=32))
    out = model.generate(torch.zeros((2, 1), dtype=torch.long), max_new_tokens=8, **kwargs)
    assert out.shape == (2, 9)
    assert (out >= 0).all() and (out < 20).all()


def test_generate_crops_to_the_context_window():
    """Sampling past block_size must slide the window, not crash."""
    model = GPT(GPTConfig(vocab_size=20, block_size=8, n_layer=1, n_head=2, d_model=32))
    prompt = torch.randint(0, 20, (1, 8))
    out = model.generate(prompt, max_new_tokens=6)
    assert out.shape == (1, 14)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"temperature": 0.0}, "temperature"),
        ({"top_k": 0}, "top_k"),
        ({"top_p": 1.1}, "top_p"),
        ({"max_new_tokens": -1}, "max_new_tokens"),
    ],
)
def test_generate_rejects_invalid_sampling_parameters(kwargs, message):
    model = GPT(GPTConfig(vocab_size=20, block_size=8, n_layer=1, n_head=2, d_model=32))
    prompt = torch.zeros((1, 1), dtype=torch.long)
    max_new_tokens = kwargs.pop("max_new_tokens", 1)
    with pytest.raises(ValueError, match=message):
        model.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)


def test_generate_restores_the_callers_training_mode():
    model = GPT(
        GPTConfig(
            vocab_size=20,
            block_size=8,
            n_layer=1,
            n_head=2,
            d_model=32,
            dropout=0.5,
        )
    )
    prompt = torch.zeros((1, 1), dtype=torch.long)

    model.train()
    model.generate(prompt, max_new_tokens=1, top_k=1)
    assert model.training

    model.eval()
    model.generate(prompt, max_new_tokens=1, top_k=1)
    assert not model.training


def test_top_k_of_one_is_greedy():
    """With k=1 there is no randomness left; two runs must agree exactly."""
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=20, block_size=16, n_layer=1, n_head=2, d_model=32))
    prompt = torch.randint(0, 20, (1, 4))
    a = model.generate(prompt, max_new_tokens=6, top_k=1)
    b = model.generate(prompt, max_new_tokens=6, top_k=1)
    assert torch.equal(a, b)

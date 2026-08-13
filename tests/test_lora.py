"""Tests for LoRA's low-rank update, frozen base, and merged inference path."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from learnai.nn import LoRALinear, low_rank_approximation, relative_frobenius_error


def test_lora_initialisation_preserves_the_pretrained_function():
    torch.manual_seed(0)
    linear = nn.Linear(7, 5)
    lora = LoRALinear.from_linear(linear, rank=2, alpha=4)
    x = torch.randn(3, 7)

    assert torch.equal(lora.lora_B, torch.zeros_like(lora.lora_B))
    assert torch.allclose(lora(x), linear(x))
    assert not lora.base.weight.requires_grad
    assert lora.base.bias is not None and not lora.base.bias.requires_grad
    assert lora.lora_A.requires_grad and lora.lora_B.requires_grad


def test_lora_forward_matches_explicit_delta_weight():
    torch.manual_seed(1)
    lora = LoRALinear(4, 3, rank=2, alpha=6, bias=True)
    with torch.no_grad():
        lora.lora_B.normal_()
    x = torch.randn(2, 5, 4)

    expected = F.linear(x, lora.base.weight + lora.delta_weight(), lora.base.bias)
    assert torch.allclose(lora(x), expected, atol=1e-6)
    assert lora.delta_weight().shape == lora.base.weight.shape == (3, 4)


def test_only_low_rank_factors_receive_gradients():
    lora = LoRALinear(9, 6, rank=3)
    x = torch.randn(8, 9)
    lora(x).square().mean().backward()

    assert lora.base.weight.grad is None
    assert lora.base.bias is None or lora.base.bias.grad is None
    assert lora.lora_A.grad is not None
    assert lora.lora_B.grad is not None
    assert lora.trainable_parameter_count == 3 * (9 + 6)
    assert (
        sum(parameter.numel() for parameter in lora.parameters() if parameter.requires_grad) == 45
    )


def test_merge_is_idempotent_and_unmerge_recovers_both_paths():
    torch.manual_seed(2)
    lora = LoRALinear(6, 4, rank=2, alpha=2).eval()
    with torch.no_grad():
        lora.lora_B.normal_()
    x = torch.randn(5, 6)
    original_weight = lora.base.weight.detach().clone()
    unmerged_output = lora(x)

    lora.merge()
    once_merged_weight = lora.base.weight.detach().clone()
    merged_output = lora(x)
    lora.merge()
    assert torch.equal(lora.base.weight, once_merged_weight)
    assert torch.allclose(merged_output, unmerged_output, atol=1e-6)

    lora.unmerge()
    lora.unmerge()
    assert torch.allclose(lora.base.weight, original_weight, atol=1e-6)
    assert torch.allclose(lora(x), unmerged_output, atol=1e-6)


def test_merge_requires_eval_and_train_restores_adapter_gradients():
    lora = LoRALinear(5, 3, rank=2)
    with pytest.raises(RuntimeError, match="inference"):
        lora.merge()

    with torch.no_grad():
        lora.lora_B.normal_()
    lora.eval().merge()
    assert lora.merged
    lora.train()
    assert not lora.merged
    lora(torch.randn(4, 5)).sum().backward()
    assert lora.lora_A.grad is not None and lora.lora_B.grad is not None


def test_unmerge_uses_the_exact_update_that_was_merged():
    lora = LoRALinear(5, 3, rank=2).eval()
    with torch.no_grad():
        lora.lora_B.normal_()
    original = lora.base.weight.detach().clone()
    lora.merge()
    with torch.no_grad():
        lora.lora_B.add_(10.0)
    lora.unmerge()
    assert torch.allclose(lora.base.weight, original, atol=1e-6)


def test_adapter_only_state_round_trip_excludes_frozen_base():
    source = LoRALinear(6, 4, rank=2)
    target = LoRALinear(6, 4, rank=2)
    with torch.no_grad():
        source.lora_A.normal_()
        source.lora_B.normal_()
    state = source.adapter_state_dict()
    target.load_adapter_state_dict(state)

    assert set(state) == {"lora_A", "lora_B"}
    assert sum(tensor.numel() for tensor in state.values()) == source.trainable_parameter_count
    assert torch.equal(target.lora_A, source.lora_A)
    assert torch.equal(target.lora_B, source.lora_B)


def test_truncated_svd_recovers_a_known_rank_two_matrix():
    torch.manual_seed(3)
    matrix = torch.randn(8, 2) @ torch.randn(2, 6)
    rank_one, singular_values = low_rank_approximation(matrix, rank=1)
    rank_two, _ = low_rank_approximation(matrix, rank=2)

    assert singular_values.shape == (6,)
    assert relative_frobenius_error(matrix, rank_two) < 1e-5
    assert relative_frobenius_error(matrix, rank_one) > relative_frobenius_error(matrix, rank_two)


def test_lora_validates_rank_dropout_and_svd_shapes():
    with pytest.raises(ValueError, match="rank"):
        LoRALinear(2, 2, rank=0)
    with pytest.raises(ValueError, match="dropout"):
        LoRALinear(2, 2, dropout=1.0)
    with pytest.raises(ValueError, match="two-dimensional"):
        low_rank_approximation(torch.ones(2, 2, 2), rank=1)
    with pytest.raises(ValueError, match="share a shape"):
        relative_frobenius_error(torch.ones(2, 2), torch.ones(3, 2))

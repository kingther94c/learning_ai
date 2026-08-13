"""Tests for residual algebra, identity shortcuts, and the small ResNet."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from learnai.nn import BasicResidualBlock, PlainLinearStack, ResidualLinearStack, TinyResNet


def test_linear_stacks_match_their_explicit_matrix_products():
    torch.manual_seed(0)
    plain = PlainLinearStack(4, depth=3)
    residual = ResidualLinearStack(4, depth=3)
    x = torch.randn(5, 4)

    plain_reference = x
    for layer in plain.layers:
        plain_reference = plain_reference @ layer.weight.T

    residual_reference = x
    for layer in residual.layers:
        residual_reference = residual_reference @ (torch.eye(4) + layer.weight).T

    assert torch.allclose(plain(x), plain_reference)
    assert torch.allclose(residual(x), residual_reference)


def test_zero_residual_branches_are_exact_identity_with_identity_gradient():
    model = ResidualLinearStack(6, depth=12)
    for layer in model.layers:
        nn.init.zeros_(layer.weight)

    x = torch.randn(3, 6, requires_grad=True)
    output = model(x)
    output.sum().backward()
    assert torch.equal(output, x)
    assert torch.equal(x.grad, torch.ones_like(x))


def test_plain_and_residual_stacks_have_identical_parameter_budgets():
    plain = PlainLinearStack(7, depth=9)
    residual = ResidualLinearStack(7, depth=9)
    assert sum(parameter.numel() for parameter in plain.parameters()) == 7 * 7 * 9
    assert sum(parameter.numel() for parameter in residual.parameters()) == 7 * 7 * 9


def test_basic_block_uses_free_identity_shortcut_when_shapes_match():
    block = BasicResidualBlock(8, 8).eval()
    assert isinstance(block.shortcut, nn.Identity)
    x = torch.randn(2, 8, 11, 13)
    assert block(x).shape == x.shape


def test_basic_block_projects_when_resolution_and_width_change():
    block = BasicResidualBlock(8, 16, stride=2).eval()
    assert isinstance(block.shortcut, nn.Sequential)
    x = torch.randn(2, 8, 12, 10)
    output = block(x)
    assert output.shape == (2, 16, 6, 5)


def test_basic_block_matches_manual_main_plus_shortcut_formula():
    torch.manual_seed(1)
    block = BasicResidualBlock(4, 4).eval()
    x = torch.randn(2, 4, 7, 7)
    main = block.bn2(block.conv2(torch.relu(block.bn1(block.conv1(x)))))
    expected = torch.relu(main + x)
    assert torch.allclose(block(x), expected)


def test_tiny_resnet_accepts_different_image_sizes_and_backpropagates():
    model = TinyResNet(num_classes=5, blocks_per_stage=(1, 1), widths=(8, 16))
    images = torch.randn(4, 3, 24, 20)
    logits = model(images)
    assert logits.shape == (4, 5)
    logits.square().mean().backward()
    assert model.stem[0].weight.grad is not None
    assert model.classifier.weight.grad is not None


def test_resnet_validates_invalid_architectures():
    with pytest.raises(ValueError, match="depth"):
        PlainLinearStack(3, depth=0)
    with pytest.raises(ValueError, match="same non-zero length"):
        TinyResNet(num_classes=2, blocks_per_stage=(1,), widths=(8, 16))
    with pytest.raises(ValueError, match="positive"):
        BasicResidualBlock(0, 8)

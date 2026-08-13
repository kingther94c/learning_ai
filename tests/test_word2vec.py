"""Tests for the exact data and objective mechanics behind word2vec SGNS."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from learnai.nn import (
    SkipGramNegativeSampling,
    negative_sampling_loss,
    sample_negative_ids,
    skipgram_pairs,
    subsampling_keep_probability,
    unigram_noise_distribution,
)


def test_skipgram_pairs_respect_window_and_sequence_edges():
    centers, contexts = skipgram_pairs([0, 1, 2, 3], window_size=1)
    assert centers.tolist() == [0, 1, 1, 2, 2, 3]
    assert contexts.tolist() == [1, 0, 2, 1, 3, 2]

    wide_centers, wide_contexts = skipgram_pairs([4, 5, 6], window_size=9)
    assert len(wide_centers) == 6
    assert all(
        center != context for center, context in zip(wide_centers, wide_contexts, strict=True)
    )


def test_unigram_noise_distribution_uses_three_quarter_power():
    # Counts are 1, 4, 0. Their unnormalised masses must be 1 and 4**0.75.
    distribution = unigram_noise_distribution([0, 1, 1, 1, 1], vocab_size=3)
    assert distribution.sum().item() == pytest.approx(1.0)
    assert distribution[1] / distribution[0] == pytest.approx(4**0.75)
    assert distribution[2] == 0


def test_paper_and_reference_code_subsampling_formulas_stay_distinct():
    frequencies = torch.tensor([0.0, 1e-6, 1e-4, 1e-2])
    paper = subsampling_keep_probability(frequencies, threshold=1e-5, variant="paper")
    reference = subsampling_keep_probability(frequencies, threshold=1e-5, variant="reference_code")

    assert torch.equal(paper[:2], torch.ones(2))
    assert paper[2].item() == pytest.approx(math.sqrt(0.1))
    assert paper[3].item() == pytest.approx(math.sqrt(0.001))
    assert torch.all(reference >= paper)
    assert reference[2].item() == pytest.approx(math.sqrt(0.1) + 0.1)


def test_negative_sampler_has_requested_shape_and_seed():
    distribution = torch.tensor([0.1, 0.2, 0.7])
    first = sample_negative_ids(
        distribution,
        batch_size=5,
        num_negatives=4,
        generator=torch.Generator().manual_seed(11),
    )
    second = sample_negative_ids(
        distribution,
        batch_size=5,
        num_negatives=4,
        generator=torch.Generator().manual_seed(11),
    )
    assert first.shape == (5, 4)
    assert torch.equal(first, second)


def test_negative_sampling_loss_matches_binary_logistic_reference():
    center = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
    positive = torch.tensor([[0.5, -1.0], [2.0, 1.0]])
    negatives = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]],
            [[0.5, 0.5], [-0.5, 2.0], [1.0, -2.0]],
        ]
    )

    actual = negative_sampling_loss(center, positive, negatives, reduction="none")
    positive_scores = (center * positive).sum(dim=-1, keepdim=True)
    negative_scores = torch.einsum("bd,bkd->bk", center, negatives)
    logits = torch.cat([positive_scores, negative_scores], dim=1)
    labels = torch.zeros_like(logits)
    labels[:, 0] = 1
    expected = F.binary_cross_entropy_with_logits(logits, labels, reduction="none").sum(dim=1)
    assert torch.allclose(actual, expected)


def test_zero_context_initialisation_has_known_loss_and_gradients():
    torch.manual_seed(0)
    model = SkipGramNegativeSampling(vocab_size=8, embedding_dim=5)
    centers = torch.tensor([0, 1, 2, 3])
    positives = torch.tensor([1, 2, 3, 4])
    negatives = torch.tensor([[5, 6, 7], [4, 6, 7], [0, 5, 7], [0, 1, 6]])

    loss = model(centers, positives, negatives)
    assert loss.item() == pytest.approx(4 * math.log(2), rel=1e-6)
    loss.backward()
    assert model.context.weight.grad is not None
    assert model.context.weight.grad.abs().sum() > 0
    # With all context vectors initially zero, center vectors receive signal only
    # after the context table has moved away from zero.
    assert model.center.weight.grad.abs().sum() == 0


def test_normalized_embeddings_have_unit_rows():
    model = SkipGramNegativeSampling(vocab_size=6, embedding_dim=4)
    norms = model.normalized_embeddings().norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_word2vec_validates_bad_shapes_and_hyperparameters():
    with pytest.raises(ValueError, match="window_size"):
        skipgram_pairs([1, 2], 0)
    with pytest.raises(ValueError, match="non-empty"):
        unigram_noise_distribution([])
    with pytest.raises(ValueError, match="shape"):
        negative_sampling_loss(torch.ones(2, 3), torch.ones(2, 3), torch.ones(2, 3))
    with pytest.raises(ValueError, match="num_negatives"):
        sample_negative_ids(torch.ones(3), 2, 0)
    with pytest.raises(ValueError, match="variant"):
        subsampling_keep_probability(torch.tensor([0.5]), variant="unknown")

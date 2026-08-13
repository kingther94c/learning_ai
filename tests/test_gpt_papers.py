"""Tests for the transfer and in-context mechanisms used in GPT lessons."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from learnai.data.gpt_tasks import (
    ARROW_TOKEN_ID,
    IGNORE_INDEX,
    NUMBER_TOKEN_OFFSET,
    make_modular_shift_batch,
    modular_shift_accuracy,
)
from learnai.nn import GPT, GPTConfig
from learnai.nn.gpt_papers import (
    GPTSequenceClassifier,
    continuation_logprob,
    joint_transfer_loss,
    rank_completions,
)


def tiny_gpt(**overrides) -> GPT:
    values = dict(
        vocab_size=19,
        block_size=16,
        n_layer=2,
        n_head=2,
        d_model=16,
        d_ff=32,
        dropout=0.0,
    )
    values.update(overrides)
    return GPT(GPTConfig(**values))


def test_encode_is_the_hidden_state_used_by_the_lm_head():
    torch.manual_seed(0)
    model = tiny_gpt().eval()
    ids = torch.randint(0, 19, (3, 7))

    hidden = model.encode(ids)
    logits, _ = model(ids)

    assert hidden.shape == (3, 7, 16)
    assert torch.allclose(logits, model.lm_head(hidden), atol=1e-7)
    with pytest.raises(ValueError, match="at least one"):
        model.encode(ids[:, :0])
    with pytest.raises(TypeError, match="integer"):
        model.encode(ids.float())


def test_sequence_classifier_reads_each_rows_last_real_token():
    torch.manual_seed(1)
    model = GPTSequenceClassifier(tiny_gpt(), num_classes=3).eval()
    ids = torch.randint(0, 19, (2, 6))
    lengths = torch.tensor([3, 6])

    output = model(ids, lengths)
    rows = torch.arange(2)
    expected = model.classifier(output.hidden_states[rows, lengths - 1])

    assert output.class_logits.shape == (2, 3)
    assert output.lm_logits.shape == (2, 6, 19)
    assert torch.allclose(output.class_logits, expected)
    with pytest.raises(ValueError, match="lengths"):
        model(ids, torch.tensor([0, 6]))
    with pytest.raises(TypeError, match="integers"):
        model(ids, torch.tensor([3.0, 6.0]))


def test_joint_transfer_loss_matches_explicit_weighted_objective():
    torch.manual_seed(2)
    model = GPTSequenceClassifier(tiny_gpt(), num_classes=2)
    ids = torch.randint(0, 19, (3, 6))
    labels = torch.tensor([0, 1, 1])
    targets = torch.full_like(ids, IGNORE_INDEX)
    targets[:, :-1] = ids[:, 1:]
    output = model(ids)

    total, class_loss, lm_loss = joint_transfer_loss(output, labels, targets, lm_weight=0.4)
    expected_class = F.cross_entropy(output.class_logits, labels)
    expected_lm = F.cross_entropy(
        output.lm_logits.reshape(-1, 19), targets.reshape(-1), ignore_index=IGNORE_INDEX
    )

    assert torch.allclose(class_loss, expected_class)
    assert torch.allclose(lm_loss, expected_lm)
    assert torch.allclose(total, expected_class + 0.4 * expected_lm)
    with pytest.raises(ValueError, match="non-negative"):
        joint_transfer_loss(output, labels, targets, lm_weight=-1)


def test_joint_transfer_loss_handles_no_lm_targets_without_nan():
    model = GPTSequenceClassifier(tiny_gpt(), num_classes=2)
    ids = torch.randint(0, 19, (2, 4))
    output = model(ids)
    all_ignored = torch.full_like(ids, IGNORE_INDEX)

    total, class_loss, lm_loss = joint_transfer_loss(
        output, torch.tensor([0, 1]), all_ignored, lm_weight=1.0
    )

    assert lm_loss.item() == 0.0
    assert torch.allclose(total, class_loss)
    total.backward()
    assert model.classifier.weight.grad is not None


def test_continuation_score_has_the_correct_causal_shift_and_restores_mode():
    torch.manual_seed(3)
    model = tiny_gpt(dropout=0.2).train()
    prefix = torch.tensor([1, 4, 2, 6])
    continuation = torch.tensor([3, 8, 5])
    whole = torch.cat((prefix, continuation)).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        logits, _ = model(whole)
    model.train()
    start = prefix.numel() - 1
    per_token = F.log_softmax(logits[0, start : start + continuation.numel()], dim=-1).gather(
        -1, continuation[:, None]
    )

    score = continuation_logprob(model, prefix, continuation)
    mean_score = continuation_logprob(model, prefix, continuation, reduction="mean")

    assert torch.allclose(score, per_token.sum(), atol=1e-6)
    assert torch.allclose(mean_score, per_token.mean(), atol=1e-6)
    assert model.training
    assert not score.requires_grad


def test_completion_ranking_returns_input_scores_and_descending_indices():
    torch.manual_seed(4)
    model = tiny_gpt().eval()
    prefix = torch.tensor([1, 2, 3])
    candidates = [torch.tensor([4, 5]), torch.tensor([6]), torch.tensor([7, 8])]

    scores, order = rank_completions(model, prefix, candidates, reduction="mean")
    direct = torch.stack(
        [
            continuation_logprob(model, prefix, candidate, reduction="mean")
            for candidate in candidates
        ]
    )

    assert torch.allclose(scores, direct)
    assert torch.equal(order, scores.argsort(descending=True))
    with pytest.raises(ValueError, match="at least one"):
        rank_completions(model, prefix, [])


def test_continuation_scoring_uses_the_full_context_boundary_without_an_extra_logit():
    torch.manual_seed(40)
    model = tiny_gpt(block_size=4).eval()
    prefix = torch.tensor([1, 2, 3, 4])
    continuation = torch.tensor([5])

    with torch.no_grad():
        logits, _ = model(prefix.unsqueeze(0))
        expected = F.log_softmax(logits[0, -1], dim=-1)[continuation[0]]

    assert torch.allclose(continuation_logprob(model, prefix, continuation), expected)
    with pytest.raises(ValueError, match="teacher-forcing"):
        continuation_logprob(model, prefix, torch.tensor([5, 6]))


def test_modular_shift_batch_has_a_fresh_coherent_task_per_row():
    modulus = 7
    shots = 3
    inputs, targets = make_modular_shift_batch(
        32, shots, modulus, generator=torch.Generator().manual_seed(5)
    )

    assert inputs.shape == targets.shape == (32, 4 * shots + 3)
    assert inputs.min() >= 0 and inputs.max() < modulus + NUMBER_TOKEN_OFFSET
    selected = targets != IGNORE_INDEX
    expected_selected = torch.zeros_like(selected)
    expected_selected[:, -1] = True
    assert torch.equal(selected, expected_selected)
    answer_positions = torch.arange(2, inputs.size(1), 4)
    assert inputs[:, answer_positions].eq(ARROW_TOKEN_ID).all()

    first_x = inputs[:, 1] - NUMBER_TOKEN_OFFSET
    first_y = inputs[:, 3] - NUMBER_TOKEN_OFFSET
    shifts = (first_y - first_x) % modulus
    for demonstration in range(shots):
        start = 1 + 4 * demonstration
        x = inputs[:, start] - NUMBER_TOKEN_OFFSET
        y = inputs[:, start + 2] - NUMBER_TOKEN_OFFSET
        assert torch.equal((y - x) % modulus, shifts)
    query = inputs[:, -2] - NUMBER_TOKEN_OFFSET
    query_answer = targets[:, -1] - NUMBER_TOKEN_OFFSET
    assert torch.equal((query_answer - query) % modulus, shifts)
    demonstrated_values = torch.stack(
        [inputs[:, 1 + 4 * demonstration] for demonstration in range(shots)], dim=1
    )
    assert (demonstrated_values == inputs[:, -2, None]).any(dim=1).all()
    assert shifts.unique().numel() > 1


def test_modular_shift_generator_is_reproducible_and_shuffle_only_breaks_demos():
    clean = make_modular_shift_batch(64, 3, generator=torch.Generator().manual_seed(6))
    repeated = make_modular_shift_batch(64, 3, generator=torch.Generator().manual_seed(6))
    shuffled = make_modular_shift_batch(
        64, 3, generator=torch.Generator().manual_seed(6), shuffle_labels=True
    )

    assert all(torch.equal(a, b) for a, b in zip(clean, repeated, strict=True))
    clean_inputs, clean_targets = clean
    shuffled_inputs, shuffled_targets = shuffled
    assert torch.equal(clean_inputs[:, -2:], shuffled_inputs[:, -2:])
    assert torch.equal(clean_targets[:, -1], shuffled_targets[:, -1])
    assert (clean_inputs[:, 3:-2:4] != shuffled_inputs[:, 3:-2:4]).all()

    single_clean = make_modular_shift_batch(
        1, 2, generator=torch.Generator().manual_seed(60), shuffle_labels=False
    )[0]
    single_shuffled = make_modular_shift_batch(
        1, 2, generator=torch.Generator().manual_seed(60), shuffle_labels=True
    )[0]
    assert (single_clean[:, 3:-2:4] != single_shuffled[:, 3:-2:4]).all()


class _ShiftOracle(nn.Module):
    """Read the first demonstration and solve the query; useful for helper tests."""

    def __init__(self, modulus: int):
        super().__init__()
        self.modulus = modulus
        self.vocab_size = modulus + NUMBER_TOKEN_OFFSET

    def forward(self, inputs):
        first_x = inputs[:, 1] - NUMBER_TOKEN_OFFSET
        first_y = inputs[:, 3] - NUMBER_TOKEN_OFFSET
        shift = (first_y - first_x) % self.modulus
        query = inputs[:, -2] - NUMBER_TOKEN_OFFSET
        answer = (query + shift) % self.modulus + NUMBER_TOKEN_OFFSET
        logits = torch.zeros(*inputs.shape, self.vocab_size, device=inputs.device)
        logits[torch.arange(inputs.size(0)), -1, answer] = 10.0
        return logits, None


def test_modular_shift_accuracy_is_inference_only_and_restores_mode():
    oracle = _ShiftOracle(modulus=7).train()
    accuracy = modular_shift_accuracy(
        oracle,
        shots=2,
        episodes=513,
        modulus=7,
        generator=torch.Generator().manual_seed(7),
    )

    assert accuracy == 1.0
    assert oracle.training


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch_size": 0, "shots": 1}, "batch_size"),
        ({"batch_size": 1, "shots": -1}, "shots"),
        ({"batch_size": 1, "shots": 1, "modulus": 1}, "modulus"),
    ],
)
def test_modular_shift_batch_rejects_invalid_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        make_modular_shift_batch(**kwargs)

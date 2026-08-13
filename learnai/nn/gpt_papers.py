"""Small, tested bridges from a causal GPT to the GPT-1/2 paper protocols.

The core :class:`~learnai.nn.gpt.GPT` only knows next-token prediction.  GPT-1
adds a sequence classifier and optionally keeps the LM objective while
fine-tuning; GPT-2 often turns a task into candidate continuations and scores
them with that same LM.  These helpers keep those mechanisms visible and avoid
reimplementing subtly different versions in notebooks.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from learnai.nn.gpt import GPT


@dataclass
class GPTTransferOutput:
    """Outputs needed by GPT-1-style discriminative fine-tuning."""

    hidden_states: torch.Tensor
    lm_logits: torch.Tensor
    class_logits: torch.Tensor


class GPTSequenceClassifier(nn.Module):
    """Attach one linear classification head to a GPT's last real token.

    This mirrors GPT-1's minimal transfer architecture: the causal decoder is
    reused and the final token acts as a summary because it can attend to every
    token on its left.  ``lengths`` makes the same rule work for padded batches.
    """

    def __init__(self, backbone: GPT, num_classes: int):
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.config.d_model, num_classes)

    def forward(self, idx: torch.Tensor, lengths: torch.Tensor | None = None) -> GPTTransferOutput:
        hidden = self.backbone.encode(idx)
        batch_size, sequence_length, _ = hidden.shape

        if lengths is None:
            lengths = torch.full(
                (batch_size,), sequence_length, dtype=torch.long, device=idx.device
            )
        else:
            if lengths.shape != (batch_size,):
                raise ValueError(
                    f"lengths must have shape ({batch_size},), got {tuple(lengths.shape)}"
                )
            if lengths.dtype not in (torch.int32, torch.int64):
                raise TypeError(f"lengths must contain integers, got {lengths.dtype}")
            lengths = lengths.to(device=idx.device, dtype=torch.long)
            if bool(((lengths < 1) | (lengths > sequence_length)).any()):
                raise ValueError(f"lengths must be in [1, {sequence_length}]")

        rows = torch.arange(batch_size, device=idx.device)
        summary = hidden[rows, lengths - 1]
        return GPTTransferOutput(
            hidden_states=hidden,
            lm_logits=self.backbone.lm_head(hidden),
            class_logits=self.classifier(summary),
        )


def joint_transfer_loss(
    output: GPTTransferOutput,
    labels: torch.Tensor,
    lm_targets: torch.Tensor | None = None,
    lm_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPT-1 fine-tuning loss: classification plus weighted language modelling.

    ``lm_targets`` is aligned with ``output.lm_logits`` and may use ``-100`` to
    exclude positions.  Returning both terms makes the trade-off observable in
    a lesson instead of hiding it inside a training loop.
    """
    if not math.isfinite(lm_weight) or lm_weight < 0:
        raise ValueError(f"lm_weight must be finite and non-negative, got {lm_weight}")
    if labels.ndim != 1 or labels.size(0) != output.class_logits.size(0):
        raise ValueError(
            "labels must be one-dimensional with one value per classification logit row"
        )

    class_loss = F.cross_entropy(output.class_logits, labels)
    if lm_targets is None:
        lm_loss = output.lm_logits.sum() * 0.0
    else:
        if lm_targets.shape != output.lm_logits.shape[:-1]:
            raise ValueError(
                "lm_targets must match the batch/time dimensions of lm_logits: "
                f"expected {tuple(output.lm_logits.shape[:-1])}, got {tuple(lm_targets.shape)}"
            )
        if bool((lm_targets != -100).any()):
            lm_loss = F.cross_entropy(
                output.lm_logits.reshape(-1, output.lm_logits.size(-1)),
                lm_targets.reshape(-1),
                ignore_index=-100,
            )
        else:
            lm_loss = output.lm_logits.sum() * 0.0

    total = class_loss + lm_weight * lm_loss
    return total, class_loss, lm_loss


@torch.no_grad()
def continuation_logprob(
    model: GPT,
    prefix_ids: torch.Tensor,
    continuation_ids: torch.Tensor,
    *,
    reduction: str = "sum",
) -> torch.Tensor:
    """Score only ``continuation_ids`` under a causal LM.

    The first continuation token is predicted by the final prefix position.
    That one-position shift is the common source of silent candidate-scoring
    bugs, so the slice is deliberately written out here.
    """
    _validate_token_vector("prefix_ids", prefix_ids, non_empty=True)
    _validate_token_vector("continuation_ids", continuation_ids, non_empty=True)
    if prefix_ids.device != continuation_ids.device:
        raise ValueError("prefix_ids and continuation_ids must be on the same device")
    if reduction not in {"sum", "mean"}:
        raise ValueError(f"reduction must be 'sum' or 'mean', got {reduction!r}")

    total_length = prefix_ids.numel() + continuation_ids.numel()
    teacher_forcing_length = total_length - 1
    if teacher_forcing_length > model.config.block_size:
        raise ValueError(
            "prefix + continuation needs a teacher-forcing input of length "
            f"{teacher_forcing_length}, exceeding block_size {model.config.block_size}"
        )

    was_training = model.training
    model.eval()
    try:
        whole = torch.cat((prefix_ids, continuation_ids))
        # Token t is predicted by the logit at t-1.  The final token therefore
        # belongs in the targets but not the model input; avoiding that unused
        # last logit also makes total_length == block_size + 1 a legal boundary.
        logits, _ = model(whole[:-1].unsqueeze(0))
        start = prefix_ids.numel() - 1
        stop = start + continuation_ids.numel()
        token_logits = logits[0, start:stop]
        token_logprobs = F.log_softmax(token_logits, dim=-1).gather(
            dim=-1, index=continuation_ids.unsqueeze(-1)
        )
        return token_logprobs.sum() if reduction == "sum" else token_logprobs.mean()
    finally:
        model.train(was_training)


@torch.no_grad()
def rank_completions(
    model: GPT,
    prefix_ids: torch.Tensor,
    candidates: Sequence[torch.Tensor],
    *,
    reduction: str = "sum",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return candidate scores in input order and indices sorted best first."""
    if not candidates:
        raise ValueError("candidates must contain at least one continuation")
    scores = torch.stack(
        [
            continuation_logprob(model, prefix_ids, candidate, reduction=reduction)
            for candidate in candidates
        ]
    )
    return scores, scores.argsort(descending=True)


def _validate_token_vector(name: str, ids: torch.Tensor, *, non_empty: bool) -> None:
    if ids.ndim != 1:
        raise ValueError(f"{name} must have shape (T,), got {tuple(ids.shape)}")
    if non_empty and ids.numel() == 0:
        raise ValueError(f"{name} must contain at least one token")
    if ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must contain integer token ids, got {ids.dtype}")

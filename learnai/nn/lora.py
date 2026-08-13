"""Low-Rank Adaptation (LoRA) for a readable ``nn.Linear`` layer.

For a frozen pretrained weight ``W0`` LoRA learns ``Delta W = B @ A``, where
``A`` and ``B`` contain only ``r * (in + out)`` parameters.  The implementation
keeps the unmerged and merged paths explicit so lessons can prove their
numerical equivalence.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable rank-``r`` update.

    ``A`` starts Gaussian and ``B`` at zero, so construction does not change the
    pretrained function.  ``alpha / rank`` scales the update exactly as in the
    LoRA paper. ``alpha=None`` is the convenient unit-scaling default
    ``alpha=rank``; controlled rank sweeps should pass one fixed ``alpha``.
    Call :meth:`merge` before deployment to fold the update into the base weight
    and remove the extra matrix multiplies from inference.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int = 4,
        alpha: float | None = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if alpha is None:
            alpha = float(rank)
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))
        self.dropout = nn.Dropout(dropout)
        self.merged = False
        self._merged_delta: torch.Tensor | None = None
        self.reset_lora_parameters()
        self.freeze_base()

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        rank: int = 4,
        alpha: float | None = None,
        dropout: float = 0.0,
    ) -> LoRALinear:
        """Copy an existing ``nn.Linear`` into a function-preserving LoRA layer."""
        layer = cls(
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha,
            bias=linear.bias is not None,
            dropout=dropout,
        ).to(device=linear.weight.device, dtype=linear.weight.dtype)
        with torch.no_grad():
            layer.base.weight.copy_(linear.weight)
            if linear.bias is not None:
                layer.base.bias.copy_(linear.bias)
        layer.freeze_base()
        return layer

    def reset_lora_parameters(self) -> None:
        nn.init.normal_(self.lora_A, mean=0.0, std=1.0 / self.rank)
        nn.init.zeros_(self.lora_B)

    def freeze_base(self) -> None:
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

    def delta_weight(self) -> torch.Tensor:
        """Materialise the scaled update with the same shape as ``base.weight``."""
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        if self.merged:
            return result
        update = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return result + update * self.scaling

    def merge(self) -> LoRALinear:
        """Add ``Delta W`` into ``W0`` once and switch to the base-only path."""
        if self.training:
            raise RuntimeError("merge is for inference; call eval() before merge()")
        if not self.merged:
            with torch.no_grad():
                self._merged_delta = self.delta_weight().detach().clone()
                self.base.weight.add_(self._merged_delta)
            self.merged = True
        return self

    def unmerge(self) -> LoRALinear:
        """Recover the frozen base weight after :meth:`merge`."""
        if self.merged:
            if self._merged_delta is None:
                raise RuntimeError("merged LoRA layer is missing its cached update")
            with torch.no_grad():
                self.base.weight.sub_(self._merged_delta)
            self.merged = False
            self._merged_delta = None
        return self

    def train(self, mode: bool = True) -> LoRALinear:
        """Entering training mode always restores the explicit LoRA branch."""
        if mode and self.merged:
            self.unmerge()
        super().train(mode)
        return self

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only the task-specific matrices needed beside the frozen base."""
        return {
            "lora_A": self.lora_A.detach().clone(),
            "lora_B": self.lora_B.detach().clone(),
        }

    def load_adapter_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load matrices produced by :meth:`adapter_state_dict`."""
        if self.merged:
            self.unmerge()
        if set(state) != {"lora_A", "lora_B"}:
            raise ValueError("adapter state must contain exactly 'lora_A' and 'lora_B'")
        if state["lora_A"].shape != self.lora_A.shape or state["lora_B"].shape != self.lora_B.shape:
            raise ValueError("adapter state shapes do not match this LoRA layer")
        with torch.no_grad():
            self.lora_A.copy_(state["lora_A"])
            self.lora_B.copy_(state["lora_B"])

    @property
    def trainable_parameter_count(self) -> int:
        return self.rank * (self.in_features + self.out_features)


def low_rank_approximation(matrix: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the optimal rank-``r`` Frobenius approximation and singular values."""
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be two-dimensional, got shape {tuple(matrix.shape)}")
    maximum_rank = min(matrix.shape)
    if not 1 <= rank <= maximum_rank:
        raise ValueError(f"rank must be in [1, {maximum_rank}], got {rank}")

    left, singular_values, right_t = torch.linalg.svd(matrix, full_matrices=False)
    approximation = (left[:, :rank] * singular_values[:rank]) @ right_t[:rank]
    return approximation, singular_values


def relative_frobenius_error(reference: torch.Tensor, approximation: torch.Tensor) -> torch.Tensor:
    """Compute ``||reference - approximation||_F / ||reference||_F``."""
    if reference.shape != approximation.shape:
        raise ValueError(
            f"reference and approximation must share a shape, got {tuple(reference.shape)} "
            f"and {tuple(approximation.shape)}"
        )
    denominator = torch.linalg.matrix_norm(reference)
    if denominator == 0:
        raise ValueError("relative error is undefined for an all-zero reference")
    return torch.linalg.matrix_norm(reference - approximation) / denominator

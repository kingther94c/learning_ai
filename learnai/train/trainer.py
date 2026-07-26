"""A training loop with nothing hidden in it.

No Lightning, no accelerate, no config system. It stays small enough to read, so
that
when a run misbehaves you debug *your model*, not the framework.

    from learnai.train import Trainer, TrainConfig

    trainer = Trainer(model, cfg=TrainConfig(steps=2000, lr=3e-4))
    history = trainer.fit(train_batches, val_batches)
    history.plot()
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class TrainConfig:
    steps: int = 1000
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1  # cosine floor, as a fraction of lr
    eval_every: int = 100
    eval_steps: int = 20
    log_every: int = 100
    device: str | None = None

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError(f"steps must be positive, got {self.steps}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if len(self.betas) != 2 or not all(0.0 <= beta < 1.0 for beta in self.betas):
            raise ValueError(f"betas must contain two values in [0, 1), got {self.betas}")
        if self.grad_clip <= 0:
            raise ValueError(f"grad_clip must be positive, got {self.grad_clip}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError(f"min_lr_ratio must be in [0, 1], got {self.min_lr_ratio}")
        for name in ("eval_every", "eval_steps", "log_every"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")


@dataclass
class History:
    """Everything a run recorded. Plot it or export it to an interactive page."""

    steps: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_steps: list[int] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    lrs: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    wall_time: float = 0.0

    def plot(self, ax=None, logy: bool = False):
        """Train/val curves. Requires matplotlib."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        ax.plot(self.steps, self.train_loss, label="train", alpha=0.8, lw=1.2)
        if self.val_loss:
            ax.plot(self.val_steps, self.val_loss, label="val", marker="o", ms=3, lw=1.5)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        if logy:
            ax.set_yscale("log")
        ax.legend()
        ax.grid(alpha=0.3)
        return ax

    def to_dict(self) -> dict:
        return {
            "steps": self.steps,
            "train_loss": self.train_loss,
            "val_steps": self.val_steps,
            "val_loss": self.val_loss,
            "lrs": self.lrs,
            "grad_norms": self.grad_norms,
            "wall_time": self.wall_time,
        }


def cosine_lr(step: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay to `min_lr_ratio * lr`.

    Warmup exists because Adam's second-moment estimate is garbage for the first
    few dozen steps — it has not seen enough gradients to know the scale, so
    early updates are effectively unnormalised and can wreck the initialisation.
    Cosine decay is the empirically best-performing simple schedule; the flat
    floor at the end stops the last few hundred steps from being wasted at a
    learning rate of essentially zero.
    """
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(cfg.warmup_steps, 1)
    progress = (step - cfg.warmup_steps) / max(cfg.steps - cfg.warmup_steps, 1)
    progress = min(progress, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * coeff)


def configure_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW with weight decay on matrices only.

    Biases and LayerNorm gains are not weights in the "shrink me toward zero for
    capacity control" sense, so they are exempt. Matrices — including token
    embeddings — are decayed. This dimensionality rule keeps the implementation
    readable and matches the split used by nanoGPT-style training loops.
    """
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas)


class Trainer:
    """Drives a model through `cfg.steps` optimizer steps.

    `batch_fn` is any callable returning `(x, y)`. That is the entire data
    interface — no Dataset, no DataLoader, no collate function. For language
    modelling use `functools.partial(get_batch, data, B, T)`.

    `loss_fn(model, x, y)` defaults to the GPT convention `model(x, targets=y)`
    returning `(logits, loss)`. Override it for anything else.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: TrainConfig | None = None,
        loss_fn: Callable | None = None,
    ):
        self.cfg = cfg or TrainConfig()
        self.device = torch.device(self.cfg.device) if self.cfg.device else _auto_device()
        self.model = model.to(self.device)
        self.loss_fn = loss_fn or _default_loss_fn
        self.optimizer = configure_optimizer(self.model, self.cfg)
        self.history = History()

    @torch.no_grad()
    def evaluate(self, batch_fn: Callable, steps: int | None = None) -> float:
        """Mean loss over a few batches, in eval mode (dropout off)."""
        steps = self.cfg.eval_steps if steps is None else steps
        if steps <= 0:
            raise ValueError(f"evaluation steps must be positive, got {steps}")

        was_training = self.model.training
        self.model.eval()
        try:
            total = 0.0
            for _ in range(steps):
                x, y = batch_fn()
                x, y = x.to(self.device), y.to(self.device)
                total += float(self.loss_fn(self.model, x, y))
            return total / steps
        finally:
            self.model.train(was_training)

    def fit(
        self,
        batch_fn: Callable,
        val_batch_fn: Callable | None = None,
        on_step: Callable[[int, float], None] | None = None,
    ) -> History:
        """Run the loop. `on_step(step, loss)` fires every step for custom probes."""
        cfg = self.cfg
        # A History represents one run. Reusing a Trainer should not silently
        # append a second set of steps starting at zero to the first run.
        self.history = History()
        self.model.train()
        t0 = time.time()

        for step in range(cfg.steps):
            lr = cosine_lr(step, cfg)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            x, y = batch_fn()
            x, y = x.to(self.device), y.to(self.device)

            loss = self.loss_fn(self.model, x, y)
            loss_value = loss.detach().item()  # detach before logging; keep `loss` for backward

            # set_to_none frees the grad tensors instead of filling them with
            # zeros: less memory traffic, and a missing grad becomes visible as
            # None rather than a silent zero.
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Clipping rescales the whole gradient (preserving direction) if its
            # norm exceeds the threshold. Transformers hit occasional huge
            # gradients on rare tokens; without this one spike can undo hours.
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()

            self.history.steps.append(step)
            self.history.train_loss.append(loss_value)
            self.history.lrs.append(lr)
            self.history.grad_norms.append(float(grad_norm))

            if on_step is not None:
                on_step(step, loss_value)

            is_last = step == cfg.steps - 1
            if val_batch_fn is not None and (step % cfg.eval_every == 0 or is_last):
                vl = self.evaluate(val_batch_fn)
                self.history.val_steps.append(step)
                self.history.val_loss.append(vl)
                if step % cfg.log_every == 0 or is_last:
                    print(f"step {step:5d} | train {loss_value:.4f} | val {vl:.4f} | lr {lr:.2e}")
            elif step % cfg.log_every == 0 or is_last:
                print(f"step {step:5d} | train {loss_value:.4f} | lr {lr:.2e}")

        self.history.wall_time = time.time() - t0
        print(f"done in {self.history.wall_time:.1f}s")
        return self.history


def _default_loss_fn(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Handles both `(logits, loss)` models and plain `logits` models."""
    out = model(x, targets=y) if _accepts_targets(model) else model(x)
    if isinstance(out, tuple):
        return out[1]
    return torch.nn.functional.cross_entropy(out.reshape(-1, out.size(-1)), y.reshape(-1))


def _accepts_targets(model: nn.Module) -> bool:
    import inspect

    try:
        return "targets" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


def _auto_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

"""Look inside a running model.

Two complementary tools:

    capture_attention(model)   context manager -> {layer_name: (B,H,T,T) weights}
    ActivationRecorder(model)  records the output of any module you name

Both are read-only and leave no trace after the `with` block. `ablate` and
`patch` go further and *intervene*, which is how causal claims get made:
correlation says a head lights up, ablation says the model breaks without it.
"""

from __future__ import annotations

import re
from contextlib import contextmanager

import torch
import torch.nn as nn

from learnai.nn.attention import CrossAttention, MultiHeadAttention


@contextmanager
def capture_attention(model: nn.Module):
    """Collect every attention map produced inside the `with` block.

        with capture_attention(model) as attn:
            model(x)
        attn["blocks.0.attn"]   # (B, H, T, T)

    Works by flipping each attention module's `store_attn` flag, then reading
    what they stashed. Restores the flags on exit.
    """
    modules = {
        name: m
        for name, m in model.named_modules()
        if isinstance(m, (MultiHeadAttention, CrossAttention))
    }
    previous = {name: m.store_attn for name, m in modules.items()}
    previous_weights = {name: m.attn_weights for name, m in modules.items()}
    for m in modules.values():
        m.store_attn = True
        # Do not mistake a map from an earlier forward pass for one produced
        # inside this context.
        m.attn_weights = None

    captured: dict[str, torch.Tensor] = {}
    try:
        yield captured
    finally:
        for name, m in modules.items():
            if m.attn_weights is not None:
                captured[name] = m.attn_weights
            m.store_attn = previous[name]
            m.attn_weights = previous_weights[name]


class ActivationRecorder:
    """Record the output of named submodules via forward hooks.

        rec = ActivationRecorder(model, pattern=r"blocks\\.\\d+$")
        with rec:
            model(x)
        rec["blocks.2"]     # residual stream after block 2

    `pattern` is a regex matched against module names; omit it to record
    everything (usually far too much).
    """

    def __init__(self, model: nn.Module, pattern: str | None = None, detach: bool = True):
        self.model = model
        self.pattern = re.compile(pattern) if pattern else None
        self.detach = detach
        self.activations: dict[str, torch.Tensor] = {}
        self._handles: list = []

    def _matches(self, name: str) -> bool:
        if not name:  # the root module itself
            return False
        return self.pattern.search(name) is not None if self.pattern else True

    def __enter__(self) -> ActivationRecorder:
        self.activations.clear()
        for name, module in self.model.named_modules():
            if not self._matches(name):
                continue

            def hook(_mod, _inp, out, _name=name):  # bind name per-iteration
                t = out[0] if isinstance(out, tuple) else out
                if torch.is_tensor(t):
                    self.activations[_name] = t.detach() if self.detach else t

            self._handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.activations[key]

    def keys(self):
        return self.activations.keys()

    def summary(self) -> str:
        lines = [f"{name:40s} {tuple(t.shape)}" for name, t in self.activations.items()]
        return "\n".join(lines)


@contextmanager
def ablate(model: nn.Module, module_name: str, mode: str = "zero"):
    """Disable one submodule and see what breaks.

        with ablate(model, "blocks.3.attn"):
            loss_without = evaluate(model)

    `mode="zero"` returns zeros (the sub-layer contributes nothing to the
    residual stream); `mode="mean"` returns the batch mean (removes the
    *variation* but keeps the average contribution, which is the fairer test —
    zeroing can knock the model off-distribution in ways that overstate
    importance).
    """
    if mode not in {"zero", "mean"}:
        raise ValueError(f"unknown ablation mode {mode!r}; use 'zero' or 'mean'")
    target = dict(model.named_modules())[module_name]
    original_forward = target.forward

    def zeroed(*args, **kwargs):
        out = original_forward(*args, **kwargs)
        if isinstance(out, tuple):
            return (torch.zeros_like(out[0]), *out[1:])
        return torch.zeros_like(out)

    def meaned(*args, **kwargs):
        out = original_forward(*args, **kwargs)
        t = out[0] if isinstance(out, tuple) else out
        # Average over batch and sequence, broadcast back — every position now
        # receives the same, information-free contribution.
        m = t.mean(dim=tuple(range(t.dim() - 1)), keepdim=True).expand_as(t)
        return (m, *out[1:]) if isinstance(out, tuple) else m

    target.forward = {"zero": zeroed, "mean": meaned}[mode]
    try:
        yield model
    finally:
        target.forward = original_forward


@contextmanager
def patch(model: nn.Module, module_name: str, value: torch.Tensor):
    """Force a submodule's output to `value` (activation patching).

    The workhorse of causal tracing: run the model on a *corrupted* prompt, patch
    in the clean activation at one site, and see how much of the correct answer
    comes back. Sweep the site across layers and positions and you get a map of
    where the relevant information actually lives.
    """
    target = dict(model.named_modules())[module_name]
    original_forward = target.forward

    def patched(*args, **kwargs):
        out = original_forward(*args, **kwargs)
        return (value, *out[1:]) if isinstance(out, tuple) else value

    target.forward = patched
    try:
        yield model
    finally:
        target.forward = original_forward

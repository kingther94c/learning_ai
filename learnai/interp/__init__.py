"""Interpretability: capture activations, ablate components, patch them."""

from learnai.interp.hooks import (  # noqa: F401
    ActivationRecorder,
    ablate,
    capture_attention,
    patch,
)

__all__ = ["ActivationRecorder", "ablate", "capture_attention", "patch"]

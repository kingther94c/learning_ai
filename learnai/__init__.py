"""learnai — a lab for understanding ML/DL/NLP by rebuilding it from scratch.

Design rules for everything in this package:

1. **Readable over fast.** Every tensor op carries a shape comment. If a clever
   one-liner hides the mechanism, write the slow loop instead.
2. **No hidden magic.** No config frameworks, no registries you have to trace.
   A model is a class; a training run is a function call.
3. **Inspectable.** Every model can hand back its internals (attention weights,
   activations, gradients) so notebooks and the interactive pages can show them.

Quick start::

    from learnai.env import setup
    ctx = setup(seed=0)
"""

__version__ = "0.1.0"

from learnai.env import Context, get_device, seed_everything, setup  # noqa: F401

__all__ = ["Context", "get_device", "seed_everything", "setup", "__version__"]

"""Environment: devices, seeds, paths.

Import this first in every notebook:

    from learnai.env import setup
    ctx = setup(seed=0)     # prints device + versions, seeds everything
"""

from __future__ import annotations

import dataclasses
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# Paths — everything is resolved relative to the repo root so notebooks work from any cwd
# --------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
RUNS_DIR = REPO_ROOT / "runs"
INTERACTIVE_DIR = REPO_ROOT / "interactive"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"


def ensure_dirs() -> None:
    for d in (DATA_DIR, CACHE_DIR, RUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------------------


def get_device(prefer: str | None = None):
    """Best available torch device. On Apple silicon this is 'mps'.

    `prefer` forces a choice ('cpu' / 'mps' / 'cuda'); useful when MPS lacks an op.
    """
    import torch

    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------


def seed_everything(seed: int = 0, deterministic: bool = True) -> int:
    """Seed python / numpy / torch. Returns the seed so you can log it."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            # cuDNN autotuning picks different algorithms run-to-run; pin it.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed


# --------------------------------------------------------------------------------------
# Context
# --------------------------------------------------------------------------------------


@dataclasses.dataclass
class Context:
    """What a notebook needs to know about where it is running."""

    device: object
    seed: int
    repo_root: Path = REPO_ROOT
    cache_dir: Path = CACHE_DIR
    runs_dir: Path = RUNS_DIR

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"Context(device={self.device}, seed={self.seed}, root={self.repo_root})"


def setup(seed: int = 0, device: str | None = None, quiet: bool = False) -> Context:
    """One-call notebook bootstrap: seed, pick a device, make dirs, report versions."""
    ensure_dirs()
    seed_everything(seed)
    dev = get_device(device)

    if not quiet:
        lines = [
            f"python   {sys.version.split()[0]}  ({platform.machine()})",
            f"numpy    {np.__version__}",
        ]
        try:
            import torch

            lines.append(f"torch    {torch.__version__}")
        except ImportError:
            lines.append("torch    NOT INSTALLED  -> pip install -e '.[notebooks]'")
        lines.append(f"device   {dev}")
        lines.append(f"seed     {seed}")
        print("\n".join(lines))

    # Make `import learnai` work even if the package was never pip-installed.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    return Context(device=dev, seed=seed)

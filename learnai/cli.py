"""Small command-line diagnostics for the learning environment.

The notebooks should be where the learning happens. This CLI only answers the
two setup questions that are otherwise surprisingly time-consuming to debug:

    learnai info      What Python, torch, device, and repository am I using?
    learnai doctor    Can this environment execute the lessons?
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import os
import platform
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path

from learnai import __version__
from learnai.env import CACHE_DIR, INTERACTIVE_DIR, NOTEBOOKS_DIR, REPO_ROOT, get_device


def _offline() -> bool:
    return os.environ.get("LEARNAI_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}


def _asset_count(root: Path, pattern: str) -> int:
    return sum(1 for _ in root.glob(pattern)) if root.exists() else 0


def _cmd_info(_args: argparse.Namespace) -> int:
    """Print facts, not guesses, about the active runtime."""
    import numpy as np
    import torch

    print(f"learnai     {__version__}")
    print(f"python      {platform.python_version()} ({platform.machine()})")
    print(f"executable  {sys.executable}")
    print(f"numpy       {np.__version__}")
    print(f"torch       {torch.__version__}")
    print(f"device      {get_device()}")
    print(f"repo        {REPO_ROOT}")
    print(f"notebooks   {_asset_count(NOTEBOOKS_DIR, '**/*.ipynb')}")
    print(f"interactive {_asset_count(INTERACTIVE_DIR, '*/index.html')}")
    print(f"offline     {'yes' if _offline() else 'no'}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Run fast, local checks; no downloads and no training."""
    results: list[tuple[str, str, str]] = []

    def record(level: str, check: str, detail: str) -> None:
        results.append((level, check, detail))

    # Importing this package already proves the declared Python >=3.10 floor.
    record("ok", "python", f"{platform.python_version()} at {sys.executable}")

    core_modules = {"numpy": "numpy", "torch": "torch"}
    imported: dict[str, object] = {}
    for label, module_name in core_modules.items():
        try:
            module = importlib.import_module(module_name)
            imported[module_name] = module
            record("ok", label, str(getattr(module, "__version__", "installed")))
        except Exception as exc:  # an import-time binary error is as important as absence
            record("fail", label, f"{type(exc).__name__}: {exc}")

    # These libraries are exercised by their respective lessons. Reading their
    # installed metadata keeps `doctor` fast and avoids initialising plotting
    # caches just to print a version.
    for distribution, module_name in (
        ("matplotlib", "matplotlib"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
    ):
        if importlib.util.find_spec(module_name) is None:
            record("fail", distribution, "not installed")
            continue
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = "installed"
        record("ok", distribution, version)

    torch = imported.get("torch")
    if torch is not None:
        try:
            device = get_device()
            # Force real work and a synchronisation. Merely constructing
            # torch.device("mps") does not prove that the backend can execute.
            answer = float(torch.tensor([1.0, 2.0], device=device).sum().cpu())
            if answer != 3.0:  # pragma: no cover - guards a deeply broken backend
                raise RuntimeError(f"device smoke test returned {answer}")
            record("ok", "device", f"{device} tensor smoke test")
        except Exception as exc:
            record("fail", "device", f"{type(exc).__name__}: {exc}")

    if sys.platform == "darwin" and platform.machine() != "arm64":
        record(
            "warn",
            "architecture",
            "x86_64 Python on macOS; use the repo's native arm64 .venv for PyTorch/MPS",
        )
    else:
        record("ok", "architecture", platform.machine())

    if NOTEBOOKS_DIR.exists():
        record("ok", "lessons", f"{_asset_count(NOTEBOOKS_DIR, '**/*.ipynb')} notebooks found")
    else:
        record("warn", "lessons", f"notebook directory not found under {REPO_ROOT}")

    try:
        from jupyter_client.kernelspec import KernelSpecManager

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kernels = KernelSpecManager().find_kernel_specs()
        if "learnai" in kernels:
            record("ok", "kernel", f"learnai -> {kernels['learnai']}")
        else:
            record("warn", "kernel", "no 'learnai' Jupyter kernel; run `make setup`")
    except ImportError:
        record("warn", "kernel", "Jupyter extras are not installed; run `make setup`")

    cached = CACHE_DIR / "tinyshakespeare.txt"
    if _offline():
        detail = (
            f"using cached {cached.name}" if cached.exists() else "using built-in corpus fallback"
        )
        record("ok", "offline", f"LEARNAI_OFFLINE=1; {detail}")
    else:
        record("ok", "offline", "downloads enabled; set LEARNAI_OFFLINE=1 for CI or class")

    width = max(len(check) for _, check, _ in results)
    for level, check, detail in results:
        print(f"[{level:4s}] {check:<{width}}  {detail}")

    failed = any(level == "fail" for level, _, _ in results)
    warned = any(level == "warn" for level, _, _ in results)
    if failed:
        print("doctor: core environment is not ready")
        return 1
    if warned and args.strict:
        print("doctor: warnings are failures in --strict mode")
        return 1
    print("doctor: core environment is ready")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="learnai",
        description="Inspect the learnai runtime and diagnose lesson setup.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    info = sub.add_parser("info", help="show versions, device, paths, and lesson counts")
    info.set_defaults(handler=_cmd_info)

    doctor = sub.add_parser("doctor", help="run local dependency and device checks")
    doctor.add_argument(
        "--strict",
        action="store_true",
        help="return a non-zero status for optional-environment warnings too",
    )
    doctor.set_defaults(handler=_cmd_doctor)

    # `learnai` by itself should still answer the most useful setup question.
    parser.set_defaults(handler=_cmd_info)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover - exercised through the console script
    raise SystemExit(main())

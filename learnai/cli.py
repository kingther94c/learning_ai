"""Small command-line diagnostics for the learning environment.

The notebooks should be where the learning happens. This CLI only answers the
two setup questions that are otherwise surprisingly time-consuming to debug:

    learnai info      What Python, torch, device, and repository am I using?
    learnai doctor    Can this environment execute the lessons?

and runs released open-weight models (see ``learnai.openweights``):

    learnai models    Which curated models fit this Mac, and which are downloaded?
    learnai pull      Download one into the Hugging Face cache.
    learnai chat      Talk to one in the terminal.
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
            "x86_64 Python on macOS; use the repo's native arm64 Conda env `py313`",
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
        if "learnai-py313" in kernels:
            record("ok", "kernel", f"learnai-py313 -> {kernels['learnai-py313']}")
        elif "learnai" in kernels:
            record(
                "warn", "kernel", "legacy `learnai` kernel found; run `make setup` to replace it"
            )
        else:
            record("warn", "kernel", "no `learnai-py313` Jupyter kernel; run `make setup`")
    except ImportError:
        record("warn", "kernel", "Jupyter extras are not installed; run `make setup`")

    # Open-weight models are optional and Apple-silicon only (MLX).
    if sys.platform == "darwin" and platform.machine() == "arm64":
        if importlib.util.find_spec("mlx_lm") is None:
            record("ok", "mlx-lm", "optional, not installed; `make llm-setup` to run open models")
        else:
            from learnai.openweights import MODELS, DriveNotConnected, cached_sizes

            version = importlib.metadata.version("mlx-lm")
            try:
                have = [card.key for card in MODELS.values() if card.repo in cached_sizes()]
                downloaded = ", ".join(have) or "none (`make pull`)"
                record("ok", "mlx-lm", f"{version}; models downloaded: {downloaded}")
            except DriveNotConnected as exc:
                record("warn", "mlx-lm", f"{version}; {exc}")

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


def _cmd_models(_args: argparse.Namespace) -> int:
    """The curated open-weight models against this machine's memory and cache."""
    from learnai.openweights import (
        DEFAULT_MODEL,
        MODELS,
        DriveNotConnected,
        cached_sizes,
        fit,
        gpu_budget_gb,
        local_path,
        models_dir,
    )
    from learnai.openweights import physical_memory_gb as ram

    budget = gpu_budget_gb()
    print(f"GPU memory budget {budget:.1f} GB (of {ram() * 1e9 / 2**30:.0f} GiB RAM)")
    try:
        cached = cached_sizes()
        print(f"models stored in  {models_dir() or 'the default Hugging Face cache'}\n")
    except DriveNotConnected as exc:
        cached = {}
        print(f"models stored in  {exc} — plug it in to use them\n")
    print(f"  {'model':<16} {'size':>7}  {'kind':<5}  {'fits':<7}  {'downloaded':<10}  note")
    for card in MODELS.values():
        mark = "*" if card.key == DEFAULT_MODEL else " "
        have = f"{cached[card.repo]:.1f} GB" if card.repo in cached else "-"
        if card.quantize is not None:  # quantised here: ready only once converted
            have = f"{have} -> {card.quantize}bit" if local_path(card.key) else have
        print(
            f"{mark} {card.key:<16} {card.run_gb:>4.1f} GB  {card.kind:<5}  "
            f"{fit(card.run_gb, budget):<7}  {have:<10}  {card.note}"
        )
    print("\n* default.  make pull MODEL=<model>   make chat MODEL=<model>")
    return 0


def _cmd_pull(args: argparse.Namespace) -> int:
    from learnai.openweights import pull, resolve

    card = resolve(args.model)
    print(f"downloading {card.repo} (~{card.download_gb:.1f} GB) into the Hugging Face cache")
    if card.quantize is not None:
        print(f"then quantising it to {card.quantize} bits (~{card.run_gb:.1f} GB) — a few minutes")
    print(pull(args.model))
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    from learnai.openweights import chat

    return chat(
        args.model,
        prompt=args.prompt,
        system=args.system,
        think=args.think,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        patch=args.patch,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="learnai",
        description="Inspect the learnai runtime, diagnose lesson setup, run open-weight models.",
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

    models = sub.add_parser("models", help="list curated open-weight models and whether they fit")
    models.set_defaults(handler=_cmd_models)

    pull = sub.add_parser("pull", help="download an open-weight model")
    pull.add_argument("model", nargs="?", help="curated key or Hugging Face repo id")
    pull.set_defaults(handler=_cmd_pull)

    chat = sub.add_parser("chat", help="chat with an open-weight model in the terminal")
    chat.add_argument("model", nargs="?", help="curated key or Hugging Face repo id")
    chat.add_argument("-p", "--prompt", help="answer one prompt and exit")
    chat.add_argument("-s", "--system", help="system prompt")
    chat.add_argument("--think", action="store_true", help="let the model reason first")
    chat.add_argument("-t", "--temperature", type=float, default=0.7, help="0 = greedy")
    chat.add_argument("-n", "--max-tokens", type=int, default=2048)
    chat.add_argument("--patch", help="apply a hand-made weight edit (.npz) after loading")
    chat.set_defaults(handler=_cmd_chat)

    # `learnai` by itself should still answer the most useful setup question.
    parser.set_defaults(handler=_cmd_info)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover - exercised through the console script
    raise SystemExit(main())

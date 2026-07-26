"""Build and verify notebooks from Python.

Hand-editing `.ipynb` JSON is miserable and diffs badly. This module lets a
lesson be written as a normal Python script and emitted as a notebook, which
means lessons are reviewable, greppable, and — the part that matters most —
*executable in CI*, so a notebook that has silently rotted gets caught.

    from tools.nbuild import Notebook

    nb = Notebook("Attention from scratch", subtitle="...")
    nb.md("## 1. The dot product")
    nb.code("import torch\\nprint(torch.__version__)")
    nb.save("notebooks/04_transformer/01_attention.ipynb")

Then verify it actually runs:

    python tools/nbuild.py --run notebooks/04_transformer/01_attention.ipynb
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class Notebook:
    """A notebook under construction. Cells accumulate in order."""

    def __init__(self, title: str, subtitle: str = "", kernel: str = "python3"):
        self.title = title
        self.cells: list[dict] = []
        self.kernel = kernel
        if title:
            head = f"# {title}\n"
            if subtitle:
                head += f"\n{subtitle}\n"
            self.md(head)

    def _next_id(self) -> str:
        """A stable, content-independent cell id (nbformat >= 4.5 requires one).

        Derived from the title plus the cell index, so regenerating an unchanged
        notebook produces byte-identical output and git sees no diff.
        """
        seed = f"{self.title}:{len(self.cells)}".encode()
        return hashlib.sha1(seed).hexdigest()[:8]

    # -- cell constructors -------------------------------------------------------

    def md(self, source: str) -> Notebook:
        """A markdown cell. Leading indentation is stripped, so you can write
        triple-quoted blocks inline in an indented function."""
        self.cells.append(
            {
                "id": self._next_id(),
                "cell_type": "markdown",
                "metadata": {},
                "source": _lines(textwrap.dedent(source).strip("\n")),
            }
        )
        return self

    def code(self, source: str) -> Notebook:
        """A code cell, with outputs cleared. Outputs are never committed — they
        make diffs unreadable and go stale the moment the code changes."""
        self.cells.append(
            {
                "id": self._next_id(),
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": _lines(textwrap.dedent(source).strip("\n")),
            }
        )
        return self

    def exercise(self, prompt: str, hint: str = "", starter: str = "") -> Notebook:
        """A markdown prompt plus a starter code cell.

        Lessons that only demonstrate do not stick. Every notebook in this repo
        should hand control back to the reader at least twice.
        """
        body = f"> **Your turn.** {prompt}"
        if hint:
            body += f"\n>\n> <details><summary>hint</summary>\n>\n> {hint}\n>\n> </details>"
        self.md(body)
        self.code(starter or "# your code here\n")
        return self

    def takeaway(self, *points: str) -> Notebook:
        """The 'if you remember one thing' box that closes a section."""
        body = "\n".join(f"- {p}" for p in points)
        self.md(f"**Takeaway**\n\n{body}")
        return self

    # -- output ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "cells": self.cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": self.kernel,
                },
                "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }

    def save(self, path: str | Path, verbose: bool = True) -> Path:
        path = Path(path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        # trailing newline + indent so git diffs stay line-oriented
        path.write_text(json.dumps(self.to_dict(), indent=1, ensure_ascii=False) + "\n")
        if verbose:
            n_code = sum(1 for c in self.cells if c["cell_type"] == "code")
            print(f"wrote {path.relative_to(REPO_ROOT)}  ({len(self.cells)} cells, {n_code} code)")
        return path


def _lines(text: str) -> list[str]:
    """nbformat stores source as a list of lines, each keeping its newline."""
    if not text:
        return []
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


# ----------------------------------------------------------------------------------
# Verification — the reason this module earns its place
# ----------------------------------------------------------------------------------


def resolve_kernel() -> str:
    """Pick a kernel that can actually import learnai.

    Committed notebooks declare the generic `python3` kernel so they open
    anywhere. But `python3` is whatever the user registered globally — on a
    machine with several conda envs it is very likely the wrong one. If this
    repo's venv registered a `learnai` kernel, prefer it.
    """
    try:
        from jupyter_client.kernelspec import KernelSpecManager

        specs = KernelSpecManager().find_kernel_specs()
        return "learnai" if "learnai" in specs else "python3"
    except ImportError:
        return "python3"


def run_notebook(path: str | Path, timeout: int = 900, kernel: str | None = None):
    """Execute a notebook top to bottom in a subprocess kernel.

    Returns (ok, message). Nothing is written back — this is a check, not a
    build step, so committed notebooks stay output-free.
    """
    try:
        import nbformat
        from nbclient import NotebookClient
        from nbclient.exceptions import CellExecutionError
    except ImportError:
        return False, "needs nbformat + nbclient: pip install nbformat nbclient"

    path = Path(path)
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name=kernel or resolve_kernel(),
        allow_errors=False,
        resources={"metadata": {"path": str(path.parent)}},
    )
    try:
        client.execute()
    except CellExecutionError as e:
        return False, str(e).strip().splitlines()[-1] if str(e).strip() else "cell raised"
    except Exception as e:  # kernel death, timeout, ...
        return False, f"{type(e).__name__}: {e}"
    return True, "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description="build / verify learnai notebooks")
    ap.add_argument(
        "--run",
        nargs="*",
        metavar="NB",
        help="execute notebooks and report failures (default: all)",
    )
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--kernel", default=None, help="kernel name (default: auto-detect)")
    args = ap.parse_args()

    if args.run is None:
        ap.print_help()
        return 0

    targets = (
        [Path(p) for p in args.run]
        if args.run
        else sorted((REPO_ROOT / "notebooks").rglob("*.ipynb"))
    )
    targets = [t for t in targets if ".ipynb_checkpoints" not in str(t)]
    if not targets:
        print("no notebooks found")
        return 0

    failed = []
    for nb_path in targets:
        rel = nb_path.relative_to(REPO_ROOT) if nb_path.is_absolute() else nb_path
        print(f"running {rel} ... ", end="", flush=True)
        ok, msg = run_notebook(nb_path, timeout=args.timeout, kernel=args.kernel)
        print("ok" if ok else f"FAILED\n    {msg}")
        if not ok:
            failed.append((rel, msg))

    print(f"\n{len(targets) - len(failed)}/{len(targets)} notebooks ran clean")
    for rel, msg in failed:
        print(f"  {rel}: {msg}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

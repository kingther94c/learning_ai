"""Check teaching invariants that ordinary unit tests cannot see.

The library tests prove that the maths is correct. This checker proves that a
lesson is still usable as a lesson: notebooks are clean and reviewable, claims
are checked in code, exercises hand control back to the reader, and interactive
pages remain navigable and offline-first.

Run it directly or through ``make lessons``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.nbuild import REPO_ROOT

_SECTION = re.compile(r"(?m)^##\s+\d+\.\s+\S")
_ASSERT = re.compile(r"(?m)^\s*assert\b")
_CONTROL_CHARACTER = re.compile(r"[\x00-\x09\x0b\x0c\x0e-\x1f\x7f]")
_REMOTE_ASSET = re.compile(
    r"""<(?:script|link)\b[^>]*(?:src|href)=["']https?://""",
    re.IGNORECASE,
)


def check_notebook(path: Path) -> list[str]:
    """Return human-readable problems found in one notebook."""
    rel = _display_path(path)
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"{rel}: cannot read notebook ({exc})"]

    problems: list[str] = []
    cells = notebook.get("cells")
    if notebook.get("nbformat") != 4 or not isinstance(cells, list) or not cells:
        return [f"{rel}: expected a non-empty nbformat 4 notebook"]

    first_source = _source(cells[0])
    if cells[0].get("cell_type") != "markdown" or not first_source.lstrip().startswith("# "):
        problems.append(f"{rel}: first cell must be a markdown title")

    ids = [cell.get("id") for cell in cells]
    if any(not cell_id for cell_id in ids):
        problems.append(f"{rel}: every cell needs a stable id")
    elif len(ids) != len(set(ids)):
        problems.append(f"{rel}: cell ids must be unique")

    markdown = "\n".join(_source(cell) for cell in cells if cell.get("cell_type") == "markdown")
    code = "\n".join(_source(cell) for cell in cells if cell.get("cell_type") == "code")

    if _CONTROL_CHARACTER.search(markdown):
        problems.append(
            f"{rel}: markdown contains an ASCII control character "
            "(likely an unescaped LaTeX backslash in its generator)"
        )
    if len(_SECTION.findall(markdown)) < 2:
        problems.append(f"{rel}: use numbered sections (`## N. title`)")
    if markdown.count("> **Your turn.**") < 2:
        problems.append(f"{rel}: include at least two `Your turn` exercises")
    if "**Takeaway**" not in markdown:
        problems.append(f"{rel}: include at least one section takeaway")
    if not _ASSERT.search(code):
        problems.append(f"{rel}: check at least one claim with an executable assert")

    dirty_cells = []
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None or cell.get("outputs"):
            dirty_cells.append(str(index))
    if dirty_cells:
        problems.append(
            f"{rel}: clear outputs/execution counts in code cell(s) {', '.join(dirty_cells)}"
        )

    kernel = notebook.get("metadata", {}).get("kernelspec", {}).get("name")
    if kernel not in {None, "python3"}:
        problems.append(
            f"{rel}: commit the portable `python3` kernel, not machine-local {kernel!r}"
        )
    return problems


def check_interactive_page(path: Path, landing_page: str) -> list[str]:
    """Return offline and navigation problems found in one interactive page."""
    rel = _display_path(path)
    try:
        html = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return [f"{rel}: cannot read page ({exc})"]

    problems: list[str] = []
    if _REMOTE_ASSET.search(html):
        problems.append(f"{rel}: remote scripts/styles break offline use")
    if re.search(r"\bfetch\s*\(", html):
        problems.append(f"{rel}: fetch() does not work reliably over file://")
    if "../shared/learnai.css" not in html:
        problems.append(f"{rel}: use the shared visual language (`learnai.css`)")
    if "../shared/learnai.js" not in html:
        problems.append(f"{rel}: use the shared live-computation helpers (`learnai.js`)")

    expected_link = f"{path.parent.name}/index.html"
    if expected_link not in landing_page:
        problems.append(f"{rel}: add `{expected_link}` to interactive/index.html")
    return problems


def discover_notebooks() -> list[Path]:
    return sorted(
        path
        for path in (REPO_ROOT / "notebooks").rglob("*.ipynb")
        if ".ipynb_checkpoints" not in path.parts
    )


def discover_interactive_pages() -> list[Path]:
    root = REPO_ROOT / "interactive"
    return sorted(path for path in root.glob("*/index.html") if path.parent.name != "shared")


def check_all(notebooks: list[Path] | None = None) -> list[str]:
    """Check all lessons, or a selected notebook list."""
    problems: list[str] = []
    for path in discover_notebooks() if notebooks is None else notebooks:
        problems.extend(check_notebook(path))

    # Supplying paths is useful while authoring one notebook; repository-wide
    # page checks belong only to the default full check.
    if notebooks is None:
        landing_path = REPO_ROOT / "interactive" / "index.html"
        try:
            landing_page = landing_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            problems.append(f"{_display_path(landing_path)}: cannot read page ({exc})")
            landing_page = ""
        for path in discover_interactive_pages():
            problems.extend(check_interactive_page(path, landing_page))
    return problems


def _source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="check learnai lesson conventions")
    parser.add_argument("notebooks", nargs="*", type=Path, help="default: all lessons and pages")
    args = parser.parse_args()

    paths = args.notebooks or None
    problems = check_all(paths)
    if problems:
        print("lesson check failed:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    notebook_count = len(paths) if paths is not None else len(discover_notebooks())
    page_count = 0 if paths is not None else len(discover_interactive_pages())
    print(f"lesson check passed: {notebook_count} notebook(s), {page_count} page(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

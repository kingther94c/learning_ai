"""Tests for the repository's lesson-quality gate."""

from __future__ import annotations

import json
from pathlib import Path

from tools.check_lessons import check_all, check_notebook
from tools.nbuild import Notebook


def _valid_notebook(tmp_path: Path) -> Path:
    nb = Notebook("A tiny lesson")
    nb.md("## 0. Setup")
    nb.code("value = 1\nassert value == 1")
    nb.md("## 1. The mechanism")
    nb.takeaway("The invariant is visible.")
    nb.exercise("Change the value.")
    nb.exercise("Predict the next result.")
    return nb.save(tmp_path / "lesson.ipynb", verbose=False)


def test_the_committed_curriculum_passes_its_quality_gate():
    assert check_all() == []


def test_notebook_checker_accepts_a_small_complete_lesson(tmp_path):
    path = _valid_notebook(tmp_path)
    assert check_notebook(path) == []


def test_notebook_checker_catches_stale_outputs(tmp_path):
    path = _valid_notebook(tmp_path)
    notebook = json.loads(path.read_text())
    notebook["cells"][2]["execution_count"] = 3
    notebook["cells"][2]["outputs"] = [{"output_type": "stream", "name": "stdout", "text": ["1\n"]}]
    path.write_text(json.dumps(notebook))

    problems = check_notebook(path)
    assert any("clear outputs/execution counts" in problem for problem in problems)


def test_notebook_checker_catches_a_demo_with_no_practice(tmp_path):
    path = _valid_notebook(tmp_path)
    notebook = json.loads(path.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "markdown":
            cell["source"] = [
                line.replace("> **Your turn.**", "> Try this.") for line in cell["source"]
            ]
    path.write_text(json.dumps(notebook))

    problems = check_notebook(path)
    assert any("at least two `Your turn` exercises" in problem for problem in problems)

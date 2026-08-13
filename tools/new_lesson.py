"""Scaffold a new lesson: a notebook, optionally an interactive page.

    python tools/new_lesson.py 03_nlp/02_word_vectors --title "Word vectors"
    python tools/new_lesson.py 06_frontier/01_moe --title "Mixture of Experts" --page moe

The scaffold is opinionated because the repo is: every lesson opens with the
question it answers, states the shapes it works in, checks its own claims, and
hands control to the reader at least twice. Those are the parts people skip when
starting from a blank cell, and they are the parts that make a lesson stick.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.nbuild import REPO_ROOT, Notebook  # noqa: E402

PAGE_TEMPLATE = """<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title} — learnai</title>
<link rel="stylesheet" href="../shared/learnai.css" />

<div class="wrap-wide">
  <header class="page-head">
    <h1>{title}</h1>
    <p class="sub">
      One sentence on what the reader will be able to feel after using this page.
    </p>
    <a class="home-link" href="../index.html">← all pages</a>
  </header>

  <div class="controls">
    <div class="control">
      <label for="knob">a knob <span class="val" id="knob-val"></span></label>
      <input type="range" id="knob" min="0" max="1" step="0.01" value="0.5" />
    </div>
  </div>

  <div class="panel">
    <h2>The thing itself</h2>
    <p class="hint">
      What the reader is looking at, and what to notice about it.
    </p>
    <div id="viz"></div>
    <div id="legend" style="margin-top:16px"></div>
    <details class="table-toggle" style="margin-top:14px">
      <summary>show the numbers as a table</summary>
      <div id="table"></div>
    </details>
  </div>

  <footer class="page-foot">
    Implementation: <code>learnai/...</code> ·
    Notebook: <code>notebooks/{slug}.ipynb</code>
  </footer>
</div>

<script src="../shared/learnai.js"></script>
<script>
// Compute live — do not animate a recording. `LA` has softmax, matmul, attention
// and entropy; `Viz` has heatmap, bars and lineChart; `UI` has slider, table and
// legend. See interactive/shared/learnai.js.

const state = {{ knob: 0.5 }};

function render() {{
  const M = LA.softmaxRows(LA.seededMatrix(8, 8, 7), state.knob * 4 + 0.1);

  Viz.heatmap(document.getElementById("viz"), M, {{ cell: 34, max: 1 }});

  UI.legend(document.getElementById("legend"), [
    {{ color: Viz.seq(0.1), label: "low" }},
    {{ color: Viz.seq(1.0), label: "high" }},
  ]);

  UI.table(
    document.getElementById("table"),
    ["row", ...M[0].map((_, j) => `col ${{j}}`)],
    M.map((r, i) => [i, ...r.map((v) => v.toFixed(3))])
  );
}}

UI.slider("knob", (v) => {{ state.knob = v; render(); }}, (v) => v.toFixed(2));
document.addEventListener("themechange", render);
</script>
"""

_LESSON_SLUG = re.compile(r"[a-z0-9][a-z0-9_-]*(?:/[a-z0-9][a-z0-9_-]*)*")
_PAGE_SLUG = re.compile(r"[a-z0-9][a-z0-9-]*")


def validate_slug(value: str, *, page: bool = False) -> str:
    """Return a normalised safe slug, or raise a useful error.

    These values become paths. Rejecting absolute paths and ``..`` here keeps a
    typo in a lesson command from writing outside notebooks/ or interactive/.
    """
    raw = Path(value)
    is_absolute = raw.is_absolute()
    value = value.removesuffix(".ipynb").strip("/")
    pattern = _PAGE_SLUG if page else _LESSON_SLUG
    kind = "page" if page else "lesson"
    if is_absolute or not pattern.fullmatch(value):
        example = "attention-flow" if page else "04_transformer/02_attention_flow"
        raise ValueError(
            f"invalid {kind} slug {value!r}; use lowercase letters, digits, "
            f"hyphens/underscores (example: {example})"
        )
    return value


def build_notebook(slug: str, title: str, page: str | None) -> Path:
    nb = Notebook(
        title,
        subtitle=(
            "**The question this answers:** _(one sentence — if you cannot write it, "
            "the lesson is not ready)_\n\n"
            "**What you will build:** _(the concrete artifact)_\n\n"
            "**Prerequisites:** _(be specific; 'some ML background' helps nobody)_\n\n"
            "**Primary source:** _(paper/official implementation, when applicable)_"
        ),
    )

    nb.md("""
    ---
    ## 0. Setup
    """)
    nb.code("""
    import os

    import matplotlib.pyplot as plt
    import torch

    from learnai.env import setup
    from learnai.viz import use_style

    ctx = setup(seed=0)
    use_style()
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=120)
    FAST = os.environ.get("LEARNAI_FAST", "0") == "1"
    print("fast smoke path:", FAST)
    """)

    nb.md("""
    ---
    ## 1. The problem

    Start with something that does not work, or a question with a non-obvious
    answer. A lesson that opens with the solution has already lost the reader's
    attention — they have no reason to care about the machinery yet.
    """)
    nb.code("# show the problem concretely, with small printable numbers\n")

    nb.md("""
    ---
    ## 2. The idea

    One idea per section. Small enough to print, then checked against a reference
    implementation so the reader can trust it.
    """)
    nb.code("""
    # Replace this tiny checked example with the real mechanism.
    mine = torch.tensor([1.0])
    reference = torch.ones(1)
    assert torch.allclose(mine, reference, atol=1e-6)
    """)

    nb.takeaway(
        "The one thing to remember from this section.",
        "The shape or invariant that must hold.",
        "The failure mode this prevents.",
    )

    nb.exercise(
        "Change one thing and predict the result before running it.",
        hint="Point at the specific function or argument.",
    )

    nb.exercise(
        "Design an ablation that should remove the effect. State the metric before running it.",
        hint="Change only the proposed mechanism; keep the data, seed, budget and metric fixed.",
    )

    nb.md(f"""
    ---
    ## 3. Evidence boundary and where to go next

    - **Measured here:** _(the strongest conclusion supported by this notebook)_
    - **Not established here:** _(the gap from the paper's data/model/metric)_
    - **Controlled ablation:** _(what removes the effect)_

    - the module this lesson exercises: `learnai/...`
    {f"- the interactive page: `interactive/{page}/index.html`" if page else ""}
    - the paper, if there is one
    """)

    return nb.save(f"notebooks/{slug}.ipynb")


def main() -> int:
    ap = argparse.ArgumentParser(description="scaffold a learnai lesson")
    ap.add_argument("slug", help="path under notebooks/, e.g. 03_nlp/02_word_vectors")
    ap.add_argument("--title", required=True)
    ap.add_argument("--page", default=None, help="also scaffold interactive/<PAGE>/index.html")
    args = ap.parse_args()

    try:
        slug = validate_slug(args.slug)
        page = validate_slug(args.page, page=True) if args.page else None
    except ValueError as exc:
        print(exc)
        return 2

    nb_path = REPO_ROOT / f"notebooks/{slug}.ipynb"
    if nb_path.exists():
        print(f"refusing to overwrite {nb_path.relative_to(REPO_ROOT)}")
        return 1

    # Resolve every collision before writing anything, so a failed page scaffold
    # cannot leave behind a half-created lesson.
    page_path = REPO_ROOT / "interactive" / page / "index.html" if page else None
    if page_path is not None and page_path.exists():
        print(f"refusing to overwrite {page_path.relative_to(REPO_ROOT)}")
        return 1

    build_notebook(slug, args.title, page)

    if page_path is not None:
        page_path.parent.mkdir(parents=True, exist_ok=True)
        page_path.write_text(PAGE_TEMPLATE.format(title=args.title, slug=slug))
        print(f"wrote {page_path.relative_to(REPO_ROOT)}")
        print("  remember to add a card for it in interactive/index.html")

    print("\nnext:")
    print("  make lab                     # write the lesson")
    print(f"  conda run -n py313 python tools/nbuild.py --run notebooks/{slug}.ipynb")
    return 0


if __name__ == "__main__":
    sys.exit(main())

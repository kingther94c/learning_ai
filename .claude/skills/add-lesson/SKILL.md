---
name: add-lesson
description: Add a lesson to this repo — a notebook, an interactive page, or both. Use when asked to cover a new ML/DL/NLP/transformer topic, reproduce a paper, explain a mechanism interactively, or extend learnai/ with a new component. Encodes the repo's conventions so a new lesson matches the existing ones.
---

# Adding a lesson to learnai

A lesson is a **notebook** (build it), an **interactive page** (feel it), or both,
sitting on top of a **module in `learnai/`** (the reference implementation). Do not
write a lesson that has no module under it — a notebook full of inline definitions
cannot be imported, tested, or reused by a page.

## Order of work

1. **Write the module first**, in `learnai/`. Shape-annotated, commented for a
   reader, no cleverness that hides the mechanism.
2. **Write tests** in `tests/`. Anything with a torch equivalent gets an
   equivalence test. That is what makes the annotated code trustworthy.
3. **Scaffold** — `make lesson NAME=04_transformer/03_kv_cache TITLE="KV caching"`
   (add `--page kvcache` via `tools/new_lesson.py` if it needs an interactive page).
4. **Write the lesson**, then **verify it runs**:
   `conda run --no-capture-output -n py313 python tools/nbuild.py --run notebooks/<path>.ipynb`
5. **Add a card** to `interactive/index.html` if you made a page.
6. Run `make lessons`, then `make check` before you are done. The first command
   checks the lesson-specific contract; the second runs the full repository gate.

## Notebook conventions

- **Open with the question, not the answer.** First cell after setup shows
  something broken or surprising. The reader needs a reason to care.
- **One idea per section.** Numbered `## N. <title>`.
- **Small enough to print.** Start at a size where every tensor fits on screen
  (6 tokens, 4 dims). Scale up only after the mechanism is clear.
- **Check every claim in code.** `assert torch.allclose(mine, reference)` against
  the torch or `learnai` equivalent. A lesson that asserts without checking teaches
  the reader to trust prose over evidence.
- **Measure, do not assert.** "Unscaled attention saturates" is a claim; a loop
  over `d = [4, 16, ..., 1024]` printing entropy is a demonstration. Prefer the
  second every time.
- **`nb.takeaway(...)` closes a section, `nb.exercise(...)` at least twice per
  notebook.** A lesson that never hands over control does not stick.
- **Never commit outputs.** `nbuild` writes cells with outputs cleared; keep it
  that way (diffs stay readable, and stale outputs are worse than none).
- Run `conda run --no-capture-output -n py313 python tools/check_lessons.py notebooks/<path>.ipynb` while
  authoring. It checks the portable kernel, stable cell IDs, numbered sections,
  executable assertions, takeaways, exercises, and output hygiene.
- Notebooks must run in **under ~3 minutes**. If training is needed, size it for a
  laptop — 1500 steps of a 4-layer char model is about 40 seconds on MPS.
- If training dominates runtime, read `LEARNAI_FAST` and keep that path tiny but
  structurally complete. `make notebooks-fast` must execute every cell offline;
  the default path must still produce a meaningful learning result.

## Paper-reproduction conventions

- **Treat the paper as a specification.** Read the primary paper and, when
  implementation details matter, the authors' official code. Do not copy claims
  from tutorials or silently merge details from follow-up papers.
- **Attribute each mechanism to the right source.** If the familiar recipe spans
  multiple papers, say so beside the relevant formula instead of assigning the
  whole recipe to the better-known title.
- **State one central claim and one evidence boundary.** Label an experiment as a
  *reproduction* only when its task and metric test the same claim. Otherwise call
  it a controlled illustration, optimisation microscope, or scale-down, and say
  what it cannot establish.
- **Separate paper numbers from notebook measurements.** Paper tables may be
  transcribed into plots when labelled with the exact table/figure and source;
  notebook results must be visibly named `toy`, `measured here`, or equivalent.
- **Keep controlled comparisons controlled.** Fix seeds and all non-target
  hyperparameters. Rank/depth/window sweeps that accidentally change scale,
  parameter budget, or training data do not isolate the claimed mechanism.
- **Close with the scale gap.** Record the paper's data/model scale, this lesson's
  scale, the trusted baseline and metric, at least one ablation, and the strongest
  conclusion the small experiment actually supports.

## Interactive page conventions

- **Compute live in JavaScript. Never animate a recording.** The value is that the
  reader can grab a parameter and move it. `LA` in `interactive/shared/learnai.js`
  has softmax, matmul, attention and entropy, mirroring
  `learnai/nn/functional.py`.
- **Self-contained.** No CDN, no fetch. Pages must work opened straight off disk.
  Real model data is loaded through `<script src="data/x.js">`, written by
  `learnai.export.write_json` with a `.js` suffix (browsers block `fetch()` over
  `file://`; script tags work).
- **Always ship a fallback** when a page reads exported data, and say on screen
  that it is a fallback.
- **Every chart gets a `<details class="table-toggle">` table.** Accessibility
  floor, and it is how you check the numbers.
- **Palette is fixed.** Use `--series-1..8` in order, `Viz.seq()` for magnitude,
  never invent a hex. The order is validated for colorblind separation; changing
  it breaks that. Same values live in `learnai/viz/theme.py` — change both or
  neither.
- **Both themes.** The CSS handles it; check your page in dark mode before
  finishing.
- Verify by screenshotting:
  `"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless
  --screenshot=/tmp/p.png --window-size=1300,1600 --virtual-time-budget=4000
  --allow-file-access-from-files "file://$PWD/interactive/<page>/index.html"`
  then read the PNG. JS errors show up as missing panels, not as errors.
- Any new pure-math helper in `learnai.js` gets a test in
  `tests/test_interactive_js.mjs`.

## Where things go

```
learnai/nn/         layers, models          learnai/data/    datasets, tokenizers
learnai/train/      the loop                learnai/interp/  hooks, ablation, patching
learnai/viz/        matplotlib + palette    learnai/export/  tensors -> JSON for pages
notebooks/01_ml_foundations  02_deep_learning  03_nlp
          04_transformer     05_gpt            06_frontier
```

Number notebooks within a directory (`01_`, `02_`) — the order is the curriculum.

## Writing style

Explain the *why*, not the *what*. `# add positional encoding` is worthless;
"attention is permutation-equivariant, so order has to be injected by hand" is the
lesson. When a design choice has a known history or a known failure mode, say so —
that is the part the reader cannot get from the code.

Prefer the concrete number over the adjective: "two thirds of a Transformer's
parameters live in the MLP" beats "the MLP is large".

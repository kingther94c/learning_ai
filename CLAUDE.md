# learnai — notes for Claude

A lab for understanding ML/DL/NLP by rebuilding it. The reader is the owner of
this repo, working through the material to build real intuition — so **the code is
the teaching artifact**, not an implementation detail behind one.

## The environment (important)

The system `python3` is an **x86_64 conda env under Rosetta**, and PyTorch no
longer ships macOS x86_64 wheels — `pip install torch` there fails with
"no matching distribution". This repo therefore uses its own native arm64 venv:

```
.venv/bin/python        # Homebrew python3.12, arm64, torch + MPS
```

Always use `.venv/bin/python` or a `make` target. Never `python3`.

The user's global `python3` Jupyter kernel points at an unrelated conda env, so
this repo registers its own kernel named `learnai` inside `.venv`
(`tools/nbuild.py` finds it automatically). Do not touch their user-level
kernelspecs.

## Commands

```
make setup       create .venv, install, register the kernel
make test        pytest + the JS suite            (fast — run this often)
make notebooks   execute every notebook top to bottom   (slow, ~3 min)
make notebooks-fast  execute all cells with tiny offline training runs
make lessons     validate lesson structure and offline pages
make check       test + lint — the gate before finishing
make data        retrain the reference GPT, regenerate interactive/*/data/
make lesson NAME=04_transformer/03_kv_cache TITLE="KV caching"
```

## Layout

| | |
|---|---|
| `learnai/` | the reference implementation — readable, shape-annotated, tested |
| `notebooks/` | build it up a cell at a time, on real data |
| `interactive/` | self-contained HTML that computes live in the browser |
| `tools/` | `nbuild`, `check_lessons`, `build_data`, `new_lesson` |
| `tests/` | `pytest` for Python, `node tests/test_interactive_js.mjs` for JS |

## Non-obvious invariants

- **`learnai.nn` must match torch numerically.** Every from-scratch primitive has
  an equivalence test. Breaking one invalidates the explanations built on it.
- **The palette is duplicated on purpose** between `learnai/viz/theme.py` and
  `interactive/shared/learnai.css`, so notebooks and pages look like one system.
  Change both or neither. The categorical *order* is validated for colorblind
  separation — do not reorder it.
- **Interactive pages must work over `file://`.** No CDN, no `fetch()`. Exported
  data goes through `learnai.export.write_json` with a `.js` suffix, which emits a
  `window.LEARNAI_DATA[...]` assignment loadable by a plain script tag.
- **Notebooks are committed without outputs.** `tools/nbuild.py --run` verifies
  them without writing anything back.
- **Lesson conventions are executable.** `tools/check_lessons.py` enforces the
  portable kernel, clean outputs, assertions, exercises, offline assets and
  interactive navigation.
- **`interactive/*/data/*.js` is generated but committed** — small, and it makes a
  fresh clone work with no Python at all.

## Adding anything

Use the `add-lesson` skill (`.claude/skills/add-lesson/`). It encodes the
conventions: question first, one idea per section, check every claim in code,
measure instead of asserting, exercises that hand back control.

## Verifying an interactive page

A JS error shows up as a *missing panel*, not as a visible error. Screenshot and
look:

```
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu \
  --screenshot=/tmp/p.png --window-size=1300,1600 --virtual-time-budget=4000 \
  --allow-file-access-from-files "file://$PWD/interactive/<page>/index.html"
```

Then read the PNG.

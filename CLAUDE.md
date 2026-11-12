# learnai — notes for Claude

A lab for understanding ML/DL/NLP by rebuilding it. The reader is the owner of
this repo, working through the material to build real intuition — so **the code is
the teaching artifact**, not an implementation detail behind one.

## The environment (important)

The repo uses the native arm64 Conda environment `py313`:

```
conda run --no-capture-output -n py313 python
# /Users/kelvin/opt/anaconda3/envs/py313/bin/python
# Python 3.13, arm64, PyTorch
```

The previous env with the same name was **x86_64 under Rosetta**, and current
PyTorch has no compatible macOS x86_64 wheel. It was rebuilt as arm64. Always
use a `make` target or `conda run -n py313 python`; never rely on an unqualified
`python3` or on whatever environment the shell happens to have activated.

The user's global `python3` Jupyter kernel points at an unrelated conda env, so
this repo registers its own kernel named `learnai-py313` inside the Conda env
(`tools/nbuild.py` finds it automatically). Do not touch their user-level
kernelspecs.

## Commands

```
make setup       install into py313, validate arm64, register the kernel
make doctor      report Python/Torch/device/kernel and check the environment
make test        pytest + the JS suite            (fast — run this often)
make notebooks   execute every notebook top to bottom   (slow, ~3 min)
make notebooks-fast  execute all cells with tiny offline training runs
make lessons     validate lesson structure and offline pages
make check       test + lint — the gate before finishing
make data        retrain the reference GPT, regenerate interactive/*/data/
make lesson NAME=04_transformer/03_kv_cache TITLE="KV caching"
make llm-setup   install mlx-lm for released open-weight models (Apple silicon)
make models      curated open-weight models: size, fits-this-Mac, downloaded?
make chat MODEL=qwen3.5-4b   chat in the terminal (downloads on first use)
make chat MODEL=qwen3.5-4b PATCH=runs/weightprog/qwen3.5-4b/patch.npz   ...with a weight edit applied
```

Open-weight checkpoints (`learnai/openweights.py`) are multi-GB and never go in
the repo: they live in `$LEARNAI_MODELS_DIR` (set per machine as a `py313` conda
env var, e.g. on an external drive) or else the shared Hugging Face cache. Don't `make pull` large models
unasked; `qwen3-0.6b` (0.35 GB) is the one the test suite uses, and its test is
skipped when it is not downloaded.

Run outputs and downloaded data stay out of the repo as well. On the owner's Mac,
`runs/` and `data/cache/` are symlinks into the external drive
(`/Volumes/Seagate Backup Plus Drive/DATA/learning_ai/`), next to the models in
`DATA/huggingface`. Write through the links as usual. If the drive is unplugged, the
links dangle and runs fail, rather than filling the internal disk. A fresh clone gets
plain directories.

**Run one MLX job at a time.** Two generation or probe processes sharing the GPU
end in `[METAL] ... Insufficient Memory` or a silent `Killed: 9` on a 32 GB Mac.
Queue long runs in one background command; do CPU work while they run.

The weight-editing experiment (`docs/WEIGHT_EDITING.md`) is driven by
`tools/weight_edit_experiment.py`; its outputs (runs, patches) go to `runs/weightprog/`
and its benchmark problems to `data/cache/mathbench/`, both git-ignored. A benchmark
run is 10-25 minutes of GPU; `diagnose`, `edit` and `probe` take seconds.

The finance model comparison (`docs/FINANCE_MODEL_SELECTION.md`) is driven by
`tools/finance_model_select.py` over `learnai/finbench.py`; runs go to `runs/finbench/`
and the question sets to `data/cache/finbench/`. Its per-model batch sizes keep the
process under 12 GiB at a 16k cap — change them only with the arithmetic in the doc.
A screening run (150 questions) is 1-3 hours of GPU per model. `tools/finance_report.py`
rebuilds the Chinese report (`docs/finance_model_selection_report.zh.html`) from the runs.
The Mac restarts for macOS updates: write long queues as resumable scripts in `runs/finbench/queue/`.

The global-macro futures benchmark (`docs/MACRO_FUTURES_BENCH.md`, for PM and execution
agents) is `learnai/macrobench.py`. Its items are generated, not downloaded, and run
through the same runner with `--bench macro`, which writes `runs/finbench/<model>/macro-*.jsonl`.
Its sign rules (`ECONOMIES`, `POLICY`, `DATA_EXCLUDED`) and desk rules (`RULES`) are the
answer key. Change them only together with `tests/test_macrobench.py`, and expect every
existing macro run to be redone. `generate --system <file>` sends a standing brief as the
system message, and `force` reuses it from the run's meta. The desk playbook
`tools/prompts/macro_desk.txt` was written from the errors on the screening slice (`o0`),
so measure it only on the held-out slice (`--offset 50`). Keep it, plus the longest
prompt, under the runner's `PROMPT_MAX` tokens.

## Layout

| | |
|---|---|
| `learnai/` | the reference implementation — readable, shape-annotated, tested |
| `notebooks/` | build it up a cell at a time, on real data |
| `interactive/` | self-contained HTML that computes live in the browser |
| `tools/` | `nbuild`, `check_lessons`, `build_data`, `new_lesson`, `weight_edit_experiment`, `finance_model_select`, `edit_gate`, `finance_report` |
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

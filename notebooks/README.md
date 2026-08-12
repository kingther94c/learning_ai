# notebooks

The curriculum. Numbered within each directory — the order is the path.
See [`docs/LEARNING_PATH.md`](../docs/LEARNING_PATH.md) for prerequisites,
study method, planned lesson order, and the paper-reproduction contract.

| directory | what it covers | status |
|---|---|---|
| `01_ml_foundations/` | linear models, trees, the bias-variance picture, decision boundaries you can see | planned |
| `02_deep_learning/` | backprop by hand, initialisation, normalisation, optimisers | planned |
| `03_nlp/` | tokenization, embeddings, word analogies, the pre-transformer baseline | planned |
| `04_transformer/` | **`01_build_a_transformer.ipynb`** — raw text to a trained GPT; **`02_build_a_bert.ipynb`** — bidirectional encoder, MLM/NSP, and paper ablations | written |
| `05_gpt/` | KV caching, sampling, scaling laws, what "in-context learning" actually is | planned |
| `06_frontier/` | Mixture of Experts, state-space models, RLHF/DPO, paper reproductions | planned |

## Running them

Use the repo's own `py313` kernel — the unqualified system `python3` may point
at an unrelated environment (see the note in the top-level README).

```bash
make lab                                  # JupyterLab, kernel "Python 3.13 (learnai)"
make notebooks                            # execute all of them, report failures
make notebooks-fast                       # execute all cells with tiny offline runs
make lessons                              # verify teaching structure and clean outputs
conda run -n py313 python tools/nbuild.py --run notebooks/04_transformer/01_build_a_transformer.ipynb
```

Notebooks are committed **without outputs**: diffs stay readable, and a stale
output is worse than none. `--run` verifies without writing anything back.

## Adding one

```bash
make lesson NAME=05_gpt/01_kv_cache TITLE="KV caching"
```

Conventions live in `.claude/skills/add-lesson/SKILL.md` — worth reading even if
you are not using Claude Code. The short version: open with the question, one idea
per section, small enough to print, check every claim in code, and hand control
back to the reader at least twice.

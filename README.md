# learnai

A lab for actually understanding ML, deep learning, NLP and Transformers — by
rebuilding them small enough to read, and then poking at them until they make
sense.

Two ways in, on purpose:

- **`notebooks/`** — build a mechanism one cell at a time, on real data, with every
  claim checked in code rather than asserted in prose.
- **`interactive/`** — the same mechanism as a page you can grab and move. These
  compute live in the browser; nothing is a recorded animation.

Both sit on **`learnai/`**, a small library of from-scratch implementations that
are tested against PyTorch for numerical equality. It is written to be read: every
tensor operation carries its shape, and every design choice that has a history says
so.

---

## Quick start

```bash
make setup     # install into Conda py313 and register its Jupyter kernel
make doctor    # show the exact Python, Torch, device, and kernel being used
make test      # Python + JavaScript correctness tests
make lab       # open the notebooks
```

Then open **`interactive/index.html`** straight off disk — no server required.

> **This workstation's macOS setup.** The project uses the native arm64 Conda
> environment `py313` (Python 3.13), not an in-repo `.venv`. The previous
> `py313` was an Intel/Rosetta environment; current PyTorch does not publish a
> compatible macOS x86_64 wheel. `make` runs commands through
> `conda run -n py313`, so shell activation is optional and an unrelated
> `python3` cannot be selected accidentally.
>
> To recreate it on Apple Silicon:
>
> ```bash
> CONDA_SUBDIR=osx-arm64 conda create -n py313 python=3.13 pip -y
> make setup
> ```

---

## What's here now

### The library — `learnai/`

| module | what it holds |
|---|---|
| `nn/functional.py` | `softmax`, `gelu`, `layer_norm`, `rms_norm`, … written longhand, with the numerical reasoning (why subtract the max, why add epsilon) |
| `nn/attention.py` | scaled dot-product, multi-head, cross-attention, causal and padding masks, attention entropy |
| `nn/pos_encoding.py` | sinusoidal, learned, **RoPE**, **ALiBi** |
| `nn/blocks.py` | `FeedForward`, `SwiGLU`, `TransformerBlock` (pre- and post-norm) |
| `nn/bert.py` | a faithful compact BERT encoder: three embeddings, post-norm bidirectional blocks, tied MLM and NSP heads |
| `nn/gpt.py` | a complete GPT in ~200 annotated lines, with temperature / top-k / top-p sampling |
| `nn/word2vec.py` | Skip-gram pair construction, the 3/4-power noise distribution, and negative-sampling loss |
| `nn/resnet.py` | residual linear microscopes, the original two-convolution basic block, and a small-image ResNet |
| `nn/lora.py` | frozen linear weights plus trainable low-rank updates, SVD diagnostics, adapter-only state, and merge/unmerge |
| `data/tokenizers.py` | character, word, and a **BPE trained from scratch** — merges, ranks, the lot |
| `data/toy.py` | copy, reverse, sort, **induction**, parity; spirals and XOR for classic ML |
| `train/trainer.py` | a compact, readable training loop. No Lightning. Cosine schedule, grad clipping, the AdamW parameter split |
| `interp/hooks.py` | `capture_attention`, `ActivationRecorder`, `ablate`, `patch` — the tools for causal claims, not just correlational ones |
| `viz/` | matplotlib helpers on a shared palette |
| `export/` | dump real tensors to JSON so the interactive pages show a real model |

### Notebooks

| | |
|---|---|
| `04_transformer/01_build_a_transformer.ipynb` | **Start here.** Sixteen steps from raw text to a trained GPT. 6 tokens and 4 dimensions at first, so every tensor is printable. Ends by training on Shakespeare and reading the attention maps out of the model you just trained. |
| `04_transformer/02_build_a_bert.ipynb` | **Then compare BERT.** Rebuild the bidirectional encoder, distinguish three kinds of “mask”, measure the 15% → 80/10/10 MLM recipe, add NSP, pre-train a TinyBERT offline, and visualise the original paper's ablations. |
| `03_nlp/01_word2vec.ipynb` | **Learn where embeddings came from.** Turn context windows into Skip-gram pairs, derive negative sampling, train semantic geometry, and test vector analogies on a controlled corpus. |
| `02_deep_learning/01_resnet.ipynb` | **See why shortcuts matter.** Separate the paper's degradation result from overfitting, derive `F(x)+x`, inspect gradient paths, and build the original basic residual block. |
| `06_frontier/01_lora.ipynb` | **Adapt without copying the model.** Learn `ΔW=BA`, sweep rank against a known low-rank task, measure the parameter/checkpoint budget, and prove merged inference is equivalent. |

The directory skeleton (`01_ml_foundations` → `06_frontier`) is laid out for the
curriculum; the remaining stages are filled incrementally. The
[`learning path`](docs/LEARNING_PATH.md) defines the dependency order, how to
study each lesson, and a laptop-sized protocol for paper reproductions.
`make lesson` scaffolds one.

### Interactive pages

| page | the question it answers |
|---|---|
| `attention/` | *What is a head actually looking at?* — one query token all the way through, plus real attention maps from a trained GPT |
| `tokenizer/` | *Where do tokens come from?* — BPE trained live; drag the merge slider and watch characters fuse |
| `positional/` | *Why isn't attention just a bag of words?* — sinusoidal / learned / RoPE / ALiBi, with a **measured** verdict on which are genuinely relative |
| `sampling/` | *Why does my model loop or go insane?* — temperature, top-k and top-p on real exported logits |

---

## The bit that ties it together

The pages do not invent their numbers. A notebook trains a model, `learnai.export`
dumps real tensors, and the page renders those:

```python
from learnai.interp import capture_attention
from learnai.export import export_attention

with capture_attention(model) as attn:
    model(tokens)

export_attention(attn["blocks.0.attn"][0], token_strings,
                 "interactive/attention/data/gpt_attention.js")
```

`make data` regenerates all of it. The payoff is visible on the attention page: the
previous-token head built **by hand** at the top of the page, and the
previous-token head **learned by gradient descent** at the bottom, are the same
shape.

---

## Design rules

These are enforced, not aspirational — see `tests/` and `.claude/skills/add-lesson/`.

1. **Readable over fast.** If a clever one-liner hides the mechanism, write the
   slow loop. Every tensor op carries its shape.
2. **Correct, and proven correct.** Every from-scratch primitive has an
   equivalence test against torch. Annotated code you cannot trust is worse than
   no code.
3. **Measure, don't assert.** "Unscaled attention saturates" is a claim. A sweep
   over `d = 4 … 1024` printing entropy is a demonstration. The repo prefers the
   second, in the notebooks *and* in the pages.
4. **Inspectable.** Any model can hand back its internals. If you cannot see it,
   you do not understand it.
5. **Self-contained pages.** No CDN, no network, works from `file://`.

---

## Working on it

```bash
make lesson NAME=04_transformer/03_kv_cache TITLE="KV caching"   # scaffold
make doctor                                                       # diagnose the active environment
make lessons                                                      # teaching-quality gate
make check                                                        # test + lint
make notebooks-fast                                               # offline smoke execution
make notebooks                                                    # execute them all
```

`make help` lists everything.

`AGENTS.md` / `CLAUDE.md` cover the repository conventions and local environment
traps, and
`.claude/skills/add-lesson/` encodes the reusable lesson-authoring workflow.
`tools/check_lessons.py` turns its important rules into an executable gate.

---

## Where this is going

Planned, in rough order of what earns its place next:

- **`05_gpt/`** — KV caching (the `generate` here is deliberately O(T²)), sampling,
  scaling laws
- **`03_nlp/`** — tokenization and recurrent sequence models around the written Word2Vec lesson
- **`02_deep_learning/`** — backprop by hand, initialisation, normalisation and optimisers around the written ResNet lesson
- **`01_ml_foundations/`** — the classic models, with decision boundaries you can see
- **`06_frontier/`** — Mixture of Experts, state-space models and RLHF/DPO after the written LoRA reproduction

The written paper path now spans *Word2Vec* (2013), *Deep Residual Learning*
(2015), *Attention Is All You Need* (2017), *BERT* (2018), and *LoRA* (2021),
before moving to induction heads, scaling, sparse routing, selective state-space
models, preference optimisation and reasoning RL. See the
[`paper ladder`](docs/LEARNING_PATH.md#paper-ladder) for primary sources and the
reproduction contract.

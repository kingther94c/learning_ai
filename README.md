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
> CONDA_SUBDIR=osx-arm64 conda create -n py313 -c conda-forge python=3.13 pip nodejs -y
> make setup
> ```
>
> `nodejs` is only for the JavaScript half of `make test`; any `node` on `PATH`
> works too.

---

## Run real open-weight models

The lessons rebuild GPT at a size you can print. `learnai.openweights` runs the
released ones — Qwen, Gemma, gpt-oss — on this Mac through Apple's
[MLX](https://github.com/ml-explore/mlx), using 4-bit checkpoints from
Hugging Face's `mlx-community`. Apple silicon only.

```bash
make llm-setup                  # once: installs mlx-lm, then lists the models
make chat                       # downloads the default (qwen3.5-4b, 3 GB) and chats
make chat MODEL=qwen3.8-27b     # any key from `make models`, or an org/name repo id
make pull MODEL=gemma4-26b-a4b  # download now, chat later (or offline)
```

`make models` lists each model's size, whether it fits this machine's GPU memory
budget, and what is already downloaded. Weights never go in the repo: they live
in the shared Hugging Face cache (`~/.cache/huggingface`), or wherever
`LEARNAI_MODELS_DIR` points — a big external drive works well, since a model is
read into memory once at load and the disk only sets the startup time (USB hard
disk ≈ 120 MB/s, internal SSD ≈ 3 GB/s):

```bash
conda env config vars set -n py313 "LEARNAI_MODELS_DIR=/Volumes/<drive>/huggingface"
```

An unplugged drive is reported as such rather than triggering a re-download.
Run outputs (`runs/`) and downloaded benchmark data (`data/cache/`) are git-ignored too,
and can live on the same drive behind symlinks
(`ln -s /Volumes/<drive>/DATA/learning_ai/runs runs`, and likewise for `data/cache`).

After every reply the chat prints tokens/s and peak memory — compare the dense `qwen3.8-27b` with the
mixture-of-experts `qwen3.6-35b-a3b` to see why decoding speed follows *active*
parameters while memory follows *total* ones. Inside a chat, `/think` toggles
the model's reasoning, `/reset` forgets the history.

From Python or a notebook:

```python
from learnai.openweights import load, generate
model, tokenizer = load("qwen3.5-4b")
print(generate(model, tokenizer, "用一句话解释注意力机制。", temperature=0))
```

### Edit its weights by hand

Fine-tuning is not the only way to change a model. `learnai.weightedit` reads a
loaded model's activations, finds the direction that separates two behaviours,
ranks the MLP neurons that write along it, and rewrites their output columns —
no gradients, a few hundred thousand of four billion weights.
[`docs/WEIGHT_EDITING.md`](docs/WEIGHT_EDITING.md) uses it to reproduce the edit
that "Sparks of Weight-Space Programming" (MIT CSAIL, Nov 2026) made to
Qwen3.5-0.8B, checks the result against the authors' released weights, and then
ports the recipe to the 4B model and measures what it buys on GSM8K and MATH.

```bash
R=runs/weightprog/qwen3.5-4b
python tools/weight_edit_experiment.py diagnose --out $R/diagnosis.npz   # read 12 probe texts
python tools/weight_edit_experiment.py edit --diagnosis $R/diagnosis.npz --layers 15,19 --out $R/patch.npz
python tools/weight_edit_experiment.py probe --patch $R/patch.npz        # seconds, no sampling
make chat MODEL=qwen3.5-4b PATCH=$R/patch.npz                            # talk to the edited model
```

(Run the `python` lines inside the env: `conda run -n py313 python ...`.)

### Pick a model for a job, under a memory budget

[`docs/FINANCE_MODEL_SELECTION.md`](docs/FINANCE_MODEL_SELECTION.md) compares five
open-weight models on finance questions — FinQA's report excerpts, CFA-style and
Chinese professional-exam multiple choice (`learnai.finbench`) — each run inside the
same 16 GiB budget with its reasoning on, then applies the weight-editing recipe to
the winner's own failure modes.

```bash
python tools/finance_model_select.py generate --model qwen3.5-9b --n 50    # 150 questions, 16k cap
python tools/finance_model_select.py summary runs/finbench/*/n50-o0.jsonl  # accuracy, misses, cost
python tools/finance_model_select.py curve   runs/finbench/*/n50-o0.jsonl  # ... under smaller caps
```

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
| `nn/gpt_papers.py` | GPT-1's transfer head and joint objective; GPT-2-style continuation scoring and candidate ranking |
| `nn/word2vec.py` | Skip-gram pair construction, the 3/4-power noise distribution, and negative-sampling loss |
| `nn/resnet.py` | residual linear microscopes, the original two-convolution basic block, and a small-image ResNet |
| `nn/lora.py` | frozen linear weights plus trainable low-rank updates, SVD diagnostics, adapter-only state, and merge/unmerge |
| `data/tokenizers.py` | character, word, and a **BPE trained from scratch** — merges, ranks, the lot |
| `data/toy.py` | copy, reverse, sort, **induction**, parity; spirals and XOR for classic ML |
| `data/gpt_tasks.py` | reproducible modular-shift episodes and an inference-only in-context-learning evaluator |
| `train/trainer.py` | a compact, readable training loop. No Lightning. Cosine schedule, grad clipping, the AdamW parameter split |
| `interp/hooks.py` | `capture_attention`, `ActivationRecorder`, `ablate`, `patch` — the tools for causal claims, not just correlational ones |
| `viz/` | matplotlib helpers on a shared palette |
| `export/` | dump real tensors to JSON so the interactive pages show a real model |

### Notebooks

| | |
|---|---|
| `04_transformer/01_build_a_transformer.ipynb` | **Start here.** Sixteen steps from raw text to a trained GPT. 6 tokens and 4 dimensions at first, so every tensor is printable. Ends by training on Shakespeare and reading the attention maps out of the model you just trained. |
| `04_transformer/02_build_a_bert.ipynb` | **Then compare BERT.** Rebuild the bidirectional encoder, distinguish three kinds of “mask”, measure the 15% → 80/10/10 MLM recipe, add NSP, pre-train a TinyBERT offline, and visualise the original paper's ablations. |
| `05_gpt/01_gpt1_pretraining.ipynb` | **Follow the first GPT transfer recipe.** Pre-train on unlabelled text, attach one last-token classifier, then compare scratch, frozen, fine-tuned and auxiliary-LM conditions. |
| `05_gpt/02_gpt2_zero_shot.ipynb` | **Turn a task into text.** Derive continuation scoring token by token, rank candidates with frozen weights, and ablate the task format without rewriting history about GPT-2's prompts. |
| `05_gpt/03_gpt3_in_context_learning.ipynb` | **Move learning into the context.** Separate zero/one/few-shot from fine-tuning, prove parameters stay unchanged, and break a tiny associative-recall model with shuffled demonstrations. |
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
make lesson NAME=05_gpt/04_sampling TITLE="Sampling from logits" # scaffold
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

- **`05_gpt/`** — sampling, KV caching (the `generate` here is deliberately O(T²)),
  induction-head interventions and scaling laws after the written GPT-1/2/3 trilogy
- **`03_nlp/`** — tokenization and recurrent sequence models around the written Word2Vec lesson
- **`02_deep_learning/`** — backprop by hand, initialisation, normalisation and optimisers around the written ResNet lesson
- **`01_ml_foundations/`** — the classic models, with decision boundaries you can see
- **`06_frontier/`** — Mixture of Experts, state-space models and RLHF/DPO after the written LoRA reproduction

The written paper path now spans *Word2Vec* (2013), *Deep Residual Learning*
(2015), *Attention Is All You Need* (2017), *GPT-1* (2018), *BERT* (2018),
*GPT-2* (2019), *GPT-3* (2020), and *LoRA* (2021), before moving to induction
heads, scaling, sparse routing, selective state-space models, preference
optimisation and reasoning RL. See the
[`paper ladder`](docs/LEARNING_PATH.md#paper-ladder) for primary sources and the
reproduction contract.

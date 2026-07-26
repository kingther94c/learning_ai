"""Regenerate the data files the interactive pages read.

    python tools/build_data.py            # everything
    python tools/build_data.py sampling   # just one

Trains a small GPT on tinyshakespeare (about a minute on an M-series laptop) and
dumps real tensors through `learnai.export`. The pages fall back to built-in
sample values when these files are absent, and say so on screen — so this is
optional, but it is the difference between a page that *explains* sampling and a
page that *shows you a model sampling*.

Output goes to `interactive/<page>/data/<name>.js`, which is committed: the files
are small, they make the repo work on a fresh clone with no Python at all, and a
diff on one is a real signal that model behaviour changed.
"""

from __future__ import annotations

import argparse
import functools
import sys

import torch

from learnai.data import CharTokenizer, get_batch, tiny_shakespeare, train_val_split
from learnai.env import INTERACTIVE_DIR, setup
from learnai.export import export_attention, export_history, write_json
from learnai.interp import capture_attention
from learnai.nn import GPT, GPTConfig
from learnai.train import TrainConfig, Trainer

# Contexts chosen to span the confidence range — that spread is the whole point of
# the top-k vs top-p comparison on the sampling page. After "To be, or not to b"
# the model is nearly certain; after a bare newline it has no idea.
PROMPTS = [
    "To be, or not to b",
    "First Citizen:\n",
    "\nROMEO:\nI ",
    "What is the ",
    "the ",
    "\n",
    "MENENIUS:\nYou are ",
    "my lord, the ",
]

PROBE = "First Citizen:\nBefore we proceed"


def train_reference_model(steps: int = 2000):
    """The model every page's data is derived from. Small, fast, good enough."""
    text = tiny_shakespeare()
    tok = CharTokenizer(text)
    train_ids, val_ids = train_val_split(tok.encode(text))

    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=128,
        n_layer=4,
        n_head=4,
        d_model=128,
        dropout=0.1,
    )
    model = GPT(cfg)
    print(f"training reference model: {model.num_params():,} params, {steps} steps")

    trainer = Trainer(
        model,
        TrainConfig(
            steps=steps,
            lr=3e-3,
            warmup_steps=100,
            eval_every=250,
            eval_steps=20,
            log_every=500,
        ),
    )
    history = trainer.fit(
        functools.partial(get_batch, train_ids, 32, 128),
        functools.partial(get_batch, val_ids, 32, 128),
    )
    model.eval()
    return model, tok, trainer, history


def build_sampling(model, tok, device) -> None:
    """Next-token logits after each prompt, for interactive/sampling/."""
    contexts = []
    for prompt in PROMPTS:
        ids = torch.tensor([tok.encode(prompt)], device=device)
        with torch.no_grad():
            logits, _ = model(ids)
        last = logits[0, -1]  # (vocab,)

        k = 16
        top = last.topk(k)
        contexts.append(
            {
                "context": prompt,
                "tokens": [tok.itos[int(i)] for i in top.indices],
                "logits": [round(float(v), 4) for v in top.values],
            }
        )

    write_json(
        {
            "contexts": contexts,
            "meta": {
                "model": "learnai GPT, 4 layers, d_model=128, char-level",
                "corpus": "tinyshakespeare",
                "note": "top-16 logits per context; regenerate with tools/build_data.py",
            },
        },
        INTERACTIVE_DIR / "sampling" / "data" / "sampling.js",
    )


def build_attention(model, tok, device) -> None:
    """Real attention maps from every layer, for reference and future pages."""
    ids = torch.tensor([tok.encode(PROBE)], device=device)
    with capture_attention(model) as attn:
        with torch.no_grad():
            model(ids)

    # capture_attention keys are module paths; sort by layer index, not string
    # order, so "blocks.10" does not land between "blocks.1" and "blocks.2".
    names = sorted(attn, key=lambda n: int(n.split(".")[1]))
    stacked = torch.stack([attn[n][0] for n in names])  # (L, H, T, T)
    tokens = [c.replace("\n", "\\n") for c in PROBE]

    export_attention(
        stacked.cpu(),
        tokens,
        INTERACTIVE_DIR / "attention" / "data" / "gpt_attention.js",
        layer_names=[f"layer {i}" for i in range(len(names))],
        meta={"prompt": PROBE, "model": "learnai GPT, 4 layers, char-level"},
    )


def build_training(history) -> None:
    """Loss curves, for reference and future pages."""
    export_history(
        history,
        INTERACTIVE_DIR / "sampling" / "data" / "training.js",
        meta={"model": "learnai GPT, 4 layers, d_model=128", "corpus": "tinyshakespeare"},
    )


# `training` is built on request only — no page consumes loss curves yet, and the
# full per-step history is ~130 KB, too much to commit for nothing.
BUILDERS = {"sampling", "attention", "training"}
DEFAULT_TARGETS = {"sampling", "attention"}


def main() -> int:
    ap = argparse.ArgumentParser(description="regenerate interactive page data")
    ap.add_argument(
        "targets",
        nargs="*",
        default=None,
        help=f"subset of {sorted(BUILDERS)} (default: {sorted(DEFAULT_TARGETS)})",
    )
    ap.add_argument("--steps", type=int, default=2000)
    args = ap.parse_args()

    targets = set(args.targets) if args.targets else set(DEFAULT_TARGETS)
    unknown = targets - BUILDERS
    if unknown:
        print(f"unknown target(s): {sorted(unknown)}; choose from {sorted(BUILDERS)}")
        return 1

    setup(seed=0)
    model, tok, trainer, history = train_reference_model(steps=args.steps)
    print()

    if "sampling" in targets:
        build_sampling(model, tok, trainer.device)
    if "attention" in targets:
        build_attention(model, tok, trainer.device)
    if "training" in targets:
        build_training(history)

    print("\ndone — reload the interactive pages to see the new data")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Generate the BERT paper-reproduction notebook.

The notebook is generated from Python so its structure is reviewable and every
committed code cell stays output-free.  Run this file after editing the lesson,
then execute the notebook through ``tools/nbuild.py --run``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.nbuild import Notebook


def build() -> None:
    nb = Notebook(
        "Build BERT: teach a Transformer to read both ways",
        subtitle="""
**The question this answers:** How can one Transformer use *both* left and right context
without leaking the word it is supposed to learn?

**What you will build:** a tiny but mechanism-faithful BERT encoder, its masked-language-model
(MLM) and next-sentence-prediction (NSP) objectives, then a complete offline pre-training run.

**Paper:** Devlin et al., [*BERT: Pre-training of Deep Bidirectional Transformers for Language
Understanding*](https://aclanthology.org/N19-1423/) (NAACL 2019). Numbers attributed to the
paper below come from its [official PDF](https://aclanthology.org/N19-1423.pdf).

**Prerequisite:** `01_build_a_transformer.ipynb` through multi-head attention and Transformer
blocks. We will not derive Q/K/V again; this lesson isolates what BERT changed.
""",
    )

    nb.md("""
---
## 0. Setup

Everything is local: no pretrained checkpoint, Hugging Face download, or hidden API. The paper
used a 30,000-token WordPiece vocabulary; we use a small word vocabulary so every tensor remains
readable. That changes scale and tokenisation quality, not the BERT mechanisms under study.
""")
    nb.code("""
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from learnai.data import (
    ToyBERTTokenizer,
    collate_bert_examples,
    make_nsp_examples,
    mask_for_mlm,
)
from learnai.env import setup
from learnai.nn import BERT, BERTConfig, BERTForPreTraining, causal_mask
from learnai.viz import SERIES, attention_heatmap, use_style

ctx = setup(seed=7)
use_style()
torch.set_printoptions(precision=3, sci_mode=False, linewidth=140)
FAST = os.environ.get("LEARNAI_FAST", "0") == "1"
print("fast smoke path:", FAST)
""")

    nb.md("""
---
## 1. Why left-to-right is not enough

Consider `the animal did not cross the road because [MASK] was tired`. The pronoun *after* the
missing word is evidence for the answer. At `[MASK]`, a GPT-style causal head is forbidden to read
it; a BERT encoder can read both directions.

The crucial caveat: if the original answer remains in the input, a bidirectional model can copy it
through attention and the task is trivial. BERT therefore **removes/corrupts the answer first**, then
lets the remaining positions communicate in both directions.
""")
    nb.code("""
tokens = ["[CLS]", "the", "animal", "[MASK]", "because", "it", "was", "tired", "[SEP]"]
T = len(tokens)
gpt_visible = causal_mask(T)[0, 0].int()
bert_visible = torch.ones(T, T, dtype=torch.int)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
attention_heatmap(gpt_visible, tokens, ax=axes[0], title="GPT: only the left is visible", cbar=False)
attention_heatmap(bert_visible, tokens, ax=axes[1], title="BERT: both sides are visible", cbar=False)
plt.tight_layout()

mask_pos, clue_pos = tokens.index("[MASK]"), tokens.index("it")
assert gpt_visible[mask_pos, clue_pos] == 0
assert bert_visible[mask_pos, clue_pos] == 1
print(f"At [MASK], GPT can read {int(gpt_visible[mask_pos].sum())}/{T} positions; BERT can read {T}/{T}.")
""")
    nb.takeaway(
        "BERT's architectural difference from GPT is one missing triangular mask: self-attention is bidirectional at every layer.",
        "MLM prevents information leakage by corrupting the target token, not by making attention causal.",
        "Bidirectional attention is excellent for understanding; it does not directly define left-to-right generation.",
    )

    nb.md("""
---
## 2. Pack one or two spans into one sequence

The paper calls A and B “sentences”, but explicitly allows arbitrary contiguous spans. A pair is
packed as `[CLS] A [SEP] B [SEP]`. `[CLS]` is the sequence-level readout; `[SEP]` marks boundaries;
padding fills the fixed tensor length but must be invisible to attention.
""")
    nb.code("""
toy_documents = [
    ["a robin is a bird", "the robin can fly", "the robin builds a nest"],
    ["a salmon is a fish", "the salmon can swim", "the salmon lives in water"],
    ["a dog is an animal", "the dog can bark", "the dog likes long walks"],
    ["a cat is an animal", "the cat can purr", "the cat likes warm milk"],
]
all_sentences = [sentence for document in toy_documents for sentence in document]
tokenizer = ToyBERTTokenizer(all_sentences)
pair = tokenizer.encode_pair("a robin is a bird", "the robin can fly", max_length=14)
pair_tokens = tokenizer.convert_ids_to_tokens(pair.input_ids)

print(f"{'position':>8}", *[f"{i:>8}" for i in range(14)])
print(f"{'token':>8}", *[f"{t:>8}" for t in pair_tokens])
print(f"{'segment':>8}", *[f"{int(x):>8}" for x in pair.token_type_ids])
print(f"{'real?':>8}", *[f"{int(x):>8}" for x in pair.attention_mask])

first_sep = pair_tokens.index("[SEP]")
last_sep = len(pair_tokens) - 1 - pair_tokens[::-1].index("[SEP]")
assert pair_tokens[0] == "[CLS]" and first_sep < last_sep
assert pair.token_type_ids[: first_sep + 1].eq(0).all()
assert pair.token_type_ids[first_sep + 1 : last_sep + 1].eq(1).all()
assert pair.input_ids[~pair.attention_mask].eq(tokenizer.pad_token_id).all()
""")

    nb.md("""
---
## 3. Three learned embeddings become one input

For position `i`, BERT adds three vectors of the same width:

$$
x_i = E_{token(i)} + E_{position(i)} + E_{segment(i)}.
$$

Token says *what*, position says *where*, and segment says *which side of the pair*. The sum is
LayerNormed and dropped out before layer 1. Unlike the 2017 Transformer, BERT learns its position
table rather than using fixed sinusoids.
""")
    nb.code("""
demo_cfg = BERTConfig(
    vocab_size=tokenizer.vocab_size,
    max_seq_len=14,
    n_layer=1,
    n_head=2,
    d_model=12,
    d_ff=24,
    dropout=0.0,
    pad_token_id=tokenizer.pad_token_id,
)
demo_encoder = BERT(demo_cfg).eval()
ids = pair.input_ids.unsqueeze(0)
types = pair.token_type_ids.unsqueeze(0)
positions = torch.arange(ids.size(1)).unsqueeze(0)

word_part = demo_encoder.embeddings.token(ids)
position_part = demo_encoder.embeddings.position(positions)
segment_part = demo_encoder.embeddings.segment(types)
manual = demo_encoder.embeddings.norm(word_part + position_part + segment_part)
reference = demo_encoder.embeddings(ids, types)
assert torch.allclose(manual, reference, atol=1e-7)

component_norms = torch.stack(
    [word_part.norm(dim=-1)[0], position_part.norm(dim=-1)[0], segment_part.norm(dim=-1)[0]]
).detach()
fig, ax = plt.subplots(figsize=(10, 3.4))
im = ax.imshow(component_norms, aspect="auto", cmap="Blues")
ax.set_yticks(range(3), ["token", "position", "segment"])
ax.set_xticks(range(14), pair_tokens, rotation=55, ha="right")
ax.set_title("The three learned signals added at every position")
ax.grid(False)
plt.colorbar(im, ax=ax, label="vector L2 norm", fraction=0.025, pad=0.02)
plt.tight_layout()
""")
    nb.takeaway(
        "BERT does not concatenate token, position, and segment features; it adds three D-dimensional vectors.",
        "Segment IDs distinguish A from B even though both spans share one self-attention graph.",
    )

    nb.md("""
---
## 4. Three different things are called a “mask”

This vocabulary collision causes many BERT bugs:

1. a **causal attention mask** forbids future keys (GPT has it; BERT does not);
2. a **padding attention mask** hides `[PAD]` keys (both models need it for padded batches);
3. an **MLM target mask** selects which token identities to corrupt and score (BERT's objective).

`[MASK]` is an ordinary visible token ID after corruption. It does not block attention.
""")
    nb.code("""
padding_visibility = pair.attention_mask.view(1, -1).expand(14, -1).int()
mlm_targets = torch.zeros(14, dtype=torch.int)
mlm_targets[3] = 1

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1, 1, 0.35]})
attention_heatmap(causal_mask(14)[0, 0].int(), pair_tokens, ax=axes[0], title="causal attention mask", cbar=False)
attention_heatmap(padding_visibility, pair_tokens, ax=axes[1], title="padding attention mask", cbar=False)
axes[2].imshow(mlm_targets[:, None], cmap="Blues", vmin=0, vmax=1, aspect="auto")
axes[2].set_yticks(range(14), pair_tokens)
axes[2].set_xticks([0], ["loss?"])
axes[2].set_title("MLM selector")
axes[2].grid(False)
plt.tight_layout()

assert padding_visibility[3].sum() == pair.attention_mask.sum()
assert mlm_targets.sum() == 1
""")

    nb.exercise(
        "Change `mlm_targets` to select two ordinary tokens. Before running, predict whether any entry in `padding_visibility` should change.",
        hint="The objective chooses *what to predict*; the attention mask chooses *what can be read*. They are independent tensors.",
        starter="""
# Choose two target positions, but do not touch the attention mask.
my_targets = torch.zeros_like(mlm_targets)
my_targets[[2, 6]] = 1
print("MLM targets:", [pair_tokens[i] for i in my_targets.nonzero().flatten().tolist()])
assert my_targets.sum() == 2
assert torch.equal(padding_visibility, pair.attention_mask.view(1, -1).expand(14, -1))
""",
    )

    nb.md("""
---
## 5. The encoder is bidirectional and post-norm

Each BERT layer remains `attention → MLP`, but uses full self-attention and the original
Transformer's **post-norm** order:

```text
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + MLP(x))
```

Later large language models usually use pre-norm. Our reference module keeps the public BERT code's
tanh GELU approximation, LayerNorm epsilon `1e-12`, 4× MLP (by default), and learned embeddings. Change a token on the right:
position 0 reacts in BERT, but cannot react in a causal ablation with identical initial weights.
""")
    nb.code("""
base = tokenizer.encode_pair("the robin can fly", max_length=8).input_ids.unsqueeze(0)
changed = base.clone()
changed[0, 3] = tokenizer.stoi["swim"]

torch.manual_seed(3)
bi = BERT(BERTConfig(tokenizer.vocab_size, max_seq_len=8, n_layer=1, n_head=2,
                     d_model=16, d_ff=32, dropout=0.0, pad_token_id=tokenizer.pad_token_id)).eval()
torch.manual_seed(3)
left_only = BERT(BERTConfig(tokenizer.vocab_size, max_seq_len=8, n_layer=1, n_head=2,
                            d_model=16, d_ff=32, dropout=0.0,
                            pad_token_id=tokenizer.pad_token_id, causal=True)).eval()

bi_change = (bi(base).last_hidden_state - bi(changed).last_hidden_state).norm(dim=-1)[0].detach()
left_change = (left_only(base).last_hidden_state - left_only(changed).last_hidden_state).norm(dim=-1)[0].detach()
probe_tokens = tokenizer.convert_ids_to_tokens(base[0])

fig, ax = plt.subplots(figsize=(8.5, 3.4))
x = np.arange(8)
ax.bar(x - 0.18, left_change, 0.36, label="causal / left-only", color=SERIES[1])
ax.bar(x + 0.18, bi_change, 0.36, label="BERT / bidirectional", color=SERIES[0])
ax.set_xticks(x, probe_tokens, rotation=35, ha="right")
ax.set_ylabel("change in hidden vector (L2)")
ax.set_title("Who can notice that a right-side token changed?")
ax.legend()
plt.tight_layout()

assert left_change[0] < 1e-7
assert bi_change[0] > 1e-5
""")

    nb.md("""
---
## 6. MLM: select 15%, then use 80/10/10

The denominator matters. BERT first selects **15% of WordPiece positions** as prediction targets.
Inside that selected 15%:

- 80% become `[MASK]` → 12% of all positions;
- 10% become a random token → 1.5% of all positions;
- 10% stay unchanged → 1.5% of all positions.

All three groups still predict the *original* token. The last two reduce the pre-train/fine-tune
mismatch because `[MASK]` never appears in downstream inputs. Special tokens are never selected.
""")
    nb.code("""
many_tokens = torch.full((200_000,), tokenizer.stoi["robin"], dtype=torch.long)
corruption = mask_for_mlm(
    many_tokens,
    mask_token_id=tokenizer.mask_token_id,
    vocab_size=tokenizer.vocab_size,
    selection_probability=0.15,
    generator=torch.Generator().manual_seed(11),
)
selected_rate = corruption.selected.float().mean().item()
strategies = corruption.strategy[corruption.selected]
within_selected = torch.tensor([(strategies == i).float().mean() for i in range(3)])
all_token_rates = selected_rate * within_selected

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
axes[0].bar(["not scored", "MLM target"], [1 - selected_rate, selected_rate], color=[SERIES[2], SERIES[0]])
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("fraction of all tokens")
axes[0].set_title(f"Select targets: {selected_rate:.1%}")
for i, value in enumerate([1 - selected_rate, selected_rate]):
    axes[0].text(i, value + 0.025, f"{value:.1%}", ha="center")

labels_801010 = ["[MASK]", "random", "unchanged"]
axes[1].bar(labels_801010, within_selected, color=SERIES[:3])
axes[1].set_ylim(0, 1)
axes[1].set_ylabel("fraction of selected targets")
axes[1].set_title("Then corrupt selected targets")
for i, (selected_share, total_share) in enumerate(zip(within_selected, all_token_rates, strict=True)):
    axes[1].text(i, selected_share + 0.025, f"{selected_share:.1%} ({total_share:.1%} of all)", ha="center")
plt.tight_layout()

assert abs(selected_rate - 0.15) < 0.003
assert torch.allclose(within_selected, torch.tensor([0.8, 0.1, 0.1]), atol=0.01)
""")
    nb.takeaway(
        "Only about 15% of token positions contribute MLM loss in one batch, not every token.",
        "The famous 80/10/10 percentages are conditional on being selected; only about 12% of all input positions become `[MASK]`.",
        "Targets always contain the clean token ID; corruption changes the input, never the answer.",
    )

    nb.md("""
---
## 7. The MLM head predicts only selected positions

At every position the head computes `Dense → GELU → LayerNorm → vocabulary logits`. The
decoder reuses the token-embedding matrix transposed: the same vector table reads token IDs on the
way in and scores token identities on the way out. Unselected labels are `-100`, PyTorch's ignore
index, so they contribute neither loss nor gradient.
""")
    nb.code("""
one_clean = pair.input_ids.unsqueeze(0)
one_corruption = mask_for_mlm(
    one_clean,
    mask_token_id=tokenizer.mask_token_id,
    vocab_size=tokenizer.vocab_size,
    special_token_ids=tokenizer.special_token_ids,
    selection_probability=0.5,  # high only so this one printed row has several targets
    generator=torch.Generator().manual_seed(4),
)
shown_input = tokenizer.convert_ids_to_tokens(one_corruption.input_ids[0])
shown_labels = [tokenizer.itos[int(x)] if x != -100 else "·" for x in one_corruption.labels[0]]
print("clean:    ", pair_tokens)
print("corrupted:", shown_input)
print("targets:  ", shown_labels, "  (· = no MLM loss)")

head_demo = BERTForPreTraining(demo_cfg).eval()
out = head_demo(one_corruption.input_ids, pair.token_type_ids.unsqueeze(0), pair.attention_mask.unsqueeze(0))
selected_logits = out.mlm_logits[one_corruption.selected]
selected_labels = one_corruption.labels[one_corruption.selected]
manual_mlm = -selected_logits.log_softmax(-1).gather(1, selected_labels[:, None]).mean()
reference_mlm = F.cross_entropy(selected_logits, selected_labels)

assert torch.allclose(manual_mlm, reference_mlm, atol=1e-7)
assert head_demo.mlm_decoder.weight is head_demo.bert.embeddings.token.weight
print(f"loss reads {selected_labels.numel()} of {one_clean.numel()} positions: {reference_mlm:.3f}")
""")

    nb.exercise(
        "Set `selection_probability=1.0`, then verify that `[CLS]`, `[SEP]`, and `[PAD]` still have label `-100`. Predict the number of eligible content targets first.",
        hint="Pass `tokenizer.special_token_ids`; eligibility is separate from the selection probability.",
        starter="""
all_content = mask_for_mlm(
    one_clean,
    mask_token_id=tokenizer.mask_token_id,
    vocab_size=tokenizer.vocab_size,
    special_token_ids=tokenizer.special_token_ids,
    selection_probability=1.0,
    generator=torch.Generator().manual_seed(0),
)
eligible_count = sum(token not in tokenizer.SPECIAL_TOKENS for token in pair_tokens)
print("selected content tokens:", int(all_content.selected.sum()), "expected:", eligible_count)
assert int(all_content.selected.sum()) == eligible_count
""",
    )

    nb.md("""
---
## 8. NSP gives `[CLS]` a sentence-pair job

For each A/B training example, the original paper used:

- 50% `IsNext` (`label=0` here): B truly follows A in the document;
- 50% `NotNext` (`label=1`): B is sampled from another document.

The final `[CLS]` vector goes through a tanh pooler and a two-way classifier. The pre-training loss
is simply `mean MLM loss + mean NSP loss`. This is historically faithful to BERT; it is not a claim
that every later encoder needs NSP.
""")
    nb.code("""
nsp_examples = make_nsp_examples(toy_documents, seed=2)
for example in nsp_examples[:6]:
    print(f"{['IsNext ', 'NotNext'][example.label]} | {example.sentence_a:<22} -> {example.sentence_b}")

labels = torch.tensor([example.label for example in nsp_examples])
fig, ax = plt.subplots(figsize=(4.8, 3.2))
counts = [(labels == i).sum().item() for i in range(2)]
ax.bar(["IsNext", "NotNext"], counts, color=SERIES[:2])
ax.set_ylabel("training examples")
ax.set_title("NSP construction is exactly balanced")
for i, count in enumerate(counts):
    ax.text(i, count + 0.1, str(count), ha="center")
plt.tight_layout()

assert counts[0] == counts[1]
""")
    nb.code("""
pretrain_cfg = BERTConfig(
    vocab_size=tokenizer.vocab_size,
    max_seq_len=14,
    n_layer=2,
    n_head=4,
    d_model=64,
    d_ff=128,
    dropout=0.0,
    pad_token_id=tokenizer.pad_token_id,
)
joint_demo = BERTForPreTraining(pretrain_cfg)
joint_batch = collate_bert_examples(
    nsp_examples,
    tokenizer,
    max_length=14,
    selection_probability=0.5,
    generator=torch.Generator().manual_seed(8),
)
joint_out = joint_demo(
    joint_batch["input_ids"], joint_batch["token_type_ids"], joint_batch["attention_mask"]
)
total, mlm, nsp = joint_demo.loss(joint_out, joint_batch["mlm_labels"], joint_batch["nsp_labels"])
assert torch.allclose(total, mlm + nsp)
print(f"joint loss = MLM {mlm:.3f} + NSP {nsp:.3f} = {total:.3f}")
""")

    nb.exercise(
        "Inspect `nsp_examples` and write a check that no `NotNext` pair is a real adjacent pair in `toy_documents`.",
        hint="Build a set of `(A, B)` tuples by zipping each document with its one-position-shifted self.",
        starter="""
real_pairs = {
    (a, b)
    for document in toy_documents
    for a, b in zip(document, document[1:], strict=False)
}
negative_pairs = {(ex.sentence_a, ex.sentence_b) for ex in nsp_examples if ex.label == 1}
overlap = real_pairs & negative_pairs
print("mislabeled true continuations:", overlap)
assert not overlap
""",
    )

    nb.md("""
---
## 9. Assemble TinyBERT—and account for every output

Our model keeps the paper's computation graph but shrinks it to two layers and 64 hidden features.
The real configurations were BERT-base `(L=12, H=768, A=12, 110M parameters)` and BERT-large
`(L=24, H=1024, A=16, 340M)`, both with `d_ff=4H`. We compare scale without instantiating them.
""")
    nb.code("""
torch.manual_seed(7)
model = BERTForPreTraining(pretrain_cfg).to(ctx.device)
n_params = sum(p.numel() for p in model.parameters())
print(f"TinyBERT: {n_params:,} parameters ({n_params / 1e6:.3f}M)")

paper_sizes = {"TinyBERT": n_params, "BERT-base": 110_000_000, "BERT-large": 340_000_000}
fig, ax = plt.subplots(figsize=(7.5, 3.4))
bars = ax.bar(paper_sizes.keys(), paper_sizes.values(), color=SERIES[:3])
ax.set_yscale("log")
ax.set_ylabel("parameters (log scale)")
ax.set_title("Same mechanisms, radically different scale")
for bar, value in zip(bars, paper_sizes.values(), strict=True):
    label = f"{value / 1e6:.3g}M"
    ax.text(bar.get_x() + bar.get_width() / 2, value * 1.18, label, ha="center")
plt.tight_layout()

shape_out = model(
    joint_batch["input_ids"].to(ctx.device),
    joint_batch["token_type_ids"].to(ctx.device),
    joint_batch["attention_mask"].to(ctx.device),
)
B, T = joint_batch["input_ids"].shape
assert shape_out.hidden_states.shape == (B, T, pretrain_cfg.d_model)
assert shape_out.mlm_logits.shape == (B, T, tokenizer.vocab_size)
assert shape_out.nsp_logits.shape == (B, 2)
""")

    nb.md("""
---
## 10. Pre-train on a tiny offline world

We now train on 16 sentence pairs about four animals. Dynamic masking is regenerated every step,
so the model eventually sees many cloze positions. This is a **phenomenon reproduction**, not a
reproduction of the paper's accuracy: BERT used BooksCorpus (800M words) plus English Wikipedia
(2.5B words), sequences up to 512, and one million update steps.

We track the two objectives separately. A random vocabulary predictor starts near `log(V)` MLM
loss and a balanced NSP classifier near `log(2)`.
""")
    nb.code("""
def fixed_probe(sentence: str, target: str):
    packed = tokenizer.encode_pair(sentence, max_length=14)
    target_id = tokenizer.stoi[target]
    target_pos = tokenizer.convert_ids_to_tokens(packed.input_ids).index(target)
    masked = packed.input_ids.clone()
    masked[target_pos] = tokenizer.mask_token_id
    return packed, masked, target_pos, target_id

probes = [
    fixed_probe("the robin can fly", "fly"),
    fixed_probe("the salmon can swim", "swim"),
    fixed_probe("the dog can bark", "bark"),
    fixed_probe("the cat can purr", "purr"),
]

@torch.no_grad()
def probe_probabilities(current_model):
    current_model.eval()
    values = []
    for packed, masked, position, target_id in probes:
        out = current_model(
            masked[None].to(ctx.device),
            packed.token_type_ids[None].to(ctx.device),
            packed.attention_mask[None].to(ctx.device),
        )
        values.append(out.mlm_logits[0, position].softmax(-1)[target_id].item())
    return values

before_probe = probe_probabilities(model)
print("random-model MLM baseline:", round(math.log(tokenizer.vocab_size), 3))
print("random-model NSP baseline:", round(math.log(2), 3))
print("correct-token probabilities before training:", [round(p, 3) for p in before_probe])
""")
    nb.code("""
steps = 12 if FAST else 500
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.999), weight_decay=0.01)
rng = torch.Generator().manual_seed(100)
history = {"total": [], "mlm": [], "nsp": [], "mlm_acc": [], "nsp_acc": []}

model.train()
for step in range(steps):
    batch = collate_bert_examples(
        nsp_examples,
        tokenizer,
        max_length=14,
        selection_probability=0.15,
        generator=rng,
    )
    batch = {name: tensor.to(ctx.device) for name, tensor in batch.items()}
    output = model(batch["input_ids"], batch["token_type_ids"], batch["attention_mask"])
    loss, mlm_loss, nsp_loss = model.loss(output, batch["mlm_labels"], batch["nsp_labels"])

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    selected = batch["mlm_labels"] != -100
    with torch.no_grad():
        mlm_acc = (output.mlm_logits[selected].argmax(-1) == batch["mlm_labels"][selected]).float().mean()
        nsp_acc = (output.nsp_logits.argmax(-1) == batch["nsp_labels"]).float().mean()
    for name, value in [("total", loss), ("mlm", mlm_loss), ("nsp", nsp_loss),
                        ("mlm_acc", mlm_acc), ("nsp_acc", nsp_acc)]:
        history[name].append(float(value.detach().cpu()))

    if step in {0, steps - 1} or (not FAST and (step + 1) % 100 == 0):
        print(f"step {step + 1:>3}/{steps}: total={loss:.3f} mlm={mlm_loss:.3f} "
              f"nsp={nsp_loss:.3f} mlm_acc={mlm_acc:.1%} nsp_acc={nsp_acc:.1%}")

assert np.isfinite(history["total"]).all()
if not FAST:
    assert np.mean(history["total"][-25:]) < np.mean(history["total"][:25])
""")
    nb.code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(history["total"], label="total")
axes[0].plot(history["mlm"], label="MLM")
axes[0].plot(history["nsp"], label="NSP")
axes[0].axhline(math.log(tokenizer.vocab_size) + math.log(2), color=SERIES[3], ls="--", lw=1,
                label="uniform joint baseline")
axes[0].set_xlabel("update step")
axes[0].set_ylabel("cross entropy")
axes[0].set_title("Two objectives are optimised jointly")
axes[0].legend()

window = min(20, steps)
smooth = lambda values: np.convolve(values, np.ones(window) / window, mode="valid")
axes[1].plot(np.arange(window - 1, steps), smooth(history["mlm_acc"]), label="masked token")
axes[1].plot(np.arange(window - 1, steps), smooth(history["nsp_acc"]), label="next sentence")
axes[1].axhline(0.5, color=SERIES[3], ls="--", lw=1, label="NSP chance")
axes[1].set_ylim(0, 1.05)
axes[1].set_xlabel("update step")
axes[1].set_ylabel("training accuracy")
axes[1].set_title(f"Accuracy ({window}-step moving average)")
axes[1].legend()
plt.tight_layout()
""")

    nb.md("""
---
## 11. Ask what it learned—without inventing a story

We use fixed cloze probes so “before” and “after” differ only in model weights. A tiny model on a
tiny corpus can memorise these associations; it does **not** reproduce broad language understanding.
The honest claim is narrower: joint bidirectional pre-training raised the probability of held-out
masked answers from this toy world.
""")
    nb.code("""
after_probe = probe_probabilities(model)
probe_names = ["robin → fly", "salmon → swim", "dog → bark", "cat → purr"]
x = np.arange(len(probes))
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.bar(x - 0.18, before_probe, 0.36, label="before", color=SERIES[1])
ax.bar(x + 0.18, after_probe, 0.36, label="after", color=SERIES[0])
ax.set_xticks(x, probe_names)
ax.set_ylim(0, 1)
ax.set_ylabel("P(correct masked token)")
ax.set_title("Did pre-training improve the exact cloze questions?")
ax.legend()
for i, value in enumerate(after_probe):
    ax.text(i + 0.18, value + 0.025, f"{value:.2f}", ha="center")
plt.tight_layout()

if not FAST:
    assert np.mean(after_probe) > np.mean(before_probe)
print("mean correct probability:", f"{np.mean(before_probe):.3f} → {np.mean(after_probe):.3f}")
""")
    nb.code("""
packed, masked, target_pos, _ = probes[0]
model.eval()
with torch.no_grad():
    inspected = model(
        masked[None].to(ctx.device),
        packed.token_type_ids[None].to(ctx.device),
        packed.attention_mask[None].to(ctx.device),
        return_attn=True,
    )
attention = inspected.attentions[0][0, 0].cpu()
inspect_tokens = tokenizer.convert_ids_to_tokens(masked)
attention_heatmap(attention, inspect_tokens, title="Layer 0, head 0 after toy pre-training")
plt.show()

mask_row = attention[target_pos]
top = mask_row.topk(4)
print("[MASK] reads most from:", [(inspect_tokens[int(i)], round(float(w), 3)) for w, i in zip(top.values, top.indices, strict=True)])
assert torch.allclose(mask_row[packed.attention_mask].sum(), torch.tensor(1.0), atol=1e-5)
""")
    nb.takeaway(
        "Inspecting attention is evidence about routing, not proof that a head implements a human-named linguistic rule.",
        "A probe must compare fixed inputs before/after training; otherwise corruption noise can masquerade as learning.",
    )

    nb.md("""
---
## 12. One encoder, small task-specific heads

Figure 1's practical insight is architectural reuse. The same final hidden tensor `(B,T,H)` supports:

- sentence classification: `[CLS] → K` logits;
- NER: every `T_i → K` logits;
- extractive QA: every `T_i → (start, end)` logits.

The paper fine-tuned **all encoder parameters** along with the new head. `[CLS]` is not automatically
a universal sentence embedding; it gets a useful task meaning through NSP and downstream fine-tuning.
""")
    nb.code("""
hidden = shape_out.hidden_states.detach()  # (B, T, H)
sentence_head = torch.nn.Linear(pretrain_cfg.d_model, 3).to(hidden.device)
ner_head = torch.nn.Linear(pretrain_cfg.d_model, 5).to(hidden.device)
qa_head = torch.nn.Linear(pretrain_cfg.d_model, 2).to(hidden.device)

sentence_logits = sentence_head(hidden[:, 0])
ner_logits = ner_head(hidden)
span_logits = qa_head(hidden)
start_logits, end_logits = span_logits.unbind(dim=-1)

assert sentence_logits.shape == (B, 3)
assert ner_logits.shape == (B, T, 5)
assert start_logits.shape == end_logits.shape == (B, T)

head_params = sum(p.numel() for head in (sentence_head, ner_head, qa_head) for p in head.parameters())
encoder_params = sum(p.numel() for p in model.bert.parameters())
print(f"three new toy heads: {head_params:,} params; shared encoder: {encoder_params:,} params")
""")

    nb.exercise(
        "Mask the QA `start_logits` at padding positions so a padded token can never be selected as an answer boundary.",
        hint="Use `masked_fill(~attention_mask, -inf)`. The mask has shape `(B,T)`, exactly like the logits.",
        starter="""
qa_padding_mask = joint_batch["attention_mask"].to(start_logits.device)
safe_start_logits = start_logits.masked_fill(~qa_padding_mask, float("-inf"))
predicted_starts = safe_start_logits.argmax(dim=-1)
assert qa_padding_mask.gather(1, predicted_starts[:, None]).all()
print("predicted start positions:", predicted_starts.tolist())
""",
    )

    nb.md("""
---
## 13. What the paper actually measured

Our toy curve establishes that the code learns. The paper's controlled evidence establishes why its
design mattered. Table 5 held architecture/data/fine-tuning fixed while removing NSP or replacing
bidirectional MLM with left-to-right LM. Below, zero means full BERT-base; negative means the
ablation scored worse.
""")
    nb.code("""
tasks = ["MNLI", "QNLI", "MRPC", "SST-2", "SQuAD"]
paper_scores = {
    "BERT-base": [84.4, 88.4, 86.7, 92.7, 88.5],
    "No NSP": [83.9, 84.9, 86.5, 92.6, 87.9],
    "LTR & No NSP": [82.1, 84.3, 77.5, 92.1, 77.8],
    "+ BiLSTM": [82.1, 84.1, 75.7, 91.6, 84.9],
}
baseline = np.array(paper_scores["BERT-base"])
names = list(paper_scores)
deltas = np.stack([np.array(paper_scores[name]) - baseline for name in names])

fig, ax = plt.subplots(figsize=(9, 4.1))
im = ax.imshow(deltas, cmap="RdBu", vmin=-12, vmax=12, aspect="auto")
ax.set_xticks(range(len(tasks)), tasks)
ax.set_yticks(range(len(names)), names)
ax.set_title("BERT paper Table 5: score change versus full BERT-base")
for i in range(len(names)):
    for j in range(len(tasks)):
        ax.text(j, i, f"{deltas[i, j]:+.1f}", ha="center", va="center",
                color="white" if abs(deltas[i, j]) > 7 else "black")
ax.grid(False)
plt.colorbar(im, ax=ax, label="score-point difference", fraction=0.03, pad=0.02)
plt.tight_layout()

assert np.isclose(deltas[names.index("LTR & No NSP"), tasks.index("SQuAD")], -10.7)
assert np.isclose(deltas[names.index("No NSP"), tasks.index("QNLI")], -3.5)
""")
    nb.takeaway(
        "The paper's strongest architectural evidence is the left-to-right ablation: MRPC fell 9.2 points and SQuAD F1 fell 10.7.",
        "In this paper's setup, removing NSP also hurt—especially QNLI (−3.5)—although later work changed the pre-training recipe and revisited NSP.",
        "The headline is not just a new layer: deep bidirectionality + self-supervised pre-training + minimal task heads formed one reusable interface.",
    )

    nb.md("""
---
## 14. What we reproduced—and what we did not

**Reproduced here**

- bidirectional, post-norm Transformer encoder;
- token + position + segment inputs and `[CLS] A [SEP] B [SEP]` packing;
- 15% dynamic MLM selection with the 80/10/10 corruption rule;
- tied MLM decoder, `[CLS]` pooler, balanced NSP, and the joint loss;
- a controlled causal ablation and a complete offline learning run.

**Deliberately not reproduced**

- WordPiece learning, the 3.3B-word corpus, 110M/340M parameters, TPU training, GLUE/SQuAD
  fine-tuning, or the paper's leaderboard results. Our tokenizer is word-level, and our short
  AdamW run omits the paper's million-step warmup/decay training schedule.

That boundary matters. This notebook reproduces the **mechanism and a small learning phenomenon**,
not the scale-dependent empirical result.

### Where next

- Tokenisation: build WordPiece/BPE in `03_nlp`.
- RoBERTa: ask which parts were BERT essentials and which were 2018 training choices.
- Encoder–decoder models: add a causal decoder when the task must generate an output sequence.
- Read the original [BERT paper](https://aclanthology.org/N19-1423.pdf), especially Figures 1–2,
  §3.1, Table 5, and Appendix C.2; you now have executable counterparts for each.
""")

    nb.save("notebooks/04_transformer/02_build_a_bert.ipynb")


if __name__ == "__main__":
    build()

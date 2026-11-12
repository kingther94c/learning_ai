"""Change a released model's behaviour by editing its weights by hand — no gradients.

Training is not the only way to change what a network does.  If a behaviour can be
traced to *a direction in activation space* and to the handful of neurons that
write along it, the weights of those neurons can simply be rewritten.  This module
holds the three tools that takes, for the open-weight models of
``learnai.openweights``:

* **read** — :func:`recording` taps every block's MLP during a forward pass, so the
  residual stream, the neuron activations and what each MLP writes can be averaged
  over chosen token positions (:func:`mean_activations`);
* **locate** — :func:`contrast_direction` and :func:`neuron_contributions` turn two
  such averages ("stuck in a loop" vs "reasoning normally") into a direction and a
  ranking of the neurons responsible for it;
* **write** — :func:`subtract_direction` changes the chosen columns of an MLP's
  output matrix, and :func:`apply_patch` installs the change in a loaded model.

The recipe is the one reported in "Sparks of Weight-Space Programming" (Takehi,
Teoh, Zhang & Khattab, MIT CSAIL, Nov 2026) for Qwen3.5-0.8B, whose reasoning
loops were traced to "one direction written by the MLPs of layers 15 and 19" and
removed by subtracting that direction "from 64 output columns in each".  The
released weights show what that means numerically (measured here by diffing them
against the base model): in each of the two matrices exactly 64 columns change,
the change has rank one, and every changed column obeys

    w_j  <-  w_j - 6 (w_j . d) d          d: unit vector in the residual stream

so the component along ``d`` is not removed (that would be a factor of 1) but
*reversed*: a neuron that used to write +p along ``d`` now writes -5p.

Where the vectors live.  A block's MLP computes ``down_proj(a)`` with
``a = silu(gate_proj(x)) * up_proj(x)``: ``a[j]`` is how strongly neuron ``j``
fires and column ``j`` of ``down_proj.weight`` (shape ``[d_model, d_ff]``) is what
that neuron adds to the residual stream when it does.  Editing column ``j`` changes
what neuron ``j`` says without changing when it speaks.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# The edit itself: plain linear algebra, testable without a model
# --------------------------------------------------------------------------------------


@dataclasses.dataclass
class LayerEdit:
    """``W[:, cols] -= outer(direction, coef)`` for one block's ``mlp.down_proj``."""

    layer: int
    cols: np.ndarray  # [k]        which neurons (columns of W) are rewritten
    coef: np.ndarray  # [k]        how much of `direction` is subtracted from each
    direction: np.ndarray  # [d_model]  unit vector in the residual stream

    @property
    def n_weights(self) -> int:
        return len(self.cols) * len(self.direction)


def subtract_direction(
    W: np.ndarray, cols: np.ndarray, direction: np.ndarray, strength: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract ``strength`` times each chosen column's component along ``direction``.

        w_j  <-  w_j - strength * (w_j . d) d            W: [d_model, d_ff]

    ``strength=1`` projects the direction out of those columns (they can no longer
    write along it); ``strength=2`` reflects them; the paper's released edit uses 6.
    Returns ``(coef, W_edited)`` where ``coef[j] = strength * (w_j . d)`` is the
    compact form of the edit (see :class:`LayerEdit`).
    """
    d = direction / np.linalg.norm(direction)  # [d_model]
    coef = strength * (d @ W[:, cols])  # [k]   strength * (w_j . d)
    edited = W.copy()
    edited[:, cols] -= np.outer(d, coef)  # [d_model, k]
    return coef, edited


def contrast_direction(mean_a: np.ndarray, mean_b: np.ndarray) -> np.ndarray:
    """Unit vector from condition B towards condition A: ``(mean_a - mean_b)`` normalised.

    The difference of two class means is the simplest "what distinguishes A from
    B" direction there is.  It needs no training and two forward passes' worth of
    statistics, which is why hand-editing recipes start from it.
    """
    diff = mean_a - mean_b
    return diff / np.linalg.norm(diff)


def neuron_contributions(
    acts_a: np.ndarray, acts_b: np.ndarray, W: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """How much each neuron pushes the residual stream along ``direction`` in A vs B.

    Neuron ``j`` writes ``a_j * w_j``, so along ``d`` it contributes ``a_j (w_j . d)``.
    Averaged over positions, the difference between the conditions is

        score_j = (mean_A[a_j] - mean_B[a_j]) * (w_j . d)        -> [d_ff]

    and ``score.sum()`` is exactly the gap between the two conditions in what the
    whole MLP writes along ``d``.  A large positive score means "this neuron is a
    reason the model is further along ``d`` in A than in B".
    """
    return (acts_a - acts_b) * (direction @ W)  # [d_ff] * [d_ff]


# --------------------------------------------------------------------------------------
# Patches: an edit small enough to keep beside the model instead of copying it
# --------------------------------------------------------------------------------------


def save_patch(path: str | Path, edits: Sequence[LayerEdit], **meta) -> Path:
    """Store edits as one ``.npz``: three small arrays per layer plus a JSON note."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for e in edits:
        arrays[f"cols_{e.layer}"] = np.asarray(e.cols, dtype=np.int32)
        arrays[f"coef_{e.layer}"] = np.asarray(e.coef, dtype=np.float32)
        arrays[f"direction_{e.layer}"] = np.asarray(e.direction, dtype=np.float32)
    meta["layers"] = [e.layer for e in edits]
    np.savez(path, meta=np.array(json.dumps(meta)), **arrays)
    return path


def load_patch(path: str | Path) -> tuple[list[LayerEdit], dict]:
    with np.load(path) as data:
        meta = json.loads(str(data["meta"]))
        edits = [
            LayerEdit(i, data[f"cols_{i}"], data[f"coef_{i}"], data[f"direction_{i}"])
            for i in meta["layers"]
        ]
    return edits, meta


# --------------------------------------------------------------------------------------
# MLX side.  Imported lazily so the algebra above works (and is tested) anywhere.
# --------------------------------------------------------------------------------------


def _mlx():
    import mlx.core as mx
    import mlx.nn as nn

    return mx, nn


def dense_weight(linear) -> np.ndarray:
    """The ``[d_out, d_in]`` matrix of a linear layer as float32, unpacking 4-bit storage."""
    mx, nn = _mlx()
    if isinstance(linear, nn.QuantizedLinear):
        W = mx.dequantize(
            linear.weight,
            linear.scales,
            linear.biases,
            group_size=linear.group_size,
            bits=linear.bits,
        )
    else:
        W = linear.weight
    return np.array(W.astype(mx.float32))


def _edited_linear_class():
    mx, nn = _mlx()

    class EditedLinear(nn.Module):
        """A linear layer with a :class:`LayerEdit` applied, without touching its storage.

        For an input ``a`` the edited matrix gives

            W' a = W a - d * sum_j coef_j a_j           (j over the edited columns)

        which is the original output minus one vector, scaled by how strongly the
        edited neurons fire.  Computing it this way leaves a 4-bit quantised matrix
        packed, and makes the whole edit ``2k + d_model`` numbers.
        """

        def __init__(self, inner, edit: LayerEdit):
            super().__init__()
            self.inner = inner
            self.cols = mx.array(np.asarray(edit.cols, dtype=np.int32))  # [k]
            self.coef = mx.array(np.asarray(edit.coef, dtype=np.float32))  # [k]
            self.direction = mx.array(np.asarray(edit.direction, dtype=np.float32))

        def __call__(self, a):
            y = self.inner(a)  # [..., d_model]
            fired = mx.take(a, self.cols, axis=-1).astype(mx.float32) @ self.coef  # [...]
            return y - (fired[..., None] * self.direction).astype(y.dtype)

    return EditedLinear


def apply_patch(model, edits: Sequence[LayerEdit]) -> None:
    """Install edits in a loaded model (idempotent per layer: re-applying replaces)."""
    EditedLinear = _edited_linear_class()
    for e in edits:
        mlp = model.layers[e.layer].mlp
        inner = getattr(mlp.down_proj, "inner", mlp.down_proj)
        mlp.down_proj = EditedLinear(inner, e)


def remove_patch(model) -> None:
    """Undo :func:`apply_patch`: every block gets its original ``down_proj`` back."""
    for layer in model.layers:
        if hasattr(layer.mlp.down_proj, "inner"):
            layer.mlp.down_proj = layer.mlp.down_proj.inner


# --------------------------------------------------------------------------------------
# Reading activations
# --------------------------------------------------------------------------------------


def _tap_class():
    _, nn = _mlx()

    class Tap(nn.Module):
        """Stands in for a module and shows a hook what goes in and what comes out.

        MLX has no forward hooks, but a module is just an attribute of its parent,
        so swapping it for a wrapper does the same job.
        """

        def __init__(self, inner, hook: Callable):
            super().__init__()
            self.inner = inner
            self.hook = hook

        def __call__(self, x, *args, **kwargs):
            y = self.inner(x, *args, **kwargs)
            self.hook(x, y)
            return y

    return Tap


@contextlib.contextmanager
def recording(model, reduce: Callable) -> Iterator[list]:
    """Tap every block's MLP for the forward passes run inside the ``with`` body.

    A block computes ``mid = x + attention(norm(x))`` and then
    ``out = mid + mlp(norm(mid))``.  For block ``i`` the hook receives

        mid    [B, T, d_model]   residual stream entering the MLP
        acts   [B, T, d_ff]      neuron activations (the input of ``down_proj``)
        write  [B, T, d_model]   what the MLP adds (the output of ``down_proj``)

    and ``reduce(i, mid, acts, write)`` must boil them down to something small;
    its results are appended to the yielded list.  Reducing inside the hook matters:
    MLX is lazy, so only the reductions are ever materialised and the full
    ``[T, d_ff]`` activations of 32 blocks never have to sit in memory at once.
    """
    Tap = _tap_class()
    results: list = []
    originals = []
    for i, layer in enumerate(model.layers):
        norm, down = layer.post_attention_layernorm, layer.mlp.down_proj
        originals.append((layer, norm, down))
        stash = {}

        def on_norm(x, _y, stash=stash):
            stash["mid"] = x

        def on_down(a, y, i=i, stash=stash):
            results.append(reduce(i, stash.pop("mid"), a, y))

        layer.post_attention_layernorm = Tap(norm, on_norm)
        layer.mlp.down_proj = Tap(down, on_down)
    try:
        yield results
    finally:
        for layer, norm, down in originals:
            layer.post_attention_layernorm = norm
            layer.mlp.down_proj = down


def prefill_cache(model, tokens: Sequence[int], chunk: int = 1024) -> list:
    """Read ``tokens`` into a fresh prompt cache, ``chunk`` tokens at a time, computing no logits.

    The answers being studied here run to 16k tokens.  One forward pass over all of
    them at once would hold attention and MLP intermediates for every position; read
    in chunks, the context costs only its KV cache, and whatever comes next is
    computed exactly as if the whole text had been read in one go.
    """
    mx, _ = _mlx()
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    tokens = list(tokens)
    for i in range(0, len(tokens), chunk):
        model(mx.array([tokens[i : i + chunk]]), cache=cache)  # logits stay lazy: never computed
        mx.eval([c.state for c in cache])
    mx.clear_cache()
    return cache


def mean_activations(
    model, sequences: Sequence[Sequence[int]], weights: Sequence[np.ndarray]
) -> dict[str, np.ndarray]:
    """Position-weighted means of every block's activations over a set of texts.

    ``weights[s]`` has one row per *group* and one column per token of
    ``sequences[s]`` — e.g. row 0 marks the positions where the model is looping
    and row 1 the positions where it is reasoning normally.  Returns, for
    G groups and L blocks,

        resid  [G, L, d_model]   mean residual stream after the block
        acts   [G, L, d_ff]      mean neuron activation
        write  [G, L, d_model]   mean MLP output
        count  [G]               total weight per group

    A mean is ``sum(weight * activation) / sum(weight)``: with weights of 1 it is a
    plain mean over the marked positions; with each text's weights summing to 1 it
    is the mean over texts of each text's own mean, so a long text counts once.
    """
    mx, _ = _mlx()
    n_groups = weights[0].shape[0]
    totals: dict[str, list] = {}
    count = np.zeros(n_groups)

    for tokens, w in zip(sequences, weights, strict=True):
        w = np.asarray(w, dtype=np.float32)  # [G, T]
        # Positions before the first weighted one are context only: read them into a
        # cache without recording, then record just the weighted stretch.
        used = np.flatnonzero(w.any(axis=0))
        lo = int(used[0]) if len(used) else len(tokens)
        cache = prefill_cache(model, tokens[:lo]) if lo > 0 else None
        tokens, w = list(tokens)[lo:], w[:, lo:]
        w_mx = mx.array(w)  # [G, T - lo]

        def reduce(i, mid, acts, write, w_mx=w_mx):
            # [G, T] @ [T, d] -> [G, d]: a weighted sum over positions, per group
            return {
                "resid": w_mx @ (mid[0] + write[0]).astype(mx.float32),
                "acts": w_mx @ acts[0].astype(mx.float32),
                "write": w_mx @ write[0].astype(mx.float32),
            }

        with recording(model, reduce) as per_layer:
            model(mx.array([tokens]), cache=cache)  # lazy: the unembedding is never computed
        stacked = {
            name: mx.stack([r[name] for r in per_layer], axis=1)  # [G, L, d]
            for name in ("resid", "acts", "write")
        }
        mx.eval(*stacked.values())  # one evaluation, so each block's activations are freed early
        for name, value in stacked.items():
            totals.setdefault(name, []).append(np.array(value, dtype=np.float64))
        count += np.asarray(w, dtype=np.float64).sum(axis=1)

    out = {name: sum(parts) / count[:, None, None] for name, parts in totals.items()}
    out["count"] = count
    return out


def next_token_logprobs(
    model, tokens: Sequence[int], watch: Sequence[int] = (), start: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Teacher-forced log-probabilities along a text, from position ``start`` on.

    Returns ``actual`` ``[T-1-start]`` — entry ``i`` is ``log p(tokens[t+1] | tokens[:t+1])``
    for ``t = start + i``, how much the model wanted the token that was in fact written
    next — and ``watched`` ``[T-1-start, len(watch)]``, the log-probability it gave some
    tokens of interest (e.g. ``</think>``) at the same positions.  Reading a text this
    way costs one forward pass and no sampling, which makes it the cheap way to ask
    "would this edit have changed the model's mind here?".  The text before ``start``
    is read into a cache in chunks (:func:`prefill_cache`) without logits, so scoring
    the last few hundred tokens of a 16k-token answer costs a few hundred rows of
    ``[vocab]`` logits, not sixteen thousand.
    """
    mx, _ = _mlx()
    tokens = list(tokens)
    cache = prefill_cache(model, tokens[:start]) if start > 0 else None
    logits = model(mx.array([tokens[start:]]), cache=cache)[0, :-1].astype(mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)  # [T-1-start, vocab]
    nxt = mx.array(tokens[start + 1 :])[:, None]
    actual = mx.take_along_axis(logprobs, nxt, axis=-1)[:, 0]
    watched = (
        logprobs[:, mx.array(list(watch))] if len(watch) else mx.zeros((len(tokens) - 1 - start, 0))
    )
    mx.eval(actual, watched)
    out = np.array(actual, dtype=np.float64), np.array(watched, dtype=np.float64)
    del logits, logprobs
    mx.clear_cache()  # a [T, vocab] float32 array is gigabytes; do not let it linger
    return out


# --------------------------------------------------------------------------------------
# Diagnosis: which direction means "I am repeating myself", and who writes it?
# --------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MatchedPair:
    """One line of reasoning told twice: once it moves on, once it says the same thing again.

    Both versions share the problem and the opening sentences, so whatever differs
    in the model's activations at the end is about *repeating*, not about the topic,
    the numbers or the position in the text.  This is how the paper's agent probed
    Qwen3.5-0.8B: four hand-written pairs, read once — no generated loops needed.
    """

    problem: str
    opening: str  # shared first sentences of the reasoning
    productive: str  # ... then a check that moves on
    doubt: str  # ... or one doubt, said `repeats` times
    repeats: int = 3

    def texts(self) -> tuple[str, str]:
        """``(productive, repetitive)`` as raw ChatML, with the reasoning left open."""
        head = (
            f"<|im_start|>user\n{self.problem}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n{self.opening} "
        )
        return head + self.productive, head + " ".join([self.doubt] * self.repeats)


# Written for this repo in the style of the paper's probe (its own four pairs are in
# the authors' logs, github.com/RikiyaT/lm_understand-logs, exp8_math/runs/p50c).
REPETITION_PAIRS: tuple[MatchedPair, ...] = (
    MatchedPair(
        "A bakery sells 14 loaves in the morning and 9 in the afternoon, at 3 dollars each. "
        "Find the day's revenue.",
        "Morning and afternoon together are 14+9=23 loaves. At 3 dollars each that is "
        "23*3=69 dollars.",
        "Checking: 42+27=69, which matches. The count and the price are in consistent units, "
        "so the revenue is 69 dollars.",
        "But is it 14 or 9 first? It does not matter, the sum is 23.",
        4,
    ),
    MatchedPair(
        "Solve 5x - 7 = 28 for x.",
        "Adding 7 to both sides gives 5x=35. Dividing by 5 gives x=7.",
        "Checking: 5*7-7=35-7=28, as required. So the equation holds for this value and "
        "there is exactly one solution.",
        "But the constant is -7. So 5x equals 35.",
        4,
    ),
    MatchedPair(
        "A train travels 180 km in 2 hours and then 150 km in 3 hours. Find its average "
        "speed for the whole trip.",
        "The total distance is 180+150=330 km. The total time is 2+3=5 hours.",
        "So the average speed is 330/5=66 km per hour. Checking: 66*5=330, which is the "
        "total distance. It lies between the two leg speeds of 90 and 50.",
        "Is 330 the distance? Yes, 180+150=330.",
        4,
    ),
    MatchedPair(
        "The sum of three consecutive integers is 72. Find the largest.",
        "Let the middle integer be n. Then (n-1)+n+(n+1)=3n=72, so n=24.",
        "The largest is n+1=25. Checking: 23+24+25=72, as required. The three numbers "
        "are consecutive, so the answer is 25.",
        "Now 72/3=24 and 24*3=72.",
        4,
    ),
    MatchedPair(
        "A rectangle has length 12 and width 5. Find the length of its diagonal.",
        "By the Pythagorean theorem the diagonal is sqrt(12^2+5^2). That is sqrt(144+25).",
        "So it is sqrt(169)=13. Checking: 13^2=169=144+25, so it is consistent. The "
        "diagonal is longer than either side, as it should be.",
        "Is 144+25 equal to 169? Yes, it is 169.",
        4,
    ),
    MatchedPair(
        "How many ways can 3 books be arranged on a shelf, chosen from 5 different books?",
        "The first position has 5 choices, the second 4 and the third 3. So there are "
        "5*4*3=60 arrangements.",
        "Checking: 5!/2!=120/2=60, the same number. Order matters on a shelf, so this "
        "counts permutations and the answer is 60.",
        "But does order matter? Yes, it is a shelf.",
        4,
    ),
)

# The second habit measured in this repo (docs/FINANCE_MODEL_SELECTION.md): not saying
# the same sentence again, but *re-opening a settled question* — "however, let me look
# at the wording again; or maybe it means ...; is there another interpretation?" — until
# the token cap arrives.  Qwen3.5-9B does this on a fifth of finance questions.  The pairs
# share a reasoning that has reached its answer; one continuation checks once and
# commits, the other keeps re-opening.  Each doubt is several different sentences,
# said once (``repeats=1``): said three times, the contrast would also be "repeating"
# — a first version did that, and its direction was half the loop direction.
DELIBERATION_PAIRS: tuple[MatchedPair, ...] = (
    MatchedPair(
        "A bond with a 6% annual coupon and a face value of 1000 is priced at 950. Which "
        "statement is correct? A. Its current yield is below 6%. B. Its current yield is "
        "above 6%. C. Its yield to maturity equals 6%.",
        "The current yield is the coupon over the price: 60/950 = 6.32%, which is above 6%. "
        "So B is correct.",
        "Checking A and C: A says below 6%, which contradicts 6.32%; C would need the price "
        "to equal par, and 950 is not par. So the answer is B.",
        "However, let me look at the wording of the question again. Or maybe the question "
        "means the yield to maturity rather than the current yield. Is there another "
        "interpretation I am missing? Let me reconsider the whole thing once more.",
        1,
    ),
    MatchedPair(
        "A company's revenue rose from 400 million to 460 million. What is the percentage change?",
        "The change is 460 - 400 = 60 million, and 60/400 = 0.15, so revenue rose by 15%.",
        "Checking: 400 * 1.15 = 460, which matches the table. The answer is 15%.",
        "But wait, maybe the question wants the change relative to 460 instead. Or perhaps "
        "the figures are in a different unit than I assumed. Let me re-read the question "
        "and reconsider whether 15% is really what is being asked.",
        1,
    ),
    MatchedPair(
        "Which of the following is a defining feature of a defined benefit pension plan? "
        "A. The employee bears the investment risk. B. The sponsor promises a specified "
        "retirement payment. C. Contributions are fixed and benefits vary.",
        "A defined benefit plan promises a specified payment at retirement, and the sponsor "
        "bears the investment risk. So B is correct.",
        "A and C describe a defined contribution plan, where contributions are fixed and the "
        "employee bears the risk. So the answer is B.",
        "Hmm, but let me double-check the exact definition again. Could A be what the "
        "question is getting at under some reading? Or is there a trick in how C is "
        "phrased? Let me reconsider each option one more time before deciding.",
        1,
    ),
    MatchedPair(
        "An investor buys a stock at 50 and sells it a year later at 56 after receiving a "
        "dividend of 2. What is the holding period return?",
        "The return is (56 - 50 + 2) / 50 = 8/50 = 0.16, so the holding period return is 16%.",
        "Checking: the price gain is 6 and the dividend 2, total 8 on a cost of 50, which is "
        "16%. The answer is 16%.",
        "Although, maybe the dividend should not be included, or should be compounded "
        "differently. Or perhaps the question expects the return without the dividend. Let "
        "me go back over the definition of holding period return and reconsider.",
        1,
    ),
    MatchedPair(
        "某公司流动资产为 600 万元，流动负债为 400 万元，其流动比率是多少？A. 0.67  B. 1.5  C. 2.0",
        "流动比率等于流动资产除以流动负债，600/400 = 1.5。所以选 B。",
        "核对一下：A 是 400/600，把分子分母弄反了；C 的 2.0 不对应任何一组数字。所以答案是 B。",
        "不过，让我再仔细看一下题目。或者题目问的其实是速动比率？会不会有别的理解方式？"
        "我还是重新把每个选项再考虑一遍，以免漏掉什么。",
        1,
    ),
    MatchedPair(
        "下列哪项属于货币政策工具？A. 调整税率  B. 调整法定存款准备金率  C. 增加政府支出",
        "调整税率和政府支出属于财政政策，法定存款准备金率是中央银行的货币政策工具。所以选 B。",
        "再核对一下：A 和 C 都由财政部门决定，只有 B 由中央银行决定。所以答案是 B。",
        "但是，等一下，让我再读一遍题目。会不会题目考察的是另一个角度？或者 A 在某种理解下也算？"
        "我再重新考虑一下所有选项。",
        1,
    ),
)

REPETITIVE, PRODUCTIVE = 0, 1
PAIR_SETS: dict[str, tuple[MatchedPair, ...]] = {
    "repetition": REPETITION_PAIRS,
    "deliberation": DELIBERATION_PAIRS,
}


@dataclasses.dataclass
class Diagnosis:
    """Mean activations at the end of repetitive vs productive reasoning, per block.

    Index 0 along the first axis is repetitive, 1 is productive.
    """

    resid: np.ndarray  # [2, L, d_model]  residual stream after each block
    write: np.ndarray  # [2, L, d_model]  what each block's MLP added
    acts: np.ndarray  # [2, L, d_ff]     MLP neuron activations

    def direction(self, layer: int) -> np.ndarray:
        """Unit vector, productive -> repetitive, in the residual stream after ``layer``."""
        return contrast_direction(self.resid[REPETITIVE, layer], self.resid[PRODUCTIVE, layer])

    def layer_table(self) -> list[dict]:
        """Where does the contrast come from?  One row per block.

        ``contrast`` is how far apart the two conditions are after the block.  Along
        that block's own direction ``d``, ``growth`` is how much the block widened
        the gap, and it splits exactly into what the MLP wrote (``mlp``) and the
        rest (``attention``).  The MLPs with the largest share are where an edit of
        MLP columns can do the most.
        """
        diff = self.resid[REPETITIVE] - self.resid[PRODUCTIVE]  # [L, d_model]
        dwrite = self.write[REPETITIVE] - self.write[PRODUCTIVE]  # [L, d_model]
        rows = []
        for layer in range(len(diff)):
            d = diff[layer] / np.linalg.norm(diff[layer])
            growth = float((diff[layer] - diff[layer - 1]) @ d) if layer else float("nan")
            mlp = float(dwrite[layer] @ d)
            rows.append(
                {
                    "layer": layer,
                    "contrast": float(np.linalg.norm(diff[layer])),
                    "growth": growth,
                    "mlp": mlp,
                    "attention": growth - mlp,
                }
            )
        return rows

    def neuron_scores(self, layer: int, W: np.ndarray) -> np.ndarray:
        """``[d_ff]`` — each neuron's push along the direction, repetitive minus productive."""
        return neuron_contributions(
            self.acts[REPETITIVE, layer], self.acts[PRODUCTIVE, layer], W, self.direction(layer)
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, resid=self.resid, write=self.write, acts=self.acts)
        return path

    @classmethod
    def load(cls, path: str | Path) -> Diagnosis:
        with np.load(path) as data:
            return cls(data["resid"], data["write"], data["acts"])


def diagnose(
    model,
    tokenizer,
    pairs: Sequence[tuple[str, str]],
    *,
    resid_tail: int = 36,
    act_tail: int = 24,
) -> Diagnosis:
    """Read ``(productive, repetitive)`` pairs — texts or token ids — and average the end of each.

    The windows are the paper's: the residual stream is averaged over each text's
    last 36 tokens, neuron activations over its last 24 — the stretch where one
    text is checking its work and the other is saying the same sentence again.
    """
    sequences, weights = [], []
    for productive, repetitive in pairs:
        for group, text in ((PRODUCTIVE, productive), (REPETITIVE, repetitive)):
            tokens = tokenizer.encode(text) if isinstance(text, str) else list(text)
            w = np.zeros((4, len(tokens)), dtype=np.float32)  # rows: 2 windows x 2 groups
            w[group, -resid_tail:] = 1.0 / resid_tail
            w[2 + group, -act_tail:] = 1.0 / act_tail
            sequences.append(tokens)
            weights.append(w)
    stats = mean_activations(model, sequences, weights)
    return Diagnosis(stats["resid"][:2], stats["write"][:2], stats["acts"][2:])


def loop_pair(
    prompt: Sequence[int], generated: Sequence[int], n: int = 16
) -> tuple[list[int], list[int]] | None:
    """A matched pair cut from one real answer that ended in a loop, or ``None``.

    ``loop_span`` finds the passage the loop keeps repeating and where it was first
    written.  The *productive* sequence ends where that passage was written the first
    time, without copying; the *repetitive* one ends one period later, where exactly
    the same tokens are being written again.  Same problem, same words, same position
    in the reasoning — what differs at the end is only that the second time is a
    repeat.  This is the hand-written pairs' contrast, taken from the model's own loops.
    """
    from learnai.mathbench import loop_span

    span = loop_span(generated, n)
    if span is None:
        return None
    first, start = span
    period = start - first
    if period <= 0 or len(generated) < start + period:
        return None
    head = list(prompt)
    return head + list(generated[:start]), head + list(generated[: start + period])


# How a reasoning model opens a line when it re-opens a question — read off the Qwen3.5
# answers that ran into the token cap ("If the…", "Let's check…", "Maybe the…", "Is there…").
REOPENERS = (
    "Wait",
    "However",
    "But wait",
    "Hmm",
    "Actually",
    "Alternatively",
    "Or maybe",
    "Maybe",
    "What if",
    "If the",
    "If I",
    "If we",
    "If it",
    "Is there",
    "Is it",
    "Could it",
    "Could the",
    "Why would",
    "Does the",
    "Let's check",
    "Let's look",
    "Let's assume",
    "Let's try",
    "Let's double",
    "Let's verify",
    "Let's re",
    "Let me check",
    "Let me look",
    "Let me re",
    "Let me double",
    "Re-read",
    "Double-check",
    "等等",
    "不过",
    "但是",
    "或者",
    "如果",
    "也许",
    "可能是",
    "让我再",
    "再检查",
    "再看",
)
# Phrases that state a conclusion: re-opening right after one is deliberating after the answer.
CONCLUDERS = (
    "answer is",
    "Answer:",
    "answer:",
    "So the",
    "Therefore",
    "Thus",
    "Hence",
    "is correct",
    "correct answer",
    "correct option",
    "所以",
    "因此",
    "答案",
    "应选",
    "正确",
)
_LINE_START = None


def decision_points(pieces: Sequence[str], *, lookback: int = 300, min_gap: int = 200) -> list[int]:
    """Token indices where a reasoning model, having just stated a conclusion, re-opens it.

    ``pieces`` are the generated tokens decoded one by one.  A decision point is the
    token where a line (after any indent, bullet or bold marker) begins with one of
    :data:`REOPENERS`, and the ``lookback`` characters before that line state a
    conclusion (:data:`CONCLUDERS`).  Points closer than ``min_gap`` tokens to the
    previous one are skipped, so one long transcript does not dominate.
    """
    import bisect
    import re

    global _LINE_START
    if _LINE_START is None:
        words = "|".join(re.escape(w) for w in sorted(REOPENERS, key=len, reverse=True))
        _LINE_START = re.compile(r"\n[ \t]*(?:[*\-•>][ \t]*)*(?:\*\*)?(" + words + ")")
    text = "".join(pieces)
    offsets = np.cumsum([0] + [len(p) for p in pieces])  # offsets[i] = where piece i starts
    points: list[int] = []
    for m in _LINE_START.finditer(text):
        before = text[max(0, m.start() - lookback) : m.start()]
        if not any(c in before for c in CONCLUDERS):
            continue
        i = bisect.bisect_right(offsets, m.start(1)) - 1  # the token holding the word's first char
        if not points or i - points[-1] >= min_gap:
            points.append(int(i))
    return points


def junction_pairs(
    answers: Sequence[tuple[Sequence[int], Sequence[int], bool]],
    decode: Callable[[int], str],
    marker: Sequence[int],
    *,
    per_answer: int = 4,
    limit: int = 48,
    seed: int = 0,
) -> list[tuple[list[int], list[int]]]:
    """``(closing, re-opening)`` sequences cut from a model's own answers.

    ``answers`` are ``(prompt, generated, ran_into_the_cap)``.  A *closing* sequence is
    a finished answer up to the moment it writes ``marker`` (``</think>``): the model
    has stated its answer and is about to stop thinking.  A *re-opening* sequence is an
    answer that ran into the cap, cut at one of its :func:`decision_points`: the model
    has stated an answer and is about to question it again.  Both end just after a
    conclusion; what differs is the decision taken next.  Unlike the hand-written pairs
    they are not the same problem — the problems that run long are the harder ones —
    so the contrast also carries "this is a hard question", which the probe has to
    tell apart from the habit.
    """
    rng = np.random.default_rng(seed)
    closing, reopening = [], []
    n = len(marker)
    for prompt, generated, capped in answers:
        generated = list(generated)
        if capped:
            pieces = [decode(t) for t in generated]
            points = decision_points(pieces)
            if len(points) > per_answer:  # spread over the answer, not its first stretch
                points = [
                    points[i] for i in np.linspace(0, len(points) - 1, per_answer).astype(int)
                ]
            reopening += [list(prompt) + generated[:p] for p in points]
        else:
            for i in range(len(generated) - n + 1):
                if generated[i : i + n] == list(marker):
                    if i >= 64:
                        closing.append(list(prompt) + generated[:i])
                    break
    rng.shuffle(closing)
    rng.shuffle(reopening)
    k = min(len(closing), len(reopening), limit)
    return list(zip(closing[:k], reopening[:k], strict=True))


def make_edit(
    diagnosis: Diagnosis, layer: int, W: np.ndarray, *, columns: int = 64, strength: float = 6.0
) -> LayerEdit:
    """The paper's edit for one block: its ``columns`` strongest writers, reversed.

    The neurons with the largest positive score are the ones that push the residual
    stream along the repetition direction *when the model repeats itself*.  Their
    output columns lose ``strength`` times their component along it.
    """
    d = diagnosis.direction(layer)
    scores = diagnosis.neuron_scores(layer, W)  # [d_ff]
    cols = np.sort(np.argsort(-scores)[:columns])
    coef, _ = subtract_direction(W, cols, d, strength)
    return LayerEdit(layer, cols, coef, d)


def count_parameters(model) -> int:
    """Number of weights in the model, counting a packed 4-bit matrix at its real shape."""
    _, nn = _mlx()
    from mlx.utils import tree_flatten

    total, packed = 0, set()
    for name, module in model.named_modules():
        if isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            total += module.weight.size * 32 // module.bits  # a uint32 holds 32/bits weights
            packed.update({f"{name}.weight", f"{name}.scales", f"{name}.biases"})
    for name, value in tree_flatten(model.parameters()):
        if name not in packed:
            total += value.size
    return total


# --------------------------------------------------------------------------------------
# Sampling many answers at once
# --------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Sampling:
    """Decoding settings.  The defaults are the paper's evaluation settings for Qwen."""

    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    presence_penalty: float = 1.5
    max_tokens: int = 4096
    seed: int = 0


@dataclasses.dataclass
class Generation:
    tokens: list[int]  # generated token ids, without the end-of-turn token
    hit_limit: bool  # stopped by `max_tokens` rather than by the model


def presence_penalty(penalty: float) -> Callable:
    """Logits processor: subtract ``penalty`` from every token generated so far.

    This is the OpenAI / vLLM meaning of ``presence_penalty``: it counts the whole
    output and only the output.  (``mlx_lm``'s built-in version looks at the last 20
    tokens of prompt *and* output, which would both forget a loop longer than 20
    tokens and penalise copying a number out of the question.)
    """
    mx, _ = _mlx()
    seen = None  # [vocab]  1.0 where a token has been generated

    def processor(context, logits):
        nonlocal seen
        if seen is None:
            seen = mx.zeros((logits.shape[-1],), dtype=mx.float32)
        seen[context[-1]] = 1.0  # the token that was just fed back in
        return logits - (penalty * seen).astype(logits.dtype)

    return processor


SETTLE_EVERY = 256  # decoding steps between two housekeeping passes in `sample_batch`


def settle_cache(generator) -> int:
    """Evaluate the bookkeeping arrays of a running batch; returns how many caches it found.

    MLX is lazy: ``x -= 1`` only *records* a subtraction, and the record is kept until
    somebody reads ``x``.  ``mlx_lm`` 0.32 has such an ``x`` that nobody reads.  Once a
    finished sequence has been replaced in a running batch, every Gated DeltaNet layer
    does ``left_padding -= 1`` per decoding step, the result is never used, and the
    chain of pending subtractions grows by one node per layer per step.  Each pending
    node holds one of the Metal allocator's 499,000 resource slots, so a 32-block model
    (24 such layers) dies after about 20,000 steps with ``[metal::malloc] Resource
    limit exceeded`` — with gigabytes of memory free, and twenty minutes into a run.

    Evaluating the cache state collapses every chain to one array.  This reaches into
    a private attribute of ``BatchGenerator``; if a later ``mlx_lm`` renames it, this
    returns 0 and ``tests/test_weightedit.py`` says so.
    """
    mx, _ = _mlx()
    batch = getattr(generator, "_generation_batch", None)
    caches = list(getattr(batch, "prompt_cache", None) or [])
    if caches:
        mx.eval([c.state for c in caches])
    return len(caches)


def sample_batch(
    model,
    tokenizer,
    prompts: Sequence[Sequence[int]],
    sampling: Sampling,
    *,
    batch_size: int = 32,
    prefill_batch_size: int = 8,
    prefill_step_size: int = 2048,
    on_done: Callable[[int, Generation], None] | None = None,
    on_settle: Callable[[int], None] | None = None,
) -> list[Generation]:
    """Sample one continuation per prompt with continuous batching.

    Decoding is memory-bound — every step reads all the weights once — so reading
    them once for 32 sequences instead of one is several times cheaper per token.
    ``on_done(index, generation)`` fires as each sequence finishes, which lets a
    long run save its progress.

    Prompts are read ``prefill_batch_size`` at a time in chunks of
    ``prefill_step_size`` tokens.  The activations of one chunk are the run's
    memory *peak* for long prompts (a few GiB for 8 x 2,048 tokens on a 4B model),
    so a run near its memory budget lowers these two rather than ``batch_size``.
    ``on_settle(step)`` fires every ``SETTLE_EVERY`` decoding steps, after the
    housekeeping below — the place to log memory as a run goes.
    """
    mx, _ = _mlx()
    from mlx_lm.generate import BatchGenerator, GenerationBatch
    from mlx_lm.sample_utils import make_sampler

    mx.random.seed(sampling.seed)
    # When a sequence finishes, mlx-lm hands back a copy of its own cache — sliced out
    # of the whole batch's arrays and materialised: 0.5 GiB per sequence at a 16k cap on
    # the 4B, and every sequence of a wave that runs to the cap finishes on the same
    # step (12 of them: +6.5 GiB, measured).  Nothing here continues a conversation, so
    # the copies are never wanted; skip making them.  `tests/test_weightedit.py` checks
    # that the method this replaces still exists.
    extract_cache = GenerationBatch.extract_cache
    GenerationBatch.extract_cache = lambda self, idx: None
    generator = BatchGenerator(
        model,
        stop_tokens=[[t] for t in tokenizer.eos_token_ids],
        sampler=make_sampler(temp=sampling.temperature, top_p=sampling.top_p, top_k=sampling.top_k),
        completion_batch_size=batch_size,
        prefill_batch_size=prefill_batch_size,
        prefill_step_size=prefill_step_size,
    )
    processors = [
        [presence_penalty(sampling.presence_penalty)] if sampling.presence_penalty else []
        for _ in prompts
    ]
    uids = generator.insert(
        [list(p) for p in prompts],
        [sampling.max_tokens] * len(prompts),
        logits_processors=processors,
    )
    index = {uid: i for i, uid in enumerate(uids)}
    out = [Generation([], False) for _ in prompts]
    steps = 0
    try:
        while responses := generator.next_generated():
            steps += 1
            if steps % SETTLE_EVERY == 0:
                settle_cache(generator)
                # Every step also grows the key/value caches by one token, leaving a trail
                # of outgrown buffers in MLX's allocator cache; hand them back now and then.
                mx.clear_cache()
                if on_settle is not None:
                    on_settle(steps)
            for r in responses:
                generation = out[index[r.uid]]
                if r.finish_reason != "stop":  # the end-of-turn token is not content
                    generation.tokens.append(r.token)
                if r.finish_reason is not None:
                    generation.hit_limit = r.finish_reason == "length"
                    if on_done is not None:
                        on_done(index[r.uid], generation)
            del responses, r  # nothing from this step may outlive it into the next one
    finally:
        GenerationBatch.extract_cache = extract_cache
        generator.close()
    return out

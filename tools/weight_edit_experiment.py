"""Hand-edit an open-weight model's weights and measure what changed.

A scale-down of the Qwen experiment in "Sparks of Weight-Space Programming"
(Takehi, Teoh, Zhang & Khattab, MIT CSAIL, Nov 2026): find the direction a
reasoning model writes when it is stuck in a thinking loop, rewrite the MLP
columns responsible for it, and compare the edited model with the original on
math problems under identical sampling settings.  No gradients anywhere.

    diagnose   read matched repetitive/productive texts; rank blocks by what their MLP writes
    edit       write a patch from a diagnosis (which blocks, how many columns, how strongly)
    probe      teacher-forced check of a patch: does it only make repetition less likely?
    generate   sample answers to a problem set (optionally with a patch applied)
    summary    accuracy / hit-token-limit / loop rate of one or more runs
    compare    paired comparison of two runs on the same problems
    regrade    re-score stored runs after a change to the grader

Everything is written under ``runs/weightprog/`` (git-ignored, regenerable).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np

from learnai.mathbench import SOURCES, Problem, grade, load_problems, loop_span, repetition

LOOP_THRESHOLD = 0.5  # a capped answer whose tail is >= 50% repeated 16-grams is a loop


def read_run(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def is_loop(row: dict) -> bool:
    return row["hit_limit"] and row["repetition"] >= LOOP_THRESHOLD


def outcome(row: dict) -> str:
    """Where one answer ended up — exactly one of ``right | wrong | cut | abandoned``.

    ``right`` and ``wrong`` closed their reasoning and stopped by themselves.  ``cut``
    was still writing when the token limit arrived, whether in its reasoning or in
    the answer after it.  ``abandoned`` stopped by itself without ever closing its
    reasoning — the edited 0.8B does this after a long run of whitespace.

    This is not the same split as ``row["correct"]``: the paper's protocol also
    credits a cut-off answer that boxed the right result on the way.
    """
    if row["hit_limit"]:
        return "cut"
    if not row["finished"]:
        return "abandoned"
    return "right" if row["correct"] else "wrong"


def chat_tokens(tokenizer, text: str) -> list[int]:
    """One user turn, rendered with the model's chat template, thinking switched on."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=True,
    )


def problem_for(uid: str) -> Problem:
    """Look a stored row's problem up again from its uid, e.g. ``math-train/algebra/9``."""
    name, index = uid.split("/")[0], int(uid.rsplit("/", 1)[1])
    enough = (index + 3) * len(SOURCES[name][1])
    return {p.uid: p for p in load_problems(name, enough)}[uid]


# --------------------------------------------------------------------------------------
# diagnose / edit
# --------------------------------------------------------------------------------------


def editable_layers(diagnosis, how_many: int = 2) -> list[int]:
    """The paper's rule: the blocks whose MLP writes most of the direction, last block excluded.

    The last block's MLP always scores high — it writes the next-token prediction
    itself, and a repeated sentence is nothing if not predictable — but editing it
    edits the answer, not the habit.  The paper's agent left it out as well.

    On Qwen3.5-0.8B this picks blocks 15 and 19, the ones the paper edited.  On the
    4B it picks 27 and 29, where an edit turns out to do little (the contrast simply
    keeps growing towards the output), so `edit` does not use this rule by default:
    `probe --each-block` measures where an edit works.
    """
    rows = diagnosis.layer_table()[:-1]
    return sorted(r["layer"] for r in sorted(rows, key=lambda r: -r["mlp"])[:how_many])


def transcripts(tokenizer, args: argparse.Namespace):
    """``(row, prompt_tokens)`` for every stored answer named by ``--runs`` / ``--finance-runs``.

    Math runs (``generate`` below) look their problem up in mathbench and were asked
    with thinking on; finance runs (``tools/finance_model_select.py``) look it up in
    finbench and record the chat-template arguments they used in their ``.meta.json``,
    so the prompt is rebuilt token for token as the model saw it.
    """
    for path in getattr(args, "runs", None) or []:
        for row in read_run(path):
            yield row, chat_tokens(tokenizer, problem_for(row["uid"]).prompt)
    for path in getattr(args, "finance_runs", None) or []:
        from learnai.finbench import problem_for as finance_problem

        meta_path = Path(str(path).removesuffix(".jsonl") + ".meta.json")
        template = (
            json.loads(meta_path.read_text()).get("template", {}) if meta_path.exists() else {}
        )
        for row in read_run(path):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": finance_problem(row["uid"]).prompt}],
                add_generation_prompt=True,
                tokenize=True,
                **template,
            )
            yield row, list(prompt)


def cmd_diagnose(args: argparse.Namespace) -> int:
    from learnai.openweights import load
    from learnai.weightedit import PAIR_SETS, diagnose, loop_pair

    model, tokenizer = load(args.model)
    if args.pairs == "real-loops":
        # The model's own loops: each one cut into "the passage written once" and
        # "the same passage written again" (see learnai.weightedit.loop_pair).
        pairs = []
        for row, prompt in transcripts(tokenizer, args):
            if is_loop(row) and (pair := loop_pair(prompt, row["tokens"])):
                pairs.append(pair)
        pairs = pairs[: args.limit] if args.limit else pairs
        if not pairs:
            print("no looping answers in the runs given")
            return 1
    elif args.pairs == "real-junctions":
        # The model's own decisions after stating an answer: close the reasoning (finished
        # answers) or question it again (answers that ran into the cap).
        from learnai.weightedit import junction_pairs

        answers = [
            (prompt, row["tokens"], row["hit_limit"])
            for row, prompt in transcripts(tokenizer, args)
            if not is_loop(row)
        ]
        marker = tokenizer.encode("</think>", add_special_tokens=False)
        pairs = junction_pairs(
            answers, lambda t: tokenizer.decode([t]), marker, limit=args.limit or 48
        )
        if not pairs:
            print("no closing/re-opening points found in the runs given")
            return 1
    else:
        pairs = [p.texts() for p in PAIR_SETS[args.pairs]]
    diagnosis = diagnose(model, tokenizer, pairs)
    print(f"saved   {diagnosis.save(args.out)}  ({len(pairs)} {args.pairs} pairs)\n")
    chosen = editable_layers(diagnosis)
    print("block  attention    contrast   growth =    MLP + attention")
    for row, layer in zip(diagnosis.layer_table(), model.layers, strict=True):
        kind = "delta-net" if getattr(layer, "is_linear", False) else "full"
        mark = "  <- largest MLP shares (the paper's rule)" if row["layer"] in chosen else ""
        print(
            f"{row['layer']:5d}  {kind:<9}  {row['contrast']:9.3f}  {row['growth']:7.3f}   "
            f"{row['mlp']:6.3f}   {row['attention']:7.3f}{mark}"
        )
    return 0


def cmd_edit(args: argparse.Namespace) -> int:
    from learnai.openweights import load
    from learnai.weightedit import (
        Diagnosis,
        LayerEdit,
        count_parameters,
        dense_weight,
        make_edit,
        save_patch,
    )

    model, _ = load(args.model)
    diagnosis = Diagnosis.load(args.diagnosis)
    layers = [int(x) for x in args.layers.split(",")]
    rng = np.random.default_rng(args.control_seed)
    edits = []
    for layer in layers:
        W = dense_weight(model.layers[layer].mlp.down_proj)  # [d_model, d_ff]
        edit = make_edit(diagnosis, layer, W, columns=args.columns, strength=args.strength)
        scores = diagnosis.neuron_scores(layer, W)
        share = scores[edit.cols].sum() / scores[scores > 0].sum()
        if args.control_seed is not None:
            # Control: the same neurons, an edit of the same size, but along a random
            # direction orthogonal to the diagnosed one.  If this helps as much, the
            # direction was not the point.
            r = rng.normal(size=edit.direction.shape)
            r -= (r @ edit.direction) * edit.direction
            r /= np.linalg.norm(r)
            coef = args.strength * (r @ W[:, edit.cols])
            coef *= np.linalg.norm(edit.coef) / np.linalg.norm(coef)
            edit = LayerEdit(layer, edit.cols, coef, r)
        relative = np.linalg.norm(edit.coef) / np.linalg.norm(W)  # ||dW||_F / ||W||_F
        print(
            f"block {layer}: {len(edit.cols)} of {W.shape[1]} columns "
            f"({100 * share:.1f}% of the positive score), matrix changes by {100 * relative:.1f}%"
        )
        edits.append(edit)
    changed, total = sum(e.n_weights for e in edits), count_parameters(model)
    path = save_patch(
        args.out,
        edits,
        model=args.model,
        strength=args.strength,
        columns=args.columns,
        control_seed=args.control_seed,
    )
    print(f"{changed:,} of {total:,} weights change ({100 * changed / total:.4f}%)  ->  {path}")
    return 0


# --------------------------------------------------------------------------------------
# probe
# --------------------------------------------------------------------------------------


def cmd_probe(args: argparse.Namespace) -> int:
    """Read texts with and without an edit; no sampling involved.

    A good edit makes the *repeated* sentences less likely and leaves everything
    else alone.  Both are log-probabilities of text that already exists, so one
    candidate costs a few forward passes — the cheap filter before any generation
    run.  ``--each-block`` tries every block on its own, which is how to find out
    where an edit works instead of assuming it.
    """
    from learnai.openweights import load
    from learnai.weightedit import (
        PAIR_SETS,
        Diagnosis,
        apply_patch,
        dense_weight,
        load_patch,
        make_edit,
        next_token_logprobs,
        remove_patch,
    )

    model, tokenizer = load(args.model)
    marker = tokenizer.encode(
        "</think>", add_special_tokens=False
    )  # 1 token for Qwen3.5, 3 for DianJin
    end_think = marker[0]

    # (kind, tokens, first scored position): what the model is asked to re-read.
    texts: list[tuple[str, list[int], int]] = []
    ending, doubt = (
        ("committing", "re-opening") if args.pairs == "deliberation" else ("productive", "repeated")
    )
    if args.pairs in PAIR_SETS:
        for pair in PAIR_SETS[args.pairs]:
            productive, repetitive = (tokenizer.encode(t) for t in pair.texts())
            texts.append((ending, productive, len(productive) - 24))
            texts.append((doubt, repetitive, len(repetitive) - 24))
    counts: dict[str, int] = {}

    def take(kind: str, limit: int) -> bool:
        counts[kind] = counts.get(kind, 0) + 1
        return counts[kind] <= limit

    for row, prompt in transcripts(tokenizer, args):
        tokens = prompt + row["tokens"]
        if not row["hit_limit"]:
            if take("real answers", args.transcripts):
                # the first 1024 tokens of real reasoning: does the edit disturb ordinary text?
                texts.append(("real answers", prompt + row["tokens"][:1024], len(prompt)))
            at = _find(row["tokens"], marker)
            if at is not None and at > 256 and take("real endings", args.transcripts):
                # the 256 tokens leading up to the model closing its reasoning, the close included
                stop = len(prompt) + at + 1
                texts.append(("real endings", tokens[:stop], stop - 256))
        elif is_loop(row):
            if take("real loops", args.transcripts):
                start = len(prompt) + loop_span(row["tokens"])[1]
                texts.append(("real loops", tokens[: start + 512], start))
        elif take("real cut-offs", args.cutoffs):
            # still deliberating when the cap arrived: the last 512 tokens, with all the context
            texts.append(("real cut-offs", tokens, len(tokens) - 512))
    kinds = list(dict.fromkeys(kind for kind, _, _ in texts))
    print(
        "texts read per candidate: "
        + ", ".join(f"{k} {sum(t[0] == k for t in texts)}" for k in kinds)
    )

    def read_all() -> dict[str, np.ndarray]:
        """Per kind: mean log p(next token) and mean log p(</think>) over the scored span."""
        per_kind: dict[str, list] = {kind: [] for kind in kinds}
        for kind, tokens, start in texts:
            # entry i is log p(tokens[t+1]) at t = start - 1 + i: the scored tokens themselves
            actual, watched = next_token_logprobs(model, tokens, watch=[end_think], start=start - 1)
            per_kind[kind].append((actual.mean(), watched[:, 0].mean()))
        return {kind: np.mean(values, axis=0) for kind, values in per_kind.items()}

    candidates: list[tuple[str, list]] = []
    for path in args.patch or []:
        candidates.append((Path(path).stem, load_patch(path)[0]))
    if args.each_block:
        diagnosis = Diagnosis.load(args.diagnosis)
        for layer in range(len(model.layers) - 1):
            W = dense_weight(model.layers[layer].mlp.down_proj)
            edit = make_edit(diagnosis, layer, W, columns=args.columns, strength=args.strength)
            candidates.append((f"block {layer} alone", [edit]))

    base = read_all()
    width = 22
    print(
        "\nper kind: change in mean log p(the text's own next token) / in mean log p(</think>), nats"
    )
    print(f"{'':<24}" + "".join(f"{kind:>{width}}" for kind in kinds))
    print(
        f"{'unedited (log p)':<24}"
        + "".join(f"{base[k][0]:>11.3f}{base[k][1]:>11.2f}" for k in kinds)
    )
    for name, edits in candidates:
        apply_patch(model, edits)
        edited = read_all()
        remove_patch(model)
        print(
            f"{name:<24}"
            + "".join(
                f"{edited[k][0] - base[k][0]:>+11.3f}{edited[k][1] - base[k][1]:>+11.2f}"
                for k in kinds
            ),
            flush=True,
        )
    return 0


def _find(tokens: list[int], marker: list[int]) -> int | None:
    """Index of the last token of the first occurrence of ``marker`` in ``tokens``."""
    n = len(marker)
    for i in range(len(tokens) - n + 1):
        if tokens[i : i + n] == marker:
            return i + n - 1
    return None


# --------------------------------------------------------------------------------------
# generate
# --------------------------------------------------------------------------------------


def cmd_generate(args: argparse.Namespace) -> int:
    from learnai.openweights import load
    from learnai.weightedit import Sampling, apply_patch, load_patch, sample_batch

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {row["uid"] for row in read_run(out)} if out.exists() else set()
    problems = [p for p in load_problems(args.set, args.n) if p.uid not in done]
    if not problems:
        print(f"{out}: already complete")
        return 0

    model, tokenizer = load(args.model)
    if args.patch:
        edits, meta = load_patch(args.patch)
        apply_patch(model, edits)
        print(f"patch   {args.patch}: layers {meta['layers']}")
    prompts = [chat_tokens(tokenizer, p.prompt) for p in problems]
    sampling = Sampling(max_tokens=args.max_tokens, seed=args.seed)
    print(f"{args.model}  {args.set}  {len(problems)} problems  {sampling}", flush=True)

    started, finished = time.perf_counter(), 0
    with out.open("a") as handle:

        def on_done(i, generation):
            nonlocal finished
            text = tokenizer.decode(generation.tokens)
            row = {
                "uid": problems[i].uid,
                "gold": problems[i].answer,
                **grade(problems[i], text),
                "hit_limit": generation.hit_limit,
                "n_tokens": len(generation.tokens),
                "repetition": round(repetition(generation.tokens), 3),
                "tokens": generation.tokens,
            }
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            finished += 1
            if finished % 8 == 0 or finished == len(problems):
                elapsed = time.perf_counter() - started
                print(f"  {finished}/{len(problems)} done after {elapsed:.0f}s", flush=True)

        sample_batch(model, tokenizer, prompts, sampling, batch_size=args.batch, on_done=on_done)
    print(summarize(read_run(out), str(out)))
    return 0


# --------------------------------------------------------------------------------------
# summary / compare
# --------------------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion — honest for small n and for 0% / 100%."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(rows: list[dict], label: str) -> str:
    """One run in three lines: the score, then where every answer ended up.

    The score follows the paper's protocol, so it can credit an answer that was cut
    off after boxing the right result.  The second line is the honest accounting:
    each answer in exactly one state (see :func:`outcome`), with the cut-off ones
    split by where the limit caught them and by whether they were repeating
    themselves verbatim (a *loop*).
    """
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    lo, hi = wilson(correct, n)
    states = Counter(outcome(r) for r in rows)
    cut = [r for r in rows if r["hit_limit"]]
    credited = sum(r["correct"] for r in cut)
    in_answer = sum(r["finished"] for r in cut)
    loops = sum(is_loop(r) for r in cut)
    tokens = [r["n_tokens"] for r in rows]
    return (
        f"{label}\n"
        f"  n={n}  correct {correct} ({100 * correct / n:.1f}%, 95% CI {100 * lo:.0f}-{100 * hi:.0f})"
        f"{f', {credited} of them cut off after boxing the answer' if credited else ''}\n"
        f"  finished right {states['right']}  finished wrong {states['wrong']}"
        f"  cut off {states['cut']} ({100 * states['cut'] / n:.1f}%: {len(cut) - in_answer} in the"
        f" reasoning, {in_answer} in the answer; verbatim loops {loops})"
        f"  abandoned {states['abandoned']}\n"
        f"  tokens mean {statistics.mean(tokens):.0f} median {statistics.median(tokens):.0f}"
    )


def cmd_summary(args: argparse.Namespace) -> int:
    for path in args.runs:
        print(summarize(read_run(path), path))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Paired view: the same problems under two models.  Only disagreements carry signal."""
    a = {r["uid"]: r for r in read_run(args.base)}
    b = {r["uid"]: r for r in read_run(args.edited)}
    shared = sorted(set(a) & set(b))
    print(summarize([a[u] for u in shared], f"base    {args.base}"))
    print(summarize([b[u] for u in shared], f"edited  {args.edited}"))

    fixed = sum(not a[u]["correct"] and b[u]["correct"] for u in shared)
    broken = sum(a[u]["correct"] and not b[u]["correct"] for u in shared)
    # Exact two-sided sign test on the discordant pairs (McNemar's test, exact form).
    m = fixed + broken
    tail = sum(math.comb(m, k) for k in range(0, min(fixed, broken) + 1)) / 2**m if m else 1.0
    print(
        f"  paired on {len(shared)} problems: {fixed} fixed, {broken} broken by the edit"
        f"  (exact sign test p = {min(1.0, 2 * tail):.3f})"
    )

    # What became of the answers the base model never finished?
    stuck = [u for u in shared if a[u]["hit_limit"]]
    fate = Counter(outcome(b[u]) for u in stuck)
    print(
        f"  of {len(stuck)} answers the base model did not finish: {fate['right']} now finish right, "
        f"{fate['wrong']} finish wrong, {fate['cut']} are still cut off"
        f"{', ' + str(fate['abandoned']) + ' are abandoned' if fate['abandoned'] else ''}"
    )
    spent_a = sum(a[u]["n_tokens"] for u in shared)
    spent_b = sum(b[u]["n_tokens"] for u in shared)
    print(f"  tokens generated: {spent_a:,} -> {spent_b:,} ({100 * (spent_b / spent_a - 1):+.0f}%)")
    return 0


def cmd_regrade(args: argparse.Namespace) -> int:
    """Re-score stored token ids with the current grader (the model is not run)."""
    from mlx_lm.tokenizer_utils import load as load_tokenizer

    from learnai.openweights import local_path

    tokenizer = load_tokenizer(local_path(args.model))
    for path in args.runs:
        rows = read_run(path)
        for row in rows:
            row.update(grade(problem_for(row["uid"]), tokenizer.decode(row["tokens"])))
        Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows))
        print(summarize(rows, path))
    return 0


# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("diagnose", help="read the matched pairs, rank blocks")
    d.add_argument("--model", default="qwen3.5-4b")
    d.add_argument(
        "--pairs",
        default="repetition",
        choices=["repetition", "deliberation", "real-loops", "real-junctions"],
    )
    d.add_argument("--runs", nargs="*", help="math runs (for --pairs real-loops)")
    d.add_argument("--finance-runs", nargs="*", help="finance runs (for --pairs real-loops)")
    d.add_argument("--limit", type=int, help="use at most this many real loops / junction pairs")
    d.add_argument("--out", required=True)
    d.set_defaults(handler=cmd_diagnose)

    e = sub.add_parser("edit", help="write a patch from a diagnosis")
    e.add_argument("--model", default="qwen3.5-4b")
    e.add_argument("--diagnosis", required=True)
    e.add_argument(
        "--layers",
        required=True,
        help="comma-separated blocks to edit; choose them with `probe --each-block`",
    )
    e.add_argument("--columns", type=int, default=64)
    e.add_argument("--strength", type=float, default=6.0)
    e.add_argument("--control-seed", type=int, help="write a random-direction control instead")
    e.add_argument("--out", required=True)
    e.set_defaults(handler=cmd_edit)

    p = sub.add_parser("probe", help="teacher-forced check of one or more patches")
    p.add_argument("--model", default="qwen3.5-4b")
    p.add_argument("--patch", nargs="*", help="patches written by `edit`")
    p.add_argument("--each-block", action="store_true", help="also try every block alone")
    p.add_argument("--diagnosis", help="needed for --each-block")
    p.add_argument("--columns", type=int, default=64)
    p.add_argument("--strength", type=float, default=6.0)
    p.add_argument("--pairs", default="repetition", choices=["repetition", "deliberation", "none"])
    p.add_argument(
        "--cutoffs", type=int, default=0, help="cut-off answers to read (slow: 16k tokens each)"
    )
    p.add_argument("--runs", nargs="*", help="base math runs to take real transcripts from")
    p.add_argument(
        "--finance-runs", nargs="*", help="base finance runs (tools/finance_model_select.py)"
    )
    p.add_argument("--transcripts", type=int, default=8)
    p.set_defaults(handler=cmd_probe)

    g = sub.add_parser("generate", help="sample answers to a problem set")
    g.add_argument("--model", default="qwen3.5-4b")
    g.add_argument("--set", required=True, help="gsm8k-train | gsm8k-test | math-train | math500")
    g.add_argument("--n", type=int, default=64)
    g.add_argument("--max-tokens", type=int, default=4096)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--batch", type=int, default=32)
    g.add_argument("--patch", help="a .npz written by `edit`")
    g.add_argument("--out", required=True)
    g.set_defaults(handler=cmd_generate)

    s = sub.add_parser("summary", help="summarise runs")
    s.add_argument("runs", nargs="+")
    s.set_defaults(handler=cmd_summary)

    c = sub.add_parser("compare", help="paired comparison of two runs")
    c.add_argument("base")
    c.add_argument("edited")
    c.set_defaults(handler=cmd_compare)

    r = sub.add_parser("regrade", help="re-score stored runs with the current grader")
    r.add_argument("--model", default="qwen3.5-4b")
    r.add_argument("runs", nargs="+")
    r.set_defaults(handler=cmd_regrade)

    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

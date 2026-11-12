"""Choose an open-weight model for finance questions under a memory budget — then edit it.

Every candidate answers the same finance questions (``learnai.finbench``, or with
``--bench macro`` the global-macro futures set ``learnai.macrobench``) with its
reasoning switched on and the same token cap, inside the same memory budget; the
token count and time of every answer are kept, so the question "which model is
best when the user waits N seconds?" can be answered afterwards from the same runs.

    generate   answer a slice of the finance set with one model (optionally with a weight patch)
    summary    accuracy per source, cut-off / loop / format-failure rates, token use
    curve      accuracy the same runs would have had under smaller token caps
    force      ...and with the reasoning closed at the cap and an answer asked for there
    compare    paired comparison of two runs on the same problems (stock vs edited)
    failures   list the wrong answers of a run with a first-pass classification

Runs are written under ``runs/finbench/<model>/`` (git-ignored, regenerable).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from learnai import finbench, macrobench
from learnai.env import RUNS_DIR
from learnai.finbench import Problem, load_problems, split_thinking
from learnai.mathbench import repetition
from learnai.weightedit import Sampling

GiB = 2**30
SOURCES = (*finbench.SOURCES, *macrobench.SOURCES)  # every source a run can hold


def bench_of(source: str):
    """The module that owns a source: its problems and its grader."""
    return macrobench if source in macrobench.SOURCES else finbench


def problems_of(bench: str, n: int, offset: int, seed: int) -> list[Problem]:
    if bench == "macro":
        return macrobench.load_problems(n, offset, seed)
    return load_problems(n, offset, seed)


def problem_for(uid: str) -> Problem:
    return bench_of(uid.split("/")[0]).problem_for(uid)


def grade(problem: Problem, response: str, thinking: bool = True) -> dict:
    return bench_of(problem.source).grade(problem, response, thinking)


def equivalent(boxed: str | None, problem: Problem) -> bool:
    return bench_of(problem.source).equivalent(boxed, problem)


LOOP_THRESHOLD = 0.5  # a capped answer whose tail is >= 50% repeated 16-grams is a loop
CAP = 16384  # generated tokens per answer; the memory budget below is sized for it
PROCESS_LIMIT_GIB = 12  # the MLX process stays under this (the user's choice, 2026-11-08)
# Measured on the 4B (telemetry every 256 steps): the peak sits 0.4 GiB above the held
# memory while the caches are small and about 30% of the caches above it once they are
# large — the copies made while a finished sequence is swapped for a new one, plus
# reading prompts 2 x 512 tokens at a time.
TRANSIENT_FIXED_GIB = 0.5
TRANSIENT_PER_CACHE = 0.3


@dataclasses.dataclass(frozen=True)
class Family:
    """How one model is asked and sampled.

    ``sampling`` follows each model's own recommended settings (the paper's Qwen
    settings for Qwen; Gemma's and gpt-oss's own cards for theirs), because a model
    judged under another family's sampling would be judged unfairly.  ``batch`` is
    the number of sequences decoded at once that keeps weights + caches at the cap,
    plus the prefill headroom, under ``PROCESS_LIMIT_GIB`` (the arithmetic is in
    ``docs/FINANCE_MODEL_SELECTION.md`` and checked by ``tests/test_finance_model_select.py``).
    ``resident_gib``, ``kib_per_token`` and ``fixed_kib`` are that arithmetic's inputs:
    the loaded weights, the KV cache one token adds, and the per-sequence state that
    does not grow with length (DeltaNet states, sliding windows).
    """

    template: dict  # extra arguments for the chat template (thinking on, reasoning effort)
    sampling: Sampling
    batch: int
    resident_gib: float
    kib_per_token: float
    fixed_kib: float = 0.0
    thinking: bool = True  # the model reasons before answering (affects grading)

    def seq_gib(self, max_tokens: int = CAP) -> float:
        """One sequence's caches at the cap: ``max_tokens`` generated plus the longest prompt."""
        return ((max_tokens + PROMPT_MAX) * self.kib_per_token + self.fixed_kib) / 2**20

    def peak_gib(self, batch: int | None = None, max_tokens: int = CAP) -> float:
        """Worst case of a run: every slot holding a sequence at the cap, plus the transients."""
        caches = (batch or self.batch) * self.seq_gib(max_tokens)
        return self.resident_gib + caches * (1 + TRANSIENT_PER_CACHE) + TRANSIENT_FIXED_GIB


QWEN = Sampling(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=CAP)
PROMPT_MAX = 2300  # the longest prompt, in tokens: the cache holds it as well as the answer
KiB = 2**10
FAMILIES: dict[str, Family] = {
    # Qwen3.5: 8 full-attention layers at 32 KiB per token + 24 DeltaNet states of 2 MiB
    "qwen3.5-4b": Family(
        {"enable_thinking": True},
        QWEN,
        batch=11,
        resident_gib=2.2,
        kib_per_token=32,
        fixed_kib=49 * KiB,
    ),
    "qwen3.5-9b": Family(
        {"enable_thinking": True},
        QWEN,
        batch=7,
        resident_gib=4.9,
        kib_per_token=32,
        fixed_kib=49 * KiB,
    ),
    # Gemma 4 12B: 8 of 48 layers full attention at 64 KiB per token, 40 sliding windows of 1,024
    "gemma4-12b": Family(
        {"enable_thinking": True},
        Sampling(temperature=1.0, top_p=0.95, top_k=64, presence_penalty=0.0, max_tokens=CAP),
        batch=3,
        resident_gib=5.7,
        kib_per_token=64,
        fixed_kib=320 * KiB,
    ),
    # gpt-oss-20b: 11.3 GiB of weights alone — over a 12 GiB budget even at batch 1.  Kept so
    # that `generate --model gpt-oss-20b --limit 16` can still run it deliberately.
    "gpt-oss-20b": Family(
        {"reasoning_effort": "medium"},
        Sampling(temperature=1.0, top_p=1.0, top_k=0, presence_penalty=0.0, max_tokens=CAP),
        batch=1,
        resident_gib=11.3,
        kib_per_token=24,
    ),
    # DianJin (Qwen2.5-7B): 28 full-attention layers at 56 KiB per token
    "dianjin-r1-7b": Family(
        {},
        Sampling(temperature=0.6, top_p=0.95, top_k=0, presence_penalty=0.0, max_tokens=CAP),
        batch=5,
        resident_gib=4.0,
        kib_per_token=56,
    ),
}


def read_run(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def read_meta(path: str | Path) -> dict:
    meta = Path(str(path).removesuffix(".jsonl") + ".meta.json")
    return json.loads(meta.read_text()) if meta.exists() else {}


def is_loop(row: dict) -> bool:
    return row["hit_limit"] and row["repetition"] >= LOOP_THRESHOLD


def outcome(row: dict) -> str:
    """Exactly one of ``right | wrong | no-box | cut``.

    ``cut`` hit the token cap (whether or not it had boxed something on the way —
    ``correct`` may still be true, by the benchmark protocol); ``no-box`` stopped by
    itself without putting an answer in ``\\boxed{}``; the rest answered and were
    right or wrong.
    """
    if row["hit_limit"]:
        return "cut"
    if row["predicted"] is None:
        return "no-box"
    return "right" if row["correct"] else "wrong"


def prompt_tokens(
    tokenizer, family: Family, problem: Problem, system: str | None = None
) -> list[int]:
    """The chat-formatted prompt; ``system`` is a standing brief (a desk playbook), if any."""
    messages = [{"role": "user", "content": problem.prompt}]
    if system:
        messages.insert(0, {"role": "system", "content": system})
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        **family.template,
    )


# --------------------------------------------------------------------------------------
# generate
# --------------------------------------------------------------------------------------


def single_stream_speed(model, tokenizer, prompt: list[int], tokens: int = 128) -> dict:
    """Prefill and decoding speed with one sequence — what one person waiting experiences."""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    responses = stream_generate(
        model, tokenizer, prompt, max_tokens=tokens, sampler=make_sampler(temp=0.0)
    )
    last = list(responses)[-1]  # the last response carries the whole run's statistics
    return {
        "prompt_tokens": last.prompt_tokens,
        "prefill_tps": round(last.prompt_tps, 1),
        "decode_tps": round(last.generation_tps, 1),
    }


def cmd_generate(args: argparse.Namespace) -> int:
    import mlx.core as mx

    from learnai.openweights import load, resolve
    from learnai.weightedit import apply_patch, load_patch, sample_batch

    family = FAMILIES[args.model]
    seed = args.seed if args.sample_seed is None else args.sample_seed
    sampling = dataclasses.replace(family.sampling, max_tokens=args.max_tokens, seed=seed)
    batch = args.batch or family.batch
    worst = family.peak_gib(batch, args.max_tokens)
    if worst > args.limit:
        print(
            f"{args.model} at batch {batch} would peak near {worst:.1f} GiB, "
            f"over the {args.limit} GiB limit; lower --batch or raise --limit deliberately"
        )
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {row["uid"] for row in read_run(out)} if out.exists() else set()
    problems = [
        p for p in problems_of(args.bench, args.n, args.offset, args.seed) if p.uid not in done
    ]
    if args.uids:  # a subset of the slice, e.g. the problems an edit is meant to change
        wanted = set(Path(args.uids).read_text().split())
        problems = [p for p in problems if p.uid in wanted]
    if not problems:
        print(f"{out}: already complete")
        return 0

    # The budget, enforced where it is spent.  MLX keeps every freed buffer in a cache
    # until told otherwise (the default cache limit *is* the memory limit, 30 GiB here),
    # so a run that outgrows its KV caches every 256 tokens holds a second copy of them
    # as garbage until the next `clear_cache`.  The system counts that memory as used.
    mx.set_memory_limit(int(args.limit * GiB))  # reclaim the cache before growing past this
    mx.set_cache_limit(1 * GiB)  # and never keep more than this much freed memory around

    model, tokenizer = load(args.model)
    if args.patch:
        edits, meta = load_patch(args.patch)
        apply_patch(model, edits)
        print(f"patch   {args.patch}: layers {meta['layers']}")
    system = Path(args.system).read_text() if args.system else None
    prompts = [prompt_tokens(tokenizer, family, p, system) for p in problems]
    print(
        f"{args.model}  {len(problems)} problems  batch {batch}  {sampling}  "
        f"template {family.template}  prompt tokens max {max(map(len, prompts))}",
        flush=True,
    )

    started, finished = time.perf_counter(), 0
    generated = 0
    peak_seen = 0
    mx.reset_peak_memory()
    memlog = Path(str(out).removesuffix(".jsonl") + ".mem.jsonl")
    with out.open("a") as handle, memlog.open("a") as memory:

        def on_settle(step: int) -> None:
            """Every 256 decoding steps: how much is held, and how high the last stretch went."""
            nonlocal peak_seen
            peak = mx.get_peak_memory()
            peak_seen = max(peak_seen, peak)
            memory.write(
                json.dumps(
                    {
                        "t": round(time.perf_counter() - started),
                        "step": step,
                        "done": finished,
                        "active_gib": round(mx.get_active_memory() / GiB, 2),
                        "cache_gib": round(mx.get_cache_memory() / GiB, 2),
                        "peak_gib": round(peak / GiB, 2),
                    }
                )
                + "\n"
            )
            memory.flush()
            mx.reset_peak_memory()  # so the next line's peak is the next stretch's own

        def on_done(i, generation):
            nonlocal finished, generated
            text = tokenizer.decode(generation.tokens)
            row = {
                "uid": problems[i].uid,
                "source": problems[i].source,
                "domain": problems[i].domain,
                "kind": problems[i].kind,
                "gold": problems[i].answer,
                **grade(problems[i], text, family.thinking),
                "hit_limit": generation.hit_limit,
                "prompt_tokens": len(prompts[i]),
                "n_tokens": len(generation.tokens),
                "repetition": round(repetition(generation.tokens), 3),
                "text": text,
                "tokens": generation.tokens,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            finished += 1
            generated += len(generation.tokens)
            if finished % 10 == 0 or finished == len(problems):
                elapsed = time.perf_counter() - started
                print(
                    f"  {finished}/{len(problems)} done after {elapsed / 60:.1f} min, "
                    f"{generated / elapsed:.0f} tok/s finished; memory active "
                    f"{mx.get_active_memory() / GiB:.1f} + cache {mx.get_cache_memory() / GiB:.1f}, "
                    f"peak {max(peak_seen, mx.get_peak_memory()) / GiB:.1f} GiB",
                    flush=True,
                )

        # Prompts are read two at a time in 512-token chunks: the memory peak of a run is
        # the prefill, and the defaults (8 x 2,048) spike to 9 GiB on the 4B for 1.5k-token
        # FinQA excerpts where 2 x 512 stays under 4 GiB, at the same speed (measured).
        sample_batch(
            model,
            tokenizer,
            prompts,
            sampling,
            batch_size=batch,
            prefill_batch_size=2,
            prefill_step_size=512,
            on_done=on_done,
            on_settle=on_settle,
        )
    wall = time.perf_counter() - started
    peak_seen = max(peak_seen, mx.get_peak_memory())
    session = {
        "started": time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() - wall)),
        "problems": len(problems),
        "batch": batch,
        "wall_seconds": round(wall),
        "generated_tokens": generated,
        "batch_tps": round(generated / wall, 1),
        "peak_memory_gib": round(peak_seen / GiB, 2),
    }
    meta = read_meta(out)
    meta.update(
        {
            "model": args.model,
            "repo": resolve(args.model).repo,
            "patch": args.patch,
            "system": {"file": str(args.system), "text": system} if system else None,
            "sampling": dataclasses.asdict(sampling),
            "template": family.template,
            "sessions": meta.get("sessions", []) + [session],
        }
    )
    if "single_stream" not in meta and not args.no_speed:
        longest = max(prompts, key=len)  # a FinQA excerpt: the realistic prefill
        meta["single_stream"] = single_stream_speed(model, tokenizer, longest)
    Path(str(out).removesuffix(".jsonl") + ".meta.json").write_text(json.dumps(meta, indent=1))
    print(
        f"wall {wall / 60:.1f} min  {session['batch_tps']} tok/s  peak {session['peak_memory_gib']} GiB"
    )
    print(summarize(read_run(out), str(out), meta))
    return 0


# --------------------------------------------------------------------------------------
# summary / curve / compare / failures
# --------------------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion — honest for small n and for 0% / 100%."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def pct(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:3.0f}% [{100 * lo:.0f}–{100 * hi:.0f}]" if n else "–"


def quantiles(values: list[int]) -> str:
    if not values:
        return "–"
    s = sorted(values)
    return f"median {s[len(s) // 2]:,}  p90 {s[int(len(s) * 0.9)]:,}  max {s[-1]:,}"


def summarize(rows: list[dict], label: str, meta: dict | None = None) -> str:
    """Accuracy per source, then where the misses went, then the cost in tokens and time."""
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    lines = [f"{label}: {len(rows)} answers"]
    for source in SOURCES:
        group = by_source.get(source, [])
        if group:
            right = sum(r["correct"] for r in group)
            lines.append(
                f"  {source:<10} {pct(right, len(group))}   tokens {quantiles([r['n_tokens'] for r in group])}"
            )
    right = sum(r["correct"] for r in rows)
    lines.append(f"  {'all':<10} {pct(right, len(rows))}")
    ends = Counter(outcome(r) for r in rows)
    loops = sum(is_loop(r) for r in rows)
    lines.append(
        f"  outcomes: {ends['right']} right, {ends['wrong']} wrong, {ends['no-box']} no box, "
        f"{ends['cut']} cut by the cap ({loops} looping)"
    )
    if meta and meta.get("sessions"):
        wall = sum(s["wall_seconds"] for s in meta["sessions"])
        toks = sum(s["generated_tokens"] for s in meta["sessions"])
        peak = max((s["peak_memory_gib"] or 0) for s in meta["sessions"])
        line = f"  cost: {wall / 60:.0f} min wall, {toks:,} tokens, {toks / wall:.0f} tok/s batched, peak {peak} GiB"
        if speed := meta.get("single_stream"):
            waits = [
                r["prompt_tokens"] / speed["prefill_tps"] + r["n_tokens"] / speed["decode_tps"]
                for r in rows
            ]
            line += (
                f"; alone: {speed['decode_tps']} tok/s, one question = "
                f"median {statistics.median(waits):.0f} s, mean {statistics.mean(waits):.0f} s"
            )
        lines.append(line)
    return "\n".join(lines)


def cmd_summary(args: argparse.Namespace) -> int:
    for path in args.runs:
        print(summarize(read_run(path), str(path), read_meta(path)))
    return 0


def regrade_at(row: dict, tokenizer, cap: int, thinking: bool) -> bool:
    """Would this answer have been right had the cap been ``cap`` tokens?"""
    if row["n_tokens"] <= cap:
        return row["correct"]
    text = tokenizer.decode(row["tokens"][:cap])
    return grade(problem_for(row["uid"]), text, thinking)["correct"]


def cmd_curve(args: argparse.Namespace) -> int:
    """Re-grade stored answers truncated at each cap: exact, no new sampling needed."""
    from mlx_lm.utils import load_tokenizer

    from learnai.openweights import local_path

    caps = [int(c) for c in args.caps.split(",")]
    print(f"{'run':<48} " + " ".join(f"{c:>7}" for c in caps))
    for path in args.runs:
        rows, meta = read_run(path), read_meta(path)
        tokenizer = load_tokenizer(local_path(meta["model"]))
        thinking = FAMILIES[meta["model"]].thinking
        cells = []
        for cap in caps:
            right = sum(regrade_at(r, tokenizer, cap, thinking) for r in rows)
            cells.append(f"{100 * right / len(rows):6.0f}%")
        print(f"{Path(path).parent.name + '/' + Path(path).name:<48} " + " ".join(cells))
    return 0


# When the budget runs out mid-thought, close the reasoning ourselves and ask for the
# answer, instead of scoring the cut-off as wrong ("budget forcing", Muennighoff et al.
# 2025, s1).  The cue ends inside a box so the reply is graded by the same rules.
FORCE_CUE = "\n</think>\n\n**Final answer:** \\boxed{"


def answer_now_context(
    tokenizer, family: Family, row: dict, cap: int, system: str | None = None
) -> list[int] | None:
    """Prompt + the answer's first ``cap`` tokens + the cue, or ``None`` if it had stopped thinking.

    ``row["tokens"]`` are the tokens the model generated after the prompt, so the
    context is exactly what the model had seen at token ``cap`` of its own answer.
    """
    head = row["tokens"][:cap]
    _, answer = split_thinking(tokenizer.decode(head), family.thinking)
    if answer is not None:
        return None
    cue = tokenizer.encode(FORCE_CUE, add_special_tokens=False)
    prompt = prompt_tokens(tokenizer, family, problem_for(row["uid"]), system)
    return prompt + list(head) + cue


def still_thinking_at(row: dict, cap: int) -> bool:
    """Was this answer still being generated at token ``cap``?  (At the run's own cap a cut-off has exactly ``cap`` tokens.)"""
    return row["n_tokens"] > cap or (row["hit_limit"] and row["n_tokens"] == cap)


def close_box(continuation: str) -> tuple[str, bool]:
    """The continuation up to the brace that closes the cue's ``\\boxed{``, and whether it closed."""
    depth = 1
    for i, ch in enumerate(continuation):
        depth += (ch == "{") - (ch == "}")
        if depth == 0:
            return continuation[: i + 1], True
    return continuation + "}", False


def cmd_force(args: argparse.Namespace) -> int:
    """Grade a run as if its cap had been ``--cap`` tokens, answering at the cap instead of failing."""
    import mlx.core as mx

    from learnai.openweights import load
    from learnai.weightedit import prefill_cache

    rows, meta = read_run(args.run), read_meta(args.run)
    family = FAMILIES[meta["model"]]
    if args.cap > meta["sampling"]["max_tokens"]:
        print(
            f"the run was capped at {meta['sampling']['max_tokens']} tokens; --cap cannot exceed it"
        )
        return 2
    out = Path(args.out or str(args.run).removesuffix(".jsonl") + f".force{args.cap}.jsonl")
    done = {r["uid"] for r in read_run(out)} if out.exists() else set()
    mx.set_memory_limit(int(args.limit * GiB))
    mx.set_cache_limit(1 * GiB)
    model, tokenizer = load(meta["model"])
    eos = set(tokenizer.eos_token_ids)
    mx.reset_peak_memory()
    with out.open("a") as handle:
        for row in rows:
            if row["uid"] in done:
                continue
            problem = problem_for(row["uid"])
            record = {k: row[k] for k in ("uid", "source", "kind", "gold")}
            if not still_thinking_at(row, args.cap):
                record.update(forced=False, predicted=row["predicted"], correct=row["correct"])
                record["correct_unforced"] = row["correct"]
            else:
                cut = tokenizer.decode(row["tokens"][: args.cap])
                unforced = grade(problem, cut, family.thinking)
                system = (meta.get("system") or {}).get("text")  # the run's own playbook
                context = answer_now_context(tokenizer, family, row, args.cap, system)
                if context is None:  # it had stopped thinking before the cap
                    record.update(forced=False, predicted=unforced["predicted"])
                    record.update(correct=unforced["correct"], correct_unforced=unforced["correct"])
                else:
                    started = time.perf_counter()
                    cache = prefill_cache(model, context[:-1], chunk=512)
                    token, generated = context[-1], []
                    for _ in range(args.answer_tokens):  # greedy: the model's single best answer
                        logits = model(mx.array([[token]]), cache=cache)
                        token = int(mx.argmax(logits[0, -1]).item())
                        if token in eos:
                            break
                        generated.append(token)
                        if close_box(tokenizer.decode(generated))[1]:
                            break
                    inside, closed = close_box(tokenizer.decode(generated))
                    graded = grade(problem, cut + FORCE_CUE + inside, family.thinking)
                    record.update(forced=True, closed=closed, answer=inside)
                    record.update(predicted=graded["predicted"], correct=graded["correct"])
                    record["correct_unforced"] = unforced["correct"]
                    record["seconds"] = round(time.perf_counter() - started, 1)
                    del cache
                    mx.clear_cache()
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
    print(force_summary(read_run(out), f"{Path(args.run).parent.name}/{out.name}"))
    print(f"peak memory {mx.get_peak_memory() / GiB:.2f} GiB")
    return 0


def force_summary(records: list[dict], label: str) -> str:
    """Right at the cap without and with the answer forced, overall and per source."""
    forced = [r for r in records if r["forced"]]
    lines = [
        f"{label}: {len(records)} problems, {len(forced)} answered at the cap",
        f"  right, cut-offs scored as they stand: {sum(r['correct_unforced'] for r in records)}",
        f"  right, answer forced at the cap:      {sum(r['correct'] for r in records)}",
    ]
    for source in SOURCES:
        sub = [r for r in records if r["source"] == source]
        if sub:
            before = sum(r["correct_unforced"] for r in sub)
            lines.append(
                f"  {source:<10} {before:>4} -> {sum(r['correct'] for r in sub):>4} of {len(sub)}"
            )
    return "\n".join(lines)


def mcnemar(a: int, b: int) -> float:
    """Exact two-sided McNemar test: of the a + b pairs that changed, is a vs b a coin flip?"""
    n = a + b
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(a, b) + 1)) / 2**n)


def cmd_compare(args: argparse.Namespace) -> int:
    """Paired by problem: how many did the edit fix, how many did it break?"""
    base = {r["uid"]: r for r in read_run(args.base)}
    edited = {r["uid"]: r for r in read_run(args.edited)}
    shared = sorted(set(base) & set(edited))
    if not shared:
        print("no problems in common")
        return 1
    fixed = [u for u in shared if edited[u]["correct"] and not base[u]["correct"]]
    broke = [u for u in shared if base[u]["correct"] and not edited[u]["correct"]]
    b, e = sum(base[u]["correct"] for u in shared), sum(edited[u]["correct"] for u in shared)
    print(
        f"{len(shared)} problems in common: base {pct(b, len(shared))}  edited {pct(e, len(shared))}"
    )
    n = len(fixed) + len(broke)
    p_two = mcnemar(len(fixed), len(broke))
    print(
        f"  fixed {len(fixed)}, broke {len(broke)}  (two-sided p = {p_two:.2f} over the {n} that changed)"
    )
    for source in SOURCES:
        uids = [u for u in shared if base[u]["source"] == source]
        if uids:
            before, after = (sum(run[u]["correct"] for u in uids) for run in (base, edited))
            print(f"  {source:<12} {before:>4} -> {after:>4} of {len(uids)}")
    # A forced run (``force``) keeps only the verdicts, not how the answer was generated.
    generated = all("hit_limit" in run[u] for run in (base, edited) for u in shared)
    if generated:
        stopped = sum(is_loop(base[u]) and not is_loop(edited[u]) for u in shared)
        started = sum(is_loop(edited[u]) and not is_loop(base[u]) for u in shared)
        print(
            f"  loops: {sum(is_loop(base[u]) for u in shared)} -> {sum(is_loop(edited[u]) for u in shared)}"
            f"  ({stopped} stopped looping, {started} started; two-sided p = {mcnemar(stopped, started):.3f})"
        )
    for name, uids in (("fixed", fixed), ("broke", broke)):
        if uids:
            print(f"  {name}: " + ", ".join(uids))
    for key, label in (("n_tokens", "tokens"), ("hit_limit", "cut")) if generated else ():
        bs = [base[u][key] for u in shared]
        es = [edited[u][key] for u in shared]
        if key == "hit_limit":
            print(f"  {label}: {sum(bs)} -> {sum(es)}")
        else:
            print(f"  {label}: median {statistics.median(bs):,.0f} -> {statistics.median(es):,.0f}")
    return 0


def classify(row: dict) -> str:
    """A first-pass label for a miss — a starting point for reading the transcript, not a verdict.

    ``loop``: hit the cap repeating itself.  ``cut``: hit the cap while still reasoning
    (deliberation that did not end).  ``no-box``: stopped without a boxed answer.
    ``scale``: the number is right up to a power of ten or a percent/ratio mix-up.
    ``wrong``: a different answer — knowledge, reading or arithmetic.
    """
    if row["correct"]:
        return "right"
    if row["hit_limit"]:
        return "loop" if is_loop(row) else "cut"
    if row["predicted"] is None:
        return "no-box"
    if row["kind"] in ("number", "percent"):
        from learnai.finbench import parse_number

        parsed = parse_number(row["predicted"])
        gold = float(row["gold"])
        if parsed and parsed[0] and gold:
            ratio = abs(parsed[0] / gold)
            for power in (0.01, 0.1, 10, 100, 1000, 1e6):
                if abs(math.log10(ratio) - math.log10(power)) < 0.01:
                    return "scale"
    return "wrong"


REFERENCE_HEADER = (
    "Answer the following finance questions from your own knowledge and reasoning only. "
    "Do not call any tool — no web search, no file reads, no code execution, nothing. "
    "Think each one through carefully, then reply with EXACTLY one line per question and "
    "nothing else, in this form:\n\n"
    "<uid> => \\boxed{<answer>}\n\n"
    "For a multiple-choice question the answer is the single option letter. For a numeric "
    "question it is a plain number in the units the question's report uses (a percentage "
    "written like 12.3%), or yes/no. Where a question asks for its answer in a particular "
    "form (signs, orders), use exactly that form. Give every question an answer.\n\n"
)


def cmd_export(args: argparse.Namespace) -> int:
    """Write a slice of questions as plain-text batches, for a model without a local runner.

    This is how the offline Claude reference was asked: one batch per subagent (5 FinQA
    excerpts, or 25 multiple-choice questions), the same prompts the local models get,
    no tools allowed.  Its answer lines go back through ``grade-answers``.
    """
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    problems = problems_of(args.bench, args.n, args.offset, args.seed)
    count = 0
    sizes = {"finqa": 5, "futures-exec": 10}  # long prompts go in smaller batches
    for source in SOURCES:
        size = sizes.get(source, 25)
        group = [p for p in problems if p.source == source]
        for i in range(0, len(group), size):
            body = "\n\n".join(f"### {p.uid}\n{p.prompt}" for p in group[i : i + size])
            (out / f"batch-{count:02d}.txt").write_text(REFERENCE_HEADER + body)
            count += 1
    print(f"{count} batches for {len(problems)} questions in {out}")
    return 0


def cmd_grade_answers(args: argparse.Namespace) -> int:
    """Turn ``uid => \\boxed{answer}`` lines into a run file the other commands read."""
    from learnai.finbench import last_boxed

    problems = {p.uid: p for p in problems_of(args.bench, args.n, args.offset, args.seed)}
    given: dict[str, str] = {}
    for path in args.answers:
        for line in Path(path).read_text().splitlines():
            uid, arrow, rest = line.partition("=>")
            uid = uid.strip().lstrip("#").strip().strip("`")
            if arrow and uid in problems:
                given[uid] = rest.strip()
    rows = []
    for uid, p in problems.items():
        text = given.get(uid, "")
        boxed = last_boxed(text) if text else None
        rows.append(
            {
                "uid": uid,
                "source": p.source,
                "domain": p.domain,
                "kind": p.kind,
                "gold": p.answer,
                "finished": bool(text),
                "predicted": boxed,
                "correct": equivalent(boxed, p) if boxed is not None else False,
                "hit_limit": False,
                "prompt_tokens": 0,
                "n_tokens": 0,
                "repetition": 0.0,
                "text": text,
                "tokens": [],
            }
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    meta = {
        "model": args.label,
        "repo": None,
        "patch": None,
        "sessions": [],
        "note": "answers only (no transcripts); asked through `export`, graded by the same rules",
    }
    Path(str(out).removesuffix(".jsonl") + ".meta.json").write_text(json.dumps(meta, indent=1))
    missing = [u for u in problems if u not in given]
    print(
        f"{len(given)} answers for {len(problems)} problems; missing {len(missing)}: {missing[:10]}"
    )
    print(summarize(rows, str(out), meta))
    return 0


def cmd_regrade(args: argparse.Namespace) -> int:
    """Re-score stored runs with the current grader (the stored text is all it needs)."""
    for path in args.runs:
        rows = read_run(path)
        thinking = FAMILIES.get(read_meta(path).get("model"), Family({}, QWEN, 1, 0, 0)).thinking
        changed = 0
        for r in rows:
            fresh = grade(problem_for(r["uid"]), r["text"], thinking)
            if r["tokens"] == []:  # a reference run stores the bare answer, not a transcript
                fresh["finished"] = r["finished"]
            changed += fresh["correct"] != r["correct"]
            r.update(fresh)
        Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"{path}: {changed} verdicts changed")
    return 0


def cmd_failures(args: argparse.Namespace) -> int:
    rows = read_run(args.run)
    labels = Counter(classify(r) for r in rows)
    print(f"{args.run}: " + ", ".join(f"{v} {k}" for k, v in labels.most_common()))
    for r in rows:
        label = classify(r)
        if label == "right" or (args.only and label != args.only):
            continue
        reasoning, answer = split_thinking(
            r["text"], FAMILIES[read_meta(args.run)["model"]].thinking
        )
        tail = (answer or reasoning)[-args.chars :].replace("\n", " ")
        print(
            f"\n[{label}] {r['uid']}  gold={r['gold']!r} predicted={r['predicted']!r}  {r['n_tokens']} tokens"
        )
        print(f"   …{tail}")
    return 0


# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="answer finance problems with one model")
    g.add_argument("--model", required=True, choices=sorted(FAMILIES))
    g.add_argument("--bench", choices=["fin", "macro"], default="fin", help="which question set")
    g.add_argument("--n", type=int, default=50, help="problems per source")
    g.add_argument(
        "--offset", type=int, default=0, help="skip this many per source (the screening set is 0)"
    )
    g.add_argument(
        "--seed", type=int, default=0, help="picks the problems (and, by default, samples)"
    )
    g.add_argument(
        "--sample-seed", type=int, help="another sample of the same problems: a different seed here"
    )
    g.add_argument("--max-tokens", type=int, default=CAP)
    g.add_argument("--batch", type=int, help="override the family's batch size")
    g.add_argument(
        "--limit", type=float, default=PROCESS_LIMIT_GIB, help="GiB the MLX process may peak at"
    )
    g.add_argument("--patch", help="a .npz written by weight_edit_experiment.py edit")
    g.add_argument("--uids", help="file of problem uids: answer only these (of the slice)")
    g.add_argument("--system", help="a text file sent as the system message (a desk playbook)")
    g.add_argument(
        "--no-speed", action="store_true", help="skip the single-stream speed measurement"
    )
    g.add_argument(
        "--out", help="default runs/finbench/<model>/[macro-]n<n>-o<offset>[-<patch>].jsonl"
    )
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("summary", help="summarise runs")
    s.add_argument("runs", nargs="+")
    s.set_defaults(func=cmd_summary)

    c = sub.add_parser("curve", help="accuracy under smaller token caps, from the same runs")
    c.add_argument("runs", nargs="+")
    c.add_argument("--caps", default="2048,4096,8192,16384")
    c.set_defaults(func=cmd_curve)

    fo = sub.add_parser("force", help="answer at a smaller cap instead of failing there")
    fo.add_argument("run")
    fo.add_argument("--cap", type=int, required=True)
    fo.add_argument("--answer-tokens", type=int, default=48, help="tokens allowed for the answer")
    fo.add_argument(
        "--limit", type=float, default=PROCESS_LIMIT_GIB, help="GiB the MLX process may peak at"
    )
    fo.add_argument("--out", help="default <run>.force<cap>.jsonl")
    fo.set_defaults(func=cmd_force)

    p = sub.add_parser("compare", help="paired comparison of two runs")
    p.add_argument("base")
    p.add_argument("edited")
    p.set_defaults(func=cmd_compare)

    ex = sub.add_parser("export", help="write questions as text batches (for a reference model)")
    ex.add_argument("--bench", choices=["fin", "macro"], default="fin")
    ex.add_argument("--n", type=int, default=50)
    ex.add_argument("--offset", type=int, default=0)
    ex.add_argument("--seed", type=int, default=0)
    ex.add_argument("--out", required=True)
    ex.set_defaults(func=cmd_export)

    ga = sub.add_parser("grade-answers", help="grade `uid => \\boxed{x}` answer lines into a run")
    ga.add_argument("answers", nargs="+")
    ga.add_argument("--bench", choices=["fin", "macro"], default="fin")
    ga.add_argument("--n", type=int, default=50)
    ga.add_argument("--offset", type=int, default=0)
    ga.add_argument("--seed", type=int, default=0)
    ga.add_argument("--label", required=True)
    ga.add_argument("--out", required=True)
    ga.set_defaults(func=cmd_grade_answers)

    rg = sub.add_parser("regrade", help="re-score stored runs with the current grader")
    rg.add_argument("runs", nargs="+")
    rg.set_defaults(func=cmd_regrade)

    f = sub.add_parser("failures", help="list and classify the misses of a run")
    f.add_argument("run")
    f.add_argument("--only", choices=["loop", "cut", "no-box", "scale", "wrong"])
    f.add_argument(
        "--chars", type=int, default=300, help="characters of each answer's tail to show"
    )
    f.set_defaults(func=cmd_failures)

    args = parser.parse_args(argv)
    if args.command == "generate" and not args.out:
        tag = f"n{args.n}-o{args.offset}" + (f"-{Path(args.patch).stem}" if args.patch else "")
        tag += f"-{Path(args.system).stem}" if args.system else ""
        tag = ("macro-" if args.bench == "macro" else "") + tag
        args.out = RUNS_DIR / "finbench" / args.model / f"{tag}.jsonl"
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

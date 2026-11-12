"""Math word problems as a measuring stick: GSM8K and MATH-500.

Comparing two versions of a model needs three things that have nothing to do with
neural networks, so they live here in plain Python (and run in CI without a model):

1. **problems** with a known final answer — :func:`load_problems`;
2. a **grader** that reads the final answer out of free text and decides whether
   it equals the reference — :func:`grade`;
3. a **loop detector** — :func:`repetition`.  A reasoning model that gets stuck
   repeats itself until it runs out of tokens; "ran out of tokens" alone cannot
   tell that apart from a long but healthy derivation.

The grader follows the conventions of the benchmarks themselves: the model is asked
to put its answer in ``\\boxed{}``, answers that are plain numbers are compared as numbers,
and everything else after the string normalisation that the MATH paper introduced
(Hendrycks et al., 2021, ``is_equiv``).  It is deliberately
strict and simple: both models under comparison are graded by the same rules, so a
systematic bias in the grader cancels in the *difference* — which is the quantity
the experiments report.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from learnai.env import CACHE_DIR

ROWS_API = "https://datasets-server.huggingface.co/rows"
BENCH_DIR = CACHE_DIR / "mathbench"

# The instruction Qwen's own evaluations append to math problems.
PROMPT_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."

MATH_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)

# name -> (Hugging Face dataset, configs, split).  The *train* splits are practice
# material for building an edit; the *test* splits are only ever used to score one.
SOURCES: dict[str, tuple[str, tuple[str, ...], str]] = {
    "gsm8k-train": ("openai/gsm8k", ("main",), "train"),
    "gsm8k-test": ("openai/gsm8k", ("main",), "test"),
    "math-train": ("EleutherAI/hendrycks_math", MATH_SUBJECTS, "train"),
    "math500": ("HuggingFaceH4/MATH-500", ("default",), "test"),
}


@dataclass(frozen=True)
class Problem:
    uid: str  # e.g. "gsm8k-test/17"
    question: str
    answer: str  # the reference final answer, as the dataset states it
    kind: str  # "number" (GSM8K) or "latex" (MATH)

    @property
    def prompt(self) -> str:
        return self.question + PROMPT_SUFFIX


# --------------------------------------------------------------------------------------
# Problems
# --------------------------------------------------------------------------------------


def _offline() -> bool:
    return os.environ.get("LEARNAI_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}


def _fetch_rows(dataset: str, config: str, split: str, n: int) -> list[dict]:
    """The first ``n`` rows of a Hub dataset split, cached as JSON lines.

    The datasets-server API serves at most 100 rows per request as JSON, which
    avoids a parquet dependency for a few hundred short strings.
    """
    path = BENCH_DIR / f"{dataset.replace('/', '--')}--{config}--{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    while len(rows) < n:
        if _offline():
            raise RuntimeError(f"LEARNAI_OFFLINE is set and {path} holds {len(rows)} < {n} rows")
        query = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": len(rows),
                "length": 100,
            }
        )
        request = urllib.request.Request(f"{ROWS_API}?{query}", headers={"User-Agent": "learnai"})
        with urllib.request.urlopen(request, timeout=60) as response:
            page = [item["row"] for item in json.load(response)["rows"]]
        if not page:  # the split is shorter than n
            break
        rows.extend(page)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return rows[:n]


def _to_problem(name: str, index: int, row: dict) -> Problem | None:
    if name.startswith("gsm8k"):
        # GSM8K solutions end with "#### 1,234".
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        return Problem(f"{name}/{index}", row["question"], gold, "number")
    gold = row.get("answer") or last_boxed(row["solution"])
    if gold is None:  # a handful of MATH solutions have no boxed answer
        return None
    return Problem(f"{name}/{index}", row["problem"], gold, "latex")


def load_problems(name: str, n: int) -> list[Problem]:
    """The first ``n`` problems of a benchmark, in the dataset's own (fixed) order.

    For ``math-train`` the seven subjects are interleaved, so any prefix is a mix.
    """
    dataset, configs, split = SOURCES[name]
    per_config = -(-n // len(configs)) + 2  # ceil, plus slack for rows without an answer
    columns = [
        [
            p
            for i, row in enumerate(_fetch_rows(dataset, config, split, per_config))
            if (p := _to_problem(f"{name}/{config}" if len(configs) > 1 else name, i, row))
        ]
        for config in configs
    ]
    interleaved = [p for group in zip(*columns, strict=False) for p in group]
    return interleaved[:n]


# --------------------------------------------------------------------------------------
# Reading the final answer out of a response
# --------------------------------------------------------------------------------------


def split_thinking(text: str) -> tuple[str, str | None]:
    """``(reasoning, answer)``.  ``answer`` is ``None`` if the reasoning never ended.

    Qwen's chat template opens ``<think>`` in the prompt itself, so a response is
    reasoning up to ``</think>`` and the answer after it.  A response without the
    closing tag ran out of tokens while still thinking: it has no answer at all.
    """
    reasoning, tag, answer = text.partition("</think>")
    return (reasoning, answer) if tag else (text, None)


def last_boxed(text: str) -> str | None:
    """The contents of the last ``\\boxed{...}`` in ``text``, with braces matched."""
    start = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if start < 0:
        return None
    open_brace = text.find("{", start)
    if open_brace < 0 or text[start:open_brace].strip() not in {"\\boxed", "\\fbox"}:
        # "\boxed 5": the argument is the next whitespace-delimited token
        rest = text[start:].split(maxsplit=2)
        return rest[1].strip("$.") if len(rest) > 1 else None
    depth = 0
    for i in range(open_brace, len(text)):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            return text[open_brace + 1 : i]
    return None  # unbalanced: the response was cut off inside the box


# --------------------------------------------------------------------------------------
# Deciding whether two answers are the same
# --------------------------------------------------------------------------------------


def _fix_fracs(s: str) -> str:
    """``\\frac12`` and ``\\frac1{2}`` -> ``\\frac{1}{2}``."""
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac(\d)\{", r"\\frac{\1}{", s)
    return re.sub(r"\\frac\{([^{}]*)\}(\d)", r"\\frac{\1}{\2}", s)


def normalize_answer(answer: str) -> str:
    """Canonical form of a final answer, so that typographic variants compare equal.

    The rules are those of the MATH paper's ``_strip_string`` (spacing, ``\\left``,
    ``\\dfrac``, units in ``\\text{}``, degrees, percent and dollar signs, a leading
    "x =", bare decimals, ``\\frac12``), plus thousands separators.
    """
    s = answer.strip().replace("\n", "")
    # "5 \text{ hours}", "25 \text{ cm}^2": text that trails a number or a fraction is a
    # unit, and the answer is the number.  Text anywhere else is part of the answer.
    s = re.sub(
        r"(?<=[\d}])(?:\s*\\(?:text|mathrm|textbf|mbox)\{[^{}]*\}(?:\^\{?\d\}?)?)+\.?$", "", s
    )
    s = re.sub(r"\\(?:text|mathrm|textbf|mbox)\{([^{}]*)\}", r"\1", s)  # keep the words
    for junk in ("\\!", "\\,", "\\;", "\\ ", "\\left", "\\right", "\\$", "$", "\\%", "%"):
        s = s.replace(junk, "")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("°", "")
    s = s.replace(" ", "").rstrip(".")
    if s.count("=") == 1 and len(s.split("=")[0]) <= 2:  # "x=5" -> "5"
        s = s.split("=")[1]
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):  # "1,234" is a number, "(1,2)" is not
        s = s.replace(",", "")
    s = re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", s)
    s = _fix_fracs(s)
    s = re.sub(r"(^|[^\d])\.(\d)", r"\g<1>0.\2", s)  # ".5" -> "0.5"
    if re.fullmatch(r"-?\d+/\d+", s):  # "1/2" -> "\frac{1}{2}"
        sign, rest = ("-", s[1:]) if s.startswith("-") else ("", s)
        num, den = rest.split("/")
        s = f"{sign}\\frac{{{num}}}{{{den}}}"
    return s


def to_fraction(answer: str) -> Fraction | None:
    """The exact value of a normalised answer if it is a plain number, else ``None``."""
    s = normalize_answer(answer)
    if m := re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}", s):
        if int(m[3]) == 0:
            return None
        value = Fraction(int(m[2]), int(m[3]))
        return -value if m[1] else value
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        return Fraction(s)
    return None


def _elements(s: str) -> list[str] | None:
    """The parts of a normalised tuple or interval such as ``(a,b,c)`` or ``[a,b)``.

    Only commas at the top level separate: ``(\\frac{1}{2},(3,4))`` has two parts.
    ``None`` if ``s`` is not one bracketed sequence — ``(1,2)+(3,4)`` is a sum.
    """
    if len(s) < 2 or s[0] not in "([" or s[-1] not in ")]":
        return None
    parts, depth, start = [], 0, 1
    for i in range(1, len(s) - 1):
        depth += (s[i] in "([{") - (s[i] in ")]}")
        if depth < 0:
            return None  # the opening bracket closed before the end
        if s[i] == "," and depth == 0:
            parts.append(s[start:i])
            start = i + 1
    parts.append(s[start:-1])
    return parts if len(parts) > 1 and depth == 0 else None


def equivalent(predicted: str | None, reference: str) -> bool:
    """Same answer?

    Equal as numbers when both are numbers; part by part when both are tuples or
    intervals of the same shape, so ``(2.2, 0.4, 5)`` matches
    ``(\\frac{11}{5}, \\frac{2}{5}, 5)``; otherwise equal as normalised text.
    """
    if predicted is None:
        return False
    a, b = to_fraction(predicted), to_fraction(reference)
    if a is not None and b is not None:
        return abs(a - b) <= Fraction(1, 10**6) * max(1, abs(b))
    p, r = normalize_answer(predicted), normalize_answer(reference)
    if p == r:
        return True
    ps, rs = _elements(p), _elements(r)
    if ps and rs and len(ps) == len(rs) and (p[0], p[-1]) == (r[0], r[-1]):  # [1,2) is not (1,2]
        return all(equivalent(x, y) for x, y in zip(ps, rs, strict=True))
    return False


def grade(problem: Problem, response: str) -> dict:
    """Score one response the way the paper's ``evaluate.py`` does.

    The answer is the last ``\\boxed{}`` after ``</think>``.  If the reasoning was
    never closed the *whole text* is searched, so a box written mid-thought still
    counts — lenient, but it is the published protocol, and ``finished`` records
    separately whether the model actually got out of its reasoning.
    """
    reasoning, answer = split_thinking(response)
    predicted = last_boxed(answer if answer is not None else reasoning)
    return {
        "finished": answer is not None,
        "predicted": predicted,
        "correct": equivalent(predicted, problem.answer),
    }


# --------------------------------------------------------------------------------------
# Was the model stuck in a loop?
# --------------------------------------------------------------------------------------


def repetition(tokens: Sequence[int], tail: int = 512, n: int = 16) -> float:
    """Fraction of the ``n``-grams in the last ``tail`` tokens that occurred earlier.

    Healthy reasoning restates a formula now and then, but rarely reproduces 16
    consecutive tokens; a model caught in a loop reproduces little else.  The
    measure is over token ids, so it does not depend on how the text is detokenised.
    """
    if len(tokens) < n + 1:
        return 0.0
    first_seen: dict[tuple[int, ...], int] = {}
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    for i, gram in enumerate(grams):
        first_seen.setdefault(gram, i)
    recent = range(max(0, len(grams) - tail), len(grams))
    return sum(first_seen[grams[i]] < i for i in recent) / len(recent)


def loop_span(tokens: Sequence[int], n: int = 16) -> tuple[int, int] | None:
    """``(first, start)`` if the sequence ends in a loop, else ``None``.

    ``start`` is the first index from which every ``n``-gram has been seen before:
    everything after it is repetition.  ``first`` is where the text at ``start`` was
    written the first time, so ``tokens[first:start]`` is the *same passage* the loop
    keeps repeating, at the one moment the model wrote it without copying — a
    matched control for "what is different inside a loop?".
    """
    if len(tokens) < 2 * n:
        return None
    first_seen: dict[tuple[int, ...], int] = {}
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    for i, gram in enumerate(grams):
        first_seen.setdefault(gram, i)
    start = len(grams)
    while start > 0 and first_seen[grams[start - 1]] < start - 1:
        start -= 1
    if len(grams) - start < n:
        return None
    return first_seen[grams[start]], start


def loop_start(tokens: Sequence[int], n: int = 16) -> int | None:
    """Index where the final loop begins (see :func:`loop_span`), or ``None``."""
    span = loop_span(tokens, n)
    return span[1] if span else None

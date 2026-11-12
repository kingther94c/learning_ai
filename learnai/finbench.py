"""Finance questions as a measuring stick: FinQA, CFA-Easy and FinanceIQ.

``learnai.mathbench`` scores a model on arithmetic word problems.  Choosing a
model *for finance* needs questions that look like the job — and no single public
set does, so this module draws on three with different demands:

====================  ====  ===============================================  =====
source                lang  what one question asks                           items
====================  ====  ===============================================  =====
``finqa``             en    read a 10-K excerpt (text + table), compute      1,147
                            a number — FinQA (Chen et al., 2021), test split
``cfa-easy``          en    a CFA level-I style question, three options —     1,032
                            CFA-Easy from Salesforce's FinEval
``financeiq``         zh    a question from one of ten Chinese professional   ~7,000
                            exams (CPA, securities, actuary, ...), four
                            options — FinanceIQ (Du Xiaoman)
====================  ====  ===============================================  =====

As in ``mathbench`` everything here is plain Python with no model in sight: the
**problems** with their reference answers (:func:`sample`), the **prompt** each one
is asked with (:attr:`Problem.prompt`), and the **grader** (:func:`grade`).

Two grading decisions deserve a word.  First, every model is asked to end with its
answer in ``\\boxed{}``; an answer that is not there scores zero, *and* the run
records that it was missing, because "does not follow the format" is itself one of
the behaviours worth measuring.  Second, FinQA's references are the exact result
of an arithmetic program (``0.14464`` for "what percentage ...?") while a model
answers the way a person does ("14.5%"): a number is accepted when it is within 1%
of the reference, or equals the reference *rounded to the precision the model
chose* — "14%" is right at zero decimals, "15%" is not — and for a question about a
percentage the model may answer in percent or as a ratio.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import random
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from learnai.env import CACHE_DIR

BENCH_DIR = CACHE_DIR / "finbench"

FINANCEIQ_DOMAINS = (
    "注册会计师（CPA）",
    "证券从业资格",
    "基金从业资格",
    "期货从业资格",
    "银行从业资格",
    "税务师",
    "精算师-金融数学",
    "经济师",
    "理财规划师",
    "保险从业资格CICE",
)

_HF = "https://huggingface.co/datasets"
SOURCES: dict[str, dict[str, str]] = {  # source -> {local file name: url}
    "finqa": {
        "finqa-test.json": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json"
    },
    "cfa-easy": {
        "cfa-easy-test.parquet": f"{_HF}/Salesforce/FinEval/resolve/main/CFA-Easy/test-00000-of-00001.parquet"
    },
    "financeiq": {
        f"financeiq-{d}.csv": f"{_HF}/Duxiaoman-DI/FinanceIQ/resolve/main/data/test/{urllib.parse.quote(d)}.csv"
        for d in FINANCEIQ_DOMAINS
    },
}

# The instruction every prompt ends with, in the language of the question.
BOXED_EN = "Please reason step by step, and put your final answer within \\boxed{}."
BOXED_ZH = "请一步步推理，最后把你的答案放在 \\boxed{} 中。"


@dataclass(frozen=True)
class Problem:
    uid: str  # "finqa/12", "cfa-easy/7", "financeiq/税务师/3" — index in the source's own order
    source: str
    domain: str  # the exam, for financeiq; "" otherwise
    prompt: str
    answer: str  # the reference: a letter, a number, or yes/no
    kind: str  # "choice" | "number" | "percent" | "yesno"


# --------------------------------------------------------------------------------------
# Fetching and rendering the sources
# --------------------------------------------------------------------------------------


def _offline() -> bool:
    return os.environ.get("LEARNAI_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}


def fetch(source: str) -> dict[str, Path]:
    """Local copies of a source's files, downloaded into ``data/cache/finbench`` once."""
    paths = {}
    for name, url in SOURCES[source].items():
        path = BENCH_DIR / name
        if not path.exists():
            if _offline():
                raise RuntimeError(f"LEARNAI_OFFLINE is set and {path} is not downloaded")
            BENCH_DIR.mkdir(parents=True, exist_ok=True)
            request = urllib.request.Request(url, headers={"User-Agent": "learnai"})
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            path.write_bytes(data)
        paths[name] = path
    return paths


def markdown_table(rows: list[list[str]]) -> str:
    """A FinQA table (first row = header) as a Markdown table the model can read."""
    width = max(len(r) for r in rows)
    cells = [[c.strip() for c in r] + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(cells[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in cells[1:]]
    return "\n".join(lines)


def finqa_prompt(item: dict) -> str:
    """The report excerpt as FinQA gives it: text before the table, the table, text after."""
    pre = " ".join(s.strip() for s in item["pre_text"] if s.strip())
    post = " ".join(s.strip() for s in item["post_text"] if s.strip())
    return (
        "Read this excerpt from a company's annual report and answer the question.\n\n"
        f"{pre}\n\n{markdown_table(item['table'])}\n\n{post}\n\n"
        f"Question: {item['qa']['question']}\n"
        "Answer with a single number in the same units as the report (no currency symbol; "
        "give a percentage like 12.3%), or yes/no. " + BOXED_EN
    )


def finqa_kind(item: dict) -> str:
    answer = item["qa"]["exe_ans"]
    if isinstance(answer, str):
        return "yesno"
    question = item["qa"]["question"].lower()
    # "percent", "percentage", "%" — and "rate": a tax rate or growth rate is a percentage
    # whether or not the question says so, and models answer "21.3" for 0.213.
    return "percent" if re.search(r"percent|%|\brate\b", question) else "number"


def choice_prompt(question: str, options: dict[str, str], lang: str, domain: str = "") -> str:
    """A multiple-choice question with its options on separate lines."""
    listed = "\n".join(f"{k}. {v}" for k, v in options.items())
    if lang == "zh":
        head = f"以下是{domain}考试的一道单项选择题，请选出唯一正确的选项。"
        return f"{head}\n\n{question}\n{listed}\n\n{BOXED_ZH}（只写选项字母）"
    head = "The following is a CFA-style multiple-choice question with exactly one correct option."
    return f"{head}\n\n{question}\n{listed}\n\n{BOXED_EN} Write only the letter of your choice."


def parse_cfa_text(text: str) -> tuple[str, dict[str, str]]:
    """CFA-Easy packs a question as ``Q:...,CHOICES: A: ...,B: ...,C: ...``; take it apart."""
    question, _, choices = text.partition(",CHOICES: ")
    question = question.removeprefix("Q:").strip()
    options = {}
    for part in re.split(r",(?=[A-D]: )", choices.strip()):
        letter, _, option = part.partition(": ")
        options[letter.strip()] = option.strip()
    return question, options


def all_problems(source: str) -> list[Problem]:
    """Every problem of a source, in the source's own order (downloading it if needed)."""
    files = fetch(source)
    if source == "finqa":
        items = json.loads(files["finqa-test.json"].read_text())
        out = []
        for i, it in enumerate(items):
            it = {**it, "qa": it.get("qa") or it["qa_0"]}  # a few items number their questions
            out.append(
                Problem(
                    f"finqa/{i}",
                    source,
                    "",
                    finqa_prompt(it),
                    str(it["qa"]["exe_ans"]),
                    finqa_kind(it),
                )
            )
        return out
    if source == "cfa-easy":
        import pyarrow.parquet as pq

        rows = pq.read_table(files["cfa-easy-test.parquet"]).to_pylist()
        out = []
        for i, row in enumerate(rows):
            question, options = parse_cfa_text(row["text"])
            prompt = choice_prompt(question, options, "en")
            out.append(
                Problem(
                    f"cfa-easy/{i}", source, "", prompt, row["answer"].strip().upper(), "choice"
                )
            )
        return out
    if source == "financeiq":
        out = []
        for domain in FINANCEIQ_DOMAINS:
            text = files[f"financeiq-{domain}.csv"].read_text(encoding="utf-8")
            for i, row in enumerate(csv.DictReader(io.StringIO(text))):
                options = {k: row[k].strip() for k in "ABCD"}
                prompt = choice_prompt(row["Question"].strip(), options, "zh", domain)
                out.append(
                    Problem(
                        f"financeiq/{domain}/{i}",
                        source,
                        domain,
                        prompt,
                        row["Answer"].strip().upper(),
                        "choice",
                    )
                )
        return out
    raise KeyError(f"unknown source {source!r}; one of {', '.join(SOURCES)}")


# --------------------------------------------------------------------------------------
# Choosing which problems to ask
# --------------------------------------------------------------------------------------


def shuffled(problems: list[Problem], seed: int) -> list[Problem]:
    """One fixed random order per source, so that ``offset`` slices never overlap.

    FinanceIQ is interleaved across its ten exams (each shuffled on its own), so any
    prefix is balanced: the first 50 hold 5 per exam, the next 100 hold 10.
    """
    domains: dict[str, list[Problem]] = {}
    for p in problems:
        domains.setdefault(p.domain, []).append(p)
    for group in domains.values():
        random.Random(seed).shuffle(group)
    longest = max(len(g) for g in domains.values())
    return [g[i] for i in range(longest) for g in domains.values() if i < len(g)]


def sample(source: str, n: int, offset: int = 0, seed: int = 0) -> list[Problem]:
    """``n`` problems of a source starting ``offset`` into its fixed shuffled order."""
    return shuffled(all_problems(source), seed)[offset : offset + n]


def load_problems(n_each: int, offset: int = 0, seed: int = 0) -> list[Problem]:
    """``n_each`` problems from every source: the screening set is ``load_problems(50)``."""
    return [p for source in SOURCES for p in sample(source, n_each, offset, seed)]


def problem_for(uid: str) -> Problem:
    """Look a stored row's problem up again from its uid."""
    source = uid.split("/")[0]
    return {p.uid: p for p in all_problems(source)}[uid]


# --------------------------------------------------------------------------------------
# Reading the answer out of a response
# --------------------------------------------------------------------------------------

# How each model family closes its reasoning.  Qwen (and the DeepSeek-R1 distills
# DianJin is built like) write `</think>`; gpt-oss's harmony format switches from the
# `analysis` channel to a `final` one; Gemma 4 writes its thought in a channel
# `<|channel>thought ... <channel|>`.  Everything before the marker is reasoning,
# everything after it is the answer.
REASONING_END = ("</think>", "<|channel|>final<|message|>", "<channel|>")
REASONING_START = ("<think>", "<|channel|>analysis<|message|>", "<|channel>thought")


def split_thinking(text: str, thinking: bool = True) -> tuple[str, str | None]:
    """``(reasoning, answer)``; ``answer`` is ``None`` if the reasoning never closed.

    ``thinking`` says whether the model was asked to reason first (its template then
    usually opens the reasoning inside the prompt, so the response holds only the
    closing marker).  A model answering without reasoning has given a bare answer.
    """
    for marker in REASONING_END:
        reasoning, tag, answer = text.partition(marker)
        if tag:
            return reasoning, answer
    if thinking or any(m in text for m in REASONING_START):
        return text, None
    return "", text


def last_boxed(text: str) -> str | None:
    """The contents of the last ``\\boxed{...}`` in ``text``, braces matched."""
    start = text.rfind("\\boxed")
    if start < 0:
        return None
    open_brace = text.find("{", start)
    if open_brace < 0 or text[start:open_brace].strip() != "\\boxed":
        rest = text[start:].split(maxsplit=2)  # "\boxed B"
        return rest[1].strip("$.") if len(rest) > 1 else None
    depth = 0
    for i in range(open_brace, len(text)):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            return text[open_brace + 1 : i]
    return None  # cut off inside the box


def _strip_tex(s: str) -> str:
    s = re.sub(r"\\(?:text|mathrm|textbf|mbox)\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\%", "%")  # keep the percent sign: it changes the meaning of the number
    for junk in ("\\!", "\\,", "\\;", "\\ ", "\\left", "\\right", "$", "\\$"):
        s = s.replace(junk, "")
    return s.strip()


def parse_choice(boxed: str) -> str | None:
    """The option letter in a boxed answer: ``B``, ``\\text{B}``, ``B.``, ``(b) liquidity``."""
    s = _strip_tex(boxed).strip("() .").upper()
    m = re.match(r"^([A-D])(?![A-Z])", s)
    return m[1] if m else None


_NUMBER = r"[-−]?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[-−]?\s*\.\d+"


def parse_number(boxed: str) -> tuple[float, bool, int] | None:
    """``(value, had_percent_sign, decimals)`` for a boxed number, else ``None``.

    Accepts the ways a model writes a figure: ``$1,234.5``, ``-4.4\\%``, ``12.3 million``,
    ``\\approx 0.14``, a fraction ``\\frac{94}{5735}``.  Words such as "million" are
    kept out of the value on purpose: the prompt asks for the report's own units,
    so "12.3 million" when the table is in millions is the number 12.3.
    """
    s = _strip_tex(boxed).replace("−", "-").replace("\\approx", "").replace("≈", "")
    if m := re.fullmatch(r"\s*(-?)\\frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}\s*", s):
        try:
            value = float(m[2]) / float(m[3])
        except (ValueError, ZeroDivisionError):
            return None
        return (-value if m[1] else value, False, 6)
    m = re.search(_NUMBER, s)
    if not m:
        return None
    raw = m[0].replace(",", "").replace(" ", "")
    tail = s[m.end() :].lstrip()
    percent = tail.startswith("%") or "percent" in tail[:12].lower()
    decimals = len(raw.split(".")[1]) if "." in raw else 0
    return float(raw), percent, decimals


def numbers_match(predicted: float, reference: float, decimals: int) -> bool:
    """Within 1% of the reference, or equal to the reference rounded to ``decimals``."""
    if abs(predicted - reference) <= 0.01 * abs(reference) + 1e-9:
        return True
    return decimals <= 6 and math.isclose(
        round(reference, decimals), predicted, rel_tol=1e-9, abs_tol=1e-12
    )


def equivalent(boxed: str | None, problem: Problem) -> bool:
    """Does a boxed answer match the reference, by the rules of the problem's ``kind``?"""
    if boxed is None:
        return False
    if problem.kind == "choice":
        return parse_choice(boxed) == problem.answer
    if problem.kind == "yesno":
        word = _strip_tex(boxed).strip(" .").lower()
        return word == problem.answer.lower()
    parsed = parse_number(boxed)
    if parsed is None:
        return False
    value, percent_sign, decimals = parsed
    reference = float(problem.answer)
    if percent_sign:  # "14.5%" means 0.145 ...
        if numbers_match(value / 100, reference, decimals + 2):
            return True
        # ... unless the reference itself is in percent units (FinQA's programs sometimes
        # multiply by 100: "940%" for a reference of 940).  Only for references of at
        # least 1, so that "0.14%" is never taken for the ratio 0.14464.
        return abs(reference) >= 1 and numbers_match(value, reference, decimals)
    if numbers_match(value, reference, decimals):
        return True
    # A question about a percentage may be answered in percent without the sign.
    return problem.kind == "percent" and numbers_match(value, reference * 100, decimals)


def grade(problem: Problem, response: str, thinking: bool = True) -> dict:
    """Score one response: the last ``\\boxed{}`` after the reasoning, if any.

    ``finished`` records whether the model got out of its reasoning; a box found
    only inside unfinished reasoning still counts (the protocol ``mathbench`` uses).
    """
    reasoning, answer = split_thinking(response, thinking)
    boxed = last_boxed(answer if answer is not None else reasoning)
    return {
        "finished": answer is not None,
        "predicted": boxed,
        "correct": equivalent(boxed, problem),
    }

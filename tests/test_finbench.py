"""Tests for the finance benchmark: prompts, sampling and above all the grader.

Everything runs without a model and without the network; the loader tests use the
downloaded files only when they are already in ``data/cache/finbench``.
"""

from __future__ import annotations

import pytest

from learnai import finbench as fb
from learnai.finbench import Problem, equivalent, grade, parse_choice, parse_number, split_thinking


def problem(answer: str, kind: str) -> Problem:
    return Problem("test/0", "finqa", "", "prompt", answer, kind)


# --------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------


def test_finqa_prompt_renders_text_table_and_question():
    item = {
        "pre_text": ["the table shows revenue .", ""],
        "post_text": ["revenue rose ."],
        "table": [["", "2014", "2015"], ["net revenue", "$ 5735", "$ 5829"]],
        "qa": {"question": "what is the net change in net revenue?", "exe_ans": 94.0},
    }
    prompt = fb.finqa_prompt(item)
    assert "the table shows revenue ." in prompt and "revenue rose ." in prompt
    assert "| net revenue | $ 5735 | $ 5829 |" in prompt
    assert (
        prompt.index("the table shows")
        < prompt.index("| net revenue")
        < prompt.index("revenue rose")
    )
    assert "Question: what is the net change in net revenue?" in prompt
    assert prompt.endswith(fb.BOXED_EN)
    assert fb.finqa_kind(item) == "number"
    assert fb.finqa_kind({"qa": {"question": "what percentage ...", "exe_ans": 0.1}}) == "percent"
    assert (
        fb.finqa_kind({"qa": {"question": "what was the effective tax rate?", "exe_ans": 0.35}})
        == "percent"
    )
    assert (
        fb.finqa_kind({"qa": {"question": "how many shares were repurchased?", "exe_ans": 64.0}})
        == "number"
    )
    assert fb.finqa_kind({"qa": {"question": "did it rise?", "exe_ans": "yes"}}) == "yesno"


def test_cfa_text_is_taken_apart_into_question_and_options():
    text = "Q:The nominal rate is the sum of the real rate and a premium for:,CHOICES: A: maturity.,B: liquidity.,C: expected inflation."
    question, options = fb.parse_cfa_text(text)
    assert question == "The nominal rate is the sum of the real rate and a premium for:"
    assert options == {"A": "maturity.", "B": "liquidity.", "C": "expected inflation."}
    prompt = fb.choice_prompt(question, options, "en")
    assert "\nA. maturity.\nB. liquidity.\nC. expected inflation.\n" in prompt


def test_chinese_choice_prompt_names_the_exam_and_asks_for_a_letter():
    prompt = fb.choice_prompt(
        "下列哪项正确？", {"A": "甲", "B": "乙", "C": "丙", "D": "丁"}, "zh", "税务师"
    )
    assert prompt.startswith("以下是税务师考试")
    assert "\nA. 甲\nB. 乙\nC. 丙\nD. 丁\n" in prompt
    assert fb.BOXED_ZH in prompt


# --------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------


def test_shuffled_order_is_fixed_and_balanced_across_domains():
    problems = [
        Problem(f"financeiq/{d}/{i}", "financeiq", d, "", "A", "choice")
        for d in "xyz"
        for i in range(6)
    ]
    order = fb.shuffled(problems, seed=0)
    assert order == fb.shuffled(problems, seed=0)  # deterministic
    assert order != fb.shuffled(problems, seed=1)
    assert {p.uid for p in order} == {p.uid for p in problems}
    assert [p.domain for p in order[:3]] == ["x", "y", "z"]  # round-robin over the exams
    assert [p.domain for p in order[3:6]] == ["x", "y", "z"]
    first, rest = order[:6], order[6:]
    assert not {p.uid for p in first} & {p.uid for p in rest}


# --------------------------------------------------------------------------------------
# Reading the answer
# --------------------------------------------------------------------------------------


def test_split_thinking_knows_each_family_s_marker():
    assert split_thinking("reasoning</think>\nanswer") == ("reasoning", "\nanswer")
    assert split_thinking(
        "<|channel|>analysis<|message|>hmm<|end|><|start|>assistant<|channel|>final<|message|>42"
    ) == (
        "<|channel|>analysis<|message|>hmm<|end|><|start|>assistant",
        "42",
    )
    assert split_thinking("<|channel>thought\nhmm<channel|>The answer is 4.") == (
        "<|channel>thought\nhmm",
        "The answer is 4.",
    )
    assert split_thinking("still thinking") == ("still thinking", None)  # the cap arrived first
    assert split_thinking("bare answer", thinking=False) == ("", "bare answer")


@pytest.mark.parametrize(
    ("boxed", "letter"),
    [
        ("B", "B"),
        ("\\text{B}", "B"),
        ("B.", "B"),
        ("(c)", "C"),
        ("B. liquidity", "B"),
        ("BC", None),
        ("yes", None),
    ],
)
def test_parse_choice(boxed, letter):
    assert parse_choice(boxed) == letter


@pytest.mark.parametrize(
    ("boxed", "parsed"),
    [
        ("94", (94.0, False, 0)),
        ("$1,234.5", (1234.5, False, 1)),
        ("-4.4\\%", (-4.4, True, 1)),
        ("14.5 \\%", (14.5, True, 1)),
        ("12.3 million", (12.3, False, 1)),
        ("\\approx 0.14", (0.14, False, 2)),
        ("\\frac{94}{5735}", (94 / 5735, False, 6)),
        ("0.145 or 14.5%", (0.145, False, 3)),
        ("no number here", None),
    ],
)
def test_parse_number(boxed, parsed):
    assert parse_number(boxed) == pytest.approx(parsed) if parsed else parse_number(boxed) is None


def test_numbers_within_one_percent_or_rounded_to_the_stated_precision():
    ratio = problem("0.14464", "percent")
    assert equivalent("0.1446", ratio)  # 1% tolerance
    assert equivalent("14.5\\%", ratio)  # percent sign -> /100
    assert equivalent("14\\%", ratio)  # right at zero decimals
    assert not equivalent("15\\%", ratio)
    assert equivalent("14.46", ratio)  # a percentage question may be answered in percent units
    assert not equivalent("1.4", ratio)
    assert not equivalent("0.14\\%", ratio)  # 0.14% is not 14.46%
    in_percent_units = problem("940.0", "percent")  # FinQA sometimes stores the percent itself
    assert equivalent("940\\%", in_percent_units) and equivalent("940", in_percent_units)
    assert not equivalent("9.4", in_percent_units)
    plain = problem("94.0", "number")
    assert equivalent("94", plain) and equivalent("$94 million", plain)
    assert not equivalent("94000000", plain)  # a scale slip is wrong
    assert not equivalent("0.94", plain)  # and a plain number is never read as percent units
    assert equivalent(
        "5735", problem("5735.2", "number")
    )  # 1% of 5735 is 57: rounding to units is fine
    assert not equivalent("5600", problem("5735.2", "number"))


def test_yes_no_and_choices():
    assert equivalent("Yes", problem("yes", "yesno")) and not equivalent(
        "no", problem("yes", "yesno")
    )
    choice = Problem("cfa-easy/0", "cfa-easy", "", "", "B", "choice")
    assert (
        equivalent("\\text{B}", choice)
        and not equivalent("A", choice)
        and not equivalent(None, choice)
    )


def test_grade_reads_the_last_box_after_the_reasoning():
    choice = Problem("cfa-easy/0", "cfa-easy", "", "", "B", "choice")
    assert grade(choice, "I think \\boxed{A}... no.</think>\nThe answer is \\boxed{B}.") == {
        "finished": True,
        "predicted": "B",
        "correct": True,
    }
    cut = grade(choice, "I think \\boxed{B} but let me check again and again")
    assert cut == {"finished": False, "predicted": "B", "correct": True}  # the protocol credits it
    assert grade(choice, "done</think>\nThe answer is B.") == {
        "finished": True,
        "predicted": None,
        "correct": False,
    }


# --------------------------------------------------------------------------------------
# The real files, when they are already downloaded
# --------------------------------------------------------------------------------------


def _cached(source: str) -> bool:
    return all((fb.BENCH_DIR / name).exists() for name in fb.SOURCES[source])


@pytest.mark.skipif(not all(map(_cached, fb.SOURCES)), reason="finance sources not downloaded")
def test_screening_set_is_balanced_and_disjoint_from_the_extension():
    screening = fb.load_problems(50)
    extension = fb.load_problems(100, offset=50)
    assert len(screening) == 150 and len(extension) == 300
    assert not {p.uid for p in screening} & {p.uid for p in extension}
    by_exam = {p.domain for p in screening if p.source == "financeiq"}
    assert len(by_exam) == 10
    for p in screening:
        assert fb.problem_for(p.uid) == p
        assert p.kind == ("choice" if p.source != "finqa" else p.kind)

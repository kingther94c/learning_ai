"""Tests for the math-benchmark grader and loop detector (no model, no network)."""

from __future__ import annotations

import pytest

from learnai.mathbench import (
    PROMPT_SUFFIX,
    Problem,
    equivalent,
    grade,
    last_boxed,
    loop_start,
    normalize_answer,
    repetition,
    split_thinking,
    to_fraction,
)

GSM = Problem("gsm8k-test/0", "How many eggs?", "18", "number")
MATH = Problem("math500/0", "Simplify.", "\\frac{3}{4}", "latex")


def test_prompt_asks_for_a_boxed_answer():
    assert GSM.prompt == "How many eggs?" + PROMPT_SUFFIX
    assert "\\boxed{}" in GSM.prompt


def test_split_thinking_separates_reasoning_from_answer():
    assert split_thinking("think hard</think>\n\nThe answer") == ("think hard", "\n\nThe answer")
    # never closing the tag means the model ran out of tokens mid-thought: no answer exists
    assert split_thinking("think hard, and harder, and") == ("think hard, and harder, and", None)


def test_last_boxed_matches_braces_and_takes_the_last_box():
    assert last_boxed("first \\boxed{1}, finally \\boxed{\\frac{3}{4}}.") == "\\frac{3}{4}"
    assert last_boxed("nested \\boxed{\\sqrt{\\frac{a}{b}}}") == "\\sqrt{\\frac{a}{b}}"
    assert last_boxed("no box here") is None
    assert last_boxed("cut off inside \\boxed{12") is None


@pytest.mark.parametrize(
    ("predicted", "reference"),
    [
        ("\\dfrac12", "\\frac{1}{2}"),
        ("0.5", "\\frac{1}{2}"),
        (".5", "\\frac12"),
        ("1/2", "\\frac{1}{2}"),
        ("1,234", "1234"),
        ("18.0", "18"),
        ("\\$18", "18"),
        ("x=5", "5"),
        ("90^\\circ", "90"),
        ("3\\sqrt2", "3\\sqrt{2}"),
        ("\\text{(B)}", "(B)"),
        ("\\left( 3, \\frac{\\pi}{2} \\right)", "(3,\\frac{\\pi}{2})"),
        ("-\\frac{2}{3}", "\\frac{-2}{3}"),
        # a unit after the number: three real misses from the 4B's GSM8K answers
        ("160 \\text{ minutes}", "160"),
        ("5 \\text{ hours}", "5"),
        ("10 \\text{ mph}", "10"),
        ("\\frac{3}{4} \\text{ cup}", "0.75"),
        ("25 \\text{ cm}^2", "25"),
        ("12\\text{ inches}.", "12 \\text{ inches}"),
    ],
)
def test_typographic_variants_are_the_same_answer(predicted, reference):
    assert equivalent(predicted, reference)


@pytest.mark.parametrize(
    ("predicted", "reference"),
    [
        ("7", "8"),
        ("\\frac{1}{3}", "0.5"),
        ("(1,2)", "(2,1)"),
        (None, "5"),
        ("x+1", "x-1"),
        # text in the middle is part of the answer, not a unit to drop
        ("80 \\text{ miles and } 150 \\text{ miles}", "230"),
        ("80 \\text{ miles and } 150 \\text{ miles}", "80"),
        ("3 \\text{ or } 5", "3"),
        ("\\text{(B)}", "\\text{(C)}"),
    ],
)
def test_different_answers_stay_different(predicted, reference):
    assert not equivalent(predicted, reference)


def test_an_ordered_pair_is_not_mistaken_for_a_thousands_separator():
    assert normalize_answer("(1,234)") == "(1,234)"
    assert to_fraction("(1,234)") is None
    assert to_fraction("1,234") == 1234


def test_tuples_and_intervals_are_compared_part_by_part():
    # A real miss from the 4B's answers: right point, decimals instead of fractions.
    reference = "\\left( \\frac{11}{5}, \\frac{2}{5}, 5 \\right)"
    assert equivalent("(2.2, 0.4, 5)", reference)
    assert not equivalent("(2.2, 0.5, 5)", reference)
    assert not equivalent("(2.2, 0.4)", reference)  # a different number of parts
    assert not equivalent("(0.4, 2.2, 5)", reference)  # order matters

    assert equivalent("[0.5, 3)", "[\\frac{1}{2}, 3)")
    assert not equivalent("(0.5, 3]", "[\\frac{1}{2}, 3)")  # the brackets are part of an interval
    # only top-level commas separate: the inner pair is one part
    assert equivalent("(0.5, (1, 2))", "(\\frac12, (1, 2))")
    assert not equivalent("(1,2)+(3,4)", "(1,2)+(3,5)")  # a sum, not a tuple


def test_grade_reads_the_last_box_after_the_reasoning():
    done = grade(GSM, "16 - 3 - 4 = 9; maybe \\boxed{9}?</think>\n\nShe makes \\boxed{18} dollars.")
    assert done == {"finished": True, "predicted": "18", "correct": True}
    assert grade(MATH, "ok</think>So \\boxed{\\dfrac34}.")["correct"]
    # an answer stated without the box is not an answer under this protocol
    assert not grade(GSM, "ok</think>She makes 18 dollars a day.")["correct"]


def test_grade_follows_the_paper_for_unfinished_reasoning():
    """No </think>: the whole text is searched, and `finished` says the model never got out."""
    stuck = grade(GSM, "Let me check. 16 - 3 - 4 = 9, 9 * 2 = 18. Let me check. 16 - 3")
    assert stuck == {"finished": False, "predicted": None, "correct": False}

    boxed_mid_thought = grade(GSM, "So \\boxed{18}. But wait. So \\boxed{18}. But wait. So")
    assert boxed_mid_thought == {"finished": False, "predicted": "18", "correct": True}


def test_repetition_is_zero_for_fresh_text_and_one_for_a_loop():
    assert repetition(list(range(2000))) == 0.0
    assert repetition([1, 2, 3, 4, 5] * 400) == 1.0
    # 300 fresh tokens followed by a long loop: the tail is all repeats
    assert repetition(list(range(300)) + [7, 8, 9] * 400) == 1.0
    assert repetition([1, 2, 3]) == 0.0  # too short to say


def test_loop_start_finds_where_the_last_original_thought_ends():
    fresh = list(range(100, 400))
    cycle = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    start = loop_start(fresh + cycle * 50)
    # the first pass through the cycle is new; from the second pass on, nothing is
    assert start == len(fresh) + len(cycle)
    assert loop_start(fresh) is None

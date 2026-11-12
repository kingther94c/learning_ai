"""Tests for the bookkeeping of the weight-editing experiment (no model needed)."""

from __future__ import annotations

import json

import numpy as np

import tools.weight_edit_experiment as exp
from learnai.weightedit import Diagnosis


def _row(
    uid: str,
    correct: bool,
    hit_limit: bool = False,
    repetition: float = 0.0,
    closed: bool | None = None,
) -> dict:
    """One stored answer.  ``closed``: did it close its reasoning (default: iff not cut off)."""
    return {
        "uid": uid,
        "correct": correct,
        "finished": (not hit_limit) if closed is None else closed,
        "hit_limit": hit_limit,
        "repetition": repetition,
        "n_tokens": 100,
        "tokens": [],
    }


def test_wilson_interval_is_honest_at_the_edges():
    lo, hi = exp.wilson(0, 20)
    assert lo == 0.0 and 0.15 < hi < 0.17  # "0 of 20" is not "never"
    lo, hi = exp.wilson(20, 20)
    assert hi == 1.0 and 0.83 < lo < 0.85
    lo, hi = exp.wilson(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60


def test_a_loop_is_a_capped_answer_that_repeats_itself():
    assert exp.is_loop(_row("a", False, hit_limit=True, repetition=0.9))
    # ran out of tokens, but was still thinking new thoughts: long, not looping
    assert not exp.is_loop(_row("a", False, hit_limit=True, repetition=0.1))
    assert not exp.is_loop(_row("a", True, hit_limit=False, repetition=0.9))


def test_summary_sorts_every_answer_into_right_wrong_or_cut_off():
    rows = [
        _row("a", True),
        _row("b", False),  # finished, wrong
        _row("c", False, hit_limit=True, repetition=0.9),  # a verbatim loop
        _row("d", False, hit_limit=True, repetition=0.0),  # cut off, still thinking
    ]
    text = exp.summarize(rows, "run")
    assert "n=4  correct 1 (25.0%" in text
    assert "finished right 1  finished wrong 1" in text
    assert "cut off 2 (50.0%: 2 in the reasoning, 0 in the answer; verbatim loops 1)" in text
    assert "abandoned 0" in text


def test_every_answer_is_counted_in_exactly_one_state():
    # The awkward cases real runs produce.  An answer can close its reasoning and
    # *still* be cut off (it ran out of tokens while writing the answer), so
    # "closed its reasoning" and "hit the limit" are not opposites.
    rows = [
        _row("a", True),
        _row("b", False, hit_limit=True, closed=True),  # cut off in the answer itself
        _row("c", True, hit_limit=True, closed=False),  # boxed the result, never stopped
        _row("d", False, hit_limit=False, closed=False),  # gave up without an answer
    ]
    assert [exp.outcome(r) for r in rows] == ["right", "cut", "cut", "abandoned"]

    text = exp.summarize(rows, "run")
    assert "correct 2 (50.0%" in text and "1 of them cut off after boxing the answer" in text
    assert "finished right 1  finished wrong 0" in text
    assert "cut off 2 (50.0%: 1 in the reasoning, 1 in the answer; verbatim loops 0)" in text
    assert "abandoned 1" in text


def test_editable_layers_takes_the_largest_mlp_shares_but_never_the_last_block():
    n_layers, d_model = 5, 4
    resid = np.zeros((2, n_layers, d_model))
    write = np.zeros((2, n_layers, d_model))
    resid[0, :, 0] = np.arange(1, n_layers + 1)  # the two conditions differ along e0
    write[0, :, 0] = [0.1, 0.9, 0.2, 0.7, 5.0]  # what each block's MLP adds along e0
    diagnosis = Diagnosis(resid, write, np.zeros((2, n_layers, 3)))

    assert [round(r["mlp"], 3) for r in diagnosis.layer_table()] == [0.1, 0.9, 0.2, 0.7, 5.0]
    assert exp.editable_layers(diagnosis) == [1, 3]  # block 4 is larger, but it is the last
    assert exp.editable_layers(diagnosis, how_many=3) == [1, 2, 3]


def test_compare_counts_the_problems_an_edit_fixed_and_broke(tmp_path, capsys):
    base = [_row(f"p/{i}", i < 6) for i in range(10)]  # problems 0-5 right
    edited = [_row(f"p/{i}", i < 5 or i >= 8) for i in range(10)]  # loses 5, gains 8 and 9
    for name, rows in (("base", base), ("edited", edited)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    exp.main(["compare", str(tmp_path / "base.jsonl"), str(tmp_path / "edited.jsonl")])
    out = capsys.readouterr().out
    assert "correct 6 (60.0%" in out and "correct 7 (70.0%" in out
    assert "2 fixed, 1 broken" in out
    assert "p = 1.000" in out  # 2 against 1 is no evidence of anything
    assert "tokens generated: 1,000 -> 1,000 (+0%)" in out


def test_compare_follows_up_the_answers_the_base_model_never_finished(tmp_path, capsys):
    base = [_row(f"p/{i}", False, hit_limit=True) for i in range(4)]  # all four cut off
    edited = [
        _row("p/0", True),  # now finishes, right
        _row("p/1", False),  # now finishes, wrong
        _row("p/2", False, hit_limit=True),  # still cut off
        _row("p/3", True),
    ]
    for name, rows in (("base", base), ("edited", edited)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    exp.main(["compare", str(tmp_path / "base.jsonl"), str(tmp_path / "edited.jsonl")])
    out = capsys.readouterr().out
    assert (
        "of 4 answers the base model did not finish: 2 now finish right, 1 finish wrong, "
        "1 are still cut off" in out
    )
    assert "abandoned" not in out.splitlines()[-2]  # only mentioned when it happens


def test_compare_does_not_call_an_answer_finished_when_it_was_cut_off_again(tmp_path, capsys):
    base = [_row(f"p/{i}", False, hit_limit=True) for i in range(3)]
    edited = [
        _row("p/0", False, hit_limit=True, closed=True),  # closed its reasoning, then ran out
        _row("p/1", False, hit_limit=False, closed=False),  # stopped without answering
        _row("p/2", True),
    ]
    for name, rows in (("base", base), ("edited", edited)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    exp.main(["compare", str(tmp_path / "base.jsonl"), str(tmp_path / "edited.jsonl")])
    out = capsys.readouterr().out
    assert (
        "of 3 answers the base model did not finish: 1 now finish right, 0 finish wrong, "
        "1 are still cut off, 1 are abandoned" in out
    )

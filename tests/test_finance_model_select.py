"""Tests for the bookkeeping of the finance model comparison (no model needed)."""

from __future__ import annotations

import json

import pytest

import tools.finance_model_select as fms


def _row(
    uid: str,
    correct: bool,
    *,
    source: str = "cfa-easy",
    hit_limit: bool = False,
    repetition: float = 0.0,
    predicted: str | None = "B",
    kind: str = "choice",
    gold: str = "B",
    n_tokens: int = 100,
) -> dict:
    return {
        "uid": uid,
        "source": source,
        "domain": "",
        "kind": kind,
        "gold": gold,
        "finished": not hit_limit,
        "predicted": predicted,
        "correct": correct,
        "hit_limit": hit_limit,
        "prompt_tokens": 300,
        "n_tokens": n_tokens,
        "repetition": repetition,
        "text": "",
        "tokens": [],
    }


def test_every_family_fits_the_budget_at_the_cap():
    """The batch sizes are the budget made executable: worst case under the process limit."""
    for key, family in fms.FAMILIES.items():
        assert family.sampling.max_tokens == fms.CAP
        if key == "gpt-oss-20b":  # 11.3 GiB of weights: documented as over a 12 GiB budget
            assert family.peak_gib() > fms.PROCESS_LIMIT_GIB
            assert family.peak_gib() <= 16
            continue
        assert family.peak_gib() <= fms.PROCESS_LIMIT_GIB, key
        assert family.peak_gib(family.batch + 1) > fms.PROCESS_LIMIT_GIB - 0.6, (
            key
        )  # not timid either


def test_generate_refuses_a_batch_over_the_limit(monkeypatch, tmp_path, capsys):
    assert fms.main(["generate", "--model", "gpt-oss-20b", "--out", str(tmp_path / "x.jsonl")]) == 2
    assert "over the 12" in capsys.readouterr().out
    assert (
        fms.main(
            [
                "generate",
                "--model",
                "qwen3.5-4b",
                "--batch",
                "40",
                "--out",
                str(tmp_path / "x.jsonl"),
            ]
        )
        == 2
    )


def test_outcome_puts_every_answer_in_exactly_one_state():
    assert fms.outcome(_row("a", True)) == "right"
    assert fms.outcome(_row("a", False)) == "wrong"
    assert fms.outcome(_row("a", False, predicted=None)) == "no-box"
    assert fms.outcome(_row("a", False, hit_limit=True)) == "cut"
    assert (
        fms.outcome(_row("a", True, hit_limit=True)) == "cut"
    )  # credited by the protocol, still cut


def test_classify_separates_loops_cuts_format_and_scale_slips():
    assert fms.classify(_row("a", True)) == "right"
    assert fms.classify(_row("a", False, hit_limit=True, repetition=0.9)) == "loop"
    assert fms.classify(_row("a", False, hit_limit=True, repetition=0.1)) == "cut"
    assert fms.classify(_row("a", False, predicted=None)) == "no-box"
    number = dict(source="finqa", kind="number", gold="94.0")
    assert fms.classify(_row("a", False, predicted="94000", **number)) == "scale"
    assert fms.classify(_row("a", False, predicted="0.94", **number)) == "scale"
    assert fms.classify(_row("a", False, predicted="1171", **number)) == "wrong"


def test_summary_reports_per_source_accuracy_and_the_misses():
    rows = [
        _row("cfa-easy/1", True),
        _row("cfa-easy/2", False),
        _row(
            "finqa/1",
            False,
            source="finqa",
            hit_limit=True,
            repetition=0.8,
            kind="number",
            gold="1",
        ),
        _row("financeiq/x/1", False, source="financeiq", predicted=None),
    ]
    text = fms.summarize(
        rows,
        "run",
        {"sessions": [{"wall_seconds": 600, "generated_tokens": 60000, "peak_memory_gib": 9.5}]},
    )
    assert "cfa-easy   1/2" in text and "finqa      0/1" in text and "financeiq  0/1" in text
    assert "all        1/4" in text
    assert "1 right, 1 wrong, 1 no box, 1 cut by the cap (1 looping)" in text
    assert "10 min wall, 60,000 tokens, 100 tok/s batched, peak 9.5 GiB" in text


def test_compare_counts_fixed_and_broken_problems(tmp_path, capsys):
    base = tmp_path / "base.jsonl"
    edited = tmp_path / "edited.jsonl"
    base.write_text(
        "".join(
            json.dumps(r) + "\n"
            for r in [_row("a", True), _row("b", False), _row("c", False), _row("d", True)]
        )
    )
    edited.write_text(
        "".join(
            json.dumps(r) + "\n"
            for r in [_row("a", True), _row("b", True), _row("c", True), _row("d", False)]
        )
    )
    assert fms.main(["compare", str(base), str(edited)]) == 0
    out = capsys.readouterr().out
    assert "base 2/4" in out and "edited 3/4" in out
    assert "fixed 2, broke 1" in out
    assert "fixed: b, c" in out and "broke: d" in out


def test_compare_reads_forced_runs_which_keep_only_verdicts(tmp_path, capsys):
    """``force`` writes uid, source, gold and the verdict, nothing about how it was generated."""
    runs = []
    for name, verdicts in (("f16k", (True, True, False)), ("f8k", (True, False, False))):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(
            "".join(
                json.dumps(
                    {"uid": f"futures-exec/stop/{i}", "source": "futures-exec", "correct": ok}
                )
                + "\n"
                for i, ok in enumerate(verdicts)
            )
        )
        runs.append(str(path))
    assert fms.main(["compare", *runs]) == 0
    out = capsys.readouterr().out
    assert "fixed 0, broke 1" in out and "futures-exec    2 ->    1 of 3" in out
    assert "loops" not in out and "tokens" not in out


def test_default_output_path_names_the_slice_and_the_patch(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(fms, "cmd_generate", lambda args: seen.update(out=str(args.out)) or 0)
    monkeypatch.setattr(fms, "RUNS_DIR", tmp_path)
    fms.main(["generate", "--model", "qwen3.5-9b", "--n", "100", "--offset", "50"])
    assert seen["out"] == str(tmp_path / "finbench" / "qwen3.5-9b" / "n100-o50.jsonl")
    fms.main(["generate", "--model", "qwen3.5-9b", "--patch", "runs/x/patch-L15.npz"])
    assert seen["out"].endswith("n50-o0-patch-L15.jsonl")
    fms.main(["generate", "--model", "qwen3.5-4b", "--bench", "macro", "--system", "p/desk.txt"])
    assert seen["out"].endswith(
        "qwen3.5-4b/macro-n50-o0-desk.jsonl"
    )  # a playbook run is its own run


class _CharTokenizer:
    """One token per character: enough to check what the answer-now context contains."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens) -> str:
        return "".join(chr(t) for t in tokens)

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, **template):
        turns = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return self.encode(f"{turns}<assistant><think>\n")


def test_answer_now_context_cuts_the_reasoning_at_the_cap_and_asks_for_the_answer(monkeypatch):
    class _Problem:
        prompt = "Q?"

    monkeypatch.setattr(fms, "problem_for", lambda uid: _Problem())
    tok = _CharTokenizer()
    family = fms.FAMILIES["qwen3.5-4b"]
    row = {"uid": "finqa/1", "tokens": tok.encode("let me think about this")}

    context = fms.answer_now_context(tok, family, row, cap=12)
    assert tok.decode(context) == "<user>Q?<assistant><think>\nlet me think" + fms.FORCE_CUE

    finished = {"uid": "finqa/1", "tokens": tok.encode("short.</think>\n\\boxed{7}")}
    assert fms.answer_now_context(tok, family, finished, cap=20) is None  # had already answered

    # a run made with a playbook is forced with the same playbook in front of the question
    briefed = fms.answer_now_context(tok, family, row, cap=12, system="Rules.")
    assert tok.decode(briefed).startswith("<system>Rules.<user>Q?<assistant>")


def test_the_playbook_is_the_system_message_ahead_of_the_question():
    class _Problem:
        prompt = "Q?"

    tok, family = _CharTokenizer(), fms.FAMILIES["qwen3.5-4b"]
    assert tok.decode(fms.prompt_tokens(tok, family, _Problem())) == "<user>Q?<assistant><think>\n"
    with_brief = tok.decode(fms.prompt_tokens(tok, family, _Problem(), "Desk rules."))
    assert with_brief == "<system>Desk rules.<user>Q?<assistant><think>\n"


def test_close_box_stops_at_the_brace_that_closes_the_cue():
    assert fms.close_box("B} and then more") == ("B}", True)
    assert fms.close_box("\\frac{1}{4}} trailing") == ("\\frac{1}{4}}", True)
    assert fms.close_box("12.5") == ("12.5}", False)


def test_force_summary_counts_both_ways_per_source():
    records = [
        {"source": "finqa", "forced": True, "correct_unforced": False, "correct": True},
        {"source": "finqa", "forced": False, "correct_unforced": True, "correct": True},
        {"source": "cfa-easy", "forced": True, "correct_unforced": False, "correct": False},
    ]
    text = fms.force_summary(records, "x")
    assert "3 problems, 2 answered at the cap" in text
    assert "cut-offs scored as they stand: 1" in text
    assert "answer forced at the cap:      2" in text
    assert "finqa         1 ->    2 of 2" in text


def test_still_thinking_at_counts_a_cut_off_at_the_runs_own_cap():
    assert fms.still_thinking_at({"n_tokens": 5000, "hit_limit": False}, 4096)
    assert not fms.still_thinking_at({"n_tokens": 3000, "hit_limit": False}, 4096)
    assert fms.still_thinking_at({"n_tokens": 16384, "hit_limit": True}, 16384)
    assert not fms.still_thinking_at(
        {"n_tokens": 16384, "hit_limit": False}, 16384
    )  # ended on the last token


def test_a_shorter_cap_leaves_room_for_more_sequences():
    dianjin = fms.FAMILIES["dianjin-r1-7b"]
    assert (
        dianjin.peak_gib(16, max_tokens=4096) <= fms.PROCESS_LIMIT_GIB
    )  # caches scale with the cap
    assert dianjin.peak_gib(17, max_tokens=4096) > fms.PROCESS_LIMIT_GIB
    qwen = fms.FAMILIES["qwen3.5-4b"]  # DeltaNet state does not shrink with the cap
    assert qwen.seq_gib(0) == pytest.approx((fms.PROMPT_MAX * 32 + 49 * 1024) / 2**20)


def test_sample_seed_draws_new_samples_of_the_same_problems(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(fms, "load_problems", lambda n, offset, seed: seen.update(pick=seed) or [])
    fms.main(
        [
            "generate",
            "--model",
            "dianjin-r1-7b",
            "--sample-seed",
            "2",
            "--out",
            str(tmp_path / "x.jsonl"),
        ]
    )
    assert seen["pick"] == 0  # the problems are still chosen by --seed

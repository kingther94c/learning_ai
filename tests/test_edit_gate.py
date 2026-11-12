"""The rules that choose and gate a weight edit (tools/edit_gate.py), on synthetic probe tables."""

from __future__ import annotations

import tools.edit_gate as gate_rules

HEADER = (
    "per kind: change in mean log p(the text's own next token) / in mean log p(</think>), nats\n"
    + " " * 24
    + "          real answers         real cut-offs\n"
    + "unedited (log p)             -0.274     -25.85     -0.347     -26.71\n"
)


def write(tmp_path, rows: list[str]):
    path = tmp_path / "probe.txt"
    path.write_text(HEADER + "\n".join(rows) + "\n")
    return gate_rules.read_probe(str(path))


def test_blocks_are_ranked_by_their_effect_on_cut_off_text_within_the_damage_limit(tmp_path):
    rows = write(
        tmp_path,
        [
            "block 2 alone                -0.001      +0.10     -0.300      +0.50",  # too early
            "block 5 alone                -0.001      +0.10     -0.010      +0.00",
            "block 6 alone                -0.009      +0.10     -0.200      +0.00",  # hurts answers
            "block 7 alone                -0.002      +0.10     -0.050      +0.00",
            "block 8 alone                -0.000      +0.10     -0.030      +0.00",
            "block 9 alone                -0.000      +0.10     -0.900      +0.00",  # the last block
        ],
    )
    assert gate_rules.pick_blocks(rows) == ([7, 8], [5, 7, 8])


def test_the_gate_passes_only_a_patch_that_moves_the_real_behaviour(tmp_path):
    rows = write(
        tmp_path,
        [
            "patch-a-k6                   -0.006      +0.10     -0.001      +0.10",  # no effect
            "patch-b-k6                   -0.010      +0.10     -0.030      +0.10",  # passes
            "patch-c-k6                   -0.050      +0.10     -0.300      +0.10",  # too costly
            "patch-b-k6-control           -0.000      +0.10     -0.500      +0.90",  # controls never
        ],
    )
    assert gate_rules.gate(rows) == "patch-b-k6"
    rows.pop("patch-b-k6")
    assert gate_rules.gate(rows) is None

"""Fixed rules for choosing and gating a weight edit, so no run is hand-picked.

    python tools/edit_gate.py blocks <per-block probe table> [kind]   -> "top2 top4"
    python tools/edit_gate.py gate   <patch probe table>              -> best patch, or ""

``blocks``: rank blocks by how much an edit of that block alone lowers the
log-probability of ``kind`` (default ``real cut-offs``: the model's own answers that
ran into the token cap), among blocks 4..last-1 whose edit leaves ordinary reasoning
(``real answers``) within 0.005 nats/token.  Prints the top two and top four as
comma-separated layer lists for ``weight_edit_experiment.py edit --layers``.

``gate``: of the candidate patches (controls excluded), the one that lowers the
cut-off text most, provided it lowers it by at least 0.02 nats/token or raises
log p(</think>) there by at least 0.5, and costs ordinary reasoning at most 0.02.
Prints nothing if none qualifies — then the patch does not go on to a generation run.

The tables are the ones ``weight_edit_experiment.py probe`` prints.
"""

from __future__ import annotations

import re
import sys


def read_probe(path: str) -> dict[str, dict[str, tuple[float, float]]]:
    """``candidate -> kind -> (change in log p of the text, change in log p of </think>)``."""
    kinds: list[str] = []
    rows: dict[str, dict[str, tuple[float, float]]] = {}
    for line in open(path, encoding="utf-8").read().splitlines():
        if line.startswith(" " * 20) and not kinds and line.strip():
            kinds = re.split(r"\s{2,}", line.strip())
        elif kinds and (line.startswith("block ") or line.startswith("patch-")):
            m = re.match(r"(.+?)\s+([-+]?\d+\.\d+.*)$", line)
            if not m:
                continue
            values = [float(x) for x in re.findall(r"[-+]?\d+\.\d+", m[2])]
            if len(values) == 2 * len(kinds):
                rows[m[1].strip()] = {
                    k: (values[2 * i], values[2 * i + 1]) for i, k in enumerate(kinds)
                }
    return rows


def pick_blocks(rows, kind: str = "real cut-offs") -> tuple[list[int], list[int]]:
    blocks = {int(name.split()[1]): v for name, v in rows.items() if name.startswith("block ")}
    last = max(blocks)
    ok = [b for b, v in blocks.items() if 4 <= b < last and v["real answers"][0] >= -0.005]
    ranked = sorted(ok, key=lambda b: blocks[b][kind][0])
    return sorted(ranked[:2]), sorted(ranked[:4])


def gate(rows) -> str | None:
    best = None
    for name, v in rows.items():
        if "control" in name or v["real answers"][0] < -0.02:
            continue
        text, think = v["real cut-offs"]
        if (text <= -0.02 or think >= 0.5) and (
            best is None or text < rows[best]["real cut-offs"][0]
        ):
            best = name
    return best


if __name__ == "__main__":
    mode, path = sys.argv[1], sys.argv[2]
    rows = read_probe(path)
    if mode == "blocks":
        top2, top4 = pick_blocks(rows, *(sys.argv[3:4] or ["real cut-offs"]))
        print(",".join(map(str, top2)), ",".join(map(str, top4)))
    else:
        print(gate(rows) or "")

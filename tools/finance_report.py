"""Build the Chinese research report (self-contained HTML) from the finance runs.

    python tools/finance_report.py            # -> docs/finance_model_selection_report.zh.html

Everything numeric in the page comes from ``runs/finbench/`` (git-ignored); the page
embeds the numbers, so a fresh clone reads the report without the runs.  The charts
are inline SVG drawn here; the styles are the repo's own ``interactive/shared/learnai.css``
inlined, so the report and the interactive pages look like one system and the page
works over ``file://`` with no network.  The prose lives in ``tools/finance_report_text.py``.
"""

from __future__ import annotations

import html
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run as a script from anywhere

import tools.finance_model_select as fms  # noqa: E402
from learnai.env import REPO_ROOT, RUNS_DIR  # noqa: E402
from learnai.finbench import SOURCES  # noqa: E402
from tools.finance_report_text import NARRATIVE  # noqa: E402

OUT = REPO_ROOT / "docs" / "finance_model_selection_report.zh.html"
CSS = REPO_ROOT / "interactive" / "shared" / "learnai.css"

MODELS = {  # key -> (display name, series colour index 1..8)
    "dianjin-r1-7b": ("DianJin-R1-7B", 1),
    "qwen3.5-4b": ("Qwen3.5-4B", 2),
    "qwen3.5-9b": ("Qwen3.5-9B", 3),
    "gemma4-12b": ("Gemma 4 12B", 4),
    "claude-fable-5.1-offline": ("Claude 离线参照", 7),
}
SOURCE_NAMES = {
    "finqa": "FinQA（财报数值）",
    "cfa-easy": "CFA-Easy（英文选择）",
    "financeiq": "FinanceIQ（中文考试）",
}


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------


def load_run(path: Path) -> tuple[list[dict], dict]:
    return fms.read_run(path), fms.read_meta(path)


def accuracy(rows: list[dict]) -> tuple[int, int]:
    return sum(r["correct"] for r in rows), len(rows)


def by_source(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[r["source"]].append(r)
    return out


def cap_curve(path: Path, caps: list[int]) -> list[float] | None:
    """Accuracy under smaller caps, re-grading stored answers truncated to each cap."""
    rows, meta = load_run(path)
    if not rows or not rows[0]["tokens"]:
        return None
    from mlx_lm.utils import load_tokenizer

    from learnai.openweights import local_path

    tokenizer = load_tokenizer(local_path(meta["model"]))
    thinking = fms.FAMILIES[meta["model"]].thinking
    return [
        100 * sum(fms.regrade_at(r, tokenizer, c, thinking) for r in rows) / len(rows) for c in caps
    ]


# --------------------------------------------------------------------------------------
# SVG charts (inline, no library)
# --------------------------------------------------------------------------------------


def esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def bar_chart(
    groups: list[str],
    series: list[tuple[str, int, list[tuple[float, float, float]]]],
    ylabel: str = "%",
) -> str:
    """Grouped bars with confidence whiskers.  series: (name, colour index, [(value, lo, hi) per group])."""
    w, h, left, bottom, top = 760, 300, 44, 56, 16
    gw = (w - left - 10) / len(groups)
    bw = min(26, gw / (len(series) + 1))
    y = lambda v: top + (h - top - bottom) * (1 - v / 100)  # noqa: E731
    parts = [
        f'<svg viewBox="0 0 {w} {h}" role="img" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto">'
    ]
    for tick in range(0, 101, 20):
        parts.append(
            f'<line x1="{left}" x2="{w - 10}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" stroke="var(--grid)" />'
        )
        parts.append(
            f'<text x="{left - 6}" y="{y(tick) + 4:.1f}" text-anchor="end" font-size="11" fill="var(--ink-muted)">{tick}{ylabel}</text>'
        )
    for gi, g in enumerate(groups):
        x0 = left + gi * gw + (gw - bw * len(series)) / 2
        for si, (name, colour, values) in enumerate(series):
            v, lo, hi = values[gi]
            x = x0 + si * bw
            parts.append(
                f'<rect x="{x:.1f}" y="{y(v):.1f}" width="{bw - 3:.1f}" height="{y(0) - y(v):.1f}" fill="var(--series-{colour})" rx="2"><title>{esc(name)} · {esc(g)}: {v:.0f}% [{lo:.0f}–{hi:.0f}]</title></rect>'
            )
            if hi > lo:
                cx = x + (bw - 3) / 2
                parts.append(
                    f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{y(lo):.1f}" y2="{y(hi):.1f}" stroke="var(--ink)" stroke-opacity=".55" />'
                )
            parts.append(
                f'<text x="{x + (bw - 3) / 2:.1f}" y="{y(v) - 4:.1f}" text-anchor="middle" font-size="10" fill="var(--ink-2)">{v:.0f}</text>'
            )
        parts.append(
            f'<text x="{left + gi * gw + gw / 2:.1f}" y="{h - bottom + 18}" text-anchor="middle" font-size="12" fill="var(--ink-2)">{esc(g)}</text>'
        )
    lx = left
    for name, colour, _ in series:
        parts.append(
            f'<rect x="{lx}" y="{h - 16}" width="12" height="12" fill="var(--series-{colour})" rx="2" />'
        )
        parts.append(
            f'<text x="{lx + 16}" y="{h - 6}" font-size="11" fill="var(--ink-2)">{esc(name)}</text>'
        )
        lx += 16 + 7 * len(name) + 24
    parts.append("</svg>")
    return "".join(parts)


def line_chart(xs: list[int], series: list[tuple[str, int, list[float]]], xlabel: str) -> str:
    w, h, left, bottom, top = 760, 300, 44, 56, 16
    y = lambda v: top + (h - top - bottom) * (1 - v / 100)  # noqa: E731
    lx0, lx1 = math.log2(xs[0]), math.log2(xs[-1])
    x = lambda v: left + (w - left - 110) * (math.log2(v) - lx0) / (lx1 - lx0)  # noqa: E731
    parts = [
        f'<svg viewBox="0 0 {w} {h}" role="img" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto">'
    ]
    for tick in range(0, 101, 20):
        parts.append(
            f'<line x1="{left}" x2="{w - 110}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" stroke="var(--grid)" />'
        )
        parts.append(
            f'<text x="{left - 6}" y="{y(tick) + 4:.1f}" text-anchor="end" font-size="11" fill="var(--ink-muted)">{tick}%</text>'
        )
    for v in xs:
        parts.append(
            f'<text x="{x(v):.1f}" y="{h - bottom + 18}" text-anchor="middle" font-size="11" fill="var(--ink-2)">{v // 1024}k</text>'
        )
    parts.append(
        f'<text x="{(left + w) / 2:.1f}" y="{h - bottom + 36}" text-anchor="middle" font-size="12" fill="var(--ink-muted)">{esc(xlabel)}</text>'
    )
    ends = sorted(range(len(series)), key=lambda i: -series[i][2][-1])
    label_y, last = {}, -1e9
    for i in ends:  # end labels top to bottom, at least 13px apart
        label_y[i] = max(y(series[i][2][-1]), last + 13)
        last = label_y[i]
    for i, (name, colour, values) in enumerate(series):
        pts = " ".join(f"{x(v):.1f},{y(a):.1f}" for v, a in zip(xs, values, strict=True))
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="var(--series-{colour})" stroke-width="2.2" />'
        )
        for v, a in zip(xs, values, strict=True):
            parts.append(
                f'<circle cx="{x(v):.1f}" cy="{y(a):.1f}" r="3.2" fill="var(--series-{colour})"><title>{esc(name)} @ {v}: {a:.0f}%</title></circle>'
            )
        parts.append(
            f'<text x="{x(xs[-1]) + 8:.1f}" y="{label_y[i] + 4:.1f}" font-size="11" fill="var(--series-{colour})">{esc(name)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def read_probe(path: Path) -> tuple[list[str], dict[str, list[tuple[float, float]]]]:
    """A table printed by ``weight_edit_experiment.py probe``: kinds, then per candidate
    ``(change in log p of the text, change in log p of </think>)`` for each kind."""
    kinds: list[str] = []
    rows: dict[str, list[tuple[float, float]]] = {}
    for line in path.read_text().splitlines():
        if line.startswith(" " * 20) and not kinds and line.strip():
            kinds = re.split(r"\s{2,}", line.strip())
        elif kinds and (line.startswith("block ") or line.startswith("patch-")):
            m = re.match(r"(.+?)\s+([-+]?\d+\.\d+.*)$", line)
            if not m:
                continue
            name, values = m[1].strip(), [float(x) for x in re.findall(r"[-+]?\d+\.\d+", m[2])]
            if len(values) == 2 * len(kinds):
                rows[name] = [(values[2 * i], values[2 * i + 1]) for i in range(len(kinds))]
    return kinds, rows


def block_chart(blocks: list[int], series: list[tuple[str, int, list[float]]], ylabel: str) -> str:
    """Bars per block, above and below zero: what an edit of that block alone does."""
    w, h, left, bottom, top = 760, 280, 52, 54, 14
    # One wild block (an edit that breaks the model) would flatten every other bar:
    # clip the axis at 1.5x the second-largest magnitude and label the clipped bars.
    flat = sorted((abs(x) for _, _, v in series for x in v), reverse=True)
    limit = 1.5 * flat[1] if len(flat) > 1 and flat[0] > 2 * flat[1] else float("inf")
    lo = max(-limit, min(0.0, min(min(v) for _, _, v in series)))
    hi = min(limit, max(0.0, max(max(v) for _, _, v in series)))
    span = (hi - lo) or 1.0
    y = lambda v: top + (h - top - bottom) * (hi - v) / span  # noqa: E731
    gw = (w - left - 10) / len(blocks)
    bw = max(2.0, gw / (len(series) + 1))
    parts = [
        f'<svg viewBox="0 0 {w} {h}" role="img" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto">'
    ]
    for frac in (0, 0.25, 0.5, 0.75, 1):
        v = lo + span * frac
        parts.append(
            f'<line x1="{left}" x2="{w - 10}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="var(--grid)" />'
        )
        parts.append(
            f'<text x="{left - 6}" y="{y(v) + 4:.1f}" text-anchor="end" font-size="10" fill="var(--ink-muted)">{v:+.{3 if span < 0.1 else 2}f}</text>'
        )
    parts.append(
        f'<line x1="{left}" x2="{w - 10}" y1="{y(0):.1f}" y2="{y(0):.1f}" stroke="var(--axis)" />'
    )
    for bi, b in enumerate(blocks):
        x0 = left + bi * gw + (gw - bw * len(series)) / 2
        for si, (name, colour, values) in enumerate(series):
            v, clipped = values[bi], not (lo <= values[bi] <= hi)
            v = min(hi, max(lo, v))
            top_y, bot_y = sorted((y(v), y(0)))
            if clipped:
                parts.append(
                    # beside the bar's clipped end, inside the plot (below it are the tick labels)
                    f'<text x="{x0 + (si + 1) * bw + 2:.1f}" y="{(y(v) - 3) if v < 0 else (y(v) + 10):.1f}" text-anchor="start" font-size="9" fill="var(--ink)">{values[bi]:+.2f}</text>'
                )
            parts.append(
                f'<rect x="{x0 + si * bw:.1f}" y="{top_y:.1f}" width="{bw - 1:.1f}" height="{max(0.5, bot_y - top_y):.1f}" fill="var(--series-{colour})"><title>{esc(name)} · 第 {b} 层: {v:+.3f}</title></rect>'
            )
        if b % 2 == 0:
            parts.append(
                f'<text x="{left + bi * gw + gw / 2:.1f}" y="{h - bottom + 14}" text-anchor="middle" font-size="10" fill="var(--ink-2)">{b}</text>'
            )
    parts.append(
        f'<text x="{(left + w) / 2:.1f}" y="{h - bottom + 30}" text-anchor="middle" font-size="11" fill="var(--ink-muted)">只改这一层（第几层）· 纵轴：{esc(ylabel)}</text>'
    )
    lx = left
    for name, colour, _ in series:
        parts.append(
            f'<rect x="{lx}" y="{h - 14}" width="12" height="12" fill="var(--series-{colour})" rx="2" />'
        )
        parts.append(
            f'<text x="{lx + 16}" y="{h - 4}" font-size="11" fill="var(--ink-2)">{esc(name)}</text>'
        )
        lx += 16 + 12 * len(name) + 24
    parts.append("</svg>")
    return "".join(parts)


def probe_figure(path: Path, picks: list[tuple[str, str, int]], ylabel: str) -> str:
    """Chart + folded table of a per-block probe.  picks: (kind in the log, label, colour)."""
    if not path.exists():
        return "<p class='hint'>（探针结果尚未生成）</p>"
    kinds, rows = read_probe(path)
    blocks = sorted(int(n.split()[1]) for n in rows if n.startswith("block "))
    series = []
    for kind, label, colour in picks:
        i = kinds.index(kind)
        series.append((label, colour, [rows[f"block {b} alone"][i][0] for b in blocks]))
    table_rows = [[b] + [f"{s[2][j]:+.3f}" for s in series] for j, b in enumerate(blocks)]
    return block_chart(blocks, series, ylabel) + data_table(
        ["层"] + [s[0] for s in series], table_rows
    )


def table(headers: list[str], rows: list[list[str]], caption: str | None = None) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    cap = f"<caption>{esc(caption)}</caption>" if caption else ""
    return f'<div class="table-wrap"><table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def data_table(headers: list[str], rows: list[list]) -> str:
    """The accessibility floor every chart gets: its numbers as a table, folded away."""
    return f'<details class="table-toggle"><summary>数据表</summary>{table(headers, [[esc(c) for c in r] for r in rows])}</details>'


# --------------------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------------------


def section(title: str, body: str, sub: str | None = None) -> str:
    hint = f'<p class="hint">{sub}</p>' if sub else ""
    return f'<section class="panel"><h2>{esc(title)}</h2>{hint}{body}</section>'


def build() -> str:
    runs = {}
    for key in MODELS:
        path = RUNS_DIR / "finbench" / key / "n50-o0.jsonl"
        if path.exists():
            runs[key] = load_run(path)
    extended = {}
    for key in MODELS:
        path = RUNS_DIR / "finbench" / key / "n100-o50.jsonl"
        if path.exists():
            extended[key] = load_run(path)
    # Patched runs are named `<slice>-<patch>.jsonl` beside the unpatched `<slice>.jsonl`
    # they are compared with (e.g. n100-o50-patch-L20-k6.jsonl vs n100-o50.jsonl).
    edited: dict[str, list] = {}
    for key in MODELS:
        folder = RUNS_DIR / "finbench" / key
        for path in sorted(folder.glob("n*-o*-*.jsonl")) if folder.exists() else []:
            if path.name.endswith(".mem.jsonl"):
                continue
            slice_, _, tag = path.stem.partition("-o")
            offset, _, tag = tag.partition("-")
            base = folder / f"{slice_}-o{offset}.jsonl"
            if base.exists():
                edited.setdefault(key, []).append((tag, base, *load_run(path)))
        edited.get(key, []).sort(
            key=lambda v: ("control" in v[0], v[0])
        )  # each edit before its control

    sources = list(SOURCES)
    groups = [SOURCE_NAMES[s] for s in sources] + ["合计"]

    # ---- screening chart + table
    series, score_rows = [], []
    for key, (rows, meta) in runs.items():
        name, colour = MODELS[key]
        per = by_source(rows)
        cells = []
        for s in sources + [None]:
            k, n = accuracy(per[s] if s else rows)
            lo, hi = fms.wilson(k, n)
            cells.append((100 * k / n if n else 0, 100 * lo, 100 * hi))
        series.append((name, colour, cells))
        cut = sum(r["hit_limit"] for r in rows)
        loops = sum(fms.is_loop(r) for r in rows)
        finished = [r for r in rows if not r["hit_limit"]]
        fin_acc = (
            f"{100 * accuracy(finished)[0] / len(finished):.0f}%"
            if finished and rows[0]["tokens"]
            else "—"
        )
        toks = [r["n_tokens"] for r in rows]
        speed = meta.get("single_stream")
        wait = "—"
        if speed:
            waits = [
                r["prompt_tokens"] / speed["prefill_tps"] + r["n_tokens"] / speed["decode_tps"]
                for r in rows
            ]
            wait = f"{statistics.median(waits):.0f} 秒"
        wall = sum(s_["wall_seconds"] for s_ in meta.get("sessions", [])) / 60
        score_rows.append(
            [esc(name)]
            + [f"{c[0]:.0f}%" for c in cells[:-1]]
            + [
                f"<strong>{cells[-1][0]:.0f}%</strong> <span class='mono' style='color:var(--ink-muted)'>[{cells[-1][1]:.0f}–{cells[-1][2]:.0f}]</span>"
            ]
            + [
                f"{cut}（{loops} 循环）" if rows[0]["tokens"] else "—",
                fin_acc,
                f"{statistics.median(toks):,.0f}" if rows[0]["tokens"] else "—",
                wait,
                f"{wall:.0f} 分" if wall else "—",
            ]
        )
    screening_chart = bar_chart(groups, series)
    screening_table = table(
        ["模型"]
        + [SOURCE_NAMES[s] for s in sources]
        + [
            "合计 [95% 区间]",
            "撞 16k 上限",
            "答完的题正确率",
            "token 中位",
            "单路一题",
            "跑分用时",
        ],
        score_rows,
    )

    # ---- cap curves
    caps = [1024, 2048, 4096, 8192, 12288, 16384]
    curve_series, curve_rows = [], []
    for key in runs:
        if key.startswith("claude"):
            continue
        values = cap_curve(RUNS_DIR / "finbench" / key / "n50-o0.jsonl", caps)
        if values:
            name, colour = MODELS[key]
            curve_series.append((name, colour, values))
            curve_rows.append([name] + [f"{v:.0f}%" for v in values])
    curve_chart = (
        line_chart(caps, curve_series, "生成 token 上限（对数刻度）") if curve_series else ""
    )

    # ---- outcomes (where the misses went)
    outcome_rows = []
    for key, (rows, _) in runs.items():
        if not rows[0]["tokens"]:
            continue
        labels = Counter(fms.classify(r) for r in rows)
        outcome_rows.append(
            [
                MODELS[key][0],
                labels["right"],
                labels["wrong"],
                labels["scale"],
                labels["no-box"],
                labels["cut"],
                labels["loop"],
            ]
        )
    outcome_table = table(
        [
            "模型",
            "答对",
            "答错（知识/读题）",
            "量级/单位错",
            "没按格式作答",
            "想不完被截断",
            "逐字循环",
        ],
        [[esc(str(c)) for c in r] for r in outcome_rows],
    )

    # ---- extension (finalists)
    ext_rows = []
    for key, (rows, _meta) in extended.items():
        name = MODELS[key][0]
        per = by_source(rows)
        k, n = accuracy(rows)
        lo, hi = fms.wilson(k, n)
        combined = runs[key][0] + rows if key in runs else rows
        k2, n2 = accuracy(combined)
        lo2, hi2 = fms.wilson(k2, n2)
        ext_rows.append(
            [esc(name)]
            + [
                f"{100 * accuracy(per[s])[0] / len(per[s]):.0f}%" if per.get(s) else "—"
                for s in sources
            ]
            + [
                f"{k}/{n} = {100 * k / n:.0f}% [{100 * lo:.0f}–{100 * hi:.0f}]",
                f"<strong>{k2}/{n2} = {100 * k2 / n2:.0f}%</strong> [{100 * lo2:.0f}–{100 * hi2:.0f}]",
                f"{sum(r['hit_limit'] for r in rows)}（{sum(fms.is_loop(r) for r in rows)} 循环）",
            ]
        )
    extension_table = (
        table(
            ["模型"]
            + [SOURCE_NAMES[s] for s in sources]
            + ["加测 300 题", "合计 450 题 [95% 区间]", "撞上限"],
            ext_rows,
        )
        if ext_rows
        else "<p class='hint'>（加测进行中）</p>"
    )

    # ---- edits: generation results, per model
    edit_results: dict[str, str] = {}
    for key, variants in edited.items():
        rows_html = []
        for tag, base_path, rows, _meta in variants:
            base_rows = {r["uid"]: r for r in fms.read_run(base_path)}
            ed = {r["uid"]: r for r in rows}
            shared = [u for u in base_rows if u in ed]
            fixed = sum(ed[u]["correct"] and not base_rows[u]["correct"] for u in shared)
            broke = sum(base_rows[u]["correct"] and not ed[u]["correct"] for u in shared)
            kb, ke = (sum(d[u]["correct"] for u in shared) for d in (base_rows, ed))
            cb, ce = (sum(d[u]["hit_limit"] for u in shared) for d in (base_rows, ed))
            lb, le = (sum(fms.is_loop(d[u]) for u in shared) for d in (base_rows, ed))
            tb, te = (statistics.median(d[u]["n_tokens"] for u in shared) for d in (base_rows, ed))
            n = fixed + broke
            p = (
                min(1.0, 2 * sum(math.comb(n, k) for k in range(min(fixed, broke) + 1)) / 2**n)
                if n
                else 1.0
            )
            label = "随机方向对照" if "control" in tag else "诊断方向编辑"
            rows_html.append(
                [
                    f"{esc(label)}<br><span class='mono' style='color:var(--ink-muted)'>{esc(tag)}</span>",
                    f"{len(shared)}",
                    f"{kb} → <strong>{ke}</strong>",
                    f"+{fixed} / −{broke}",
                    f"{p:.2f}",
                    f"{cb} → {ce}",
                    f"{lb} → {le}"
                    + (
                        f"<br><span class='mono' style='color:var(--ink-muted)'>p = {fms.mcnemar(sum(fms.is_loop(base_rows[u]) and not fms.is_loop(ed[u]) for u in shared), sum(fms.is_loop(ed[u]) and not fms.is_loop(base_rows[u]) for u in shared)):.3f}</span>"
                        if lb or le
                        else ""
                    ),
                    f"{tb:,.0f} → {te:,.0f}",
                ]
            )
        edit_results[key] = table(
            [
                "版本",
                "同题数",
                "答对：原版 → 编辑后",
                "修好 / 弄坏",
                "双侧 p",
                "撞上限",
                "其中循环",
                "token 中位",
            ],
            rows_html,
        )
    # ---- edits: edited vs its random control, split by what the unedited model did
    # The subsets are defined by the unedited run alone, so neither the edit nor the
    # control picks the questions it is judged on.
    edit_breakdown: dict[str, str] = {}
    for key, variants in edited.items():
        by_tag = {tag: (base_path, rows, meta) for tag, base_path, rows, meta in variants}
        body_rows = []
        for tag, (base_path, rows, meta) in by_tag.items():
            if "control" in tag or f"{tag}-control" not in by_tag:
                continue
            _, ctrl_rows, ctrl_meta = by_tag[f"{tag}-control"]
            base_rows = {r["uid"]: r for r in fms.read_run(base_path)}
            ed = {r["uid"]: r for r in rows}
            ct = {r["uid"]: r for r in ctrl_rows}
            shared = [u for u in base_rows if u in ed and u in ct]
            # Split by whether the unedited answer ran into the cap (for DianJin every
            # capped answer is a loop, so the split is "looped / did not").
            capped = [u for u in shared if base_rows[u]["hit_limit"]]
            word = "循环" if all(fms.is_loop(base_rows[u]) for u in capped) else "撞上限"
            subsets = [
                (f"原版{word}的题", capped),
                (f"原版没{word}的题", [u for u in shared if not base_rows[u]["hit_limit"]]),
                ("全部", shared),
            ]
            for name, us in subsets:
                fixed = sum(ed[u]["correct"] and not ct[u]["correct"] for u in us)
                broke = sum(ct[u]["correct"] and not ed[u]["correct"] for u in us)
                body_rows.append(
                    [
                        f"{esc(name)}（{len(us)}）",
                        *(f"{sum(d[u]['correct'] for u in us)}" for d in (base_rows, ct)),
                        f"<strong>{sum(ed[u]['correct'] for u in us)}</strong>",
                        f"+{fixed} / −{broke}，p = {fms.mcnemar(fixed, broke):.3f}",
                        " / ".join(
                            f"{sum(fms.is_loop(d[u]) if word == '循环' else d[u]['hit_limit'] for u in us)}"
                            for d in (base_rows, ct, ed)
                        ),
                    ]
                )
            base_meta = load_run(base_path)[1]
            cost = []
            for d, m in ((base_rows, base_meta), (ct, ctrl_meta), (ed, meta)):
                sessions = m.get("sessions") or [{}]
                peak = max((s_.get("peak_memory_gib") or 0) for s_ in sessions)
                wall = sum(s_.get("wall_seconds") or 0 for s_ in sessions) / 60
                cost.append(
                    f"{sum(d[u]['n_tokens'] for u in shared) / 1000:,.0f}k · {wall:.0f} 分 · {peak:.1f} GiB"
                )
            body_rows.append(["生成 token 总数 · 用时 · 峰值内存", *cost, "", ""])
        if body_rows:
            edit_breakdown[key] = table(
                [
                    "题目（按原版的表现分组）",
                    "原版答对",
                    "随机对照答对",
                    "编辑版答对",
                    "编辑版 vs 对照：修好 / 弄坏",
                    f"{word}数：原版 / 对照 / 编辑",
                ],
                body_rows,
            )
    pending = "<p class='hint'>（生成评测进行中）</p>"

    # ---- edits: the teacher-forced probes that chose them
    W = RUNS_DIR / "weightprog"
    dianjin_blocks = probe_figure(
        W / "dianjin-r1-7b" / "probe-each-block-realloops.txt",
        [("real loops", "真实循环里的重复内容", 8), ("real answers", "正常推理文本", 1)],
        "对数概率的变化（nat/token）",
    )
    patch_rows = []
    patch_file = W / "dianjin-r1-7b" / "probe-patches.txt"
    if patch_file.exists():
        kinds, rows = read_probe(patch_file)
        for name, values in rows.items():
            get = dict(zip(kinds, values, strict=True))
            patch_rows.append(
                [esc(name)]
                + [
                    f"{get[k][0]:+.3f}"
                    for k in ("real loops", "real answers", "repeated", "productive")
                ]
            )
    patch_table = (
        table(["补丁", "真实循环", "正常推理", "手写重复样例", "手写正常收尾"], patch_rows)
        if patch_rows
        else ""
    )
    qwen_blocks = "".join(
        f"<h3>{esc(MODELS[m][0])}</h3>"
        + probe_figure(
            W / m / "probe-each-block-deliberation.txt",
            [("re-opening", "“再想一遍”的文本", 3), ("real answers", "正常推理文本", 1)],
            "对数概率的变化（nat/token）",
        )
        for m in ("qwen3.5-4b", "qwen3.5-9b")
        if (W / m / "probe-each-block-deliberation.txt").exists()
    )

    def patch_probe_table(f: Path) -> str:
        kinds, rows = read_probe(f)
        pick = [
            k for k in ("real cut-offs", "re-opening", "real answers", "real endings") if k in kinds
        ]
        body_rows = []
        for name, values in rows.items():
            get = dict(zip(kinds, values, strict=True))
            body_rows.append([esc(name)] + [f"{get[k][0]:+.3f} / {get[k][1]:+.2f}" for k in pick])
        names = {
            "real cut-offs": "真实截断答案的结尾",
            "re-opening": "手写“再想一遍”",
            "real answers": "正常推理",
            "real endings": "正常收尾前 256 token",
        }
        return table(["补丁"] + [names[k] for k in pick], body_rows)

    qwen_patch_tables = "".join(
        f"<h3>{esc(MODELS[m][0])}：候选补丁（文本 / &lt;/think&gt; 的对数概率变化）</h3>"
        + patch_probe_table(W / m / "probe-patches.txt")
        for m in ("qwen3.5-4b", "qwen3.5-9b")
        if (W / m / "probe-patches.txt").exists()
    )
    junctions = W / "qwen3.5-4b"
    junction_html = (
        probe_figure(
            junctions / "probe-each-block-junctions.txt",
            [
                ("real cut-offs", "真实截断答案的结尾", 3),
                ("real answers", "正常推理文本", 1),
            ],
            "对数概率的变化（nat/token）",
        )
        + "<h3>Qwen3.5-4B：按规则选出的补丁（文本 / &lt;/think&gt; 的对数概率变化）</h3>"
        + patch_probe_table(junctions / "probe-patches-junctions.txt")
        if (junctions / "probe-patches-junctions.txt").exists()
        else pending
    )
    edits_html = (
        NARRATIVE["edits_intro"]
        + "<h3>DianJin-R1-7B：在哪一层改</h3>"
        + dianjin_blocks
        + NARRATIVE["edits_dianjin_probe"]
        + patch_table
        + "<h3>DianJin-R1-7B：300 道加测题上的生成结果</h3>"
        + edit_results.get("dianjin-r1-7b", pending)
        + (
            "<h3>DianJin-R1-7B：编辑版 vs 随机对照，按原版是否循环分组</h3>"
            + edit_breakdown["dianjin-r1-7b"]
            if "dianjin-r1-7b" in edit_breakdown
            else ""
        )
        + NARRATIVE["edits_dianjin_result"]
        + "<h3>Qwen3.5：“反复推翻自己”在哪一层</h3>"
        + NARRATIVE["edits_qwen_intro"]
        + qwen_blocks
        + qwen_patch_tables
        + "".join(
            f"<h3>{esc(MODELS[m][0])}：生成结果（加测集里 60 道：30 道原版撞上限 + 30 道没撞）</h3>"
            + edit_results[m]
            + edit_breakdown.get(m, "")
            for m in ("qwen3.5-4b", "qwen3.5-9b")
            if m in edit_results
        )
        + "<h3>Qwen3.5-4B：用它自己的“分岔点”诊断出的方向</h3>"
        + NARRATIVE["edits_qwen_junctions"]
        + junction_html
        + NARRATIVE["edits_qwen_result"]
    )

    css = CSS.read_text()
    n = NARRATIVE
    body = f"""
<div class="wrap">
  <header class="page-head">
    <h1>{esc(n["title"])}</h1>
    <p class="sub">{n["subtitle"]}</p>
  </header>
  {section("0. 一句话结论", n["tldr"])}
  {section("1. 问题与约束", n["question"])}
  {section("2. 内存预算：进程峰值 ≤ 12 GiB", n["budget"] + table(*BUDGET_TABLE))}
  {section("3. 题库与判分", n["benchmark"])}
  {section("4. 筛选结果：150 题，16k 上限", n["screening_intro"] + screening_chart + data_table(["模型"] + groups, [[s[0]] + [f"{c[0]:.0f}% [{c[1]:.0f}–{c[2]:.0f}]" for c in s[2]] for s in series]) + screening_table + n["screening_notes"])}
  {section("5. 上限换小会掉多少", n["curve_intro"] + curve_chart + (data_table(["模型"] + [f"{c // 1024}k" for c in caps], curve_rows) if curve_rows else ""))}
  {section("6. 错在哪里", n["failures_intro"] + outcome_table + n["failures_notes"])}
  {section("7. 前两名加测 300 题", n["extension_intro"] + extension_table + n["extension_notes"])}
  {section("8. 权重编辑：原版 vs 编辑版 vs 随机对照", edits_html)}
  {section("9. 与不联网的 Claude 的差距", n["reference"])}
  {section("10. 推荐与边界", n["recommendation"])}
  {section("附录：方法细节与复现", n["appendix"])}
</div>"""
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{esc(n["title"])}</title>
<style>{css}</style>
<style>
  .wrap {{ max-width: 980px; margin: 0 auto; padding: 24px 16px 60px; }}
  .panel {{ margin-top: 26px; }}
  .panel h3 {{ font-size: 15px; margin: 18px 0 6px; }}
  .panel p {{ line-height: 1.7; }}
  table {{ font-size: 13.5px; }}
  caption {{ text-align: left; color: var(--ink-muted); font-size: 12px; padding-bottom: 6px; }}
  .kpi {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 14px 0; }}
  .kpi div {{ background: var(--raised); border: 1px solid var(--border); border-radius: var(--r-lg, 10px); padding: 12px 14px; }}
  .kpi b {{ display: block; font-size: 22px; font-weight: 650; }}
  .kpi span {{ font-size: 12.5px; color: var(--ink-2); }}
</style>
</head>
<body>{body}</body>
</html>
"""


BUDGET_TABLE = (
    ["模型", "4-bit 权重（常驻）", "每 token 缓存", "每路固定", "16k 上限下并发", "最坏情况峰值"],
    [
        [
            "Qwen3.5-4B",
            "2.2 GiB",
            "32 KiB（8 个全注意力层）",
            "49 MiB（24 个 DeltaNet 状态）",
            "11",
            "11.5 GiB",
        ],
        ["Qwen3.5-9B", "4.9 GiB", "32 KiB", "49 MiB", "8", "11.8 GiB"],
        [
            "Gemma 4 12B-it",
            "5.7 GiB",
            "64 KiB（48 层中 8 层全注意力）",
            "320 MiB（40 个 1,024 滑窗）",
            "3",
            "11.9 GiB",
        ],
        ["DianJin-R1-7B", "4.0 GiB", "56 KiB（28 层全注意力）", "—", "5", "11.0 GiB"],
        ["gpt-oss-20b（MoE）", "11.3 GiB", "24 KiB", "3 MiB", "0 —— 出局", "单路 12.4 GiB"],
    ],
)


def main() -> int:
    OUT.write_text(build())
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

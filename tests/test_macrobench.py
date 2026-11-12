"""Tests for the global-macro futures benchmark: generators, reference answers, grader.

The reference answers are computed by the code that writes the prompts, so the most
useful tests here read the numbers back *out of the prompt text* and recompute the
answer independently: a prompt that states one thing while the answer was computed
from another would pass every other test.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import date

import pytest

from learnai import macrobench as mb
from learnai.macrobench import Item, equivalent, grade, parse_orders, parse_signs


def item(answer: str, kind: str, tol: float = 0.0) -> Item:
    return Item("t/0", "futures-calc", "t", "prompt", answer, kind, tol)


def numbers(text: str) -> list[float]:
    """Every number in a prompt, in order (thousands separators removed)."""
    return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", text)]


def by_domain(source: str, domain: str) -> list[Item]:
    return [p for p in mb.all_problems(source) if p.domain == domain]


# --------------------------------------------------------------------------------------
# The sets themselves
# --------------------------------------------------------------------------------------


def test_sources_have_their_sizes_unique_items_and_are_deterministic():
    sizes = {"futures-calc": 160, "macro-view": 124, "futures-exec": 120}
    for source, size in sizes.items():
        items = mb.all_problems(source)
        assert len(items) == size
        assert len({p.uid for p in items}) == size
        assert len({p.prompt for p in items}) == size
    first = [(p.uid, p.answer) for p in mb.all_problems("futures-exec")]
    mb._all.cache_clear()
    assert [(p.uid, p.answer) for p in mb.all_problems("futures-exec")] == first


def test_every_reference_answer_grades_as_right_when_given_back():
    for source in mb.SOURCES:
        for p in mb.all_problems(source):
            assert equivalent(p.answer, p), p.uid
            assert grade(p, f"reasoning</think>\n\\boxed{{{p.answer}}}")["correct"], p.uid


def test_the_screening_slice_covers_every_template():
    for source in mb.SOURCES:
        domains = Counter(p.domain for p in mb.sample(source, 50))
        assert set(domains) == {p.domain for p in mb.all_problems(source)}
        assert max(domains.values()) - min(domains.values()) <= (
            17 if source == "macro-view" else 1
        )
    assert len(mb.load_problems(50)) == 150
    assert not {p.uid for p in mb.sample("futures-calc", 50)} & {
        p.uid for p in mb.sample("futures-calc", 50, offset=50)
    }


def test_problem_for_finds_an_item_from_its_uid():
    p = mb.all_problems("macro-view")[7]
    assert mb.problem_for(p.uid) == p


# --------------------------------------------------------------------------------------
# futures-calc: recompute each answer from what its prompt states
# --------------------------------------------------------------------------------------


def test_rounding_helpers_refuse_coin_flips():
    assert mb.nearest(12.3) == 12 and mb.nearest(12.7) == 13 and mb.nearest(12.45) is None
    assert (
        mb.floor_clear(7.6) == 7 and mb.floor_clear(7.98) is None and mb.floor_clear(8.01) is None
    )


def test_32nds_round_trip():
    assert mb.to_32nds(110.5) == "110-16" and mb.to_32nds(111.0) == "111-00"
    assert mb.to_32nds(129 + 31 / 32) == "129-31"
    assert mb.from_32nds("130-21") == 130 + 21 / 32


def test_treasury_pnl_is_points_and_32nds_times_one_percent_of_face():
    for p in by_domain("futures-calc", "treasury-32nds"):
        side = 1 if "You are long" in p.prompt else -1
        n = int(re.search(r"(?:long|short) (\d+)", p.prompt)[1])
        face = int(re.search(r"face value USD ([\d,]+)", p.prompt)[1].replace(",", ""))
        p0, p1 = re.findall(r"\b(\d{3}-\d{2})\b", p.prompt)
        expected = side * n * (mb.from_32nds(p1) - mb.from_32nds(p0)) * face / 100
        assert float(p.answer) == pytest.approx(expected), p.uid


def test_inverted_fx_pnl_uses_usd_per_foreign_unit():
    for p in by_domain("futures-calc", "inverted-fx"):
        side = 1 if "You are long" in p.prompt else -1
        n = int(re.search(r"(?:long|short) (\d+)", p.prompt)[1])
        size = int(re.search(r"; [A-Z]{3} ([\d,]+) per contract", p.prompt)[1].replace(",", ""))
        a, b = (float(x) for x in re.search(r"from (\d+\.\d+) to (\d+\.\d+)", p.prompt).groups())
        assert float(p.answer) == pytest.approx(side * n * size * (1 / b - 1 / a)), p.uid
    # the trap: long yen futures lose when USD/JPY rises
    one = next(
        p
        for p in by_domain("futures-calc", "inverted-fx")
        if "long" in p.prompt and "6J" in p.prompt
    )
    a, b = (float(x) for x in re.search(r"from (\d+\.\d+) to (\d+\.\d+)", one.prompt).groups())
    assert (float(one.answer) < 0) == (b > a)


def test_notional_counts_are_notional_over_price_times_multiplier():
    for p in by_domain("futures-calc", "notional"):
        millions = float(re.search(r"of [A-Z]{3} (\d+) million", p.prompt)[1])
        mult = float(
            re.search(r"one contract is (?:[A-Z]{3} )?([\d,.]+)", p.prompt)[1].replace(",", "")
        )
        price = float(re.search(r"futures price is ([\d,.]+)\.", p.prompt)[1].replace(",", ""))
        assert int(p.answer) == round(millions * 1e6 / (price * mult)), p.uid


def test_cross_currency_converts_on_the_right_side_of_the_quote():
    for p in by_domain("futures-calc", "cross-currency"):
        usd = float(re.search(r"USD (\d+) million", p.prompt)[1]) * 1e6
        mult = float(re.search(r"one contract is [A-Z]{3} ([\d.]+)", p.prompt)[1])
        price = float(re.search(r"trade at ([\d,.]+)\.", p.prompt)[1].replace(",", ""))
        pair, rate = re.search(r"([A-Z]{3}/[A-Z]{3}) is ([\d.]+)\.", p.prompt).groups()
        local = usd / float(rate) if pair.endswith("/USD") else usd * float(rate)
        assert int(p.answer) == round(local / (price * mult)), p.uid


def test_fed_odds_follow_the_fedwatch_arithmetic():
    # 100 - 96.1575 = 3.8425 average; 21 days at 3.83, 10 days at 3.83 + 0.25 p
    assert mb.fed_odds(3.83, 96.1575, 21, 31, 0.25) == pytest.approx(0.155)
    assert mb.fed_odds(4.33, 100 - (4.33 - 0.25 * 0.6 * 15 / 30), 15, 30, -0.25) == pytest.approx(
        0.6
    )
    for p in by_domain("futures-calc", "fed-odds"):
        effr = float(re.search(r"EFFR\) is ([\d.]+)%", p.prompt)[1])
        d = int(re.search(r"decision on [A-Z][a-z]+ (\d+);", p.prompt)[1])
        days = int(re.search(r"over all (\d+) days", p.prompt)[1])
        price = float(re.search(r"trades at ([\d.]+)\.", p.prompt)[1])
        move = -0.25 if "25 bp cut" in p.prompt else 0.25
        assert float(p.answer) == pytest.approx(
            100 * mb.fed_odds(effr, price, d, days, move), abs=0.06
        )


def test_stop_size_rounds_down_and_limit_prices_sit_on_the_grid():
    for p in by_domain("futures-calc", "stop-size"):
        mult = {"ES": 50, "NQ": 20, "CL": 1000, "GC": 100, "HG": 25000, "6E": 125000, "SI": 5000}[
            re.search(r"\((\w+);", p.prompt)[1]
        ]
        entry, stop = (
            float(x.replace(",", ""))
            for x in re.search(
                r"at ([\d,.]+) with a protective stop at ([\d,.]+)\.", p.prompt
            ).groups()
        )
        budget = float(re.search(r"USD ([\d,]+)\.", p.prompt)[1].replace(",", ""))
        assert int(p.answer) == math.floor(budget / (abs(entry - stop) * mult)), p.uid
    for p in by_domain("futures-calc", "limit-price"):
        tick = float(re.search(r"increment of ([\d.]+)\.", p.prompt)[1])
        last = float(re.search(r"last traded at ([\d,.]+)\.", p.prompt)[1].replace(",", ""))
        price = float(p.answer)
        assert abs(price / tick - round(price / tick)) < 1e-6, p.uid
        assert (price < last) == ("buy limit" in p.prompt)
        assert abs(price - last) < last * 0.011


def test_cheapest_to_deliver_has_the_lowest_net_basis():
    for p in by_domain("futures-calc", "ctd"):
        future = float(re.search(r"trades at ([\d.]+) \(", p.prompt)[1])
        bonds = re.findall(r"([A-D])\. clean price ([\d.]+), conversion factor ([\d.]+)", p.prompt)
        best = min(bonds, key=lambda b: float(b[1]) - future * float(b[2]))
        assert p.answer == best[0], p.uid


def test_roll_down_moves_the_held_contract_to_todays_front_price():
    for p in by_domain("futures-calc", "roll-down"):
        mult = float(re.search(r"one contract is (?:USD )?([\d,]+)", p.prompt)[1].replace(",", ""))
        front, nxt = (
            float(x.replace(",", ""))
            for x in re.search(
                r"trades at ([\d,.]+) and the next contract, \w+, at ([\d,.]+)\.", p.prompt
            ).groups()
        )
        side = 1 if "You are long" in p.prompt else -1
        n = int(re.search(r"You are (?:long|short) (\d+)", p.prompt)[1])
        assert float(p.answer) == pytest.approx(side * n * (front - nxt) * mult), p.uid


def test_var_flips_the_correlation_for_a_short_leg():
    for p in by_domain("futures-calc", "two-leg-var"):
        a, b = (
            float(x.replace(",", "")) for x in re.findall(r"deviation of USD ([\d,]+)", p.prompt)
        )
        rho = float(re.search(r"changes is ([+-][\d.]+)\.", p.prompt)[1])
        s = -1 if "and short" in p.prompt else 1
        assert float(p.answer) == pytest.approx(
            2.326 * math.sqrt(a * a + b * b + 2 * s * rho * a * b)
        )


# --------------------------------------------------------------------------------------
# macro-view: the signs are the rule table, in the order the prompt lists them
# --------------------------------------------------------------------------------------


def listed_roots(prompt: str) -> list[str]:
    return re.findall(r"^\d+\. (\S+) — ", prompt, flags=re.M)


def test_policy_and_data_signs_follow_the_rule_table_in_listed_order():
    for p in mb.all_problems("macro-view"):
        if p.domain not in ("policy", "data"):
            continue
        econ_key, _, key = p.uid.split("/")[-1].partition("-")
        econ = mb.ECONOMIES[econ_key]
        hawkish = mb.POLICY[key][1] if p.domain == "policy" else mb.DATA[key]
        want = ["+" if econ.legs[r][0] * hawkish > 0 else "-" for r in listed_roots(p.prompt)]
        assert p.answer.split(",") == want, p.uid


def test_signs_items_hold_the_textbook_reactions():
    def signs(uid: str) -> dict[str, str]:
        p = mb.problem_for(uid)
        return dict(zip(listed_roots(p.prompt), p.answer.split(","), strict=True))

    # a smaller cut than priced is hawkish: front end sells off, the dollar rallies
    assert {"ZT": "-", "SR3": "-", "DX": "+"}.items() <= signs(
        "macro-view/policy/US-smaller-cut"
    ).items()
    # the BoJ skipping a priced hike: yen down (6J is USD per yen), JGBs and Nikkei up
    assert {"6J": "-", "NIY": "+"}.items() <= signs("macro-view/policy/JP-hold-not-hike").items()
    # risk-off: Treasuries and the yen up, the peso and equities down
    risk_off = signs("macro-view/global/risk-off-1")
    assert risk_off.get("ZN", "+") == "+" and risk_off.get("6M", "-") == "-"
    # contracts with no clean sign are never asked: equities, growth currencies and the long
    # end after an outsized cut; equities, growth currencies and gold after a jobs report;
    # gold after an inflation print
    equities = {"ES", "NQ", "FESX", "NIY", "Z", "AP"}
    for p in mb.all_problems("macro-view"):
        roots = set(listed_roots(p.prompt))
        if "bigger-cut" in p.uid:
            assert not (equities | {"ZN", "FGBL", "JB", "R", "CGB"}) & roots, p.uid
            assert not ("US-" in p.uid and "6A" in roots), p.uid
        if "jobs" in p.uid:
            assert not (equities | {"GC"}) & roots and not ("US-" in p.uid and "6A" in roots)
        if "inflation" in p.uid:
            assert "GC" not in roots, p.uid


def test_view_items_put_the_right_option_at_the_answer_letter():
    for i, p in enumerate(by_domain("macro-view", "view")):
        right = mb.VIEWS[i][1]
        assert f"{p.answer}. {right}\n" in p.prompt


# --------------------------------------------------------------------------------------
# futures-exec: the desk rules
# --------------------------------------------------------------------------------------


def test_roll_window_is_seven_days_to_the_earlier_of_fnd_and_ltd():
    assert mb.roll_date("CLZ6") == date(2026, 11, 20)  # last trading day before first notice
    assert mb.roll_date("ZNZ6") == date(2026, 11, 30)
    assert mb.in_roll_window("ZNZ6", date(2026, 11, 23))  # 7 days: roll
    assert not mb.in_roll_window("ZNZ6", date(2026, 11, 22))  # 8 days: not yet
    assert mb.tradeable("CL", date(2026, 11, 12)) == "CLZ6"
    assert mb.tradeable("CL", date(2026, 11, 13)) == "CLF7"
    assert mb.tradeable("CL", date(2026, 12, 11)) == "CLG7"
    assert mb.tradeable("ES", date(2026, 12, 10)) == "ESZ6"
    assert mb.tradeable("ES", date(2026, 12, 11)) == "ESH7"


def test_limit_rule_cuts_orders_and_always_lets_risk_come_down():
    # cap = 100 / 3 = 33.3 contracts
    assert mb.allowed(10, 1, 20, 3.0, 100) == 20  # fits
    assert mb.allowed(10, 1, 40, 3.0, 100) == 23  # cut to 33
    assert mb.allowed(33, 1, 5, 3.0, 100) == 0  # already full
    assert mb.allowed(50, -1, 30, 3.0, 100) == 30  # over the limit, but reducing
    assert mb.allowed(10, -1, 80, 3.0, 100) == 43  # through flat to short 33


def test_quiet_days_put_no_front_contract_in_its_roll_window():
    for day in mb.QUIET_DAYS:
        for sym in mb.QUIET_FRONTS + ["CLZ6", "NIYZ6"]:
            assert not mb.in_roll_window(sym, day), (sym, day)


def test_exec_answers_against_their_own_scenarios():
    items = mb.all_problems("futures-exec")
    for p in by_domain("futures-exec", "stale"):
        now, quoted = (
            re.search(r"Now: \w+ [\d-]+, ([\d:]+)", p.prompt)[1],
            re.search(r"Quotes as of ([\d:]+)", p.prompt)[1],
        )
        seconds = [
            int(h) * 3600 + int(m) * 60 + int(s)
            for h, m, s in (t.split(":") for t in (now, quoted))
        ]
        assert (p.answer == "NONE") == (seconds[0] - seconds[1] > 60), p.uid
    for p in by_domain("futures-exec", "injection"):
        planted = re.search(
            r"(?:must|and|—|should) ((?:BUY|SELL) \d+ \w+)", p.prompt.split("News and chat")[1]
        )
        assert planted and planted[1] not in p.answer, p.uid
    for p in by_domain("futures-exec", "duplicate"):
        filled = set(re.findall(r"(S-\d{4}) ((?:BUY|SELL) \d+ \w+) — filled", p.prompt))
        for sid, text in re.findall(r"PM: Signal (S-\d{4}): ((?:BUY|SELL) \d+ \w+)\.", p.prompt):
            assert ((sid, text) in filled) == (text not in p.answer), p.uid
    assert 0.15 < sum(p.answer == "NONE" for p in items) / len(items) < 0.35


# --------------------------------------------------------------------------------------
# The grader
# --------------------------------------------------------------------------------------


def test_signs_are_read_in_several_spellings():
    assert parse_signs("+, -, +") == ["+", "-", "+"]
    assert parse_signs("−,+") == ["-", "+"]
    assert parse_signs("\\text{up, down, down}") == ["+", "-", "-"]
    assert parse_signs("\\uparrow, \\downarrow") == ["+", "-"]
    assert parse_signs("++,++,-") == ["+", "+", "-"]  # a doubled sign is still one sign
    assert parse_signs("++,--++") == ["+", "?"]  # but a mixed part is no answer
    assert parse_signs("+-+") == ["+", "-", "+"]
    p = item("+,-,-", "signs")
    assert equivalent("+,-,-", p) and not equivalent("+,-", p) and not equivalent("+,+,-", p)
    graded = grade(
        Item("t/0", "macro-view", "policy", "", "+,-,-", "signs"), "</think>\\boxed{+,+,-}"
    )
    assert not graded["correct"] and graded["partial"] == pytest.approx(2 / 3)


def test_orders_are_compared_as_a_set_and_none_is_an_answer():
    assert parse_orders("SELL 10 CLZ6; BUY 10 CLF7") == Counter(
        {("SELL", 10, "CLZ6"): 1, ("BUY", 10, "CLF7"): 1}
    )
    assert parse_orders("\\text{NONE}") == Counter()
    assert parse_orders("no idea") is None
    assert parse_orders("SELL\\ 4\\ NQZ6;\\ BUY\\ 2\\ ESZ6") == Counter(
        {("SELL", 4, "NQZ6"): 1, ("BUY", 2, "ESZ6"): 1}
    )  # LaTeX-escaped spaces, as models write inside \\boxed
    p = item("BUY 10 CLF7; SELL 10 CLZ6", "orders")
    assert equivalent("SELL 10 CLZ6; BUY 10 CLF7", p)  # order of the orders does not matter
    assert not equivalent("SELL 10 CLZ6", p)
    assert not equivalent("SELL 10 CLZ6; BUY 10 CLF7; BUY 1 ESZ6", p)
    assert not equivalent("SELL 10 CLZ26; BUY 10 CLF27", p)  # the symbol format is part of it
    assert equivalent("NONE", item("NONE", "orders")) and not equivalent(
        "BUY 1 ESZ6", item("NONE", "orders")
    )
    # orders split across boxes after the reasoning are read together
    split = "plan</think>\nFirst \\boxed{SELL 10 CLZ6} then \\boxed{BUY 10 CLF7}"
    assert grade(p, split)["correct"]


def test_numeric_kinds_have_their_own_tolerances():
    assert equivalent("37", item("37", "integer")) and equivalent("-37", item("37", "integer"))
    assert not equivalent("38", item("37", "integer"))
    assert equivalent("$-26,880", item("-26882.10", "number"))  # within 1%
    assert not equivalent("26882", item("-26882.10", "number"))  # the sign matters
    assert equivalent("-76.4", item("-76.5", "value", 0.3)) and not equivalent(
        "-76", item("-76.5", "value", 0.3)
    )
    assert equivalent("15.9\\%", item("15.5", "pct", 1.5)) and equivalent(
        "0.155", item("15.5", "pct", 1.5)
    )
    assert not equivalent("18%", item("15.5", "pct", 1.5))
    assert equivalent("1.11040", item("1.11040", "value", 5e-6))
    assert not equivalent("1.11045", item("1.11040", "value", 5e-6))

# A global-macro futures benchmark for PM and execution agents

The finance comparison (`docs/FINANCE_MODEL_SELECTION.md`) chose a model on FinQA,
CFA-Easy and FinanceIQ. Those questions are mostly accounting and exam knowledge. Under a
third of them touch markets, and none asks what to trade. This set asks what a global-macro
desk does all day when every view goes through futures. It is
`learnai/macrobench.py`, and it runs through the same runner as the finance set, with
`--bench macro`.

## 1. What it measures, and what it does not

A PM or execution agent built on a small model has three jobs the model has to get right:

1. **Arithmetic on contracts**: sizing, P&L and pricing, where one quoting convention
   gotten wrong is a loss.
2. **Translating news into positions**: which way each futures price moves, and which
   position expresses a view.
3. **Following the desk's rules**: stops, rolls, limits, duplicate signals, stale data,
   and instructions hidden in the news.

It does **not** measure whether a model makes money. Backtested or live P&L cannot rank
models at any affordable sample size:

- A Sharpe ratio estimated over *n* trading days has a standard error of about √(252/n).
  StockBench's 82 days give ±1.75, wider than the gaps between the models it ranks.
- On LiveTradeBench, the Spearman correlation between a model's LMArena rank and its
  trading return is 0.05.
- On FINSABER-2, the LLM strategies' edge disappears once transaction costs are counted.
- Next-day movement prediction (ACL18 and similar sets) is near chance for every model
  (MCC ≤ 0.08).

These figures come from the papers and project pages and were not re-measured here. A P&L
run belongs at the end, as a smoke test of the whole agent, not as a way to choose the
model inside it.

## 2. The three sources

Every item is generated from a fixed seed, and its answer is computed by the same code
that writes the prompt. Nothing is downloaded and there is no licence to check. No model
can have trained on these items. That matters here: DianJin-R1 trained on CFLUE and
FinQA. Each prompt states the contract facts it needs (multiplier, face value, how the
price is quoted), as an agent's tool documentation would. The tests then check whether the
model uses those facts correctly.

| source | items | what one item asks | graded as |
|---|---|---|---|
| `futures-calc` | 160 (20 templates × 8) | one sizing, P&L or pricing number | exact count / within 1% / within a stated tolerance / one letter |
| `macro-view` | 124 | the sign of each listed future's first-order reaction to an event (98), or the position that expresses a view (26) | all signs right / one letter |
| `futures-exec` | 120 (10 templates × 12) | the orders to send now, given a book, desk rules, quotes, a contract calendar and the PM's line | the exact set of orders, or `NONE` |

The screening set is the first 50 of each source (`--n 50`), interleaved so that every
template appears in it.

**`futures-calc` templates.**
- Sizing:
  - contracts for a notional;
  - contracts for a USD notional in a foreign index (which side of the FX quote to divide by);
  - beta hedge to a target beta;
  - contracts for a DV01 target (in a foreign currency too);
  - the other leg of a DV01-neutral curve trade;
  - contracts for a volatility target;
  - the most contracts a stop and a risk budget allow (rounded *down*);
  - contracts that fit under a margin cap (0 when it is used up).
- P&L:
  - equity-index futures, converted to USD;
  - Treasury futures quoted in 32nds;
  - Bund, Gilt and JGB futures;
  - SOFR, Euribor and CORRA futures, at 100 minus the rate;
  - yen, Canadian-dollar and franc futures, quoted in USD per foreign unit (the inverse of
    the market's USD/JPY);
  - calendar spreads at negative prices;
  - roll-down when the curve stays unchanged.
- Pricing:
  - the rate change a STIR strip implies;
  - FOMC odds from a fed funds future (the CME FedWatch arithmetic);
  - rich or cheap against cost-of-carry fair value;
  - a limit price on the tick grid;
  - the cheapest-to-deliver bond.
- Risk: the 99% VaR of two legs, where a short leg flips the sign of the correlation.

**`macro-view`.**
- Policy decisions, 60 items: ten kinds of surprise from six central banks (Fed, ECB, BoJ,
  BoE, RBA, BoC). Several are worded so that the verb points the wrong way: a cut
  smaller than priced is hawkish, and so is an expected cut with fewer cuts signalled.
- Inflation and jobs surprises, 20 items.
- Global shocks, 18 items: risk-off, risk-on, an oil supply cut or surge, and a China
  stimulus or slowdown.
- Views, 26 items: a view and four positions to choose from.

The sign rules are data in `ECONOMIES` and `GLOBAL_SHOCKS`, so they can be read and argued
with. When an economy's rate outlook turns hawkish:
- its STIR and bond futures fall;
- its currency future rises, since the futures are quoted in USD per foreign unit;
- for the Fed, it is the other currencies' futures that fall, and the Dollar Index that
  rises;
- the local equity index falls;
- gold falls on a hawkish Fed.

Reactions that are ambiguous in practice are left out rather than guessed:
- After a cut larger than priced: equities, growth currencies and the 10-year. A 50 bp cut
  can read as panic, and 10-year yields *rose* after the Fed's 50 bp cut in September 2024.
- After a jobs report: equities, growth currencies and gold.
- After an inflation print: gold. "Real yields up" and "inflation hedge" point opposite
  ways.
- In a trade deal: gold.
- Canada's TSX 60 is left out entirely. It is mostly banks and energy, so it has no clean
  sign to a rate surprise.

**`futures-exec`.** Every scenario carries the same seven desk rules, so the model has to
find the one that applies. The ten templates:
- stops hit, or just missed;
- rolls inside a 7-day window, keyed on the earlier of first notice and last trading day;
- a DV01 limit that cuts an order down (risk-reducing orders always pass);
- a gross notional limit across USD, EUR and JPY contracts;
- a margin cap;
- a target position, where the order is the difference;
- a trade instruction planted in news or chat;
- a signal already filled;
- stale quotes;
- choosing the contract month for a new position.

About a fifth of the right answers are `NONE`.

## 3. Checking the items

`tests/test_macrobench.py` covers four things:
- It reads the numbers back out of each prompt and recomputes the answer independently,
  so a prompt that states one thing while its answer was computed from another fails.
- It checks the sign answers against the rule table, in the order the prompt lists the
  contracts.
- It pins the desk rules at their edges: a roll date 7 days away rolls and 8 days away
  does not; a limit cuts an order down, but always lets risk come down.
- It feeds every reference answer back through the grader.

Count rules refuse coin flips: a draw whose exact count lies within 0.1 of a half, or a
round-down within 0.05 of a whole number, is thrown away.

**A blind reference pass.** Before any local model ran, three subagents (Claude Opus 5.5)
answered the screening set. They had no tools and saw only the questions, and were asked
to flag any item they found ambiguous. All 150 answers matched the references. That rules
out a broken generator or an unanswerable item, though it does not prove the set is
fair. The flags changed the following:
- Arguable signs were removed from `macro-view`, as listed in §2.
- Three desk rules were spelled out, each in the direction the reference had already
  taken:
  - a new position goes in a contract shown in the calendar, not in a serial month the
    model knows from outside;
  - a new signal ID repeating an earlier trade is still a new signal;
  - a position already beyond a limit, or a book already over the margin cap, is not
    forced down; only new orders are cut.

The 12 screening `macro-view` items these changes touched were answered again.

**A second blind pass, 2026-11-11.** After those changes, fresh subagents answered the
screening `macro-view` and `futures-exec` items again, in their final wording (the
`futures-calc` prompts had not changed). They also answered the next 150 items, which no
reference had checked: the held-out slice (`--offset 50`) used below to measure a
playbook. All 300 answers matched the keys. The flags they raised change no answer, but a
later version of the set should fix them:
- The short-rate legs (`CRA`, `TONA3`) name no expiry. After a decision that came in as
  priced, a front contract whose rate period is already fixed may hardly move. Naming the
  contracts "two to four quarters ahead", as the view items do, would remove the doubt.
- In `duplicate` scenarios the book does not reflect the fills already logged today.
  Either build the book after those fills, or label it the start-of-day book.
- Some calendars quote neighbouring months implausibly far apart (CLF7 against CLG7, 34%).
- Rule 6 does not say how to cut an order that flips a position through zero. The key cuts
  the whole order down to the limit, and the reference read it the same way.
- One `risk-on` text reads as a ceasefire between the US and China, an odd premise.

## 4. Results

The screening set has 150 items, 50 per source. DianJin ran at the finance study's
recommended setting: 4-bit, a 4,096-token cap, and the answer forced at the cap
(`FINANCE_MODEL_SELECTION.md` §8.1). The brackets are 95% Wilson intervals.

| | futures-calc | macro-view | futures-exec | all |
|---|---|---|---|---|
| DianJin-R1-7B, 4k, forced | 26 | 22 | 24 | **72 (48%)** [40–56] |
| Claude Opus 5.5, no tools (reference) | 50 | 50 | 50 | 150 (100%) |

For comparison, DianJin scored 76% on the finance set (FinQA, CFA-Easy and FinanceIQ). On
the macro set it gets under half, and the misses are not spread evenly:

- **Arithmetic it does, conventions it does not.**
  - It gets most of the simple sizing and P&L items right: notional, DV01, curve legs,
    stops, margin, bonds in decimal points.
  - It gets none of these right:
    - FOMC odds from fed funds futures (0/3);
    - the cost-of-carry fair value (0/2);
    - two-leg VaR, where a short leg flips the correlation (0/2);
    - P&L on yen, Canadian-dollar and franc futures, quoted in USD per foreign unit (0/2);
    - the cheapest-to-deliver bond (0/2).
- **Signs: a coin flip exactly where a trader is tested.**
  - Global shocks: 85% of individual signs right.
  - Plain policy and data surprises: 77%.
  - Decisions whose verb points the other way from the surprise: 50%, chance level. These
    are a cut smaller than priced, a hold when a cut was priced, and guidance against the
    decision.
  - Currency futures: 11 of 22 signs right, also chance. 6J, 6C and 6A are the worst.
  - Only 12 of 38 sign items are right in full.
  - Choosing a position for a stated view is easier: 10 of 12.
- **Execution: rule-following that breaks on the details.**
  - It handles signals cut by a limit or by margin, duplicate IDs, and planted
    instructions (4 of 5 each).
  - It fails the target-position items: 0 of 5. It sends the target size, or the wrong
    side, instead of the difference.
  - It mostly fails choosing the contract month: 1 of 5. It picks the later contract when
    the front one is still outside its roll window.
  - It mostly fails stops: 1 of 5. It closes positions whose stop was not hit, or writes
    `CLOSE` instead of an order.
  - It gets 2 of 5 rolls right. Short positions get rolled the wrong way.
- 10 answers reached the cap, 7 of them looping. Forcing an answer at the cap recovered
  one.
- Cost:
  - 23 minutes of wall time for the 150 items at batch 16, peak 9.0 GiB.
  - One question alone takes a median 34 s.

**The grader was fixed once after this run.** The model writes LaTeX spaces inside the
box (`SELL\ 4\ NQZ6`), and the shared TeX clean-up deleted them, which fused each order
into one word. It also sometimes doubles a sign (`++` per contract). Both are now read as
intended, with tests. Re-grading the stored transcripts changed 4 verdicts, all to right.
The numbers above are after that fix.

**What the reference does and does not show.** 150/150 means the items are well defined
and answerable from the prompt alone. It does not prove the keys are right where the
reference and the generator share a blind spot, since both came from the same model
family. The sign rules are written out in §2, so they can be argued with.

**Qwen3.5-4B, 2026-11-11.** It ran at the finance study's 4B setting: 4-bit, thinking on,
a 16,384-token cap, batch 11, and the answer forced at the cap afterwards.

| | futures-calc | macro-view | futures-exec | all |
|---|---|---|---|---|
| Qwen3.5-4B, 16k, as it stands | 32 | 18 | 27 | 77 (51%) [43–59] |
| Qwen3.5-4B, 16k, forced | 41 | 40 | 41 | **122 (81%)** [74–87] |

- **Its main failure is not stopping.**
  - 81 of the 150 answers ran into the 16k cap, 36 of them looping. Of the 69 that
    finished, 59 were right (86%).
  - A typical sign item has the right signs within the first few thousand tokens. It then
    repeats the same three-line check ("Wait. If the market priced in 50bp…") until the cap.
  - Forcing an answer at the cap recovers 45 items. `macro-view` goes from 18 to 40.
- **Cost.** 256 minutes of wall time for the 150 items, peak 11.46 GiB. That is eleven
  times DianJin's 23 minutes. A median question asked alone would take about five minutes.
- **What stays wrong after forcing** (28 items) is a few conventions and one blind spot in
  the desk rules:
  - Stops and rolls. It answers `NONE` to 9 of the 10 scenarios, and is right only where
    `NONE` is the key (5 of 5), plus one stop. Every one of these scenarios has
    "PM: no new trades today", and the model lets that line override the mandatory stop
    and roll rules. In one roll it worked out that the roll date was 4 days away, and still
    answered `NONE`.
  - Inverted FX quotes (0/2): it uses the market's USD/CAD rate as the 6C futures price.
  - STIR P&L (0/3): it reads a price change of 0.245 as 0.245 bp, not 24.5 bp.
  - Treasury P&L in 32nds (1/3): one answer drops the number of contracts, reading
    "31 2-year Treasury note futures" as a typo.
  - Limits (3/5 and 4/5): it asserts that the book is already beyond the limit without
    computing it, and sends the whole order.
  - Global shocks (6/12) are its weakest `macro-view` family. Three forced answers came
    out as `---` with no commas, and were scored wrong.
- **Against DianJin**, at each model's own setting, 4B wins on each source. The biggest
  gains are signs (data surprises 13/13, policy 10/13) and the execution templates DianJin
  failed: target positions (5/5) and choosing the contract month (5/5).

**A smaller budget, with the answer forced.** The first *c* tokens of a 16k answer are what
a *c*-token cap would have produced. So `force --cap c` on the same transcripts shows each
smaller cap without another generation run.

| 4B, forced at | futures-calc | macro-view | futures-exec | all | longest wait alone |
|---|---|---|---|---|---|
| 2,048 | 24 | 30 | 34 | 88 (59%) [51–66] | ~40 s |
| 4,096 | 29 | 33 | 39 | 101 (67%) [59–74] | ~75 s |
| 8,192 | 36 | 36 | 40 | 112 (75%) [67–81] | ~2.5 min |
| 16,384 | 41 | 40 | 41 | 122 (81%) [74–87] | ~5 min |

- Each halving of the budget costs 10 to 13 items, and each step is significant in a paired
  test (16k against 8k: 12 lost, 2 gained, p = 0.01; 8k against 4k: p = 0.03; 4k against
  2k: p = 0.02).
- Execution holds up best: 39 of 50 at 4k against 41 at 16k. Arithmetic loses most.
- At the same 4k budget, 4B scores 101 to DianJin's 72 (51 items gained, 22 lost,
  p < 0.001). Even at 2k, 4B is ahead.

## Reproducing

```
# the screening set (50 per source) with one model, at the finance study's settings
python tools/finance_model_select.py generate --bench macro --model dianjin-r1-7b \
    --max-tokens 4096 --batch 16
python tools/finance_model_select.py force runs/finbench/dianjin-r1-7b/macro-n50-o0.jsonl --cap 4096

# the same questions as text, for a model without a local runner, and its graded answers
python tools/finance_model_select.py export --bench macro --out <dir>
python tools/finance_model_select.py grade-answers --bench macro <answer files> --label <name> --out <run>
```

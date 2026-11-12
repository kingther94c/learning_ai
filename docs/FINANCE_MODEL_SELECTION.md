# Choosing an open-weight model for finance questions

*Five candidates, one memory budget, three kinds of finance question — then the
weight-editing recipe of `docs/WEIGHT_EDITING.md` applied to the winner's own bad habits.*

**The question.** On a 32 GB Mac, which released open-weight model answers finance
questions best when it must share the machine with everything else — and can a
few thousand hand-edited weights make it better still, without any training?

**Why a budget first.** "Fits in memory" is not one number. The weights are fixed,
but every token of context costs cache memory, and every sequence decoded in
parallel multiplies that cost. A 4B model with 16k tokens of context is cheap; the
same model answering 16 questions at once needs 11 GiB. So the comparison is made
*under a stated budget*, and each model gets whatever context and parallelism fit
inside it.

Everything below is **measured here** (this Mac, the code in this repo) unless
labelled otherwise.

---

## 1. The budget

The machine has 32 GiB. Metal lets a process keep 25.0 GiB resident on the GPU
(`mx.device_info()`); past that a run dies with `[METAL] Insufficient Memory`
rather than slowing down, because GPU memory is wired and never swapped.

The budget chosen — by the owner, after seeing the first run's numbers: **the MLX
process peaks at ≤ 12 GiB** (first set at 16; lowered on 2026-11-08). This Mac's
baseline without any model is ~3 GiB wired, so 12 GiB leaves more than half the
machine for a Python or Jupyter session, a browser and whatever else runs while a
benchmark goes on in the background.

Cache cost per model, from each model's config (`learnai/finbench` has no say in
this; the arithmetic is in `tools/finance_model_select.py` and the numbers for the
4B were verified with `mx.get_active_memory`):

| model | 4-bit weights, resident | cache per token | fixed per sequence | context at batch 1 | sequences at a 16k cap | worst-case peak |
|---|---|---|---|---|---|---|
| Qwen3.5-4B | 2.2 GiB | 32 KiB (8 full-attention layers) | 49 MiB (24 DeltaNet states) | ~300k | 11 | 11.5 GiB |
| Qwen3.5-9B | 4.9 GiB | 32 KiB | 49 MiB | ~220k | 8 | 11.8 GiB |
| Gemma 4 12B-it | 5.7 GiB | 64 KiB (8 of 48 layers are full attention) | 320 MiB (40 sliding windows of 1,024) | ~70k | 3 | 11.9 GiB |
| DianJin-R1-7B (Qwen2.5 base) | 4.0 GiB | 56 KiB (28 layers, all full) | – | ~120k | 5 | 11.0 GiB |
| gpt-oss-20b (MoE, 3.6B active) | 11.3 GiB | 24 KiB (12 of 24 layers) | 3 MiB | — | 0 | 12.4 GiB at batch 1 |

The worst case is every slot holding a sequence at the cap (16,384 generated plus a
2,300-token prompt), plus the transients a batched run was measured to add on top of
what it holds: about 0.5 GiB, and 30% of the caches (telemetry every 256 decoding
steps on the 4B — the copies made while a finished sequence is swapped for a new one,
and reading prompts 2 × 512 tokens at a time). gpt-oss-20b is out: its weights alone
are 11.3 GiB. It stays in the code for anyone who wants the data point with
`--limit 16`.

**Three things made the first attempts use far more than the arithmetic says.**
With mlx-lm's default prompt processing (8 prompts at once in 2,048-token chunks) a
batch of 1.5k-token FinQA excerpts took the 4B to a 9.0 GiB peak; reading 2 prompts
at a time in 512-token chunks gave 4.0 GiB at the same speed. MLX keeps every freed
buffer in an allocator cache whose limit defaults to the *memory* limit (30 GiB on
this machine): a batched run outgrows its KV caches every 256 tokens, so the
outgrown copies pile up as cached garbage until the next `clear_cache`, and the
system counts them as used. And when a sequence finishes, mlx-lm hands back a copy of its own cache, cut out of the
whole batch's arrays — 0.5 GiB per sequence at 16k on the 4B — and every sequence of a
wave that runs to the cap finishes on the same step: at step 16,384 of one run the
held memory jumped from 8.6 to 15.1 GiB and was back at 5.9 one interval later. Nothing
here continues a conversation, so `sample_batch` now switches that extraction off.
Every run here reads prompts 2 × 512 at a time, sets `mx.set_memory_limit(12 GiB)` and
`mx.set_cache_limit(1 GiB)`, and logs active, cached and peak memory every 256 steps
into `<run>.mem.jsonl`.

## 2. The questions

| source | language | one question | pool | screening | finalists |
|---|---|---|---|---|---|
| FinQA (Chen et al., 2021; test split, MIT) | en | a 10-K excerpt — text and a table — and a numeric question about it | 1,147 | 50 | +100 |
| CFA-Easy (Salesforce FinEval; CC-BY-NC-4.0) | en | a CFA level-I style question, three options | 1,032 | 50 | +100 |
| FinanceIQ (Du Xiaoman; CC-BY-NC-SA-4.0) | zh | a question from one of ten professional exams (CPA, securities, funds, futures, banking, tax, actuarial maths, economist, financial planner, insurance), four options | 7,123 | 50 (5 per exam) | +100 (10 per exam) |

Problems are drawn in one fixed shuffled order per source (seed 0), so the
finalists' extra 100 never overlap the screening 50. Every prompt asks for
step-by-step reasoning and the final answer in `\boxed{}`.

**Grading** (`learnai/finbench.py`, tested without a model):

- a choice is right when the boxed letter is the reference letter;
- a FinQA number is right when it is within 1% of the reference, or equals the
  reference rounded to the precision the model chose ("14%" for 0.14464 is right,
  "15%" is not); a question that asks for a percentage or a rate may be answered in
  percent or as a ratio; an answer with a `%` sign is read as a ratio, unless the
  reference is itself at least 1 (FinQA's programs sometimes multiply by 100, so 940
  stands for 940%);
- an answer that is not in a box scores zero *and is counted separately* as a
  format failure; an answer cut off by the cap is counted as such, and whether it
  was looping is measured by the repeated-16-gram rate of its tail.

Sampling follows each model's own recommendation, because a model judged under
another family's settings is judged unfairly: Qwen 3.5 — temperature 0.6, top-p
0.95, top-k 20, presence penalty 1.5 (the paper's settings); Gemma 4 — 1.0 / 0.95 /
64; gpt-oss — 1.0 / 1.0, reasoning effort `medium`; DianJin — 0.6 / 0.95. One
sample per problem, seed 0. Thinking mode on for every model that has one.

Two of these rules were added after reading misses, not before: the percent-unit
reference (found among the offline reference's misses) and "rate" questions (a model
answering 21.3 for an effective-tax-rate difference of 0.213). Every run was re-scored
afterwards (`regrade`); one or two verdicts changed per run, and all numbers below are
the re-scored ones.

## 3. Screening: 150 questions, 16k cap

| model | FinQA | CFA-Easy | FinanceIQ | all [95% Wilson] | cut by the cap (looping) | right among finished | median tokens | one question alone (median) | run time |
|---|---|---|---|---|---|---|---|---|---|
| DianJin-R1-7B | 78% | 88% | 66% | **77%** [70–83] | 9 (9) | 82% | 744 | 22 s | 53 min |
| Qwen3.5-4B | 76% | 86% | 66% | **76%** [69–82] | 19 (4) | 86% | 4,934 | 88 s | 111 min |
| Qwen3.5-9B | 80% | 84% | 54% | **73%** [65–79] | 31 (0) | 92% | 7,171 | 231 s | 241 min |
| Gemma 4 12B-it | 72% | 88% | 60% | **73%** [66–80] | 19 (8) | 84% | 4,495 | 231 s | 387 min |
| *Claude, offline (reference, §6)* | 88% | 96% | 92% | 92% [87–95] | – | – | – | – | – |

"One question alone" is the prompt read at the measured prefill speed plus the
answer's tokens at the measured single-sequence decoding speed (DianJin 33.9 tok/s,
4B 56.2, 9B 31.1, Gemma 19.6).

All four land within 73–77% and their intervals overlap: 150 questions cannot rank
them. What separates them is how they answer. DianJin answers in a few hundred to a
thousand tokens; the three general models think for four to seven thousand. Gemma 4
is no better and by far the slowest, and drops out here. The 9B scores lowest overall
but is right on 92% of the answers it finishes — it loses on the fifth of the
questions where it is still thinking at 16,384 tokens.

**The same answers under smaller caps** (each stored answer truncated and re-graded,
`curve`):

| cap | 1k | 2k | 4k | 8k | 12k | 16k |
|---|---|---|---|---|---|---|
| DianJin-R1-7B | 56% | 75% | 76% | 77% | 77% | 77% |
| Qwen3.5-4B | 7% | 23% | 47% | 64% | 73% | 75% |
| Qwen3.5-9B | 1% | 11% | 35% | 55% | 67% | 73% |
| Gemma 4 12B-it | 9% | 27% | 45% | 56% | 67% | 73% |

So the budget's real constraint is not how big a model fits — every candidate gets
70k+ tokens of context alone inside 12 GiB — but how long it may think, which sets
both the wait for one question and the parallelism of a benchmark.

## 4. What goes wrong, and the finalists on 450 questions

| model | right | wrong (finished) | scale/unit slip | no box | cut, still thinking | cut, looping |
|---|---|---|---|---|---|---|
| DianJin-R1-7B | 116 | 25 | 0 | 0 | 0 | 9 |
| Qwen3.5-4B | 114 | 18 | 0 | 0 | 15 | 3 |
| Qwen3.5-9B | 109 | 10 | 0 | 0 | 31 | 0 |
| Gemma 4 12B-it | 110 | 21 | 0 | 0 | 11 | 8 |

Finished-but-wrong answers are capability; an edit of a few columns will not add
knowledge. Answers cut off by the cap are behaviour, and the three Qwen-family models
(DianJin is a Qwen2.5 fine-tune) each have a different one: every one of DianJin's
unfinished answers is a verbatim loop — the defect the paper removed from
Qwen3.5-0.8B; none of the 9B's 31 is a loop — it keeps re-opening a settled question
("However, let's look at the wording again… Is there another interpretation?");
the 4B does both.

The two finalists, DianJin and the 4B, answered 100 more questions per source
(offset 50, disjoint from the screening slice):

| model | 300 new: FinQA | CFA-Easy | FinanceIQ | all | all 450 [95% Wilson] | cut (looping), new 300 |
|---|---|---|---|---|---|---|
| DianJin-R1-7B | 67% | 74% | 68% | 70% | **325 = 72.2%** [68–76] | 25 (25) |
| Qwen3.5-4B | 75% | 81% | 62% | 73% | **332 = 73.8%** [70–78] | 30 (8) |

A tie: seven questions apart. Their strengths are complementary — the 4B is better
at the English report arithmetic and CFA concepts, DianJin at the Chinese
professional exams — and their costs differ tenfold (2.1 vs 4.1 hours for the 300;
22 vs 88 s per question alone).

## 5. Weight edits

The recipe is the one `docs/WEIGHT_EDITING.md` reproduces: matched pairs that differ
only in the habit → the mean residual difference at their ends is a direction `d` →
the 64 MLP neurons per block whose output most pushes along `d` in the habit →
`w_j ← w_j − 6 (w_j·d) d` on those columns → a random-direction control of the same
size. Two rules keep it honest here:

- **Design on the screening slice, judge on the extension.** Pairs, block choice and
  patch choice use only the 150 screening questions (offset 0) and the models' answers
  to them; generation runs that judge an edit use the 300 extension questions
  (offset 50), which the design never saw.
- **Read before running.** A candidate is first applied to the model *reading*
  existing text (`probe`, teacher-forced, seconds per candidate): how much less
  likely does the habit's text become, how much does ordinary reasoning suffer, and
  — the filter — does it change the model's *own* answers that show the habit? Only
  a patch that passes goes on to hours of generation. Since the second edit below the
  choice of blocks and the pass/fail decision are fixed rules (`tools/edit_gate.py`),
  not judgement calls.

Long transcripts are read through a chunked KV cache (`prefill_cache`): context costs
only its cache, and the scored stretch is computed exactly as in one pass (tested
against a full forward pass on the small model).

### 5.1 DianJin: its own loops

Pairs from the model's own nine screening-set loops (`loop_pair`): the looping passage
the first time it was written, against the same passage written again — same words,
same position, differing only in being a repeat. Per-block probe
(`runs/weightprog/dianjin-r1-7b/probe-each-block-realloops.txt`), change in mean
log p per token:

| block edited alone | 18 | 19 | 20 | 26 |
|---|---|---|---|---|
| repeated tokens in real loops | −0.105 | −0.061 | −0.096 | −0.185 |
| ordinary reasoning | −0.045 | −0.013 | −0.018 | −0.065 |
| hand-written repetition pairs (independent of the diagnosis) | −0.029 | −0.085 | −0.131 | −0.010 |

Combined patches (`probe-patches.txt`): blocks 19+20 lower the loops by 0.30 for 0.05
on ordinary reasoning; 18–20 by 0.89 but 0.18 on ordinary reasoning (too costly);
the three random controls change nothing (|Δ| ≤ 0.004). **Chosen: blocks 19+20, 64
columns each, strength 6** — 458,752 of 7.6 billion weights (0.006%).

Judged on the 300 extension questions (never seen while designing), against the
unedited model and a random-direction control of the same size on the same columns:

| | unedited | random control | edited |
|---|---|---|---|
| correct, all 300 | 209 | 224 | **215** |
| correct, the 25 the unedited model looped on | 0 | 10 | 14 |
| correct, the 275 it did not loop on | 209 | 214 | 201 |
| answers looping to the cap | 25 | 20 | **3** |
| tokens generated · wall time · peak | 662k · 125 min · 9.5 GiB | 620k · 115 min · 10.2 GiB | 467k · 80 min · 9.6 GiB |

The subsets are defined by the unedited run alone, so neither the edit nor the control
chooses the questions it is judged on.

- **The loops are gone, and it is the edit's doing**: edited vs control, 20 stopped
  looping and 3 started (exact McNemar p < 0.001). 29% fewer tokens.
- **Accuracy did not improve.** On the questions that looped, the edit recovers a few
  more than the control does (14 vs 10; +6/−2, p = 0.29); on the questions that did
  not, it is behind the control (201 vs 214; +10/−23, p = 0.035) and the unedited
  model (201 vs 209; p = 0.20). A small cost on ordinary questions, about 3–4 points,
  cancels the gain on loops.
- **The control is the lesson.** It changes no log-probability in the probe, yet
  scored 15 more than the unedited model (+28/−13, p = 0.03). At temperature 0.6
  with one sample per question, two runs that should be equivalent disagree on 41 of
  300 questions, and looping itself is half chance: the control looped again on only
  10 of the unedited model's 25 loops. Without the control, the edit's "+6" would
  read as a win. Single-run differences under about five points — including the 4B's
  7-question lead over DianJin on 450 — are not differences.

A token cap does the practical part for free: DianJin's accuracy at 4,096 tokens
equals its accuracy at 16,384 (screening 76% vs 77%, extension 70% vs 70%) with
40–47% fewer tokens, because a looping answer is wrong either way. The edit's value
is the mechanism — the loop is a direction that can be located and switched off in a
finance-tuned model too — not a better finance model.

### 5.2 The Qwens: deliberating after the answer

First attempt, the paper's way: six hand-written pairs (English and Chinese) whose
reasoning has reached its answer; one ending checks once and commits, the other
re-opens with several different doubts. (A first version said one doubt three times;
its direction was half "repeating" — on the 4B it acted most on loops. Doubts are now
said once.) The per-block probe looks promising — block 18 of the 4B lowers the
hand-written doubts by 0.24 nats/token for 0.002 on ordinary reasoning — but the
filter fails: on the models' own answers that ran into the cap, the two- and
four-block patches move the text by 0.001–0.007 nats/token and leave log p(</think>)
where it was, on the 4B and on the 9B alike. The direction captured the style of the
hand-written doubts, not the models' deliberation; the gate rejects every patch and
none was run.

Second attempt, from the 4B's own answers (`junction_pairs`): moments just after it
states a conclusion where a finished answer goes on to write `</think>`, against
moments where an answer that later ran into the cap starts a new line questioning it
("Wait", "If the…", "Let's check…", "不过…"; `decision_points`). 48 of each. These
are not the same problems — questions that run long are harder — so the probe and the
control carry more weight here. Block choice, patches and the go/no-go are fixed rules
in `tools/edit_gate.py`, decided before the probe ran.

It fails the same filter. Edited one at a time, no block moves the model's own cut-off
text by more than 0.006 nats/token, and most cost ordinary reasoning more than that
(`probe-each-block-junctions.txt`). The two patches the rules picked (blocks 4+18 and
4+5+7+18) move it by −0.004 and −0.008 (gate: −0.02) and raise log p(`</think>`) there
by 0.16 and 0.19 (gate: 0.5), against −0.001/−0.06 and −0.003/+0.06 for their controls
(`probe-patches-junctions.txt`) — and they make the hand-written doubts *more* likely
(+0.05, +0.10). No generation run.

The baselines say why the recipe stops DianJin's loops and not this. Inside a loop the
repeated text is near-certain (mean log p −0.004 per token); the DianJin patch takes
0.30 nats off every token, so each copied token is 74% instead of 99.6% likely and the
loop breaks within tens of tokens. At the 4B's cut-offs log p(`</think>`) averages
−26.7: closing there needs twenty-odd nats, and a two-block, 64-column edit delivers
0.2. A verbatim loop is one narrow, confident mechanism; "think again" is a tendency
spread through the whole reasoning, not a switch at one place. The negative result is
for this recipe only (one direction, MLP output columns, strength 6) — larger edits,
attention, or fine-tuning may work, but are outside "read the weights, then edit".

## 6. The gap to an offline frontier model

The same 150 screening prompts went to Claude (Fable 5.1 at the time) as 14 subagents —
five FinQA excerpts or 25 multiple-choice questions each — with every tool forbidden:
no search, no files, no code. Their `uid => \boxed{answer}` lines were graded by the
same grader (`finance_model_select.py export` writes the batches, `grade-answers`
scores the replies). **138/150 = 92%** [87–95]: FinQA 88%, CFA-Easy 96%,
FinanceIQ 92% — fifteen points above the best local model, and 26–38 points on the
Chinese exams.

Where the local models lose to it differs. Of the questions the reference got right
and a model got wrong, the 9B was cut off by the cap on 27 of 30, the 4B on 16 of 26,
Gemma on 18 of 32, DianJin looped on 9 of 25: the 9B's gap is almost entirely not
finishing (on the answers it finishes it matches the reference's 92%), DianJin's is
mostly knowledge. Seven questions defeated all five; at least two of them have
questionable references (`finqa/969` asks for total off-balance-sheet obligations,
the reference is 670, the excerpt lists three items totalling 1,171; `finqa/1032`
has its sign reversed). Such items lower every score alike and leave comparisons
intact.

Not a controlled comparison: a cloud model, no memory budget or token cap, 5–25
questions per call, reasoning not visible. It measures how far the local models are
from what a frontier model knows and works out without help.

## 7. The recommendation and its boundary

**DianJin-R1-7B, 4-bit, capped at 4,096 generated tokens** — and, since §8.1, with
the answer forced at the cap (76.4% on 450).

1. *As accurate as the best candidate*: 72.2% on 450 against the 4B's 73.8% — seven
   questions, well inside run-to-run noise (§5.1).
2. *It does not need to think long*: 16k → 4k costs one question in 450 (325 → 324)
   and halves generation; a single question takes a median 22 s against the 4B's 88 s.
   The 4B falls to 37–47% at 4k, so it cannot be capped the same way.
3. *Memory to spare*: 4.0 GiB of weights and at most ~0.35 GiB of cache per stream at
   4k — about 5 GiB for one stream, far inside 12.
4. *Chinese exams no worse*: FinanceIQ 100 vs 96 of 150 (66%/66% screening, 67%/63%
   extension) — within noise.

Choose otherwise when: the work is mostly English annual-report arithmetic and a
minute or two per question is acceptable (Qwen3.5-4B at 16k: FinQA 75% vs 71%, CFA
83% vs 79%, both within noise); or a question may take ten minutes — Qwen3.5-9B is
right on 92% of the answers it finishes (the offline reference's rate) and may be the
most accurate local model with a 32k+ cap, which was not measured here.

The edited DianJin is not recommended: it removes the loops but not the errors (§5.1),
and the cap gets the same savings without touching a weight.

**Boundary.** One sample per question (±4 points at 450); multiple-choice and
single-number questions only — no open analysis, report writing or follow-ups; a few
FinQA references are wrong (§6), equally for all; DianJin was trained on finance
data and public exam questions may be in it, so discount its Chinese-exam showing;
measured on this Mac with MLX 4-bit weights only.


## 8. More accuracy inside the same budget

Weight edits removed DianJin's loops but not its errors (§5). What else raises
accuracy without raising the 12 GiB peak? Three things were measured on the stored
runs and the GPU; the rest is from the literature and marked as such.

### 8.1 Answer at the cap

When an answer reaches the cap still thinking, close the thinking and ask for the
answer instead of scoring the cut-off as wrong: append
`\n</think>\n\n**Final answer:** \boxed{` and let the model fill the box (greedy, at most
48 tokens). This is the budget forcing of s1 (Muennighoff et al. 2025,
arXiv:2501.19393). It needs no memory — the cache is already there — and about ten
tokens. `finance_model_select.py force` re-reads each stored answer up to the cap
(chunked, `prefill_cache`) and appends the cue, so these are exactly the answers a
capped run with the cue would have given.

| model, cap | cut-offs scored wrong | answer forced at the cap | forcing fixed / broke | one question alone (median) |
|---|---|---|---|---|
| DianJin-R1-7B, 2k | 315 (70.0%) | **340 (75.6%)** [71–79] | 25 / 0 | 22 s |
| DianJin-R1-7B, 4k | 324 (72.0%) | **344 (76.4%)** [72–80] | 20 / 0 | 22 s |
| DianJin-R1-7B, 16k | 325 (72.2%) | 342 (76.0%) | 17 / 0 | 22 s |
| Qwen3.5-4B, 4k | 181 (40.2%) | **332 (73.8%)** [70–78] | 153 / 2 | 73 s |
| Qwen3.5-4B, 8k | 281 (62.4%) | 341 (75.8%) | 62 / 2 | 95 s |
| Qwen3.5-4B, 16k | 332 (73.8%) | **352 (78.2%)** [74–82] | 21 / 1 | 95 s |
| Qwen3.5-9B, 8k (screening 150 only) | 83 (55.3%) | **130 (86.7%)** [80–91] | 47 / 0 | 231 s |
| Qwen3.5-9B, 16k (screening 150 only) | 109 (72.7%) | 129 (86.0%) [80–91] | 20 / 0 | 231 s |

All 450 questions unless marked; peak memory 3.2–6.4 GiB, one sequence at a time. The
times are medians over questions answered alone; most answers end before 8k, so the
9B's median is the same at both caps (its 90th percentile is 264 s at 8k, 527 s at 16k).

- **Forcing never costs and often pays.** It broke at most 2 answers in any row. Of
  the cut-off answers it was applied to, 54% (DianJin at 4k), 67% (4B at 4k) and 81%
  (9B at 8k) ended with the right answer — on numeric FinQA questions too, where a
  guess cannot be right. By the cap, most of the work is usually done.
- **The length curves flatten.** The 4B forced at 4k equals its old 16k score, at a
  bounded 73 s per question instead of up to 5 minutes. DianJin forced at 2k is within
  four questions of 4k.
- **It is what the loop edit was for.** DianJin's loops usually come after the answer
  has been worked out: on the same 300 extension questions forcing at 16k scores 220,
  against 209 for the stock model and 215 for the edited one — with no weight touched.
- **The 9B forced at 8k is the best local score measured**: 130 of 150, eight short of
  the offline reference (138) — but on the 150 screening questions only, and against
  DianJin forced at 4k it is 15 questions right where DianJin is wrong against 7 the
  other way (McNemar p = 0.13). It needs the 300 extension questions before it can be
  called better, and it is ten times slower than DianJin. Doubling its cap to 16k adds
  nothing (129 forced).

### 8.2 Several answers, one vote

Sample each question three times and take the most common answer (self-consistency,
Wang et al. 2022, arXiv:2203.11171). Measured on DianJin at 4k with forcing over the 300
extension questions: the stored seed-0 answers cut at 4k, plus two new runs with
`--max-tokens 4096 --sample-seed 1` and `2`.

| | right of 300 |
|---|---|
| one sample (seeds 0, 1, 2) | 222, 221, 224 — mean 222.3 (74.1%) |
| majority of three (ties go to seed 0) | **225 (75.0%)** |
| right in at least one of the three (pass@3) | 247 (82.3%) |
| right in all three | 196 |

- **Three votes buy three questions.** Against each single run the majority is right
  where the run is wrong on 9–11 questions and the reverse on 6–9 (McNemar p ≥ 0.48);
  with ties sent to seed 1 or 2 instead, 223 and 227. Not distinguishable from one sample.
- **Forcing took the gain voting would have had.** Before forcing, three stored runs at
  4k (stock, loop-edited, random control) voted 8 above their average — mostly by
  outvoting a sample that looped. Forced answers leave few cut-offs to outvote.
- **Agreement is a usable confidence signal.** When all three agree (195 questions) the
  answer is right 84% of the time; two of three, 61% (90); all different, 40% (15). A
  deployment could answer the agreed two thirds and flag the rest.
- **The ceiling is a selector, not more votes.** 51 questions are right in one or two of
  the three samples; one sample gets 26 of them on average, the majority 29. The 22 it
  misses are what a verifier able to pick the right sample (Fin-PRM in §8.5) would go
  after.

Memory: three samples of one question run as a batch of three at ~5.8 GiB instead of
~4.9, each stream somewhat slower than alone (one stream decodes at 34 tokens/s, sixteen
share 160); the slowest of the three sets the latency (median 1,112 tokens against 724
for one sample, 90th percentile the full 4,096). For a batch job it is three times the
tokens.

### 8.3 Two models: hand DianJin's cut-offs to the 4B

From the stored runs: take DianJin at 4k, and where it is cut off, the 4B's 16k answer
instead. 338 of 450 (+14), with the 4B consulted on 8% of the questions and both
models resident (6.2 GiB of weights). Forcing DianJin's own answer gets 344 with one
model — the hand-off is not worth its complexity.

### 8.4 What DianJin's FinQA misses are

Of DianJin's 38 finished-but-wrong FinQA answers, 4 are a percentage written without
its sign where the reference is a ratio and 3 have the sign of a change reversed —
conventions of FinQA's references more than errors — and the reference value
appears somewhere in the reasoning of 15 of the 37 numeric ones. Reading them, most of the rest picked the wrong
line of the table or the wrong formula; plain arithmetic slips (`finqa/24`: 26,900 /
16,900 written as 1.61) are a handful. A calculator would fix few of them.

### 8.5 The rest, from the literature (not measured here)

| method | memory under the budget | what is reported | caveat |
|---|---|---|---|
| Run Python for arithmetic (program-of-thought) | none | FinQA 40 → 65 with Codex (Chen et al., arXiv:2211.12588); LoRA-tuned Mistral-7B writing Python 77% (Phogat et al., arXiv:2408.12337) | DianJin's chain of thought already scores 78% on FinQA by its authors' count, and §8.4 finds few arithmetic slips |
| LoRA fine-tuning on the model's own right answers (rejection sampling, or GRPO) | QLoRA on a 4-bit 7B: weights ~4–5 GiB, plus activations; logits of a 4k-token trace over a 152k vocabulary are ~2.3 GiB in fp32 — fits 12 with gradient checkpointing and batch 1 | LLaMA-7B GSM8K 36 → 42 with 47k self-generated examples (RFT, arXiv:2308.01825); DianJin's own recipe (38k SFT + GRPO) is what made it | needs training questions disjoint from the test slices (FinQA's train split; CFA and FinanceIQ have none here); hours of generation plus training |
| A finance process reward model choosing among N answers | a second 7B (Fin-PRM is Qwen2.5-7B-based): ~4 GiB more at 4-bit | best-of-16 beats majority voting by 5 points on CFLUE (Fin-PRM, arXiv:2508.15202, self-reported) | no gain over voting on MATH-500 in the same paper; weights' licence not checked |
| 8-bit weights for DianJin | 7.6 GiB of weights; one stream at 4k ≈ 8.5 GiB | typically small gains over 4-bit on 7B reasoning | not measured; the bf16 checkpoint is on disk |
| Quantised KV cache | saves cache, which is small at a 4k cap | Qwen2.5-based reasoning models collapse with 4-bit KV, 8-bit near-lossless (Liu et al., arXiv:2504.04823) | unnecessary at 4k; only worth it for many long streams |
| Retrieval of reference texts | a small embedding model, < 1 GiB | mixed gains on CFA; none measured for Chinese exams | needs a corpus; can break answers that were right |
| Another finance model | similar | ODA-Fin-RL-8B (Qwen3-8B, Apache-2.0, March 2026) claims +4.3 over DianJin-R1-7B on nine benchmarks | self-reported; a screening run would settle it |
| Speculative decoding, prompt caching | a draft model / a cached prefix | speed only | — |

### 8.6 What changes

**DianJin-R1-7B, 4-bit, capped at 4,096 tokens, with the answer forced at the cap:
344 of 450 (76.4% [72–80])**, up from 324, at the same ~5 GiB and the same median 22 s.
The cue is one string appended to the cut-off text; nothing is retrained and no memory
is added. A 2k cap with forcing (340) trims the slow tail (90th percentile 61 s against
83 s) at a cost of four questions.

| | of 450 | of the 150 screening questions | median per question | peak, one stream |
|---|---|---|---|---|
| DianJin-R1-7B, 4k, forced (recommended) | 344 (76.4%) | 122 (81.3%) | 22 s | ~4.9 GiB |
| Qwen3.5-4B, 16k, forced | 352 (78.2%) | 123 (82.0%) | 95 s | ~3.5 GiB |
| Qwen3.5-9B, 8k, forced | not run | 130 (86.7%) | 231 s | ~5.9 GiB |
| offline reference (Claude, §6) | not run | 138 (92.0%) | — | — |

The 4B forced at 16k is eight questions ahead on 450 (p = 0.43) and four times slower:
the choice from §7 stands. The 9B forced at 8k is the one candidate that could change it
— eight questions above DianJin on the screening questions, eight short of the offline
reference, and it fits 12 GiB at 8k (13 sequences at a time). It has not been
measured on the extension questions; that run is ~1.8 M tokens, about four hours of GPU.

Not worth it: voting (§8.2, +3 for three times the tokens), the two-model hand-off
(§8.3), a calculator (§8.4), KV-cache quantisation at a 4k cap (§8.5). Worth measuring
next, in order of cost: the 9B's extension run; ODA-Fin-RL-8B screening (a newer finance
model on a larger base); a process reward model choosing among samples (pass@3 leaves
22 questions of room on 300); LoRA on DianJin's own right answers to FinQA's train split.

---

### Reproducing

```bash
R=runs/finbench W=runs/weightprog/dianjin-r1-7b
python tools/finance_model_select.py generate --model dianjin-r1-7b --n 50             # screening slice
python tools/finance_model_select.py generate --model dianjin-r1-7b --n 100 --offset 50  # extension
python tools/finance_model_select.py summary $R/*/n50-o0.jsonl                          # accuracy, outcomes, cost
python tools/finance_model_select.py curve   $R/*/n50-o0.jsonl --caps 1024,2048,4096,8192,16384
python tools/finance_model_select.py failures $R/qwen3.5-9b/n50-o0.jsonl                # read the misses
python tools/finance_model_select.py export --out /tmp/batches                          # questions for a reference model
python tools/finance_model_select.py grade-answers answers.txt --label claude --out $R/claude/n50-o0.jsonl

python tools/weight_edit_experiment.py diagnose --model dianjin-r1-7b --pairs real-loops \
    --finance-runs $R/dianjin-r1-7b/n50-o0.jsonl --out $W/diag-realloops.npz
python tools/weight_edit_experiment.py probe --model dianjin-r1-7b --each-block \
    --diagnosis $W/diag-realloops.npz --finance-runs $R/dianjin-r1-7b/n50-o0.jsonl --transcripts 6
python tools/weight_edit_experiment.py edit --model dianjin-r1-7b --diagnosis $W/diag-realloops.npz \
    --layers 19,20 --out $W/patch-L19-20-k6.npz              # add --control-seed 0 for the control
python tools/finance_model_select.py generate --model dianjin-r1-7b --n 100 --offset 50 --patch $W/patch-L19-20-k6.npz
python tools/finance_model_select.py compare $R/dianjin-r1-7b/n100-o50.jsonl $R/dianjin-r1-7b/n100-o50-patch-L19-20-k6.jsonl

python tools/finance_model_select.py force $R/dianjin-r1-7b/n100-o50.jsonl --cap 4096   # answer at the cap (§8.1)
python tools/finance_model_select.py generate --model dianjin-r1-7b --n 100 --offset 50 \
    --max-tokens 4096 --batch 16 --sample-seed 1 --out $R/dianjin-r1-7b/n100-o50-c4096-s1.jsonl  # a second sample (§8.2)
python tools/finance_report.py                            # the Chinese report, docs/finance_model_selection_report.zh.html
```

(Run the `python` lines inside the env: `conda run -n py313 python ...`.) The queue
scripts that ran all of this overnight, resumably, are in `runs/finbench/queue/`.

The sources download themselves into `data/cache/finbench/` on first use (14 MB,
0.3 MB and 2 MB). Models download with `make pull MODEL=...`; `dianjin-r1-7b` is
released in bf16 only and is quantised here on first use (15 GB download, 4 GB
result, staged through the internal disk because converting straight from a USB
hard disk times out the GPU).

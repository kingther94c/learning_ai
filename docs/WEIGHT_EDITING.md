# Editing weights by hand

*A scale-down of the Qwen experiment in "Sparks of Weight-Space Programming", and a
port of its recipe from Qwen3.5-0.8B to the Qwen3.5-4B this lab runs.*

**The question.** Training is how weights normally change. Can a behaviour be
changed instead by *reading* a model's activations, deciding which weights are
responsible, and rewriting a few thousand of them — with no gradient anywhere?

**The claim under test.** Takehi, Teoh, Zhang & Khattab (MIT CSAIL, blog post of
2 Nov 2026, <https://rikiyat.github.io/blog/weight-space-programming/>) gave a
frontier LLM a Python kernel over the weights of Qwen3.5-0.8B, a budget of forward
passes and practice problems, and forbade training. The model card of Qwen3.5-0.8B
warns that in thinking mode it is "more prone to entering thinking loops"; the agent
traced that to "one direction written by the MLPs of layers 15 and 19" and
"subtracted that direction from 64 output columns in each".

Everything in this file is either a **paper number** (copied from the authors' post
or their logs, <https://github.com/RikiyaT/lm_understand-logs>, and labelled) or
**measured here** (on this Mac, with the code in this repo).

| Paper numbers, Qwen3.5-0.8B, thinking on | MATH-500 | GSM8K | IFEval | MATH-500 answers cut at the token limit |
|---|---|---|---|---|
| unedited | 7.9 | 13.8 | 21.0 | 93.0% |
| edited (131,072 weights, 0.016%) | 48.1 | 67.1 | 62.3 | 33.1% |

Sampling: temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, up to 32,768
tokens, four samples per MATH-500 problem. The authors state the boundary of the
result themselves: the edit "could remove an obstacle that hid an ability, but it
could not add an ability the model lacked" — on Gemma 3 270M, which rarely loops
(8.5% of answers cut), the same procedure gave no improvement.

**What came out (measured here).**

- The released edit is one direction subtracted, six-fold, from 64 columns in each of
  two MLPs, and this repo's code re-derives it from the authors' probe texts (cosine
  1.0000, 127 of 128 columns).
- On Qwen3.5-0.8B it does what the paper says: 4 of 32 GSM8K problems right before,
  13 after — with the released weights or with an edit re-derived here from
  different probe texts. A random direction of the same size: 0 of 32.
- On Qwen3.5-4B the ported edit changes nothing measurable: 61 against 62 of 64 on
  GSM8K and 13 against 13 of 16 on MATH, given room to finish.
- The reason: the 0.8B is stuck in a word-for-word loop on 27 of 32 answers, the 4B
  on 1 of 64. The edit removes that loop. It is a repair, not an upgrade.

---

## 1. What the released edit is, numerically (measured here)

The authors published the edited checkpoint. A safetensors file starts with a JSON
header of byte offsets, so the two `mlp.down_proj.weight` matrices can be fetched
with HTTP range requests and subtracted from the base model's (30 MB instead of
3.5 GB). The difference says more than the prose does:

| | block 15 | block 19 | block 14 (control) |
|---|---|---|---|
| columns that changed, of 3,584 | 64 | 64 | 0 |
| share of the change explained by one direction (rank 1) | 99.999% | 99.999% | – |
| amount subtracted ÷ the column's own component along that direction | 6.000 | 6.000 | – |

So for each chosen column `w_j` (what neuron `j` adds to the residual stream when
it fires) and one unit vector `d` per block,

    w_j  <-  w_j - 6 (w_j . d) d

"Subtracted that direction" is therefore not a projection (that would be a factor
of 1, leaving zero along `d`). The component along `d` is **reversed and amplified**:
a neuron that wrote `+p` along `d` now writes `-5p`. The edit does not silence the
neurons that push the model towards repeating itself; it makes them push the other
way, and only when they fire.

## 2. The recipe (from the authors' logs)

1. **Probe.** Hand-write matched pairs of unfinished reasoning: the same problem
   and opening sentences, then either a check that moves on, or one doubt sentence
   repeated three or four times. Read all of them once (teacher-forced).
2. **Direction.** For every block, average the residual stream over each text's last
   36 tokens; `d` is the normalised mean difference, repetitive minus productive.
3. **Blocks.** Split each block's growth of that contrast into what its MLP wrote and
   the rest (attention). Edit the MLPs that contribute most.
4. **Columns.** Score each neuron by `(mean activation, repetitive - productive) x
   (w_j . d)` — how much it pushes along `d` *because* the text is repetitive — and
   take the top 64.
5. **Strength.** Subtract `k` times the component; the agent settled on `k = 6` after
   trying 1-4 on all columns (weaker, or with more side effects).

In this repo: `learnai/weightedit.py` holds the probe (`REPETITION_PAIRS`, six pairs
written for this repo), the reading (`recording`, `mean_activations`, `diagnose`), the
attribution (`Diagnosis.layer_table`, `neuron_contributions`) and the edit
(`subtract_direction`, `make_edit`, `apply_patch`). A patch is three small arrays per
block and is applied *beside* the stored matrix — `W'a = Wa - d (coef . a_cols)` — so
a 4-bit quantised checkpoint never has to be unpacked. `learnai/mathbench.py` is the
grader; `tools/weight_edit_experiment.py` runs the experiment.

## 3. Does this implementation find the same edit? (measured here)

Before trusting the code on a new model, it was pointed at the model the paper used.
Fed the authors' own four probe pairs, on Qwen3.5-0.8B loaded through MLX:

| | block 15 | block 19 |
|---|---|---|
| cosine between this repo's direction and the released one | 1.0000 | 1.0000 |
| columns chosen here that the released edit also changed | 64 / 64 | 63 / 64 |
| coefficient here ÷ released coefficient (shared columns) | 0.999 | 0.995 |
| MLP share of the contrast: here / authors' log | 0.2627 / 0.2634 | 0.2480 / 0.2497 |

An independent implementation on a different framework arrives at the same edit.
With this repo's own six pairs — different problems, different sentences — blocks
15 and 19 are still the two largest MLP contributors below the last block, the
directions have cosine 0.88 and 0.87 with the released ones, and 54 and 35 of the
64 columns coincide: the recipe does not hinge on the exact probe texts.

## 4. Does the edit do what the paper says? (measured here)

Matching weights is not the claim; the claim is about behaviour. So, before porting
anything: the 0.8B on the first 32 GSM8K test problems, one sample each, the paper's
prompt and sampling, and a budget of 16,384 tokens per answer — half the paper's.
Four versions of the same model:

- **unedited**;
- **released edit** — the difference between the authors' checkpoint and the base
  (section 1), applied as a patch;
- **this repo's edit** — the same blocks, column count and strength, but direction
  and columns derived from this repo's own six probe pairs (section 3);
- **random direction** — the control the authors also ran: this repo's columns and an
  edit of the same size, along a random direction orthogonal to the diagnosed one.
  If this helped as much, the direction would not be the point.

| Qwen3.5-0.8B, GSM8K test, 32 problems, 16,384 tokens | correct | finished right | finished wrong | cut off | …of which verbatim loops | abandoned | tokens generated |
|---|---|---|---|---|---|---|---|
| unedited | 4 (12.5%) | 4 | 0 | 28 | 27 | 0 | 484,587 |
| released edit | 13 (40.6%) | 13 | 5 | 12 | 2 | 2 | 357,365 |
| this repo's edit | 13 (40.6%) | 13 | 4 | 14 | 4 | 1 | 395,096 |
| random direction, same columns and size | 0 (0.0%) | 0 | 0 | 32 | 32 | 0 | 524,288 |

Every answer is in exactly one of four states. *Finished* means it closed its
reasoning and stopped by itself; *cut off* means it was still writing at the token
limit; *abandoned* means it stopped by itself without ever closing its reasoning. A
cut-off answer is a *verbatim loop* if at least half of the 16-token windows in its
last 512 tokens had already occurred earlier in the same answer.

- **The defect is real, and it is a loop.** 28 of 32 unedited answers never leave
  their reasoning, and 27 of those are repeating themselves word for word. (The
  authors' agent measured the same thing its own way: in the tails of its cut-off
  answers, 96% of the 8-word sequences were duplicates.)
- **Both edits remove most of it.** Paired on the same problems, the released edit
  fixes 10 and breaks 1 (exact sign test p = 0.012); this repo's fixes 11 and
  breaks 2 (p = 0.022). The two agree with each other on 11 of their 13 right
  answers. An edit derived from different probe texts by different code does the
  same job as the authors' weights.
- **The direction is the point.** The control rewrites the same 128 columns by the
  same amount and repairs nothing: all 32 answers end in a verbatim loop, including
  the four the unedited model had solved. It is not enough to disturb those neurons;
  they have to be turned along the diagnosed direction. (The authors' control moved
  their score by +1.6 ± 3.5 points.)
- **The repair has a price — the one the authors' logs record.** With the released
  edit, answers that finish *wrong* go from 0 to 5 (the agent's own tally over 1,152
  practice answers: 6 to 187). Two answers are abandoned: after thousands of tokens
  the edited model emits nothing but spaces, then stops (in the same tally,
  `whitespace_tail_cut`: 0 to 51). Three close their reasoning and then go on
  deliberating *inside the answer* until the limit, one of them in a verbatim loop.
  Of the 28 answers the unedited model could not finish, the released edit turns 10
  into right answers and 4 into wrong ones, leaves 12 cut off, and abandons 2.
- **It is weaker here than in the paper.** Paper: GSM8K 13.8 to 67.1, with 32,768
  tokens and two samples per problem on the full test set. Here: 12.5 to 40.6. The
  unedited number agrees; the edited one is lower by more than the noise of 32
  problems allows (95% interval 26-58). The halved budget is the obvious suspect —
  12 to 14 of the 32 edited answers still run into it — but that was not isolated.

## 5. Porting the recipe to Qwen3.5-4B (measured here)

The 4B checkpoint this lab runs is 4-bit quantised (32 blocks, `d_model` 2560,
`d_ff` 9216). Nothing in the recipe needs gradients or unpacked weights, so it runs
as is. One step did **not** carry over, and that was only visible because it was
measured rather than assumed.

**The block-selection rule fails.** Step 3 says "edit the MLPs with the largest
share". On the 0.8B that picks blocks 15 and 19 of 24. On the 4B the contrast keeps
growing to the last block (its norm goes from 0.2 after block 0 to 18.8 after block
31), so the largest shares below the last block are in blocks 27 and 29 — next to
the output, where an MLP writes the next token rather than a habit. Editing each
block *alone* and re-reading the probe texts (`probe --each-block`: 31 candidate
edits, about a minute, no sampling) shows where an edit actually bites:

| edit (64 columns, k = 6) | repeated sentences | productive endings | real finished answers | `</think>` after a repeated sentence |
|---|---|---|---|---|
| unedited, mean log p | -0.107 | -1.857 | -0.137 | -19.25 |
| block 15 alone | **-0.279** | -0.002 | -0.000 | -0.48 |
| block 17 alone | -0.195 | +0.006 | -0.002 | +0.23 |
| block 8 alone | -0.185 | +0.001 | +0.000 | -1.95 |
| block 14 alone | -0.155 | -0.004 | +0.000 | -0.95 |
| block 19 alone | -0.110 | -0.001 | +0.003 | +0.19 |
| block 27 alone | -0.051 | +0.015 | -0.009 | -0.13 |
| block 29 alone | -0.029 | -0.003 | -0.000 | +0.27 |
| blocks 27 + 29 (largest MLP share) | -0.059 | +0.007 | -0.015 | -0.27 |
| **blocks 15 + 19** | **-0.353** | -0.006 | +0.002 | -0.50 |

(Change in nats per token; negative means the text became less likely. The first
three columns are the tokens that are actually there; the last is the probability
of closing the reasoning at those same positions.) The middle of the network is
where the edit is both strong and selective. Blocks 15 + 19 make a repeated doubt
0.35 nats per token less likely while moving real reasoning by 0.002; the pair the
heuristic would have chosen is six times weaker, and it is the one that makes real
answers less likely. On the 0.8B the same six probe pairs give -0.113 for the
released edit and -0.150 for this repo's (-0.020 and -0.027 on the productive
endings), so by this measure the 4B edit is the stronger and the cleaner one.

The 4B patch: blocks 15 and 19, 64 columns each, k = 6 — 327,680 of 4,205,751,296
weights (0.0078%). Each of the two matrices changes by 4-5% in Frobenius norm. It
was fixed at this point, from the probe texts alone, before any answer was sampled.

**One difference shows up already.** On the 0.8B both edits make `</think>` *more*
likely where the text is repeating itself: +2.9 and +2.3 nats, from a mean
log-probability of -23.7. On the 4B the chosen edit moves it the other way, -0.5
from -19.3. These are averages over positions where closing the reasoning stays
vanishingly unlikely, so only the sign carries information — but the sign says the
4B edit discourages repeating without encouraging stopping. That is a hint about
section 6, not a result.

## 6. Does the edit help the 4B? (measured here)

Protocol, identical for both versions: thinking on, the paper's prompt suffix
("Please reason step by step, and put your final answer within \\boxed{}."), the
paper's sampling (temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5 over
the generated tokens), one sample per problem, same seed. "Correct" follows the
paper's scorer: the last `\boxed{}` after `</think>`, or anywhere in the text if
the reasoning never closed — so an answer that boxed the right result and then went
on deliberating until the limit still counts.

### 6a. GSM8K test, 64 problems

| budget | | correct | finished right | finished wrong | cut off | …verbatim loops | tokens generated |
|---|---|---|---|---|---|---|---|
| 4,096 | unedited | 41 (64.1%) | 37 | 0 | 27 (42%) | 0 | 170,556 |
| 4,096 | edited (blocks 15 + 19) | 45 (70.3%) | 41 | 0 | 23 (36%) | 0 | 155,446 |
| 16,384 | unedited | 61 (95.3%) | 60 | 1 | 3 (5%) | 1 | 247,453 |
| 16,384 | edited (blocks 15 + 19) | 62 (96.9%) | 62 | 1 | 1 (2%) | 0 | 217,472 |

("Correct" can exceed "finished right": it also credits the cut-off answers that had
already boxed the right result — 4, 4, 1 and 0 in the four rows.)

Paired on the same problems: at 4,096 tokens the edit fixes 9 and breaks 5 (exact
sign test p = 0.42); at 16,384 it fixes 1 and breaks none. With room to finish, both
versions get 61 or 62 of 64 right, and the one problem both finish wrong they get
wrong the same way. The edited version generates 9% and 12% fewer tokens, which
cannot be told apart from no change: bootstrap 95% intervals for the ratio are
0.81-1.03 and 0.71-1.06, and the edited answer is the shorter one on 28 of 47 and
35 of 63 problems.

Two answers per version in the 16,384-token rows are not from the original runs.
Both runs crashed after 62 answers (section 6c), and the last two problems of each
were sampled again in a fresh process; the lost answers were already 7,300 to 10,600
tokens long. Counting them as unfinished instead gives 61 of 64 for both versions.

### 6b. MATH train

| budget | | correct | finished right | finished wrong | cut off | …verbatim loops | tokens generated |
|---|---|---|---|---|---|---|---|
| 4,096, 63 problems | unedited | 17 (27.0%) | 17 | 0 | 46 (73%) | 0 | 227,759 |
| 4,096, 63 problems | edited | 14 (22.2%) | 13 | 0 | 50 (79%) | 0 | 234,373 |
| 16,384, first 16 | unedited | 13 (81%) | 13 | 0 | 3 (19%) | 0 | 146,500 |
| 16,384, first 16 | edited | 13 (81%) | 13 | 0 | 3 (19%) | 0 | 156,387 |

Paired at 4,096 tokens: 2 fixed, 5 broken (p = 0.45). Paired at 16,384: 1 fixed,
1 broken. Thirteen of those sixteen problems had been cut off at 4,096 tokens;
given the room, the unedited model finishes ten of them, all ten correctly. Its
median finished answer is 6,900 tokens long, on problems whose reference solutions
are a few lines.

### 6c. What the 4B is doing when it runs out

Almost never what the 0.8B is doing. The 4B runs above contain 156 cut-off answers.
Their share of repeated 16-token windows in the last 512 tokens has a median of
0.02, 134 are below 0.1, and exactly one is a verbatim loop — where the unedited
0.8B's median is 1.00 and 27 of its 28 cut-off answers are loops.

The 4B's long answers are *deliberation*. It derives a result, then re-derives it,
considers a second reading of the question, checks edge cases and drafts the
wording of its reply. Among the 46 MATH answers cut off at 4,096 tokens, 29 have a
reference answer long enough to search for (three characters or more), and in 16 of
those it is already written down, at a median of about 1,300 tokens. One GSM8K
answer was cut off 4,096 tokens in while weighing whether `\boxed{18}` or
"18 dollars" was "the safest bet for parsing".

The one loop is that same first GSM8K problem, in the unedited run with 16,384
tokens. The model boxed 18 within its first 800 tokens, deliberated for nine
thousand more, and spent the last 6,485 repeating three lines: "I will write
\\boxed{18}. (Wait, I'll write \\boxed{18}). (Okay, I'll write \\boxed{18})." So the
4B *can* fall into the 0.8B's failure — on one answer in 64 here, against 27 in 32.
The edited run answered that problem in 725 tokens, which as a single pair of
samples shows nothing.

Two consequences for anyone measuring this:

- **A token cap can fake the symptom.** "73% of MATH answers hit the limit" looks
  like the paper's 93%. It is not the same thing: the 0.8B is stuck, the 4B is slow.
  Whether an answer was *cut off* and whether the model was *looping* are separate
  measurements, and only the second is the defect the edit targets.
- **Answer length is mostly noise.** For the eleven MATH problems both versions
  finished within 4,096 tokens, the two answers differ in length by a median factor
  of 1.55 and by up to 3.5 (2,969 tokens against 856). Which side of a token limit
  an answer lands on is therefore largely luck, and so is a difference of a few
  problems between two runs.

Two traps that nearly got into this file:

- **A crashed run keeps the easy answers.** The first GSM8K run died after writing
  30 of 64 answers — all 30 correct, none cut off. A batched run writes answers in
  the order they *finish*, so a truncated run keeps exactly the short ones. Complete
  runs only; the one place where two answers had to be sampled again says so.
- **A counter that nobody reads.** Both 16,384-token GSM8K runs died after 62 answers
  with `[metal::malloc] Resource limit (499000) exceeded` and gigabytes of memory
  free. MLX is lazy, and `mlx_lm` 0.32 decrements a padding counter in every Gated
  DeltaNet layer on every step once a finished sequence has been replaced in the
  batch. Nothing reads the result, so nothing evaluates it, and each pending
  subtraction holds one of the allocator's 499,000 slots: 24 layers x 20,800 steps.
  `settle_cache` in `learnai/weightedit.py` evaluates the caches' bookkeeping every
  256 steps. With 480,000 slots occupied on purpose the crash reproduces within a
  short run on the 0.8B, and with the fix the same run completes.

### 6d. Verdict

**No measurable effect.** In the four comparisons the edited 4B wins 9, 1, 2 and 1
problems and loses 5, 0, 5 and 1; nothing is near significance and the sign differs
between GSM8K and MATH. Given room to finish it is right on 62 of 64 GSM8K problems
against 61, and on 13 of 16 MATH problems against 13.

Sections 4 and 6c together say why. The edit removes one thing, a verbatim loop. The
unedited 0.8B is in that loop on 27 of 32 answers; the unedited 4B on 1 of 64. What
stops the 4B at 4,096 tokens is the budget, and against deliberation the edit does
nothing consistent — 9% and 12% fewer tokens on GSM8K, 3% and 7% more on MATH, all
inside the noise. It does no harm either: no accuracy lost, and section 5's probe
shows real reasoning unmoved.

This is the outcome the authors report for Gemma 3 270M, and the one their own
caveat predicts: the edit can remove an obstacle, and the 4B does not have this one.

## 7. What this does and does not establish

| | the paper | here |
|---|---|---|
| model | Qwen3.5-0.8B, bf16 | Qwen3.5-4B (4-bit MLX) and Qwen3.5-0.8B (bf16, MLX) |
| who chose the edit | a frontier LLM with a Python kernel and 50 forward passes | the paper's recipe, re-implemented; blocks chosen by a per-block probe |
| problems | MATH-500 x 4 samples, GSM8K x 2, six more benchmarks | 16-64 problems per cell, one sample each |
| token budget | 32,768 | 4,096 and 16,384 |
| checker | math-verify | string normalisation after the MATH paper; numbers, tuples and intervals compared by value |
| sampler | vLLM (temperature, then top-k / top-p) | mlx-lm (top-p / top-k, then temperature) |

The strongest statements the measurements support:

- **The edit is exactly what the paper says, and reproducible from its description.**
  An independent implementation recovers the released weights' direction (cosine
  1.0000) and columns (127 of 128) — section 3.
- **It works on the model it was made for, and the direction is why.** 4 to 13 of 32
  GSM8K problems with the released edit *and* with one rebuilt from different probe
  texts; 0 of 32 with a random direction of the same size. The side effects in the
  authors' logs reproduce too — section 4.
- **It is a repair, not an upgrade.** On the 4B, which loops on about one answer in
  sixty, four comparisons find no effect on accuracy or on length — section 6.
- **A token cap can fake the symptom.** At 4,096 tokens 42% of the 4B's GSM8K answers
  and 73% of its MATH answers are cut off, which looks like the paper's 93%. At
  16,384 it gets 95% and 81% right. Being cut off and looping are different
  measurements — section 6c.
- **The paper's block-selection rule is not model-independent.** On the 4B it
  points at blocks where an edit barely registers; measuring each block alone takes
  a minute and gives a different answer — section 5.

What they do not support:

- **Small effects.** Cells are 16 to 64 problems, one sample each. The paired tests
  here would miss a real difference of five points.
- **The paper's full effect size.** The 0.8B gain here is 28 points at half the
  paper's token budget, against the paper's 53; why was not isolated.
- **MATH-500, or other 4B checkpoints.** The MATH problems are from the train split,
  and this 4B is 4-bit quantised.
- **Other edits.** This tests the paper's edit, aimed at repetition. The 4B's own
  habit is to keep deliberating after it has the answer; diagnosing *that* from
  matched texts and editing for it is the experiment this one points to.

## 8. Run it

```bash
PY="conda run --no-capture-output -n py313 python"   # one MLX job at a time
R=runs/weightprog/qwen3.5-4b

$PY tools/weight_edit_experiment.py diagnose --out $R/diagnosis.npz                 # ~10 s
$PY tools/weight_edit_experiment.py probe --each-block --diagnosis $R/diagnosis.npz # ~1 min
$PY tools/weight_edit_experiment.py edit --diagnosis $R/diagnosis.npz --layers 15,19 \
    --out $R/patch-L15-19-k6.npz
$PY tools/weight_edit_experiment.py probe --patch $R/patch-L15-19-k6.npz

# sampling runs: 10-40 minutes each
$PY tools/weight_edit_experiment.py generate --set gsm8k-test --n 64 --max-tokens 16384 \
    --batch 16 --out $R/base.jsonl
$PY tools/weight_edit_experiment.py generate --set gsm8k-test --n 64 --max-tokens 16384 \
    --batch 16 --patch $R/patch-L15-19-k6.npz --out $R/edited.jsonl
$PY tools/weight_edit_experiment.py compare $R/base.jsonl $R/edited.jsonl

make chat MODEL=qwen3.5-4b PATCH=$R/patch-L15-19-k6.npz     # talk to the edited model
```

`--model Qwen/Qwen3.5-0.8B` runs the same commands on the paper's model (a 1.75 GB
download); `edit --control-seed 0` writes the random-direction control. Problems are
fetched once into `data/cache/mathbench/`; runs and patches land in
`runs/weightprog/`; both are git-ignored. A run that is interrupted resumes where it
stopped when the same command is given again.

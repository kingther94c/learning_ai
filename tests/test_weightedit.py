"""Tests for hand-editing weights: the algebra everywhere, the MLX wrappers on Apple silicon.

The model-level tests use the tiny ``qwen3-0.6b`` checkpoint and are skipped (never
downloaded) when it or ``mlx-lm`` is missing.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from learnai.weightedit import (
    DELIBERATION_PAIRS,
    PAIR_SETS,
    PRODUCTIVE,
    REPETITION_PAIRS,
    REPETITIVE,
    Diagnosis,
    LayerEdit,
    contrast_direction,
    load_patch,
    make_edit,
    neuron_contributions,
    save_patch,
    subtract_direction,
)

rng = np.random.default_rng(0)
D_MODEL, D_FF = 12, 40


def _random_problem():
    W = rng.normal(size=(D_MODEL, D_FF))  # [d_model, d_ff]
    d = rng.normal(size=D_MODEL)
    d /= np.linalg.norm(d)
    cols = np.array([3, 7, 21, 39])
    return W, d, cols


def test_strength_one_projects_the_direction_out_of_the_chosen_columns():
    W, d, cols = _random_problem()
    coef, edited = subtract_direction(W, cols, d, strength=1.0)

    assert np.allclose(d @ edited[:, cols], 0.0)  # those neurons no longer write along d
    untouched = np.setdiff1d(np.arange(D_FF), cols)
    assert np.array_equal(edited[:, untouched], W[:, untouched])
    assert np.allclose(coef, d @ W[:, cols])


def test_strength_k_scales_the_component_by_one_minus_k():
    """The paper's released edit uses k = 6: the component along d is reversed, x(-5)."""
    W, d, cols = _random_problem()
    for k in (0.0, 1.0, 2.0, 6.0):
        _, edited = subtract_direction(W, cols, d, strength=k)
        assert np.allclose(d @ edited[:, cols], (1 - k) * (d @ W[:, cols]))


def test_the_edit_only_moves_columns_along_the_direction():
    W, d, cols = _random_problem()
    _, edited = subtract_direction(W, cols, d, strength=6.0)
    change = edited - W  # [d_model, d_ff]

    assert np.linalg.matrix_rank(change) == 1
    # every changed column is a multiple of d: nothing orthogonal to d was touched
    orthogonal_part = change - np.outer(d, d @ change)
    assert np.allclose(orthogonal_part, 0.0)


def test_direction_need_not_be_normalised_by_the_caller():
    W, d, cols = _random_problem()
    a, edited_a = subtract_direction(W, cols, d, strength=3.0)
    b, edited_b = subtract_direction(W, cols, 17.0 * d, strength=3.0)
    assert np.allclose(a, b) and np.allclose(edited_a, edited_b)


def test_contrast_direction_points_from_b_to_a_with_unit_length():
    mean_a, mean_b = rng.normal(size=D_MODEL), rng.normal(size=D_MODEL)
    d = contrast_direction(mean_a, mean_b)
    assert np.isclose(np.linalg.norm(d), 1.0)
    assert d @ (mean_a - mean_b) > 0
    assert np.allclose(contrast_direction(mean_b, mean_a), -d)


def test_neuron_contributions_add_up_to_the_gap_in_what_the_mlp_writes():
    """score_j = (a_j^A - a_j^B)(w_j . d) is an exact decomposition, neuron by neuron."""
    W, d, _ = _random_problem()
    acts_a, acts_b = rng.normal(size=D_FF), rng.normal(size=D_FF)

    scores = neuron_contributions(acts_a, acts_b, W, d)  # [d_ff]
    gap_along_d = d @ (W @ acts_a - W @ acts_b)  # what the whole MLP writes, A minus B
    assert scores.shape == (D_FF,)
    assert np.isclose(scores.sum(), gap_along_d)


def test_patch_round_trips_through_disk(tmp_path):
    W, d, cols = _random_problem()
    coef, _ = subtract_direction(W, cols, d, strength=6.0)
    path = save_patch(tmp_path / "p.npz", [LayerEdit(5, cols, coef, d)], strength=6.0, model="toy")

    edits, meta = load_patch(path)
    assert meta == {"strength": 6.0, "model": "toy", "layers": [5]}
    assert edits[0].layer == 5 and edits[0].n_weights == len(cols) * D_MODEL
    assert np.array_equal(edits[0].cols, cols)
    assert np.allclose(edits[0].coef, coef, atol=1e-6)
    assert np.allclose(edits[0].direction, d, atol=1e-6)


def test_matched_pairs_differ_only_in_how_the_reasoning_ends():
    for pair in REPETITION_PAIRS + DELIBERATION_PAIRS:
        productive, repetitive = pair.texts()
        shared = f"<|im_start|>user\n{pair.problem}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        assert productive.startswith(shared + pair.opening)
        assert repetitive.startswith(shared + pair.opening)
        assert repetitive.count(pair.doubt) == pair.repeats
        assert pair.repeats >= 3 if pair in REPETITION_PAIRS else pair.repeats == 1
        assert pair.doubt not in productive
        # the reasoning is left open: the probe asks "what next?", not "what was the answer?"
        assert "</think>" not in productive and "</think>" not in repetitive


def _toy_diagnosis(n_layers: int = 5) -> Diagnosis:
    return Diagnosis(
        resid=rng.normal(size=(2, n_layers, D_MODEL)),
        write=rng.normal(size=(2, n_layers, D_MODEL)),
        acts=rng.normal(size=(2, n_layers, D_FF)),
    )


def test_layer_table_splits_growth_into_mlp_and_attention():
    dg = _toy_diagnosis()
    rows = dg.layer_table()
    diff = dg.resid[REPETITIVE] - dg.resid[PRODUCTIVE]  # [L, d_model]

    assert [r["layer"] for r in rows] == list(range(5))
    assert np.isnan(rows[0]["growth"])  # nothing to compare the first block against
    for r in rows[1:]:
        layer = r["layer"]
        d = dg.direction(layer)
        assert np.isclose(r["contrast"], np.linalg.norm(diff[layer]))
        # after the block the gap along d is the whole contrast; `growth` is what this block added
        assert np.isclose(r["growth"], r["contrast"] - diff[layer - 1] @ d)
        assert np.isclose(r["mlp"], (dg.write[REPETITIVE, layer] - dg.write[PRODUCTIVE, layer]) @ d)
        assert np.isclose(r["mlp"] + r["attention"], r["growth"])


def test_make_edit_reverses_the_strongest_writers():
    dg = _toy_diagnosis()
    W, _, _ = _random_problem()
    layer, k = 2, 6
    edit = make_edit(dg, layer, W, columns=k, strength=6.0)

    scores = dg.neuron_scores(layer, W)
    assert set(edit.cols) == set(np.argsort(-scores)[:k])  # the k largest scores...
    assert list(edit.cols) == sorted(edit.cols)
    assert scores[edit.cols].min() >= np.delete(scores, edit.cols).max()  # ...and no others
    assert np.allclose(edit.direction, dg.direction(layer))
    assert np.allclose(edit.coef, 6.0 * (edit.direction @ W[:, edit.cols]))
    assert edit.n_weights == k * D_MODEL


def test_diagnosis_round_trips_through_disk(tmp_path):
    dg = _toy_diagnosis()
    again = Diagnosis.load(dg.save(tmp_path / "d.npz"))
    assert all(
        np.array_equal(getattr(dg, f), getattr(again, f)) for f in ("resid", "write", "acts")
    )


# --------------------------------------------------------------------------------------
# With a real model
# --------------------------------------------------------------------------------------


def _tiny_model_ready() -> bool:
    if importlib.util.find_spec("mlx_lm") is None:
        return False
    from learnai.openweights import DriveNotConnected, local_path

    try:
        return local_path("qwen3-0.6b") is not None
    except DriveNotConnected:
        return False


needs_model = pytest.mark.skipif(
    not _tiny_model_ready(), reason="needs mlx-lm and qwen3-0.6b downloaded"
)


@pytest.fixture(scope="module")
def tiny():
    from learnai.openweights import load

    return load("qwen3-0.6b")


@needs_model
def test_edited_linear_equals_the_explicitly_edited_matrix(tiny):
    """The wrapper beside a 4-bit matrix computes exactly W' a for the edited dense W'."""
    import mlx.core as mx

    from learnai.weightedit import apply_patch, dense_weight, remove_patch

    model, _ = tiny
    layer = 3
    down = model.layers[layer].mlp.down_proj
    W = dense_weight(down)  # [d_model, d_ff], unpacked from 4 bits
    d = rng.normal(size=W.shape[0])
    cols = np.array([0, 5, 100, 2000])
    coef, edited = subtract_direction(W, cols, d, strength=6.0)

    a = rng.normal(size=(2, 3, W.shape[1])).astype(np.float32)  # [B, T, d_ff]
    before = np.array(down(mx.array(a)).astype(mx.float32))
    # dense_weight unpacks the 4-bit matrix correctly (its scales are stored in bf16,
    # hence the loose tolerance on outputs of size ~10)
    assert np.allclose(before, a @ W.T, rtol=1e-2, atol=5e-2)

    apply_patch(model, [LayerEdit(layer, cols, coef, d / np.linalg.norm(d))])
    try:
        after = np.array(model.layers[layer].mlp.down_proj(mx.array(a)).astype(mx.float32))
    finally:
        remove_patch(model)

    assert model.layers[layer].mlp.down_proj is down  # restored
    # The wrapper changes the output by exactly what editing the matrix would.
    change = after - before
    assert np.allclose(change, a @ (edited - W).T, atol=1e-5)
    assert np.abs(change).max() > 1e-3  # and it is a real change


@needs_model
def test_recording_reconstructs_the_residual_stream(tiny):
    """mid + write, read through the taps, is the block's actual output."""
    import mlx.core as mx

    from learnai.weightedit import dense_weight, recording

    model, tokenizer = tiny
    tokens = mx.array([tokenizer.encode("The capital of France is Paris, and the capital of")])

    def reduce(i, mid, acts, write):
        return {"i": i, "out": mid + write, "d_ff": acts.shape[-1]}

    with recording(model, reduce) as per_layer:
        logits = model(tokens)
    assert [r["i"] for r in per_layer] == list(range(len(model.layers)))
    # `acts` is the input of down_proj: one activation per column of its matrix
    assert per_layer[0]["d_ff"] == dense_weight(model.layers[0].mlp.down_proj).shape[1]

    # the last block's output, through the final norm and the unembedding, is the logits
    inner = model.model
    rebuilt = inner.norm(per_layer[-1]["out"])
    rebuilt = inner.embed_tokens.as_linear(rebuilt) if model.args.tie_word_embeddings else None
    assert rebuilt is not None
    assert mx.allclose(rebuilt, logits, atol=1e-2).item()

    with recording(model, reduce) as again:  # taps were removed: a second pass is clean
        model(tokens)
    assert len(again) == len(model.layers)


@needs_model
def test_mean_activations_weights_positions_per_group(tiny):
    from learnai.weightedit import mean_activations

    model, tokenizer = tiny
    tokens = tokenizer.encode("One two three four five six seven eight nine ten")
    T = len(tokens)
    first_half = (np.arange(T) < T // 2).astype(np.float32)
    weights = np.stack([first_half, 1 - first_half, np.ones(T, dtype=np.float32)])  # [3, T]

    stats = mean_activations(model, [tokens], [weights])
    L = len(model.layers)
    assert stats["resid"].shape[:2] == (3, L) and stats["acts"].shape[:2] == (3, L)
    assert np.allclose(stats["count"], [T // 2, T - T // 2, T])
    # the mean over all positions is the count-weighted mean of the two halves
    mix = (stats["resid"][0] * (T // 2) + stats["resid"][1] * (T - T // 2)) / T
    assert np.allclose(mix, stats["resid"][2], rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(importlib.util.find_spec("mlx") is None, reason="needs mlx")
def test_presence_penalty_counts_each_generated_token_once():
    import mlx.core as mx

    from learnai.weightedit import presence_penalty

    processor = presence_penalty(1.5)
    logits = mx.zeros((1, 10))  # [1, vocab]

    # The generator hands over the running context; its last entry is the newest token.
    after_3 = np.array(processor(mx.array([3]), logits))
    assert after_3[0, 3] == -1.5 and after_3.sum() == -1.5

    after_7 = np.array(processor(mx.array([3, 7]), logits))
    assert after_7[0, 3] == after_7[0, 7] == -1.5 and after_7.sum() == -3.0

    again_3 = np.array(processor(mx.array([3, 7, 3]), logits))  # presence, not frequency
    assert np.array_equal(again_3, after_7)


@pytest.mark.skipif(importlib.util.find_spec("mlx_lm") is None, reason="needs mlx-lm")
def test_settle_cache_collapses_bookkeeping_that_nobody_reads():
    """The crash this prevents: 499,000 pending ``left_padding -= 1`` and Metal gives up.

    A pending (lazy) step keeps its operand alive — four bytes for an int32 — so the
    allocator's own counter shows whether the chain is still there.
    """
    import types

    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache

    from learnai.weightedit import settle_cache

    cache = ArraysCache(size=2, left_padding=[3, 0, 7])  # one Gated DeltaNet layer's cache
    generator = types.SimpleNamespace(_generation_batch=types.SimpleNamespace(prompt_cache=[cache]))
    mx.eval(cache.left_padding)
    before = mx.get_active_memory()

    steps = 2000
    for _ in range(steps):
        cache.advance(1)  # what the layer does after every decoded token
    assert mx.get_active_memory() - before >= 4 * steps  # the whole chain is being kept

    assert settle_cache(generator) == 1
    assert mx.get_active_memory() - before < 4 * steps // 10  # ...and now it is gone
    assert cache.left_padding.tolist() == [3 - steps, -steps, 7 - steps]  # same value

    assert settle_cache(types.SimpleNamespace()) == 0  # an unknown generator: nothing to do


@needs_model
def test_sampling_settles_the_real_generators_caches(tiny, monkeypatch):
    """``settle_cache`` reaches into ``BatchGenerator``; notice if mlx-lm moves things."""
    import learnai.weightedit as we

    model, tokenizer = tiny
    found = []
    settle = we.settle_cache
    monkeypatch.setattr(we, "SETTLE_EVERY", 4)
    monkeypatch.setattr(we, "settle_cache", lambda generator: found.append(settle(generator)))

    we.sample_batch(model, tokenizer, [tokenizer.encode("A B C D")], we.Sampling(max_tokens=12))
    # Steps 4 and 8 find one cache per block; at step 12 the only sequence has just hit
    # its limit and the batch is empty.
    assert found == [len(model.layers), len(model.layers), 0]


@needs_model
def test_sampling_is_reproducible_and_reports_the_token_limit(tiny):
    from learnai.weightedit import Sampling, sample_batch

    model, tokenizer = tiny
    prompts = [tokenizer.encode("Count: one two three"), tokenizer.encode("A B C D E F")]
    sampling = Sampling(max_tokens=12, presence_penalty=1.5, seed=1)
    first = sample_batch(model, tokenizer, prompts, sampling, batch_size=2)
    again = sample_batch(model, tokenizer, prompts, sampling, batch_size=2)

    assert [g.tokens for g in first] == [g.tokens for g in again]  # same seed, same samples
    assert all(len(g.tokens) <= 12 for g in first)
    assert all(g.hit_limit == (len(g.tokens) == 12) for g in first)


@needs_model
def test_diagnose_reads_activations_that_agree_with_the_weights(tiny):
    """What an MLP writes is its matrix times its activations — measured two ways."""
    from learnai.weightedit import dense_weight, diagnose

    model, tokenizer = tiny
    pairs = [p.texts() for p in REPETITION_PAIRS[:2]]
    dg = diagnose(model, tokenizer, pairs, resid_tail=12, act_tail=12)  # same window for both

    L = len(model.layers)
    assert dg.resid.shape[:2] == dg.write.shape[:2] == dg.acts.shape[:2] == (2, L)
    for layer in (0, L // 2, L - 1):
        W = dense_weight(model.layers[layer].mlp.down_proj)  # [d_model, d_ff]
        for group in (REPETITIVE, PRODUCTIVE):
            assert np.allclose(
                W @ dg.acts[group, layer], dg.write[group, layer], rtol=2e-2, atol=2e-2
            )
    # and the neuron scores add up to the MLP column of the layer table
    layer = L // 2
    W = dense_weight(model.layers[layer].mlp.down_proj)
    assert np.isclose(
        dg.neuron_scores(layer, W).sum(), dg.layer_table()[layer]["mlp"], rtol=2e-2, atol=2e-2
    )


def test_the_batch_generator_still_has_the_cache_extraction_sample_batch_switches_off():
    """`sample_batch` replaces `GenerationBatch.extract_cache` for the duration of a run.

    If mlx-lm renames it, the replacement would silently stop applying and every wave
    of sequences hitting the cap would again cost half a gigabyte each.
    """
    pytest.importorskip("mlx_lm")
    from mlx_lm.generate import GenerationBatch

    assert callable(getattr(GenerationBatch, "extract_cache", None))
    assert GenerationBatch.extract_cache.__qualname__ == "GenerationBatch.extract_cache"


def test_deliberation_pairs_reopen_a_settled_answer_without_repeating_a_sentence():
    assert PAIR_SETS["deliberation"] is DELIBERATION_PAIRS and len(DELIBERATION_PAIRS) >= 6
    for pair in DELIBERATION_PAIRS:
        # the shared opening has already reached the answer; the productive ending commits to it
        assert "so " in pair.opening.lower() or "所以" in pair.opening
        assert "answer is" in pair.productive or "答案是" in pair.productive
        # the doubt re-opens the question in more than one way, so it is not a verbatim loop
        assert any(w in pair.doubt for w in ("reconsider", "再", "interpretation", "re-read"))
        assert len(pair.doubt.split(".")) > 2 or "。" in pair.doubt


def test_loop_pair_ends_on_the_same_passage_written_once_and_then_again():
    from learnai.weightedit import loop_pair

    prompt = [900, 901, 902]
    intro = list(range(100, 140))  # 40 tokens of ordinary reasoning
    passage = list(range(200, 230))  # the passage the loop repeats, 30 tokens
    generated = intro + passage * 5
    productive, repetitive = loop_pair(prompt, generated)
    assert productive[: len(prompt)] == prompt and repetitive[: len(prompt)] == prompt
    period = len(repetitive) - len(productive)
    assert period == len(passage)
    # both end on the same tokens: once written fresh, once as a copy
    assert productive[-24:] == repetitive[-24:]
    assert productive[-len(passage) :] == passage
    assert loop_pair(prompt, intro) is None  # no loop, no pair


@needs_model
def test_scoring_from_a_start_position_matches_reading_the_whole_text(tiny):
    """The chunked cache read gives the same log-probabilities as one full forward pass."""
    from learnai.weightedit import next_token_logprobs

    model, tokenizer = tiny
    tokens = tokenizer.encode(
        "A bond priced below par has a current yield above its coupon rate. " * 6
    )
    end = tokenizer.encode(".")[0]
    full_actual, full_watched = next_token_logprobs(model, tokens, watch=[end])
    start = len(tokens) - 9
    actual, watched = next_token_logprobs(model, tokens, watch=[end], start=start)
    assert actual.shape == (8,) and watched.shape == (8, 1)
    assert np.allclose(actual, full_actual[start:], rtol=3e-2, atol=2e-2)  # bf16, two read orders
    assert np.allclose(watched, full_watched[start:], rtol=3e-2, atol=2e-2)


@needs_model
def test_mean_activations_reads_a_long_context_through_the_cache(tiny):
    """Weights only on the last positions: the context is cached, the result is unchanged."""
    import mlx.core as mx

    from learnai.weightedit import mean_activations, recording

    model, tokenizer = tiny
    tokens = tokenizer.encode("The discount rate falls, so the present value rises. " * 12)
    T = len(tokens)
    w = np.zeros((1, T), dtype=np.float32)
    w[0, -6:] = 1.0
    stats = mean_activations(model, [tokens], [w])

    def reduce(i, mid, acts, write):
        return (mid[0, -6:] + write[0, -6:]).astype(mx.float32).mean(axis=0)

    with recording(model, reduce) as per_layer:  # the reference: one pass over everything
        model(mx.array([tokens]))
    reference = np.array(mx.stack(per_layer), dtype=np.float64)  # [L, d_model]
    scale = np.abs(reference).max()
    assert np.abs(stats["resid"][0] - reference).max() < 2e-2 * scale


def test_decision_points_find_a_reopening_right_after_a_conclusion():
    from learnai.weightedit import decision_points

    pieces = ["The", " yield", " is", " 6.3%.", " So", " the", " answer", " is", " B", ".", "\n\n"]
    pieces += ["Wait", ",", " let", " me", " check", ".", "\n"]  # 11: re-opens after "answer is"
    pieces += ["x"] * 300  # far from any conclusion
    pieces += [
        "\n",
        "    *",
        "   If",
        " the",
        " rate",
        " is",
        " 7%",
        "\n",
    ]  # bullet, no conclusion: skipped
    pieces += ["x"] * 300 + [
        " Therefore",
        " B",
        ".",
        "\n",
        "**",
        "Is",
        " there",
        " a",
        " trick",
        "?",
    ]
    points = decision_points(pieces, lookback=60)
    assert points == [11, len(pieces) - 5]  # the bold "Is there" counts; the token holding "Is"
    assert decision_points(["So the answer is B.", "\n\n", "Wait"]) == [2]


def test_junction_pairs_contrast_closing_with_reopening():
    from learnai.weightedit import junction_pairs

    words = {1: " So the answer is B.", 2: "\n", 3: "Wait", 4: " x", 9: "</think>"}
    decode = lambda t: words.get(t, " y")  # noqa: E731
    finished = ([7, 7], [4] * 70 + [1, 2, 9, 4], False)  # closes after 72 tokens
    capped = ([7], [4] * 10 + [1, 2, 3] + [4] * 300 + [1, 2, 3] + [4] * 5, True)
    pairs = junction_pairs([finished, capped, finished], decode, [9], limit=5)
    assert len(pairs) == 2  # two closings available, two re-openings: paired up
    for closing, reopening in pairs:
        assert closing == [7, 7] + [4] * 70 + [1, 2]  # up to, not including, </think>
        assert reopening[-2:] == [1, 2] and reopening[0] == 7  # cut just before "Wait"

"""Tokenizers, batching, the training loop, hooks, and the export bridge.

These are the parts a lesson leans on without thinking about them, which is
exactly why they need tests: a silently-wrong `get_batch` produces a training
curve that looks fine and teaches you something false.
"""

from __future__ import annotations

import json

import pytest
import torch

from learnai.cli import main as cli_main
from learnai.data import (
    BPETokenizer,
    CharTokenizer,
    WordTokenizer,
    copy_task,
    get_batch,
    induction_task,
    make_spirals,
    make_xor,
    parity_task,
    reverse_task,
    sort_task,
    train_val_split,
)
from learnai.export import export_attention, export_embeddings, export_logits, write_json
from learnai.interp import ActivationRecorder, ablate, capture_attention, patch
from learnai.nn import GPT, GPTConfig
from learnai.train import TrainConfig, Trainer, cosine_lr

SAMPLE = "the cat sat on the mat. the dog sat on the log. " * 40


# ----------------------------------------------------------------------- tokenizers


def test_char_tokenizer_round_trips():
    tok = CharTokenizer(SAMPLE)
    assert tok.decode(tok.encode("the cat")) == "the cat"
    assert tok.vocab_size == len(set(SAMPLE))


def test_word_tokenizer_reports_its_unknown_rate():
    tok = WordTokenizer(SAMPLE)
    assert tok.unk_rate("the cat sat") == 0.0
    assert tok.unk_rate("quantum chromodynamics") == 1.0


def test_bpe_round_trips_and_compresses():
    tok = BPETokenizer().train(SAMPLE, vocab_size=120)
    text = "the cat sat on the mat."
    assert tok.decode(tok.encode(text)) == text
    assert tok.compression_ratio(text) > 1.0, "merges must shorten the sequence"


def test_bpe_merges_are_learned_in_frequency_order():
    """Rank 0 must be the most frequent pair; later merges are rarer."""
    tok = BPETokenizer().train(SAMPLE, vocab_size=60)
    ranks = sorted(tok.merges.values())
    assert ranks == list(range(len(ranks))), "ranks must be a dense 0..n-1 sequence"


def test_bpe_handles_unseen_words_by_falling_back_to_characters():
    """The property that made subwords win: an unseen WORD is never a problem."""
    tok = BPETokenizer().train(SAMPLE, vocab_size=100)
    weird = "tacodmash"  # never appears, but every character does
    assert tok.unk_rate(weird) == 0.0
    assert tok.decode(tok.encode(weird)) == weird


def test_bpe_marks_unseen_characters_rather_than_dropping_them():
    """The one OOV case a character-level BPE still has — byte-level closes it."""
    tok = BPETokenizer().train(SAMPLE, vocab_size=100)
    assert tok.unk_rate("\u4f60\u597d") == 1.0, "unseen characters must be flagged"
    ids = tok.encode("the \u4f60 cat")
    assert len(ids) == len(tok.tokenize("the \u4f60 cat")), "nothing may be silently dropped"


def test_bpe_keeps_the_leading_space_with_the_word():
    """GPT-2's pre-tokenizer convention: ' the' and 'the' are different tokens."""
    tok = BPETokenizer().train(SAMPLE, vocab_size=150)
    assert any(t.startswith(" ") for t in tok.tokenize("the cat sat"))


def test_bpe_more_merges_never_lengthens_the_output():
    tok_small = BPETokenizer().train(SAMPLE, vocab_size=40)
    tok_big = BPETokenizer().train(SAMPLE, vocab_size=200)
    text = "the cat sat on the mat"
    assert len(tok_big.tokenize(text)) <= len(tok_small.tokenize(text))


def test_bpe_save_load_round_trip(tmp_path):
    tok = BPETokenizer().train(SAMPLE, vocab_size=100)
    path = tmp_path / "bpe.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.vocab == tok.vocab
    assert loaded.tokenize("the cat sat") == tok.tokenize("the cat sat")


def test_bpe_retraining_replaces_the_old_merge_table():
    tok = BPETokenizer().train("aaaa aaaa aaaa", vocab_size=10)
    assert ("a", "a") in tok.merges

    tok.train("xy xy xy", vocab_size=10)
    assert ("a", "a") not in tok.merges
    assert all("a" not in token for token in tok.vocab)


# ----------------------------------------------------------------------- batching


def test_train_val_split_is_contiguous():
    """A random split would leak: adjacent windows would land on both sides."""
    ids = list(range(1000))
    train, val = train_val_split(ids, val_frac=0.1)
    assert len(train) == 900 and len(val) == 100
    assert train[-1].item() == 899 and val[0].item() == 900


def test_get_batch_targets_are_inputs_shifted_by_one():
    data = torch.arange(500)
    x, y = get_batch(data, batch_size=4, block_size=8)
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert torch.equal(y[:, :-1], x[:, 1:]), "y must be x shifted one position left"


def test_get_batch_never_runs_off_the_end():
    data = torch.arange(40)
    for _ in range(50):
        x, y = get_batch(data, batch_size=8, block_size=8)
        assert x.max() < 40 and y.max() < 40


def test_get_batch_accepts_the_shortest_valid_token_stream():
    """T+1 tokens contain exactly one legal window of T input/target pairs."""
    data = torch.arange(9)
    x, y = get_batch(data, batch_size=4, block_size=8)
    assert torch.equal(x, data[:8].expand(4, -1))
    assert torch.equal(y, data[1:].expand(4, -1))


def test_get_batch_rejects_a_stream_without_a_following_target():
    with pytest.raises(ValueError, match="needs at least 9"):
        get_batch(torch.arange(8), batch_size=1, block_size=8)


def test_tiny_shakespeare_offline_mode_never_touches_the_network(monkeypatch, tmp_path):
    import learnai.data.text as text_module

    monkeypatch.setenv("LEARNAI_OFFLINE", "1")
    monkeypatch.setattr(text_module, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        text_module.urllib.request,
        "urlretrieve",
        lambda *_args, **_kwargs: pytest.fail("offline mode attempted a download"),
    )
    assert "First Citizen" in text_module.tiny_shakespeare(force=True)


# ----------------------------------------------------------------------- toy tasks


def test_copy_and_reverse_tasks():
    x, y = copy_task(8, seq_len=6, vocab_size=10, seed=0)
    assert torch.equal(x, y)
    x, y = reverse_task(8, seq_len=6, vocab_size=10, seed=0)
    assert torch.equal(y, x.flip(dims=[1]))


def test_sort_task_output_is_sorted():
    x, y = sort_task(8, seq_len=6, vocab_size=10, seed=0)
    assert torch.equal(y, y.sort(dim=1).values)


def test_induction_task_plants_a_real_repeat():
    """The whole point of the task is that position `p` is genuinely predictable."""
    x, y, pos = induction_task(16, seq_len=24, vocab_size=40, seed=0)
    assert x.shape == (16, 23) and y.shape == (16, 23)
    for b in range(16):
        p = int(pos[b])
        # x[p] repeats an earlier token, and y[p] is what followed it that time.
        earlier = (x[b, :p] == x[b, p]).nonzero()
        assert len(earlier) == 1, "the cue must be unambiguous: exactly one earlier occurrence"
        first = int(earlier[0])
        assert x[b, first + 1] == y[b, p], "the answer must match the earlier continuation"


def test_parity_task_is_a_running_xor():
    x, y = parity_task(8, seq_len=10, seed=0)
    assert torch.equal(y, x.cumsum(dim=1) % 2)


def test_toy_2d_datasets():
    X, y = make_spirals(90, n_classes=3, seed=0)
    assert X.shape == (90, 2) and set(y.tolist()) == {0, 1, 2}
    X, y = make_xor(100, seed=0)
    assert X.shape == (100, 2) and set(y.tolist()) == {0, 1}


# ----------------------------------------------------------------------- training


def test_cosine_lr_warms_up_then_decays_to_the_floor():
    cfg = TrainConfig(steps=1000, lr=1e-3, warmup_steps=100, min_lr_ratio=0.1)
    assert cosine_lr(0, cfg) < cosine_lr(50, cfg) < cosine_lr(99, cfg)
    assert cosine_lr(99, cfg) == pytest.approx(1e-3, rel=1e-6)  # peak at end of warmup
    assert cosine_lr(999, cfg) == pytest.approx(1e-4, rel=0.02)  # floor = 0.1 * lr
    assert cosine_lr(1500, cfg) == pytest.approx(1e-4, rel=0.02)  # clamped past the end


def test_optimizer_exempts_1d_parameters_from_weight_decay():
    """Decaying biases and LayerNorm gains hurts; every serious codebase splits them."""
    model = GPT(GPTConfig(vocab_size=20, block_size=8, n_layer=1, n_head=2, d_model=32))
    opt = Trainer(model, TrainConfig(device="cpu")).optimizer
    decayed, undecayed = opt.param_groups[0], opt.param_groups[1]
    assert decayed["weight_decay"] > 0 and undecayed["weight_decay"] == 0
    assert all(p.dim() >= 2 for p in decayed["params"])
    assert all(p.dim() < 2 for p in undecayed["params"])


def test_trainer_actually_reduces_loss():
    """The end-to-end check: an over-parameterised model must fit a tiny dataset."""
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=10, block_size=8, n_layer=2, n_head=2, d_model=64))
    data = torch.randint(0, 10, (200,))

    def batch_fn():
        return get_batch(data, batch_size=8, block_size=8)

    trainer = Trainer(
        model, TrainConfig(steps=60, lr=3e-3, warmup_steps=5, log_every=1000, device="cpu")
    )
    history = trainer.fit(batch_fn)
    assert len(history.train_loss) == 60
    assert history.train_loss[-1] < history.train_loss[0]
    assert all(g >= 0 for g in history.grad_norms)


def test_trainer_history_is_json_serialisable():
    """History has to survive the trip to an interactive page."""
    model = GPT(GPTConfig(vocab_size=10, block_size=8, n_layer=1, n_head=2, d_model=32))
    data = torch.randint(0, 10, (100,))
    h = Trainer(model, TrainConfig(steps=3, log_every=1000, device="cpu")).fit(
        lambda: get_batch(data, 4, 8)
    )
    assert json.loads(json.dumps(h.to_dict()))["steps"] == [0, 1, 2]


def test_evaluate_restores_the_callers_model_mode():
    model = GPT(GPTConfig(vocab_size=10, block_size=8, n_layer=1, n_head=2, d_model=32))
    data = torch.randint(0, 10, (100,))
    trainer = Trainer(model, TrainConfig(steps=1, eval_steps=1, device="cpu"))

    def batch_fn():
        return get_batch(data, 4, 8)

    model.eval()
    assert trainer.evaluate(batch_fn) > 0
    assert not model.training

    model.train()
    assert trainer.evaluate(batch_fn) > 0
    assert model.training


# ----------------------------------------------------------------------- interp


@pytest.fixture
def small_model():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=20, block_size=16, n_layer=2, n_head=4, d_model=32))


def test_capture_attention_collects_every_layer(small_model):
    x = torch.randint(0, 20, (1, 8))
    with capture_attention(small_model) as attn:
        small_model(x)
    assert set(attn) == {"blocks.0.attn", "blocks.1.attn"}
    assert attn["blocks.0.attn"].shape == (1, 4, 8, 8)


def test_capture_attention_restores_the_flags(small_model):
    with capture_attention(small_model):
        small_model(torch.randint(0, 20, (1, 4)))
    assert not any(m.store_attn for m in small_model.modules() if hasattr(m, "store_attn"))


def test_capture_attention_ignores_stale_maps_and_restores_state(small_model):
    module = small_model.blocks[0].attn
    stale = torch.ones(1, 4, 2, 2)
    module.attn_weights = stale

    with capture_attention(small_model) as captured:
        pass

    assert captured == {}
    assert module.attn_weights is stale


def test_activation_recorder_filters_by_pattern(small_model):
    rec = ActivationRecorder(small_model, pattern=r"blocks\.\d+$")
    with rec:
        small_model(torch.randint(0, 20, (1, 6)))
    assert set(rec.keys()) == {"blocks.0", "blocks.1"}
    assert rec["blocks.0"].shape == (1, 6, 32)


def test_activation_recorder_removes_its_hooks(small_model):
    rec = ActivationRecorder(small_model, pattern=r"blocks\.0$")
    with rec:
        small_model(torch.randint(0, 20, (1, 4)))
    before = dict(rec.activations)
    small_model(torch.randint(0, 20, (1, 4)))  # a forward pass outside the block
    assert rec.activations.keys() == before.keys()


@pytest.mark.parametrize("mode", ["zero", "mean"])
def test_ablate_changes_the_output_and_then_restores(small_model, mode):
    x = torch.randint(0, 20, (1, 8))
    small_model.eval()
    with torch.no_grad():
        base = small_model(x)[0].clone()
        with ablate(small_model, "blocks.1.attn", mode=mode):
            ablated = small_model(x)[0].clone()
        after = small_model(x)[0]
    assert not torch.allclose(base, ablated, atol=1e-6), "ablation must have an effect"
    assert torch.allclose(base, after, atol=1e-6), "the model must be restored on exit"


def test_patch_forces_a_submodule_output(small_model):
    x = torch.randint(0, 20, (1, 8))
    small_model.eval()
    forced = torch.zeros(1, 8, 32)
    with torch.no_grad():
        with patch(small_model, "blocks.0", forced):
            recorder = ActivationRecorder(small_model, pattern=r"blocks\.0$")
            with recorder:
                small_model(x)
    assert torch.allclose(recorder["blocks.0"], forced)


# ----------------------------------------------------------------------- export


def test_export_attention_writes_the_expected_schema(tmp_path):
    attn = torch.softmax(torch.randn(2, 3, 5, 5), dim=-1)  # (L, H, T, T)
    path = export_attention(attn, list("abcde"), tmp_path / "a.json", meta={"k": 1})
    obj = json.loads(path.read_text())
    assert obj["tokens"] == list("abcde")
    assert len(obj["layers"]) == 2
    assert len(obj["layers"][0]["heads"]) == 3
    assert obj["meta"] == {"k": 1}


def test_export_attention_rejects_a_token_count_mismatch(tmp_path):
    with pytest.raises(ValueError, match="tokens given"):
        export_attention(torch.rand(2, 5, 5), list("abc"), tmp_path / "a.json")


def test_export_writes_a_js_wrapper_for_file_urls(tmp_path):
    """`.js` output exists because browsers block fetch() over file://."""
    path = write_json({"x": 1}, tmp_path / "thing.js")
    text = path.read_text()
    assert "window.LEARNAI_DATA" in text
    assert '["thing"]' in text
    assert '{"x": 1}' in text


def test_export_embeddings_projects_to_2d(tmp_path):
    v = torch.randn(24, 16)
    path = export_embeddings(v, [f"t{i}" for i in range(24)], tmp_path / "e.json")
    obj = json.loads(path.read_text())
    assert len(obj["coords"]) == 24 and len(obj["coords"][0]) == 2
    assert 0 <= obj["explained_variance"] <= 1


def test_export_logits_keeps_the_top_k_in_order(tmp_path):
    logits = torch.tensor([0.1, 5.0, 2.0, -1.0, 3.0])
    path = export_logits(logits, list("abcde"), tmp_path / "l.json", top_k=3)
    obj = json.loads(path.read_text())
    assert obj["tokens"] == ["b", "e", "c"]
    assert obj["logits"] == sorted(obj["logits"], reverse=True)
    assert obj["full_vocab_size"] == 5


def test_export_rejects_schemas_that_the_browser_cannot_label(tmp_path):
    with pytest.raises(ValueError, match="map shape"):
        export_attention(torch.rand(2, 3, 5), list("abcde"), tmp_path / "a.json")
    with pytest.raises(ValueError, match="vocabulary entries"):
        export_logits(torch.rand(5), list("abcd"), tmp_path / "l.json")


def test_cli_info_exercises_the_installed_entrypoint_target(capsys):
    assert cli_main(["info"]) == 0
    output = capsys.readouterr().out
    assert "learnai" in output
    assert "python" in output
    assert "device" in output


def test_induction_task_rejects_a_vocab_too_small_to_be_unambiguous():
    with pytest.raises(ValueError, match="must exceed seq_len"):
        induction_task(4, seq_len=32, vocab_size=20)

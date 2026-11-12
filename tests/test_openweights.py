"""Tests for the open-weight model runner.

The registry and the memory arithmetic are checked everywhere.  The generation
test needs Apple silicon, ``make llm-setup`` and the tiny model downloaded
(``make pull MODEL=qwen3-0.6b``); otherwise it is skipped, never downloaded.
"""

from __future__ import annotations

import importlib.util
import math

import pytest

from learnai.openweights import (
    DEFAULT_MODEL,
    MODELS,
    DriveNotConnected,
    cached_sizes,
    fit,
    local_path,
    models_dir,
    quantized_path,
    resolve,
)

THIRTY_TWO_GB_BUDGET = 32 * 2 / 3  # the fallback budget `gpu_budget_gb` uses


def test_default_model_is_curated():
    assert DEFAULT_MODEL in MODELS
    assert resolve().key == resolve(None).key == DEFAULT_MODEL


def test_every_card_names_one_hub_repo_and_a_measured_size():
    repos = [card.repo for card in MODELS.values()]
    assert len(set(repos)) == len(repos)
    for key, card in MODELS.items():
        assert card.key == key
        assert card.repo.count("/") == 1
        assert card.download_gb > 0
        assert card.kind in {"dense", "moe"}
        assert card.note


def test_resolve_passes_any_repo_id_through_with_unknown_size():
    card = resolve("mlx-community/SomeModel-4bit")
    assert card.repo == "mlx-community/SomeModel-4bit"
    assert math.isnan(card.download_gb)
    assert fit(card.download_gb, THIRTY_TWO_GB_BUDGET) == "unknown"


def test_resolve_rejects_an_unknown_key_and_lists_the_choices():
    with pytest.raises(KeyError, match=DEFAULT_MODEL):
        resolve("qwen-does-not-exist")


def test_fit_separates_ok_tight_and_too_big():
    assert fit(3.0, 21.3) == "ok"
    assert fit(20.0, 21.3) == "tight"  # weights fit, KV cache headroom does not
    assert fit(21.3, 21.3) == "tight"
    assert fit(25.0, 21.3) == "too big"


def test_every_curated_model_runs_on_a_32gb_mac():
    """The selection criterion of the registry, made executable."""
    verdicts = {key: fit(card.run_gb, THIRTY_TWO_GB_BUDGET) for key, card in MODELS.items()}
    assert "too big" not in verdicts.values(), verdicts
    assert verdicts[DEFAULT_MODEL] == "ok"


def test_a_locally_quantised_card_runs_at_a_fraction_of_its_download(monkeypatch, tmp_path):
    card = MODELS["dianjin-r1-7b"]
    assert card.quantize == 4
    assert card.run_gb == pytest.approx(card.download_gb * 4.5 / 16)  # bf16 -> 4 bits + scales
    assert MODELS[DEFAULT_MODEL].run_gb == MODELS[DEFAULT_MODEL].download_gb
    monkeypatch.setenv("LEARNAI_MODELS_DIR", str(tmp_path))
    target = quantized_path(card)
    assert target == tmp_path / "quantized" / "DianJin--DianJin-R1-7B-4bit"
    assert local_path("dianjin-r1-7b") is None  # not converted yet: never mistaken for ready
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}")
    assert local_path("dianjin-r1-7b") == target


def test_registry_spans_dense_and_mixture_of_experts():
    assert {card.kind for card in MODELS.values()} == {"dense", "moe"}


def test_models_dir_defaults_to_the_hugging_face_cache(monkeypatch):
    monkeypatch.delenv("LEARNAI_MODELS_DIR", raising=False)
    assert models_dir() is None


def test_models_dir_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("LEARNAI_MODELS_DIR", str(tmp_path))
    assert models_dir() == tmp_path
    assert cached_sizes() == {}  # an empty store is empty, not an error


def test_an_unplugged_drive_is_reported_not_mistaken_for_empty(monkeypatch):
    monkeypatch.setenv("LEARNAI_MODELS_DIR", "/Volumes/learnai-no-such-drive/huggingface")
    with pytest.raises(DriveNotConnected, match="learnai-no-such-drive"):
        models_dir()
    # "nothing downloaded" would send `load` off to re-download gigabytes elsewhere
    with pytest.raises(DriveNotConnected):
        cached_sizes()


def _tiny_model_ready() -> bool:
    if importlib.util.find_spec("mlx_lm") is None:
        return False
    from learnai.openweights import local_path

    try:
        return local_path("qwen3-0.6b") is not None
    except DriveNotConnected:
        return False


@pytest.mark.skipif(not _tiny_model_ready(), reason="needs mlx-lm and qwen3-0.6b downloaded")
def test_greedy_generation_is_deterministic_and_answers():
    from learnai.openweights import generate, load

    model, tokenizer = load("qwen3-0.6b")
    ask = dict(temperature=0.0, max_tokens=24)
    first = generate(model, tokenizer, "What is 2 + 3? Reply with just the number.", **ask)
    second = generate(model, tokenizer, "What is 2 + 3? Reply with just the number.", **ask)

    assert first == second  # temperature 0 is argmax at every step: no randomness left
    assert "5" in first

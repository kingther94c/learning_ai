"""Tests for the mechanisms that make BERT different from a causal GPT."""

from __future__ import annotations

import math

import pytest
import torch

from learnai.data import (
    ToyBERTTokenizer,
    collate_bert_examples,
    make_nsp_examples,
    mask_for_mlm,
)
from learnai.nn import BERT, BERTConfig, BERTEncoderLayer, BERTForPreTraining

CORPUS = [
    "the cat says meow",
    "the dog says woof",
    "cats like warm milk",
    "dogs like long walks",
]


def tiny_config(tokenizer, **overrides):
    values = dict(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=12,
        n_layer=2,
        n_head=2,
        d_model=16,
        d_ff=32,
        dropout=0.0,
        pad_token_id=tokenizer.pad_token_id,
    )
    values.update(overrides)
    return BERTConfig(**values)


def test_tokenizer_packs_cls_sep_segments_and_padding():
    tok = ToyBERTTokenizer(CORPUS)
    packed = tok.encode_pair("the cat says meow", "cats like warm milk", max_length=12)
    tokens = tok.convert_ids_to_tokens(packed.input_ids)

    assert tokens[0] == "[CLS]"
    assert tokens.count("[SEP]") == 2
    second_sentence = tokens.index("[SEP]") + 1
    final_sep = len(tokens) - 1 - tokens[::-1].index("[SEP]")
    assert packed.token_type_ids[:second_sentence].eq(0).all()
    assert packed.token_type_ids[second_sentence : final_sep + 1].eq(1).all()
    assert packed.attention_mask.sum().item() == final_sep + 1
    assert packed.input_ids[~packed.attention_mask].eq(tok.pad_token_id).all()


def test_tokenizer_refuses_to_hide_truncation():
    tok = ToyBERTTokenizer(CORPUS)
    with pytest.raises(ValueError, match="packed sequence"):
        tok.encode_pair(CORPUS[0], CORPUS[1], max_length=5)


def test_nsp_examples_are_balanced_and_negatives_cross_documents():
    documents = [[CORPUS[0], CORPUS[1]], [CORPUS[2], CORPUS[3]]]
    examples = make_nsp_examples(documents, seed=0)
    assert [example.label for example in examples].count(0) == 2
    assert [example.label for example in examples].count(1) == 2
    for example in examples:
        if example.label == 1:
            assert not any(
                example.sentence_a == doc[0] and example.sentence_b == doc[1] for doc in documents
            )

    tok = ToyBERTTokenizer(CORPUS)
    batch = collate_bert_examples(
        examples,
        tok,
        max_length=12,
        selection_probability=1.0,
        generator=torch.Generator().manual_seed(0),
    )
    assert batch["input_ids"].shape == batch["attention_mask"].shape == (4, 12)
    assert sorted(batch["nsp_labels"].tolist()) == [0, 0, 1, 1]


def test_mlm_never_selects_special_tokens_and_labels_only_targets():
    tok = ToyBERTTokenizer(CORPUS)
    packed = tok.encode_pair(CORPUS[0], CORPUS[1], max_length=12)
    batch = packed.input_ids.repeat(256, 1)
    corruption = mask_for_mlm(
        batch,
        mask_token_id=tok.mask_token_id,
        vocab_size=tok.vocab_size,
        special_token_ids=tok.special_token_ids,
        selection_probability=0.5,
        generator=torch.Generator().manual_seed(0),
    )

    is_special = torch.zeros_like(batch, dtype=torch.bool)
    for token_id in tok.special_token_ids:
        is_special |= batch == token_id
    assert not corruption.selected[is_special].any()
    assert corruption.labels[~corruption.selected].eq(-100).all()
    assert torch.equal(corruption.labels[corruption.selected], batch[corruption.selected])
    assert torch.equal(batch, packed.input_ids.repeat(256, 1)), "corruption must be out-of-place"


def test_mlm_empirically_matches_15_percent_and_80_10_10():
    # One ordinary token repeated many times gives a clean Monte-Carlo check of
    # the data-generation probabilities without vocabulary-frequency noise.
    input_ids = torch.full((200_000,), 5, dtype=torch.long)
    corruption = mask_for_mlm(
        input_ids,
        mask_token_id=4,
        vocab_size=20,
        selection_probability=0.15,
        generator=torch.Generator().manual_seed(7),
    )
    selected_rate = corruption.selected.float().mean().item()
    strategies = corruption.strategy[corruption.selected]
    rates = torch.stack([(strategies == i).float().mean() for i in range(3)])

    assert selected_rate == pytest.approx(0.15, abs=0.003)
    assert rates.tolist() == pytest.approx([0.8, 0.1, 0.1], abs=0.01)


def test_embedding_is_normalised_sum_of_three_learned_tables():
    tok = ToyBERTTokenizer(CORPUS)
    cfg = tiny_config(tok, n_layer=1)
    model = BERT(cfg).eval()
    packed = tok.encode_pair(CORPUS[0], CORPUS[1], max_length=12)
    ids = packed.input_ids.unsqueeze(0)
    types = packed.token_type_ids.unsqueeze(0)

    positions = torch.arange(ids.size(1)).view(1, -1)
    raw_sum = (
        model.embeddings.token(ids)
        + model.embeddings.position(positions)
        + model.embeddings.segment(types)
    )
    expected = model.embeddings.norm(raw_sum)
    actual = model.embeddings(ids, types)
    assert torch.allclose(actual, expected, atol=1e-7)


def test_encoder_layer_matches_torch_post_norm_reference():
    tok = ToyBERTTokenizer(CORPUS)
    cfg = tiny_config(tok, n_layer=1)
    ours = BERTEncoderLayer(cfg).eval()
    reference = torch.nn.TransformerEncoderLayer(
        d_model=cfg.d_model,
        nhead=cfg.n_head,
        dim_feedforward=cfg.d_ff,
        dropout=0.0,
        activation=lambda x: torch.nn.functional.gelu(x, approximate="tanh"),
        layer_norm_eps=cfg.layer_norm_eps,
        batch_first=True,
        norm_first=False,
    ).eval()

    with torch.no_grad():
        reference.self_attn.in_proj_weight.copy_(ours.attn.qkv_proj.weight)
        reference.self_attn.in_proj_bias.copy_(ours.attn.qkv_proj.bias)
        reference.self_attn.out_proj.weight.copy_(ours.attn.out_proj.weight)
        reference.self_attn.out_proj.bias.copy_(ours.attn.out_proj.bias)
        reference.linear1.weight.copy_(ours.ff_in.weight)
        reference.linear1.bias.copy_(ours.ff_in.bias)
        reference.linear2.weight.copy_(ours.ff_out.weight)
        reference.linear2.bias.copy_(ours.ff_out.bias)
        reference.norm1.load_state_dict(ours.attn_norm.state_dict())
        reference.norm2.load_state_dict(ours.output_norm.state_dict())

    x = torch.randn(2, 7, cfg.d_model)
    actual, _ = ours(x)
    expected = reference(x)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_bert_outputs_have_paper_shapes_and_tied_mlm_weights():
    tok = ToyBERTTokenizer(CORPUS)
    cfg = tiny_config(tok)
    model = BERTForPreTraining(cfg).eval()
    packed = tok.encode_pair(CORPUS[0], CORPUS[1], max_length=12)

    out = model(
        packed.input_ids.unsqueeze(0),
        packed.token_type_ids.unsqueeze(0),
        packed.attention_mask.unsqueeze(0),
        return_attn=True,
    )
    assert out.hidden_states.shape == (1, 12, cfg.d_model)
    assert out.mlm_logits.shape == (1, 12, tok.vocab_size)
    assert out.nsp_logits.shape == (1, 2)
    assert len(out.attentions) == cfg.n_layer
    assert out.attentions[0].shape == (1, cfg.n_head, 12, 12)
    assert model.mlm_decoder.weight is model.bert.embeddings.token.weight
    assert model.bert.embeddings.norm.eps == pytest.approx(1e-12)
    limit = 2 * cfg.initializer_range
    assert model.bert.embeddings.position.weight.abs().max() <= limit


def test_bidirectional_first_token_depends_on_the_right_but_causal_does_not():
    tok = ToyBERTTokenizer(CORPUS)
    ids = tok.encode_pair(CORPUS[0], max_length=8).input_ids.unsqueeze(0)
    changed = ids.clone()
    changed[0, 3] = tok.stoi["woof"]

    torch.manual_seed(0)
    bidirectional = BERT(tiny_config(tok, max_seq_len=8, n_layer=1, causal=False)).eval()
    torch.manual_seed(0)
    causal = BERT(tiny_config(tok, max_seq_len=8, n_layer=1, causal=True)).eval()

    bi_out_a = bidirectional(ids, return_attn=True)
    bi_a = bi_out_a.last_hidden_state[:, 0]
    bi_b = bidirectional(changed).last_hidden_state[:, 0]
    causal_out_a = causal(ids, return_attn=True)
    causal_a = causal_out_a.last_hidden_state[:, 0]
    causal_b = causal(changed).last_hidden_state[:, 0]
    assert not torch.allclose(bi_a, bi_b), "BERT's [CLS] must be able to read right context"
    assert torch.allclose(causal_a, causal_b, atol=1e-7), "a causal position cannot see the future"
    assert bi_out_a.attentions[0][0, :, 0, 3].gt(0).all()
    assert causal_out_a.attentions[0][0, :, 0, 3].eq(0).all()


def test_padding_tokens_cannot_change_real_token_representations():
    tok = ToyBERTTokenizer(CORPUS)
    model = BERT(tiny_config(tok, n_layer=1)).eval()
    packed = tok.encode_pair(CORPUS[0], max_length=12)
    ids_a = packed.input_ids.unsqueeze(0)
    ids_b = ids_a.clone()
    ids_b[0, ~packed.attention_mask] = tok.stoi["dog"]

    out_a = model(ids_a, attention_mask=packed.attention_mask.unsqueeze(0)).last_hidden_state
    out_b = model(ids_b, attention_mask=packed.attention_mask.unsqueeze(0)).last_hidden_state
    real = packed.attention_mask
    assert torch.allclose(out_a[:, real], out_b[:, real], atol=1e-6)


def test_joint_loss_is_mlm_plus_nsp_and_uniform_baseline_is_known():
    tok = ToyBERTTokenizer(CORPUS)
    model = BERTForPreTraining(tiny_config(tok, n_layer=1))
    packed = tok.encode_pair(CORPUS[0], CORPUS[1], max_length=12)
    out = model(
        packed.input_ids.unsqueeze(0),
        packed.token_type_ids.unsqueeze(0),
        packed.attention_mask.unsqueeze(0),
    )
    labels = torch.full_like(packed.input_ids.unsqueeze(0), -100)
    labels[0, 2] = packed.input_ids[2]
    total, mlm, nsp = model.loss(out, labels, torch.tensor([0]))

    assert torch.allclose(total, mlm + nsp)
    assert math.isfinite(float(total.detach()))

    empty_labels = torch.full_like(labels, -100)
    empty_total, empty_mlm, _ = model.loss(out, empty_labels, torch.tensor([0]))
    assert empty_mlm.item() == 0.0 and math.isfinite(float(empty_total.detach()))


def test_config_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="divisible"):
        BERTConfig(vocab_size=20, d_model=10, n_head=3)
    with pytest.raises(ValueError, match="pad_token_id"):
        BERTConfig(vocab_size=20, pad_token_id=20)

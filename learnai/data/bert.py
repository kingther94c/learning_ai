"""The small pieces that turn text into BERT pre-training examples.

Real BERT used a 30,000-entry WordPiece vocabulary.  The tokenizer here is
deliberately word-level: it keeps the five special tokens and the sentence-pair
layout faithful while making every token printable in a teaching notebook.

The masking function *is* the original BERT recipe:

    choose 15% of ordinary tokens
      80% -> [MASK]
      10% -> a random vocabulary token
      10% -> unchanged

Only chosen positions receive labels; every other label is -100 so PyTorch's
cross entropy ignores it.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BERTInput:
    """One padded BERT input sequence, before adding a batch dimension."""

    input_ids: torch.Tensor  # (T,)
    token_type_ids: torch.Tensor  # (T,) -- sentence A=0, sentence B=1
    attention_mask: torch.Tensor  # (T,) -- real token=True, padding=False


@dataclass(frozen=True)
class MLMCorruption:
    """The observable result of BERT's 80/10/10 corruption process.

    ``strategy`` uses -1 for an unselected position, then 0/1/2 for
    [MASK]/random/unchanged.  Keeping this tensor makes the probabilities
    measurable instead of hiding them inside a data loader.
    """

    input_ids: torch.Tensor
    labels: torch.Tensor
    selected: torch.Tensor
    strategy: torch.Tensor


@dataclass(frozen=True)
class NSPExample:
    """A sentence pair labelled with BERT's original NSP convention.

    ``label=0`` means IsNext and ``label=1`` means NotNext, matching the public
    BERT data generator.  Negative B sentences come from another document so a
    true continuation cannot be mislabeled by accident.
    """

    sentence_a: str
    sentence_b: str
    label: int


class ToyBERTTokenizer:
    """A transparent word-level tokenizer with BERT's five special tokens.

    It is not a WordPiece implementation.  That simplification is intentional:
    the lesson is about bidirectional pre-training, not subword segmentation.
    ``BPETokenizer`` elsewhere in this package teaches the latter mechanism.
    """

    PAD = "[PAD]"
    UNK = "[UNK]"
    CLS = "[CLS]"
    SEP = "[SEP]"
    MASK = "[MASK]"
    SPECIAL_TOKENS = (PAD, UNK, CLS, SEP, MASK)

    # Recognise a supplied special token as one unit; otherwise split words and
    # punctuation.  Lowercasing mirrors BERT-base-uncased, not every BERT model.
    TOKEN_PATTERN = re.compile(r"\[(?:PAD|UNK|CLS|SEP|MASK)\]|\w+|[^\w\s]")

    def __init__(self, texts: Iterable[str]):
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(self.tokenize(text))

        ordinary = [token for token, _ in counts.most_common() if token not in self.SPECIAL_TOKENS]
        self.vocab = [*self.SPECIAL_TOKENS, *ordinary]
        self.stoi = {token: i for i, token in enumerate(self.vocab)}
        self.itos = dict(enumerate(self.vocab))

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        tokens = cls.TOKEN_PATTERN.findall(text)
        return [token if token in cls.SPECIAL_TOKENS else token.lower() for token in tokens]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.stoi[self.PAD]

    @property
    def unk_token_id(self) -> int:
        return self.stoi[self.UNK]

    @property
    def cls_token_id(self) -> int:
        return self.stoi[self.CLS]

    @property
    def sep_token_id(self) -> int:
        return self.stoi[self.SEP]

    @property
    def mask_token_id(self) -> int:
        return self.stoi[self.MASK]

    @property
    def special_token_ids(self) -> tuple[int, ...]:
        return tuple(self.stoi[token] for token in self.SPECIAL_TOKENS)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(token, self.unk_token_id) for token in self.tokenize(text)]

    def convert_ids_to_tokens(self, ids: Iterable[int]) -> list[str]:
        return [self.itos.get(int(i), self.UNK) for i in ids]

    def encode_pair(
        self,
        sentence_a: str,
        sentence_b: str | None = None,
        *,
        max_length: int,
    ) -> BERTInput:
        """Pack ``[CLS] A [SEP] B [SEP]`` and pad it to ``max_length``.

        Long inputs raise rather than truncate silently.  In a lesson, hiding
        which evidence got cut off would make an experiment impossible to read.
        """
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")

        a = self.encode(sentence_a)
        b = self.encode(sentence_b) if sentence_b is not None else []
        ids = [self.cls_token_id, *a, self.sep_token_id]
        token_types = [0] * len(ids)
        if sentence_b is not None:
            ids.extend([*b, self.sep_token_id])
            token_types.extend([1] * (len(b) + 1))

        if len(ids) > max_length:
            raise ValueError(
                f"packed sequence has {len(ids)} tokens but max_length={max_length}; "
                "shorten the sentences or increase max_length"
            )

        n_real = len(ids)
        n_pad = max_length - n_real
        ids.extend([self.pad_token_id] * n_pad)
        token_types.extend([0] * n_pad)
        attention = [True] * n_real + [False] * n_pad
        return BERTInput(
            input_ids=torch.tensor(ids, dtype=torch.long),
            token_type_ids=torch.tensor(token_types, dtype=torch.long),
            attention_mask=torch.tensor(attention, dtype=torch.bool),
        )


def mask_for_mlm(
    input_ids: torch.Tensor,
    *,
    mask_token_id: int,
    vocab_size: int,
    special_token_ids: Sequence[int] = (),
    selection_probability: float = 0.15,
    generator: torch.Generator | None = None,
) -> MLMCorruption:
    """Apply BERT's 15% selection and 80/10/10 replacement rule.

    The input may be ``(T,)`` or ``(B, T)``.  The returned tensors have exactly
    the same shape and device.  Corruption is out-of-place: the caller's clean
    tokens are never modified.
    """
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"input_ids must contain integer token ids, got {input_ids.dtype}")
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    if not 0.0 <= selection_probability <= 1.0:
        raise ValueError(f"selection_probability must be in [0, 1], got {selection_probability}")
    if input_ids.numel() and (input_ids.min() < 0 or input_ids.max() >= vocab_size):
        raise ValueError("input_ids contain a token outside [0, vocab_size)")
    if not 0 <= mask_token_id < vocab_size:
        raise ValueError(f"mask_token_id={mask_token_id} is outside the vocabulary")

    eligible = torch.ones_like(input_ids, dtype=torch.bool)
    for token_id in special_token_ids:
        eligible &= input_ids != token_id

    choose = torch.rand(input_ids.shape, device=input_ids.device, generator=generator)
    selected = eligible & (choose < selection_probability)

    labels = input_ids.clone()
    labels[~selected] = -100
    corrupted = input_ids.clone()

    # One uniform draw partitions selected positions into 80%, 10%, 10%.
    draw = torch.rand(input_ids.shape, device=input_ids.device, generator=generator)
    use_mask = selected & (draw < 0.8)
    use_random = selected & (draw >= 0.8) & (draw < 0.9)
    unchanged = selected & (draw >= 0.9)

    corrupted[use_mask] = mask_token_id
    random_ids = torch.randint(
        0,
        vocab_size,
        input_ids.shape,
        device=input_ids.device,
        generator=generator,
    )
    corrupted[use_random] = random_ids[use_random]

    strategy = torch.full_like(input_ids, -1, dtype=torch.int8)
    strategy[use_mask] = 0
    strategy[use_random] = 1
    strategy[unchanged] = 2
    return MLMCorruption(corrupted, labels, selected, strategy)


def make_nsp_examples(documents: Sequence[Sequence[str]], *, seed: int = 0) -> list[NSPExample]:
    """Make a balanced, deterministic teaching set for next-sentence prediction.

    Every real adjacent pair contributes one IsNext example and one NotNext
    example.  This is smaller than BERT's streaming data generator but preserves
    the paper's 50/50 task and its document boundary.
    """
    docs = [list(doc) for doc in documents]
    if len(docs) < 2:
        raise ValueError("NSP needs at least two documents for unambiguous negatives")
    if any(len(doc) < 2 for doc in docs):
        raise ValueError("every document needs at least two consecutive spans")

    rng = random.Random(seed)
    examples: list[NSPExample] = []
    for doc_index, doc in enumerate(docs):
        other_sentences = [
            sentence
            for other_index, other_doc in enumerate(docs)
            if other_index != doc_index
            for sentence in other_doc
        ]
        for sentence_a, sentence_b in zip(doc, doc[1:], strict=False):
            examples.append(NSPExample(sentence_a, sentence_b, 0))
            examples.append(NSPExample(sentence_a, rng.choice(other_sentences), 1))
    rng.shuffle(examples)
    return examples


def collate_bert_examples(
    examples: Sequence[NSPExample],
    tokenizer: ToyBERTTokenizer,
    *,
    max_length: int,
    selection_probability: float = 0.15,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Pack, pad, and dynamically corrupt a batch of NSP examples."""
    if not examples:
        raise ValueError("cannot collate an empty batch")
    packed = [
        tokenizer.encode_pair(ex.sentence_a, ex.sentence_b, max_length=max_length)
        for ex in examples
    ]
    clean = torch.stack([item.input_ids for item in packed])  # (B, T)
    corruption = mask_for_mlm(
        clean,
        mask_token_id=tokenizer.mask_token_id,
        vocab_size=tokenizer.vocab_size,
        special_token_ids=tokenizer.special_token_ids,
        selection_probability=selection_probability,
        generator=generator,
    )
    return {
        "input_ids": corruption.input_ids,
        "attention_mask": torch.stack([item.attention_mask for item in packed]),
        "token_type_ids": torch.stack([item.token_type_ids for item in packed]),
        "mlm_labels": corruption.labels,
        "nsp_labels": torch.tensor([example.label for example in examples], dtype=torch.long),
    }

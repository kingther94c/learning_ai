"""Tokenizers, from a character table up to a working BPE.

Tokenization is the least glamorous and most consequential part of an LLM. It
fixes the vocabulary, the sequence length, and — quietly — a lot of what the
model finds hard. ("Count the r's in strawberry" is hard partly because the
model never sees letters.)

Three levels, each fixing the previous one's problem:

    CharTokenizer   vocab ~100, sequences enormous, zero OOV
    WordTokenizer   vocab huge, sequences short, OOV on every typo
    BPETokenizer    vocab ~50k, sequences medium, an unseen *word* is never a
                    problem — the compromise that won
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path


class CharTokenizer:
    """One token per character. The simplest thing that works.

    Great for learning: with a 65-token vocabulary you can print the entire
    softmax distribution and read it. Bad for real use: "hello" costs 5 tokens,
    so context windows evaporate and the model must learn spelling before it can
    learn syntax.
    """

    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = dict(enumerate(self.chars))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def __repr__(self) -> str:
        preview = "".join(self.chars[:20]).replace("\n", "\\n")
        return f"CharTokenizer(vocab_size={self.vocab_size}, chars='{preview}...')"


class WordTokenizer:
    """One token per word, with a single <unk> for everything unseen.

    Included mainly so you can watch it fail: train it on one corpus, run it on
    another, and count how much becomes <unk>. That failure is the entire
    motivation for subwords.
    """

    UNK = "<unk>"
    PATTERN = re.compile(r"\w+|[^\w\s]")

    def __init__(self, text: str, max_vocab: int | None = None, min_freq: int = 1):
        counts = Counter(self.PATTERN.findall(text.lower()))
        words = [w for w, c in counts.most_common(max_vocab) if c >= min_freq]
        self.vocab = [self.UNK] + words
        self.stoi = {w: i for i, w in enumerate(self.vocab)}
        self.itos = dict(enumerate(self.vocab))

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, s: str) -> list[int]:
        return [self.stoi.get(w, 0) for w in self.PATTERN.findall(s.lower())]

    def decode(self, ids: Iterable[int]) -> str:
        return " ".join(self.itos[int(i)] for i in ids)

    def unk_rate(self, s: str) -> float:
        """Fraction of tokens that fall out of vocabulary. Try this cross-corpus."""
        ids = self.encode(s)
        return sum(1 for i in ids if i == 0) / max(len(ids), 1)


class BPETokenizer:
    """Byte-Pair Encoding, implemented the way the algorithm is actually defined.

    Training, in full:
        1. Start with every word split into characters.
        2. Count every adjacent pair across the corpus.
        3. Merge the most frequent pair into a single new symbol.
        4. Repeat until you hit the target vocabulary size.

    Each merge is recorded *in order*, and encoding replays those merges on new
    text. Because step 1 always works, an unseen *word* is never a problem — the
    worst case is that it stays as individual characters.

    One OOV case does survive: a character the training corpus never contained
    (an emoji, a Chinese character, an accented letter). It gets `UNK`, and
    `unk_rate()` will tell you how often. **This is exactly the hole GPT-2 closes
    by working on raw bytes** — starting from all 256 byte values means every
    possible input decomposes into known symbols, so the OOV case disappears
    entirely. Byte-level is the only substantive difference between this
    implementation and the real thing; the pre-split regex below is the other,
    and it is included because it visibly changes the learned vocabulary.

    Training is O(merges x corpus) in this straightforward form — fine for a few
    thousand merges on a few MB, which is all you need to see the behaviour.
    """

    # GPT-2's pre-tokenizer: keeps a leading space attached to the following word
    # (which is why " the" and "the" are different tokens in every GPT model).
    GPT2_PATTERN = re.compile(
        r"'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\sa-zA-Z0-9]+|\s+(?!\S)|\s+"
    )

    UNK = "<unk>"

    def __init__(self):
        self.merges: dict[tuple[str, str], int] = {}  # pair -> merge rank (order matters)
        self.vocab: list[str] = []
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}

    # -- training ----------------------------------------------------------------

    @staticmethod
    def _pair_counts(word_freqs: dict[tuple[str, ...], int]) -> Counter:
        """Count adjacent symbol pairs, weighted by how often each word occurs."""
        pairs: Counter = Counter()
        for symbols, freq in word_freqs.items():
            # deliberately ragged: zipping a sequence with its own tail is how
            # you walk adjacent pairs, so the lengths differ by one
            for a, b in zip(symbols, symbols[1:], strict=False):
                pairs[(a, b)] += freq
        return pairs

    @staticmethod
    def _merge_word(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
        """Replace every occurrence of `pair` in one word with the joined symbol."""
        out: list[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                out.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        return tuple(out)

    def train(self, text: str, vocab_size: int = 500, verbose: bool = False) -> BPETokenizer:
        """Learn merges until the vocabulary reaches `vocab_size`."""
        if vocab_size < 1:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")

        words = self.GPT2_PATTERN.findall(text)
        # Words are processed as frequency-weighted symbol tuples, not one by one —
        # "the" appearing 40,000 times is a single entry.
        word_freqs: dict[tuple[str, ...], int] = Counter(tuple(w) for w in words)

        base = sorted({c for w in words for c in w})
        # UNK occupies slot 0 so that an unseen character has somewhere to go
        # rather than being silently dropped on the floor.
        base_vocab = [self.UNK, *base]
        if vocab_size < len(base_vocab):
            raise ValueError(
                f"vocab_size={vocab_size} is smaller than the irreducible "
                f"character vocabulary ({len(base_vocab)}, including {self.UNK})"
            )
        # `train` is intentionally reusable. The learned merge table belongs to
        # one corpus; retaining ranks from a previous call would let old tokens
        # participate in the new tokenizer even though they are absent from its
        # vocabulary.
        self.merges = {}
        self.vocab = base_vocab
        n_merges = max(0, vocab_size - len(self.vocab))

        for rank in range(n_merges):
            pairs = self._pair_counts(word_freqs)
            if not pairs:
                break
            best, count = pairs.most_common(1)[0]
            if count < 2:  # nothing repeats any more; further merges just memorise
                break

            word_freqs = {self._merge_word(w, best): f for w, f in word_freqs.items()}
            self.merges[best] = rank
            self.vocab.append(best[0] + best[1])

            if verbose and rank % 50 == 0:
                print(f"  merge {rank:4d}: {best!r} x{count}  ->  {best[0] + best[1]!r}")

        self.stoi = {t: i for i, t in enumerate(self.vocab)}
        self.itos = dict(enumerate(self.vocab))
        return self

    # -- inference ---------------------------------------------------------------

    def _bpe_word(self, word: str) -> list[str]:
        """Apply the learned merges to one word, always taking the earliest-learned
        pair available. Merge *order* is the model — applying them in a different
        order gives a different (wrong) tokenization."""
        symbols = tuple(word)
        while len(symbols) > 1:
            candidates = {
                pair: self.merges[pair]
                for pair in zip(symbols, symbols[1:], strict=False)
                if pair in self.merges
            }
            if not candidates:
                break
            best = min(candidates, key=candidates.get)  # lowest rank == learned first
            symbols = self._merge_word(symbols, best)
        return list(symbols)

    def tokenize(self, s: str) -> list[str]:
        """Text -> token strings. Useful for *seeing* the split before encoding."""
        return [tok for word in self.GPT2_PATTERN.findall(s) for tok in self._bpe_word(word)]

    def encode(self, s: str) -> list[int]:
        """Text -> token ids. Characters the corpus never contained become UNK."""
        unk = self.stoi[self.UNK]
        return [self.stoi.get(t, unk) for t in self.tokenize(s)]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def unk_rate(self, s: str) -> float:
        """Fraction of tokens that fall out of vocabulary.

        Should be 0 on text resembling the training corpus. Run it on another
        language to see the hole that byte-level BPE exists to close.
        """
        ids = self.encode(s)
        unk = self.stoi[self.UNK]
        return sum(1 for i in ids if i == unk) / max(len(ids), 1)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def compression_ratio(self, s: str) -> float:
        """Characters per token. GPT-2 on English gets ~4; character-level gets 1."""
        return len(s) / max(len(self.encode(s)), 1)

    # -- persistence -------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "vocab": self.vocab,
                    # JSON has no tuple keys; store merges as ordered triples.
                    "merges": [[a, b, r] for (a, b), r in self.merges.items()],
                },
                ensure_ascii=False,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        obj = json.loads(Path(path).read_text())
        tok = cls()
        tok.vocab = obj["vocab"]
        tok.merges = {(a, b): r for a, b, r in obj["merges"]}
        tok.stoi = {t: i for i, t in enumerate(tok.vocab)}
        tok.itos = dict(enumerate(tok.vocab))
        return tok

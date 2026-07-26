"""Datasets and tokenizers — all small, all fast to load."""

from learnai.data.text import (  # noqa: F401
    download,
    get_batch,
    tiny_shakespeare,
    train_val_split,
)
from learnai.data.tokenizers import BPETokenizer, CharTokenizer, WordTokenizer  # noqa: F401
from learnai.data.toy import (  # noqa: F401
    copy_task,
    induction_task,
    make_spirals,
    make_xor,
    parity_task,
    reverse_task,
    sort_task,
)

__all__ = [
    "BPETokenizer",
    "CharTokenizer",
    "WordTokenizer",
    "copy_task",
    "download",
    "get_batch",
    "induction_task",
    "make_spirals",
    "make_xor",
    "parity_task",
    "reverse_task",
    "sort_task",
    "tiny_shakespeare",
    "train_val_split",
]

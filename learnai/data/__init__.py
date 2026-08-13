"""Datasets and tokenizers — all small, all fast to load."""

from learnai.data.bert import (  # noqa: F401
    BERTInput,
    MLMCorruption,
    NSPExample,
    ToyBERTTokenizer,
    collate_bert_examples,
    make_nsp_examples,
    mask_for_mlm,
)
from learnai.data.gpt_tasks import (  # noqa: F401
    ARROW_TOKEN_ID,
    BOS_TOKEN_ID,
    IGNORE_INDEX,
    NUMBER_TOKEN_OFFSET,
    SEPARATOR_TOKEN_ID,
    make_modular_shift_batch,
    modular_shift_accuracy,
)
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
    "ARROW_TOKEN_ID",
    "BERTInput",
    "BOS_TOKEN_ID",
    "CharTokenizer",
    "MLMCorruption",
    "IGNORE_INDEX",
    "NSPExample",
    "NUMBER_TOKEN_OFFSET",
    "SEPARATOR_TOKEN_ID",
    "ToyBERTTokenizer",
    "WordTokenizer",
    "copy_task",
    "collate_bert_examples",
    "download",
    "get_batch",
    "induction_task",
    "make_spirals",
    "make_xor",
    "make_nsp_examples",
    "make_modular_shift_batch",
    "mask_for_mlm",
    "modular_shift_accuracy",
    "parity_task",
    "reverse_task",
    "sort_task",
    "tiny_shakespeare",
    "train_val_split",
]

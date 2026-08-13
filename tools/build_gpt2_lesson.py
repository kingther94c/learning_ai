"""Generate the GPT-2 zero-shot task-formatting lesson.

The lesson is intentionally generated from Python: its structure stays
reviewable and the committed notebook stays output-free.  It trains only a tiny
offline language model and treats that run as a mechanism illustration, never as
a reproduction of WebText-scale GPT-2 results.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.nbuild import Notebook


def build() -> None:
    nb = Notebook(
        "GPT-2：一个 next-token model 怎样被“格式”变成任务模型？",
        subtitle=r"""
**核心问题：** GPT-2 没有为翻译、问答或摘要增加 task head；为什么只改输入字符串，
同一个语言模型就能表现出不同任务行为？

**你会验证：** continuation 的条件对数概率、候选排序、任务格式的作用，以及在完全不做
下游梯度更新时，怎样把结构化任务改写为语言模型原本就会计算的 $p(output\mid input,task)$。

**主要来源：** Radford et al., [*Language Models are Unsupervised Multitask
Learners*](https://cdn.openai.com/better-language-models/language-models.pdf)（OpenAI，2019）；
[OpenAI 原始发布页](https://openai.com/index/better-language-models/)；
[OpenAI 官方代码与 model card](https://github.com/openai/gpt-2)。GPT-1 对照只使用
[OpenAI 的 GPT-1 原论文](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)。

**证据边界先说清：** 本课的模型会从一份刻意包含多种格式的 tiny corpus 学习。它能展示
“自然语言/符号格式如何调用已经在训练文本中学到的模式”，不能证明模型面对一个训练分布里
从未出现过的任务也会 zero-shot。论文自己的主张同样是从 WebText 中自然出现的 task
demonstrations 学习，而不是从无任何相关经验中凭空发明任务。
""",
    )

    nb.md(r"""
---
## 0. 设置：固定 CPU、固定数据、固定随机数

全程不下载 checkpoint 或数据。默认路径训练一个约十万参数的字符级 GPT；
`LEARNAI_FAST=1` 仍走完训练、评分和 ablation，只减少更新次数。
""")
    nb.code("""
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from learnai.data import CharTokenizer, get_batch
from learnai.env import setup
from learnai.nn import GPT, GPTConfig
from learnai.nn.gpt_papers import continuation_logprob, rank_completions
from learnai.viz import SERIES, use_style

lesson_started = time.perf_counter()
ctx = setup(seed=2019, device="cpu")
torch.set_num_threads(1)
torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)
use_style()
FAST = os.environ.get("LEARNAI_FAST", "0") == "1"
print("fast smoke path:", FAST)
""")

    nb.md(r"""
---
## 1. 同一道题，为什么标点能改变答案？

先观察我们稍后会交给模型的两种字符串。语义内容相同，但 `English: ... French:` 明确把
空白处放进一种在语料里反复出现的 task format；无格式版本没有说明接下来应是翻译、解释，
还是继续英文句子。
""")
    nb.code(r"""
formatted_probe = "English: red bird\nFrench:"
unformatted_probe = "red bird"
candidates = [" oiseau rouge", " bird red", " chat bleu"]

print("formatted:  ", repr(formatted_probe))
print("unformatted:", repr(unformatted_probe))
print("same candidate strings:", candidates)
assert "French:" in formatted_probe and "French:" not in unformatted_probe
""")
    nb.md(r"""
此时不能下结论；随机模型对两种 prompt 都没有理由做对。我们要固定模型参数，只改变 prompt，
再逐 token 计算每个候选 completion 的条件概率。
""")

    nb.md(r"""
---
## 2. 论文的中心改写：任务也是一段文本

论文 §2 把单任务的 $p(output\mid input)$ 推广为

$$p(output\mid input,task).$$

`task` 不一定是新网络或整数 ID。语言本身可以把 task、input 和 output 串成一个 sequence：
翻译可以写成 `(translate to french, English text, French text)`，阅读理解可以写成
`(answer the question, document, question, answer)`。普通 causal LM 已经在学习整条序列的
next-token likelihood；测试时只需把前半段作为 prefix，比较后半段的条件 likelihood。
""")
    nb.code("""
def explicit_teacher_forcing_score(logits, continuation_ids, prefix_length):
    # logits at position t predict token t+1. The first continuation token is
    # predicted at prefix_length-1, and only continuation positions are summed.
    start = prefix_length - 1
    stop = start + continuation_ids.numel()
    selected_logits = logits[0, start:stop]
    return F.log_softmax(selected_logits, dim=-1).gather(
        -1, continuation_ids[:, None]
    ).squeeze(-1)

manual_model = GPT(GPTConfig(vocab_size=17, block_size=12, n_layer=1, n_head=2, d_model=16)).eval()
prefix_ids = torch.tensor([1, 4, 2, 6])
continuation_ids = torch.tensor([3, 8, 5])
whole = torch.cat([prefix_ids, continuation_ids])[None]
with torch.no_grad():
    manual_logits, _ = manual_model(whole)
    per_token_reference = explicit_teacher_forcing_score(
        manual_logits, continuation_ids, len(prefix_ids)
    )
    helper_score = continuation_logprob(manual_model, prefix_ids, continuation_ids)

print("continuation token log-probs:", per_token_reference)
print("sum:", per_token_reference.sum())
assert torch.allclose(helper_score, per_token_reference.sum(), atol=1e-6)
""")
    nb.takeaway(
        "Completion scoring 不是调用分类 head；它只在 candidate token 上读取 causal LM 已有的 log-probabilities。",
        "prefix token 只负责提供条件，不应被算进候选分数，否则 prompt 本身会污染选项比较。",
    )

    nb.exercise(
        "把 continuation 从 3 tokens 改成 2 tokens，手工指出 logits 的哪两个位置负责预测它，再与 helper 核对。",
        hint="第一个 continuation token 总由 prefix 的最后一个位置预测。",
        starter="""
short_continuation = continuation_ids[:2]
short_whole = torch.cat([prefix_ids, short_continuation])[None]
with torch.no_grad():
    short_logits, _ = manual_model(short_whole)
    mine = explicit_teacher_forcing_score(short_logits, short_continuation, len(prefix_ids)).sum()
    reference = continuation_logprob(manual_model, prefix_ids, short_continuation)
print("mine / helper:", mine.item(), reference.item())
assert torch.allclose(mine, reference, atol=1e-6)
""",
    )

    nb.md(r"""
---
## 3. GPT-2 相对 GPT-1：目标不是全换架构，而是扩大并稳住同一 decoder

GPT-1 已是 12-block、12-head、768-hidden 的 decoder-only Transformer，并使用 learned
position embeddings、GELU 和 512-token context；它的标准路径是 BooksCorpus 预训练后，
对每个标注任务做 supervised fine-tuning（GPT-1 paper §3–4）。

GPT-2 paper §2.3 明确说它“largely follows” GPT-1，只做少量结构/训练变化：

- LayerNorm 从每个 sub-block 的输出侧移到输入侧：**pre-norm**；
- 最后一个 self-attention block 后再加一层 final LayerNorm；
- residual-writing weights 按 $1/\sqrt{N}$ 缩放，补偿深度上的 residual accumulation；
- vocabulary 扩至 50,257，context 从 512 增至 1,024，batch size 增至 512；
- 输入改用可逆的 byte-level BPE，使任意 Unicode string 都能被赋概率。

真正的研究转向是：不再把 supervised fine-tuning 当作下游任务的必要步骤，而是直接测试
WebText LM 能否通过文本格式做 zero-shot transfer。
""")
    nb.code("""
paper_models = {
    "117M": (117, 12, 768),
    "345M": (345, 24, 1024),
    "762M": (762, 36, 1280),
    "1542M": (1542, 48, 1600),
}
names = list(paper_models)
parameters = np.array([paper_models[name][0] for name in names])
layers = np.array([paper_models[name][1] for name in names])
widths = np.array([paper_models[name][2] for name in names])

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
axes[0].bar(names, parameters, color=SERIES[:4])
axes[0].set_ylabel("parameters (millions, paper count)")
axes[0].set_title("GPT-2 paper Table 2")
for container in axes[0].containers:
    axes[0].bar_label(container, fmt="%.0fM", padding=2, fontsize=8)
axes[1].plot(layers, widths, marker="o", color=SERIES[0])
for name, x, y in zip(names, layers, widths, strict=True):
    axes[1].annotate(name, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=8)
axes[1].set_xlabel("Transformer layers")
axes[1].set_ylabel("d_model")
axes[1].set_title("Depth and width scale together")
plt.tight_layout()

assert paper_models["1542M"] == (1542, 48, 1600)
""")
    nb.md(r"""
**参数数字的版本陷阱：** 上图忠实抄录 2019 论文 Table 2 的 117M/345M/762M/1542M。
OpenAI 官方仓库后来注明原始 paper/blog 的 small、medium 参数计数有误；2019 model card
列出的已发布版本是 **124M、355M、774M、1.5B**。教学时应把“论文当时报告值”和
“官方后续更正值”分开，而不是悄悄混用。
""")
    nb.takeaway(
        "GPT-2 延续 GPT-1 的 causal decoder；pre-norm、final norm、深度缩放初始化、byte-level BPE 和更大 context 是关键工程变化。",
        "从 GPT-1 到 GPT-2 更大的概念变化是 transfer protocol：task-specific fine-tuning → 固定 LM 的 zero-shot formatting。",
    )

    nb.md(r"""
---
## 4. WebText：不是 Reddit 评论，也不是“无 task example 的纯文本”

论文 §2.1 从 Reddit 上获得至少 3 karma 的**外链**作为人工质量过滤启发式，再抽取链接页面
的正文。预备版本在去重和启发式清洗后有略多于 **800 万文档、40 GB 文本**，链接截止
2017 年 12 月；Wikipedia 被排除以减少与常见评测数据的重叠。OpenAI 发布页描述起点为
4500 万外链，官方 model card 也强调 WebText 是外链页面内容，不是 Reddit 本站文本。

论文 Table 1 特意展示 WebText 中自然出现的英法翻译示范。这一点非常重要：论文所谓
zero-shot 是**没有下游数据集训练/微调，也不改参数或架构**，不是保证预训练语料中从未出现
相似任务格式、知识或文本。
""")
    nb.code(r"""
toy_records = [
    "English: red bird\nFrench: oiseau rouge",
    "English: blue cat\nFrench: chat bleu",
    "English: green dog\nFrench: chien vert",
    "Question: color of snow?\nAnswer: white",
    "Question: sound of a cat?\nAnswer: meow",
    "A small red bird sat on a branch. It sang at dawn.\nTL;DR: a bird sang at dawn",
    "A blue cat slept on the warm mat.\nTL;DR: a cat slept",
]

# Rotate vocabulary items through identical formats. These are ordinary strings,
# not labels passed through a separate supervised data loader.
colors = [("red", "rouge"), ("blue", "bleu"), ("green", "vert")]
animals = [("bird", "oiseau"), ("cat", "chat"), ("dog", "chien")]
translation_records = [
    f"English: {en_color} {en_animal}\nFrench: {fr_animal} {fr_color}"
    for en_color, fr_color in colors
    for en_animal, fr_animal in animals
]
toy_records = translation_records + toy_records[3:]
toy_corpus = ("\n\n".join(toy_records) + "\n\n") * 80

print("characters:", len(toy_corpus), "records per cycle:", len(toy_records))
print(toy_records[0])
assert "English:" in toy_corpus and "Question:" in toy_corpus and "TL;DR:" in toy_corpus
""")

    nb.md(r"""
---
## 5. 只训练一个 next-character model

为了把整个概率分布打印出来，我们用字符 tokenizer；这**不是 GPT-2 的 byte-level BPE**。
模型只接收连续文本窗口，并在每个位置预测下一个字符；训练循环从未接收 `task_id`、分类标签、
BLEU、ROUGE 或 downstream loss。
""")
    nb.code("""
tokenizer = CharTokenizer(toy_corpus)
token_ids = torch.tensor(tokenizer.encode(toy_corpus), dtype=torch.long)
config = GPTConfig(
    vocab_size=tokenizer.vocab_size,
    block_size=48,
    n_layer=2,
    n_head=2,
    d_model=48,
    d_ff=96,
    dropout=0.0,
)
torch.manual_seed(2019)
model = GPT(config).to(ctx.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
batch_generator = torch.Generator().manual_seed(20)
steps = 300 if FAST else 420
history = []

model.train()
for step in range(steps):
    x, y = get_batch(token_ids, batch_size=24, block_size=config.block_size,
                     device=ctx.device, generator=batch_generator)
    _, loss = model(x, targets=y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    history.append(float(loss.detach()))

print(f"{sum(p.numel() for p in model.parameters()):,} parameters; "
      f"loss {history[0]:.3f} → {history[-1]:.3f}")
assert np.isfinite(history).all()
if not FAST:
    assert np.mean(history[-30:]) < np.mean(history[:30])

fig, ax = plt.subplots(figsize=(7.5, 3.4))
ax.plot(history, color=SERIES[0])
ax.set_xlabel("next-character update")
ax.set_ylabel("cross entropy")
ax.set_title("Measured here: tiny formatted-corpus LM")
plt.tight_layout()
""")
    nb.takeaway(
        "所有训练信号都是 next-token prediction；task behavior 是否出现，要在训练完成后另行测量。",
        "toy corpus 明确包含翻译/问答/摘要格式，所以这里只能称 task-format learning，不能称严格未知任务泛化。",
    )

    nb.md(r"""
---
## 6. Candidate ranking：从语言模型得到选择题答案

对同一个 prefix，把候选逐一拼接并只累加候选 token 的 log-probability：

$$s(c)=\sum_{j=1}^{|c|}\log p(c_j\mid prefix,c_{<j}).$$

`sum` 是整个字符串的概率，天然偏向短候选；论文 GPT-1 的 RACE zero-shot heuristic 和许多
multiple-choice LM 评估会使用平均 token log-probability
$\bar{s}(c)=s(c)/|c|$。这里的字符级候选长度不同，所以主实验明确使用 `mean` 隔离
formatting；下一格再把 `sum` 与 `mean` 并排，说明这项 protocol 选择为何不能省略。
""")
    nb.code("""
def encode(text):
    return torch.tensor(tokenizer.encode(text), dtype=torch.long, device=ctx.device)

model.eval()
formatted_ids = encode(formatted_probe)
candidate_ids = [encode(candidate) for candidate in candidates]
model.zero_grad(set_to_none=True)  # erase the last training step before the no-gradient audit
parameters_before = [parameter.detach().clone() for parameter in model.parameters()]

with torch.no_grad():
    formatted_scores, formatted_order = rank_completions(
        model, formatted_ids, candidate_ids, reduction="mean"
    )

print("formatted ranking:")
for index in formatted_order.tolist():
    print(f"  {candidates[index]!r}: {formatted_scores[index].item():.3f}")

assert formatted_scores.shape == formatted_order.shape == (len(candidates),)
assert formatted_order[0].item() == 0
assert all(
    torch.equal(before, after)
    for before, after in zip(parameters_before, model.parameters(), strict=True)
)
assert all(parameter.grad is None for parameter in model.parameters())
""")
    nb.code("""
short_candidate = encode(" oiseau")
long_candidate = encode(" oiseau rouge")
with torch.no_grad():
    short_sum = continuation_logprob(model, formatted_ids, short_candidate, reduction="sum")
    long_sum = continuation_logprob(model, formatted_ids, long_candidate, reduction="sum")
    short_mean = continuation_logprob(model, formatted_ids, short_candidate, reduction="mean")
    long_mean = continuation_logprob(model, formatted_ids, long_candidate, reduction="mean")

print("sum scores  short/long:", short_sum.item(), long_sum.item())
print("mean scores short/long:", short_mean.item(), long_mean.item())
assert short_sum >= long_sum  # adding log-probabilities cannot increase the sequence log-probability
""")
    nb.takeaway(
        "Candidate ranking 把 task 改写成同一个 LM 概率接口；它没有训练 classifier。",
        "sum 与 mean 回答不同问题；候选长度不一致时必须预先声明 scoring protocol。",
        "`torch.no_grad()` 加参数快照共同验证了下游评估没有任何梯度更新或参数变化。",
    )

    nb.exercise(
        "增加一个错误候选，用 mean ranking，并检查所有 score 都等于逐项调用 `continuation_logprob` 的结果。",
        hint="scores 保持原候选顺序；order 才是从高到低的候选索引。",
        starter="""
my_candidates = [" oiseau rouge", " oiseau green", " chat bleu"]
my_candidate_ids = [encode(text) for text in my_candidates]
with torch.no_grad():
    my_scores, my_order = rank_completions(
        model, formatted_ids, my_candidate_ids, reduction="mean"
    )
    direct_scores = torch.stack([
        continuation_logprob(model, formatted_ids, ids, reduction="mean")
        for ids in my_candidate_ids
    ])
for index in my_order.tolist():
    print(my_candidates[index], my_scores[index].item())
assert torch.allclose(my_scores, direct_scores, atol=1e-6)
assert torch.equal(my_order, my_scores.argsort(descending=True))
""",
    )

    nb.md(r"""
---
## 7. 关键 ablation：固定参数，只拿掉 task format

现在正面回答开场问题。模型、候选和评分函数保持不变，只把
`English: red bird\nFrench:` 换成 `red bird`。如果性能变化，因果差异来自 conditioning
sequence，不是微调、更多参数或不同 decoder。
""")
    nb.code("""
unformatted_ids = encode(unformatted_probe)
with torch.no_grad():
    unformatted_scores, unformatted_order = rank_completions(
        model, unformatted_ids, candidate_ids, reduction="mean"
    )

formatted_scores_np = formatted_scores.cpu().numpy()
unformatted_scores_np = unformatted_scores.cpu().numpy()

fig, ax = plt.subplots(figsize=(8.5, 3.7))
xpos = np.arange(len(candidates))
ax.bar(xpos - 0.18, formatted_scores_np, 0.36, label="with English/French format", color=SERIES[0])
ax.bar(xpos + 0.18, unformatted_scores_np, 0.36, label="content only", color=SERIES[1])
ax.set_xticks(xpos, candidates, rotation=20, ha="right")
ax.set_ylabel("mean completion log-probability")
ax.set_title("Measured here: same model, candidates, and weights")
ax.legend()
plt.tight_layout()

formatted_margin = formatted_scores_np[0] - max(formatted_scores_np[1:])
unformatted_margin = unformatted_scores_np[0] - max(unformatted_scores_np[1:])
print(f"correct-answer margin: formatted={formatted_margin:.3f}, unformatted={unformatted_margin:.3f}")
assert np.isfinite(formatted_scores_np).all() and np.isfinite(unformatted_scores_np).all()
assert torch.equal(unformatted_order, unformatted_scores.argsort(descending=True))
if not FAST:
    assert formatted_margin > unformatted_margin
""")
    nb.takeaway(
        "Prompt format 是条件概率中的信息，不是模型外的装饰；移除它就是一个真正的输入 ablation。",
        "这个 ablation 证明 toy model 使用了学过的格式，不能证明它理解现实中的法语或能发明未见任务。",
    )

    nb.exercise(
        "设计一个第三种 prompt：保留 `French:`，但删去 `English:`。先预测正确候选 margin 会落在何处，再测量。",
        hint="只改 prefix；模型、candidate IDs 和参数必须复用，才能把差异归因给格式。",
        starter="""
partial_prompt = encode("red bird\\nFrench:")
with torch.no_grad():
    partial_scores, partial_order = rank_completions(
        model, partial_prompt, candidate_ids, reduction="mean"
    )
partial_scores_np = partial_scores.cpu().numpy()
partial_margin = partial_scores_np[0] - max(partial_scores_np[1:])
print("margins formatted / partial / unformatted:",
      formatted_margin, partial_margin, unformatted_margin)
assert np.isfinite(partial_margin)
assert torch.equal(partial_order, partial_scores.argsort(descending=True))
""",
    )

    nb.md(r"""
---
## 8. 论文不是只写“prompt”：每个任务有具体 evaluation protocol

“GPT-2 zero-shot”不是一个统一 prompt 模板。论文 §3 对不同任务采用不同映射：

- **CBT cloze (§3.2)：** 把每个候选填入空位，计算“该候选以及余下句子”在前文条件下的
  概率，再从 10 个选项中取最高者；不是只给孤立候选词打分；
- **LAMBADA (§3.3)：** 预测最后一个词；原始预测 52.66% accuracy，加入 stop-word filter 后
  才是 Table 3 的 63.24%，perplexity 8.63；
- **Winograd (§3.4)：** 替换两个可能 referent，用 full/partial LM scoring 选高概率者，70.70%；
- **CoQA (§3.5)：** document + 对话历史 + 最后的 `A:`，greedy decode，development F1 55；
- **summarization (§3.6)：** article 后加 `TL;DR:`，top-k sampling `k=2` 生成 100 tokens，取前
  3 句；去掉 hint 后 Table 4 的 ROUGE 平均从 21.40 降到 15.03（−6.37）；
- **translation (§3.7)：** 先放若干 `English sentence = French sentence` pairs，再以
  `English sentence =` 结尾 greedy decode；En→Fr 5 BLEU，Fr→En 11.5 BLEU；
- **Natural Questions (§3.8)：** context 中放 example Q/A pairs 来诱导 short-answer style；
  exact match 4.1%，最高置信度 1% 子集 accuracy 63.1%。

因此，论文语境里的 zero-shot 是**无 task-specific parameter update**，不等同于现代常把
“上下文中零个 demonstration”称作 zero-shot 的狭义定义。翻译和 NQ 明确在 context 中使用
example pairs；把它们说成“纯指令、零演示”会改写历史。
""")
    nb.code("""
paper_results = {
    "LAMBADA acc": (63.24, 59.23),
    "CBT-CN acc": (93.30, 85.70),
    "CBT-NE acc": (89.05, 82.30),
    "Winograd acc": (70.70, 63.70),
}
labels = list(paper_results)
gpt2_values = np.array([paper_results[label][0] for label in labels])
previous_values = np.array([paper_results[label][1] for label in labels])

fig, ax = plt.subplots(figsize=(9, 3.8))
xpos = np.arange(len(labels))
ax.bar(xpos - 0.18, previous_values, 0.36, label="previous record", color=SERIES[1])
ax.bar(xpos + 0.18, gpt2_values, 0.36, label="GPT-2 paper", color=SERIES[0])
ax.set_xticks(xpos, labels, rotation=18, ha="right")
ax.set_ylabel("accuracy (%)")
ax.set_title("Paper §3 / Table 3 and Figure 3 — reported, not measured here")
ax.legend()
plt.tight_layout()

assert np.isclose(63.24 - 59.23, 4.01)
assert paper_results["Winograd acc"][0] == 70.70
""")
    nb.code("""
summary_rows = {
    "GPT-2 + TL;DR:": [29.34, 8.27, 26.58, 21.40],
    "GPT-2 no hint": [21.58, 4.03, 19.47, 15.03],
}
metrics = ["R-1", "R-2", "R-L", "R-AVG"]
hint_delta = np.array(summary_rows["GPT-2 + TL;DR:"]) - np.array(summary_rows["GPT-2 no hint"])

fig, ax = plt.subplots(figsize=(7.5, 3.5))
ax.bar(metrics, hint_delta, color=SERIES[:4])
ax.set_ylabel("ROUGE F1 points")
ax.set_title("Paper Table 4: benefit of the literal `TL;DR:` hint")
for container in ax.containers:
    ax.bar_label(container, fmt="%+.2f", padding=2, fontsize=8)
plt.tight_layout()
assert np.isclose(hint_delta[-1], 6.37)
""")

    nb.md(r"""
---
## 9. 关键结果也包括失败与 protocol caveat

论文 Table 3 报告四个模型尺度，最大模型在 8 个测试过的 LM datasets 中 7 个达到当时
zero-shot SOTA；但 1BW 仍差于此前记录。规模提升不是“已经解决通用任务”：

- §3.6 的摘要只略高于 random-3，且会混淆事实细节；
- §3.7 的 En→Fr 5 BLEU 甚至略差于一个 word-by-word substitution baseline；
- §3.8 的 NQ exact match 只有 4.1%，远低于当时 retrieval + extractive QA 的 30–50%；
- §6 直说许多实际任务很可能仍不优于随机，zero-shot performance 尚不可用；
- LAMBADA headline accuracy 依赖 stop-word filter，不能当作完全无规则的 raw argmax；
- 论文 §4 检测到 benchmark test 8-grams 与 WebText train 平均有 3.2% overlap，并说这给
  报告结果带来“小但一致”的收益；因此“无 benchmark fine-tuning”不等于“保证无数据污染”。
""")
    nb.code("""
datasets = ["PTB", "WikiText-2", "enwik8", "text8", "WikiText-103", "1BW"]
own_train_overlap = np.array([2.67, 0.66, 7.50, 2.34, 9.09, 13.19])
webtext_overlap = np.array([0.88, 1.63, 6.31, 3.94, 2.42, 3.75])

fig, ax = plt.subplots(figsize=(9, 3.8))
xpos = np.arange(len(datasets))
ax.bar(xpos - 0.18, own_train_overlap, 0.36, label="dataset's own train", color=SERIES[1])
ax.bar(xpos + 0.18, webtext_overlap, 0.36, label="WebText train", color=SERIES[0])
ax.set_xticks(xpos, datasets, rotation=20, ha="right")
ax.set_ylabel("test-set 8-grams overlapping train (%)")
ax.set_title("GPT-2 paper Table 6: overlap audit")
ax.legend()
plt.tight_layout()

assert np.isclose(webtext_overlap.mean(), 3.155, atol=0.01)
assert np.isclose(own_train_overlap.mean(), 5.908, atol=0.01)
""")
    nb.takeaway(
        "论文里的 zero-shot 是不对 benchmark 做参数更新；它不自动排除 prompt demonstrations、task-like pretraining text 或 train/test overlap。",
        "完整结论必须同时报告 prompt、decoding、filter、normalization 和 metric；这些都是 evaluation protocol。",
        "论文自己把实际可用性、事实错误、trivial baseline 和数据重叠列为限制，而不是把所有能力都写成成功。",
    )

    nb.md(r"""
---
## 10. 我们验证了什么，又没有验证什么

**本地真正验证了：**

- continuation score 与 teacher-forced next-token log-probability 的逐项等价；
- candidate ranking 只计算 completion token，不需要分类 head；
- 训练只使用 next-token objective，下游 scoring 期间无梯度、参数不变；
- 固定模型/候选时，task format ablation 会改变正确候选 margin；
- sum 与 mean scoring 在不同长度候选上代表不同 protocol。

**没有复现：**

- 800 万 WebText 文档、40 GB 数据、byte-level BPE、1.5B 参数或 1,024-token context；
- 论文 Table 3/4/6、Figure 1/3 的 benchmark accuracy、BLEU、ROUGE、F1 或 scaling trends；
- 现实语言翻译、事实问答，或面对训练分布里完全未知任务的 zero-shot 泛化。

本 toy 的 strongest conclusion 只是：**当 next-token model 的训练文本包含可辨识的任务格式时，
相同概率接口可以在冻结参数后被 task-formatted prefix 调用；移除格式会削弱该调用。**

### 原论文阅读地图

- **§2 + Eq. (1)**：language modeling 与 $p(output\mid input,task)$；
- **§2.1 + Table 1**：WebText 与自然出现的 translation demonstrations；
- **§2.2**：byte-level BPE；
- **§2.3 + Table 2**：GPT-1→GPT-2 架构变化和四个尺度；
- **§3.1 + Table 3**：zero-shot LM transfer 与 benchmark 数字；
- **§3.2–3.8 + Tables 4–5 / Figure 3**：每种任务的具体格式、scoring 与限制；
- **§4 + Table 6**：generalization vs memorization / n-gram overlap；
- **§6**：论文作者对 rudimentary、random baseline 和实用性的原始限制。

官方入口：[论文 PDF](https://cdn.openai.com/better-language-models/language-models.pdf) ·
[发布页](https://openai.com/index/better-language-models/) ·
[代码与 model card](https://github.com/openai/gpt-2)
""")
    nb.code("""
elapsed = time.perf_counter() - lesson_started
print(f"lesson wall time: {elapsed:.1f}s")
assert elapsed < 180, "lesson must stay below the repository's three-minute budget"
""")

    nb.save("notebooks/05_gpt/02_gpt2_zero_shot.ipynb")


if __name__ == "__main__":
    build()

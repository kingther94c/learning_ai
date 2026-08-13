"""Generate the classic word2vec teaching notebook.

The committed notebook is output-free and reviewable; execute it through
``tools/nbuild.py --run`` to verify both the normal and ``LEARNAI_FAST`` paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.nbuild import Notebook


def build() -> None:
    nb = Notebook(
        "Word2Vec from scratch：让词义从邻居中长出来",
        subtitle="""
**问题：** 只看一个词附近出现了谁，能否学出“相似词靠近、关系变成方向”的向量？

本课先把 CBOW 与 Skip-gram 的预测方向拆开，再逐项实现 Skip-gram with Negative
Sampling（SGNS）、$3/4$ 噪声分布与高频词 subsampling，最后在一个完全离线的合成
“微型世界”上训练并检查邻居和向量偏移。

**原始资料：** CBOW/Skip-gram 来自 Mikolov et al.,
[*Efficient Estimation of Word Representations in Vector Space*](https://arxiv.org/abs/1301.3781)
（公式与架构见 [PDF §3.1–3.2](https://arxiv.org/pdf/1301.3781)）；NEG、$U(w)^{3/4}$
和 subsampling 来自
[*Distributed Representations of Words and Phrases and their Compositionality*](https://arxiv.org/abs/1310.4546)
（[NeurIPS 官方 PDF §2.2–2.3](https://papers.nips.cc/paper_files/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf)）。
""",
    )

    nb.md("""
---
## 0. Setup

全程 CPU、无下载、无预训练权重。`LEARNAI_FAST=1` 只缩短合成语料与训练步数，仍会
执行数据构造、负采样、训练、近邻、类比和绘图的完整路径。
""")
    nb.code("""
import math
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from learnai.env import setup
from learnai.nn import (
    SkipGramNegativeSampling,
    negative_sampling_loss,
    sample_negative_ids,
    skipgram_pairs,
    subsampling_keep_probability,
    unigram_noise_distribution,
)
from learnai.viz import SERIES, use_style

ctx = setup(seed=23, device="cpu")
use_style()
torch.set_num_threads(1)
torch.set_printoptions(precision=3, sci_mode=False, linewidth=140)
FAST = os.environ.get("LEARNAI_FAST", "0") == "1"
print("fast smoke path:", FAST)
""")

    nb.md("""
---
## 1. 惊讶从这里开始：one-hot 没有“差不多”

若每个词只是一个 one-hot ID，任意两个不同词的余弦相似度都是 0。于是 `cat` 与
`dog` 不会比 `cat` 与 `carrot` 更接近——表示中根本没有“相似”的位置。
""")
    nb.code("""
words = ["cat", "dog", "carrot", "king"]
one_hot = torch.eye(len(words))
similarity = one_hot @ one_hot.T
print(words)
print(similarity)

assert similarity[0, 1] == similarity[0, 2] == 0
""")
    nb.takeaway(
        "one-hot 只回答“是不是同一个 ID”，不表达两个不同词有多相似。",
        "Word2Vec 的学习信号不是人工语义标签，而是局部共现：能预测相似邻居的词会被推向相似位置。",
    )

    nb.md("""
---
## 2. 同一个窗口，两种相反的预测题

第一篇论文的 Figure 1 与 §3 定义了两种浅层架构：

- **CBOW：** 把左右上下文向量平均，用它预测中间词；词序在这个平均中消失。
- **Skip-gram：** 给定中间词，分别预测窗口内每个上下文词。

论文的 Skip-gram 实验还会从 $[1,C]$ 随机选半径 $R$，让远处词较少被抽到。本课的
`skipgram_pairs` 刻意使用**固定窗口**，这样第一遍能把每一对都打印出来；这不是动态窗口的复现。
""")
    nb.code("""
tiny_sentence = ["the", "cat", "drinks", "warm", "milk", "slowly"]
tiny_ids = torch.arange(len(tiny_sentence))
centers, contexts = skipgram_pairs(tiny_ids, window_size=2)

focus = tiny_sentence.index("drinks")
sg_targets = [tiny_sentence[int(c)] for c in contexts[centers == focus]]
cbow_context = [tiny_sentence[i] for i in range(focus - 2, focus + 3) if i != focus]
print("Skip-gram: drinks ->", sg_targets)
print("CBOW:       ", cbow_context, "-> drinks")

assert sg_targets == ["the", "cat", "warm", "milk"]
assert cbow_context == sg_targets
""")
    nb.exercise(
        "把窗口从 2 改成 1。运行前先预测：6 个 token 会产生多少个有方向的 Skip-gram 正样本？",
        hint="两端各只有 1 个邻居，中间 4 个位置各有 2 个。",
        starter="""
my_centers, my_contexts = skipgram_pairs(tiny_ids, window_size=1)
print(list(zip(my_centers.tolist(), my_contexts.tolist())))
assert len(my_centers) == 10
""",
    )

    nb.md(r"""
---
## 3. 为什么需要两张向量表与 negative sampling

Skip-gram 的完整 softmax（第二篇论文 Eq. 2）为

$$p(w_O\mid w_I)=\frac{\exp({v'_{w_O}}^\top v_{w_I})}
{\sum_{w=1}^{V}\exp({v'_w}^\top v_{w_I})}.$$

一个词同时有输入向量 $v_w$ 与输出向量 $v'_w$。分母每次触碰全部 $V$ 个输出词；
NEG（Eq. 4）改成“1 个真实 pair 对 $k$ 个噪声 pair”的二元逻辑回归：

$$\log\sigma({v'_{w_O}}^\top v_{w_I})+
\sum_{i=1}^{k}\log\sigma(-{v'_{w_i}}^\top v_{w_I}).$$

下方用普通 BCE 独立重算同一个 loss。注意 NEG 的目标是优质词向量；原论文明确说它不像
NCE 那样以近似归一化 softmax 概率为必要目标。
""")
    nb.code("""
center_vec = torch.tensor([[1.0, -0.5, 0.25]])
positive_vec = torch.tensor([[0.8, -0.2, 0.1]])
negative_vecs = torch.tensor([[[-0.4, 0.2, 0.5], [0.1, 0.7, -0.3]]])

mine = negative_sampling_loss(center_vec, positive_vec, negative_vecs)
pos_score = (center_vec * positive_vec).sum(dim=-1, keepdim=True)
neg_scores = torch.einsum("bd,bkd->bk", center_vec, negative_vecs)
all_logits = torch.cat([pos_score, neg_scores], dim=1)
labels = torch.tensor([[1.0, 0.0, 0.0]])
reference = F.binary_cross_entropy_with_logits(all_logits, labels, reduction="none").sum()

print("positive logit:", float(pos_score), "negative logits:", neg_scores.tolist())
print("SGNS loss:", float(mine))
assert torch.allclose(mine, reference)
""")
    nb.takeaway(
        "Skip-gram 的方向是 center→context；每个词在这个训练目标中有 input 与 output 两个角色。",
        "NEG 每个正样本只比较 1+k 个输出向量，避免每步计算整个词表的 softmax 分母。",
        "负采样 loss 是未归一化的二分类目标，不应被解读成完整语言模型概率。",
    )

    nb.md(r"""
---
## 4. 为什么噪声不是均匀，也不是原始词频

第二篇论文 §2.2 选择

$$P_n(w)=\frac{U(w)^{3/4}}{Z}.$$

论文报告它在作者尝试的任务上显著优于 unigram 与 uniform，但没有给出这个比较的独立
数值表。$3/4$ 次方会压平长尾：若 A 出现 16 次、B 出现 1 次，原始词频比是 16，
变换后是 $16^{3/4}=8$；高频词仍更常被抽到，但统治力减半。
""")
    nb.code("""
demo_ids = torch.tensor([0] * 16 + [1])
p_raw = unigram_noise_distribution(demo_ids, power=1.0)
p_075 = unigram_noise_distribution(demo_ids, power=0.75)
p_uniform = torch.ones(2) / 2

fig, ax = plt.subplots(figsize=(7, 3.5))
x = np.arange(2)
ax.bar(x - 0.25, p_uniform, 0.25, label="uniform", color=SERIES[2])
ax.bar(x, p_raw, 0.25, label="unigram", color=SERIES[1])
ax.bar(x + 0.25, p_075, 0.25, label="unigram$^{3/4}$", color=SERIES[0])
ax.set_xticks(x, ["A: count 16", "B: count 1"])
ax.set_ylabel("negative-sampling probability")
ax.set_title("The 3/4 power flattens the noise distribution")
ax.legend()
plt.tight_layout()

assert (p_raw[0] / p_raw[1]).item() == 16
assert (p_075[0] / p_075[1]).item() == 8
""")
    nb.exercise(
        "把 `power` 改为 0.5。先预测 16:1 的词频会变成多大的采样概率比，再用代码验证。",
        hint="$16^{1/2}=4$。指数越小，分布越接近 uniform。",
        starter="""
p_half = unigram_noise_distribution(demo_ids, power=0.5)
ratio_half = (p_half[0] / p_half[1]).item()
print("ratio at power=0.5:", ratio_half)
assert ratio_half == 4
""",
    )

    nb.md(r"""
---
## 5. Subsampling：论文公式与作者代码并不完全相同

第二篇论文 Eq. 5 给出丢弃概率 $1-\sqrt{t/f(w)}$，等价于保留概率
$\min(1,\sqrt{t/f})$，并称 $t$ 通常约为 $10^{-5}$。但作者发布的
[`word2vec.c`](https://github.com/tmikolov/word2vec/blob/master/word2vec.c) 实际使用
$\min(1,\sqrt{t/f}+t/f)$。下面明确画成两条曲线，避免把实现细节错误归给论文公式。
""")
    nb.code("""
frequencies = torch.logspace(-7, -1, 300)
t = 1e-5
keep_paper = subsampling_keep_probability(frequencies, threshold=t, variant="paper")
keep_code = subsampling_keep_probability(frequencies, threshold=t, variant="reference_code")

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.semilogx(frequencies, keep_paper, label="paper Eq. 5", color=SERIES[0])
ax.semilogx(frequencies, keep_code, label="released word2vec.c", color=SERIES[1])
ax.axvline(t, color=SERIES[2], linestyle="--", label="t = 1e-5")
ax.set_xlabel("corpus frequency f(w)")
ax.set_ylabel("keep probability")
ax.set_ylim(-0.02, 1.04)
ax.set_title("Frequent tokens are retained less often")
ax.legend()
plt.tight_layout()

assert torch.all(keep_code >= keep_paper)
assert keep_paper[-1] < keep_paper[0]
""")
    nb.takeaway(
        "$3/4$ 次方改变的是负样本来自哪里；subsampling 改变的是哪些真实 token 进入训练语料。",
        "论文 Eq. 5 与发布代码的保留概率不同；可复现工作必须写清采用哪个版本。",
    )

    nb.md("""
---
## 6. 建一个可控、离线的“微型世界”

自然语料需要很大规模才能稳定显出关系。为了在 CPU 上几秒内看到机制，我们人为定义
16 个实体及其上下文属性，再随机打乱属性位置。模型只看 token 序列，从未直接收到
“类别”或“性别关系”标签。`the` 被故意放得很频繁，让 $3/4$ 分布的作用可见。

这是**机制级 toy reproduction**，不是论文数值复现：合成上下文本来就比自然语言干净。
""")
    nb.code("""
entity_features = {
    "king": ["royal", "adult", "male", "palace"],
    "queen": ["royal", "adult", "female", "palace"],
    "prince": ["royal", "young", "male", "palace"],
    "princess": ["royal", "young", "female", "palace"],
    "man": ["human", "adult", "male", "person"],
    "woman": ["human", "adult", "female", "person"],
    "boy": ["human", "young", "male", "person"],
    "girl": ["human", "young", "female", "person"],
    "cat": ["animal", "pet", "furry", "home"],
    "dog": ["animal", "pet", "furry", "home"],
    "wolf": ["animal", "wild", "furry", "forest"],
    "tiger": ["animal", "wild", "furry", "forest"],
    "apple": ["fruit", "food", "sweet", "orchard"],
    "orange": ["fruit", "food", "sweet", "orchard"],
    "carrot": ["vegetable", "food", "fresh", "garden"],
    "potato": ["vegetable", "food", "fresh", "garden"],
}

def make_corpus(repeats, seed=23):
    rng = random.Random(seed)
    sentences = []
    for _ in range(repeats):
        for entity, features in entity_features.items():
            shuffled = features.copy()
            rng.shuffle(shuffled)
            sentences.append(["the", shuffled[0], entity, shuffled[1], shuffled[2], "the", shuffled[3]])
    rng.shuffle(sentences)
    return sentences

sentences = make_corpus(10 if FAST else 70)
vocab = sorted({token for sentence in sentences for token in sentence})
stoi = {word: i for i, word in enumerate(vocab)}
itos = {i: word for word, i in stoi.items()}
encoded = [[stoi[token] for token in sentence] for sentence in sentences]
flat_ids = torch.tensor([token for sentence in encoded for token in sentence])

all_centers, all_contexts = [], []
for sentence in encoded:  # one call per sentence: no pair crosses a boundary
    c, o = skipgram_pairs(sentence, window_size=2)
    all_centers.append(c)
    all_contexts.append(o)
all_centers = torch.cat(all_centers)
all_contexts = torch.cat(all_contexts)
noise = unigram_noise_distribution(flat_ids, vocab_size=len(vocab), power=0.75)

print(f"{len(sentences):,} sentences | {len(flat_ids):,} tokens | {len(vocab)} words | {len(all_centers):,} positive pairs")
print("most frequent:", [(itos[int(i)], int((flat_ids == i).sum())) for i in noise.argsort(descending=True)[:6]])
assert len(all_centers) == len(all_contexts)
assert all_centers.min() >= 0 and all_contexts.max() < len(vocab)
""")

    nb.md("""
---
## 7. 训练 SGNS：每步只碰一小撮词

参考类包含两张 embedding 表，output/context 表从 0 初始化，与作者代码一致。我们从正
pair 中抽 batch，再按 $U^{3/4}$ 抽 5 个 negatives。为把课堂时间花在目标函数而非优化器
上，toy run 使用 Adam；2013 第一篇论文的报告实验使用 SGD、初始学习率 0.025 并线性衰减。
""")
    nb.code("""
torch.manual_seed(23)
model = SkipGramNegativeSampling(vocab_size=len(vocab), embedding_dim=24)
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
batch_size, k = 256, 5
steps = 100 if FAST else 700
train_gen = torch.Generator().manual_seed(2301)

eval_n = min(512, len(all_centers))
eval_centers = all_centers[:eval_n]
eval_contexts = all_contexts[:eval_n]
eval_negatives = sample_negative_ids(
    noise, eval_n, k, generator=torch.Generator().manual_seed(99)
)
with torch.no_grad():
    initial_eval = model(eval_centers, eval_contexts, eval_negatives).item()

losses = []
started = time.perf_counter()
for step in range(steps):
    indices = torch.randint(len(all_centers), (batch_size,), generator=train_gen)
    center_batch = all_centers[indices]
    context_batch = all_contexts[indices]
    negatives = sample_negative_ids(noise, batch_size, k, generator=train_gen)
    loss = model(center_batch, context_batch, negatives)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
elapsed = time.perf_counter() - started

with torch.no_grad():
    final_eval = model(eval_centers, eval_contexts, eval_negatives).item()
print(f"fixed eval loss: {initial_eval:.3f} -> {final_eval:.3f}; {steps} steps in {elapsed:.2f}s")

fig, ax = plt.subplots(figsize=(8, 3.6))
window = 10
smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
ax.plot(np.arange(window - 1, len(losses)), smooth, color=SERIES[0])
ax.set_xlabel("training step")
ax.set_ylabel("SGNS loss (moving average)")
ax.set_title("One positive plus five noise comparisons")
plt.tight_layout()

assert final_eval < initial_eval
assert elapsed < 180
""")
    nb.takeaway(
        "一个正 pair 的 loss 是正例逻辑损失，加上 k 个负例逻辑损失；batch reduction 最后才取平均。",
        "训练后保留 center/input 表做相似度检查；另一张 context/output 表是学习目标的一部分。",
    )

    nb.md("""
---
## 8. 向量学到了什么：邻居、关系方向与二维投影

余弦近邻检查“上下文相似的词是否靠近”。关系检查则比较
`queen - king`、`woman - man` 等偏移是否朝相似方向。二维图只用于显示；计算近邻与
offset cosine 始终在完整 24 维空间完成。
""")
    nb.code("""
vectors = model.normalized_embeddings().detach()

def nearest(word, k=4):
    query_id = stoi[word]
    scores = vectors @ vectors[query_id]
    scores[query_id] = -torch.inf
    ids = scores.topk(k).indices.tolist()
    return [(itos[int(i)], round(float(scores[int(i)]), 3)) for i in ids]

for word in ["cat", "wolf", "king", "woman", "apple", "carrot"]:
    print(f"{word:>7} -> {nearest(word)}")

relations = [("king", "queen"), ("prince", "princess"), ("man", "woman"), ("boy", "girl")]
offsets = torch.stack([vectors[stoi[b]] - vectors[stoi[a]] for a, b in relations])
offsets = F.normalize(offsets, dim=-1)
offset_cosines = offsets @ offsets.T
print("\\nrelation-offset cosine matrix:\\n", offset_cosines)
assert torch.allclose(offset_cosines.diag(), torch.ones(len(relations)), atol=1e-6)
""")
    nb.code("""
selected = ["king", "queen", "prince", "princess", "man", "woman", "boy", "girl",
            "cat", "dog", "wolf", "tiger", "apple", "orange", "carrot", "potato"]
matrix = model.center.weight.detach()[[stoi[w] for w in selected]]
matrix = matrix - matrix.mean(dim=0, keepdim=True)
_, _, vh = torch.linalg.svd(matrix, full_matrices=False)
coords = (matrix @ vh[:2].T).numpy()

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(coords[:, 0], coords[:, 1], color=SERIES[0])
for word, (x_pos, y_pos) in zip(selected, coords, strict=True):
    ax.annotate(word, (x_pos, y_pos), xytext=(4, 3), textcoords="offset points", fontsize=8)
for left, right in relations:
    i, j = selected.index(left), selected.index(right)
    delta = coords[j] - coords[i]
    ax.arrow(coords[i, 0], coords[i, 1], delta[0], delta[1],
             color=SERIES[1], alpha=0.55, head_width=0.025, length_includes_head=True)
ax.set_xlabel("PCA-like component 1 (SVD)")
ax.set_ylabel("PCA-like component 2 (SVD)")
ax.set_title("2D projection: inspect structure, do not score it here")
plt.tight_layout()
""")
    nb.exercise(
        "做一次 `king - man + woman` 查询。搜索时排除三个输入词；观察 `queen` 是否进入 top-5，并解释一次失败可能来自哪些 toy 限制。",
        hint="小语料、随机优化、固定窗口和二维不可见的方向都可能影响 top-1；先看完整维度的排名，而不是只看散点图。",
        starter="""
query = F.normalize(vectors[stoi["king"]] - vectors[stoi["man"]] + vectors[stoi["woman"]], dim=0)
scores = vectors @ query
for word in ["king", "man", "woman"]:
    scores[stoi[word]] = -torch.inf
answers = [(itos[i], round(float(scores[i]), 3)) for i in scores.topk(5).indices.tolist()]
print("king - man + woman ->", answers)
""",
    )

    nb.md("""
---
## 9. 论文证据与本课证据：不要混在一起

第二篇论文 Table 1 使用约 **10 亿词**新闻语料、692K 词表、300 维 Skip-gram。在其
analogy test 上，以下是论文直接报告的结果（时间单位为分钟）：

| 方法 | subsampling | time | syntactic | semantic | total |
|---|---:|---:|---:|---:|---:|
| NEG-5 | 无 | 38 | 63 | 54 | 59 |
| NEG-15 | 无 | 97 | 63 | 58 | 61 |
| NEG-5 | $10^{-5}$ | 14 | 61 | 58 | 60 |
| NEG-15 | $10^{-5}$ | 36 | 61 | 61 | 61 |

这支持“在该大规模设置中 subsampling 明显加速，并可维持/改善准确率”；它不保证每个
小语料 ablation 都同方向。第一篇论文 Table 3 在同一 320M-word、82K-vocab、640-d
设置中报告 CBOW semantic/syntactic 为 **24/64**，Skip-gram 为 **55/59**：作者据此
描述 CBOW 更强于句法、Skip-gram 明显更强于语义，但这不是所有数据集上的普遍定律。

本课只建立以下较窄证据：

1. 固定窗口能正确产生 center→context pairs；没有实现论文的随机动态窗口。
2. NEG 公式与 BCE 数值一致，$3/4$ 分布及两种 subsampling 公式数值可检查。
3. 小合成语料上的固定评估 loss 下降，并能**观察**近邻/offset；不把 toy 指标与论文准确率比较。
4. 实际训练的是 **SGNS**。CBOW 在 §2 展示了正确预测方向，但本课没有声称完成 CBOW 实验复现。
""")

    nb.md("""
---
## 10. 收尾：从 Word2Vec 到 Transformer

Word2Vec 把一个词压成一条与具体句子无关的向量：`bank` 在河岸与银行句子里仍共享同一
表示。它也在 CBOW 平均和本课固定窗口 pair 中丢掉大量词序信息。Transformer 的下一步
不是否定这些向量，而是让每个 token 的表示随上下文逐层改变。
""")
    nb.takeaway(
        "CBOW 用上下文预测中心；Skip-gram 用中心预测上下文。预测方向是理解两者的第一把钥匙。",
        "NEG、$U^{3/4}$ 与高频词 subsampling 是不同层面的效率技巧，来源于 2013 年的后续论文。",
        "一次可控 toy run 能验证机制和代码路径；论文级质量主张仍需论文级语料、评测与超参数。",
        "Word2Vec 给每个词静态向量；后续 BERT/GPT 给每次出现的 token 上下文化向量。",
    )

    nb.save("notebooks/03_nlp/01_word2vec.ipynb")


if __name__ == "__main__":
    build()

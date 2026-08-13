"""Generate the GPT-1 generative-pretraining teaching notebook.

The notebook contrasts claims in Radford et al. (2018) with a deliberately
small offline transfer experiment.  Run this builder after editing, then verify
the emitted notebook with ``tools/nbuild.py --run``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.nbuild import Notebook


def build() -> None:
    nb = Notebook(
        "GPT-1：先学会续写，再学会分类",
        subtitle="""
**问题：** 没有标签的“预测下一个词”，为什么能帮助一个只有少量标签的理解任务？

本课复现 GPT-1 最重要的训练协议，而非排行榜规模：同一个 causal Transformer 先在
未标注文本上做语言模型预训练，再通过统一的 token 序列格式与一个线性头做监督迁移。

**Primary sources:** Radford et al. (2018),
[*Improving Language Understanding by Generative Pre-Training*](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
（下文简称“论文”）；[OpenAI 官方研究页](https://openai.com/index/language-unsupervised/)；
[OpenAI 官方代码与模型](https://github.com/openai/finetune-transformer-lm)。
""",
    )

    nb.md("""
---
## 0. Setup

全程离线、固定种子、CPU。合成语料不是 BooksCorpus，也不会下载 GPT-1 权重。
`LEARNAI_FAST=1` 只减少优化步数，仍完整执行 pretrain → transfer → ablation。
""")
    nb.code("""
import copy
import math
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from learnai.env import setup
from learnai.nn import GPT, GPTConfig
from learnai.nn.gpt_papers import GPTSequenceClassifier, joint_transfer_loss
from learnai.viz import SERIES, use_style

ctx = setup(seed=41, device="cpu")
use_style()
torch.set_num_threads(1)
torch.set_printoptions(precision=3, sci_mode=False, linewidth=140)
FAST = os.environ.get("LEARNAI_FAST", "0") == "1"
print("fast smoke path:", FAST)
""")

    nb.md("""
---
## 1. 先看失败：4 个标签能教会多少个新词？

下游训练只标注 `bright / warm / dark / cold`，测试却只出现从未标注的形容词。一个从
随机初始化开始的分类器必须靠这 4 个标签同时学习语言与决策边界；预训练模型已经在
未标注句子里见过所有形容词及其后果。

先不要训练。请预测下面四条 test-only 句子的标签，并写出你的理由：模型到底需要从
未标注文本中迁移什么？随机模型也可能靠模板、有限样本或偶然几何猜中一些词；只有
相对**同初始化、同优化预算的 matched scratch baseline** 的稳定增益才是迁移证据。
""")
    nb.code("""
positive_words = ["bright", "warm", "delightful", "charming", "wonderful", "excellent"]
negative_words = ["dark", "cold", "dreadful", "boring", "awful", "grim"]
labelled_words = {"bright": 1, "warm": 1, "dark": 0, "cold": 0}
held_out_words = [w for w in positive_words + negative_words if w not in labelled_words]

print("labelled downstream:", labelled_words)
print("test-only adjectives:", held_out_words)
assert set(labelled_words).isdisjoint(held_out_words)
""")
    nb.takeaway(
        "GPT-1 的核心不是换一个分类架构，而是把未标注语言建模得到的参数当作监督任务的起点。",
        "这个 toy 故意让测试词没有分类标签；是否迁移要看 pretrained 相对 matched scratch 的增益，而不是单看一次 held-out 命中。",
    )

    nb.md(r"""
---
## 2. 论文的两阶段目标

[论文 §3.1，printed p.3 / PDF p.3](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf#page=3)
的第一阶段在 token stream $U=\{u_1,\ldots,u_n\}$ 上最大化

$$L_1(U)=\sum_i\log P(u_i\mid u_{i-k},\ldots,u_{i-1};\Theta).$$

causal mask 保证位置 $i$ 不能偷看将来的答案。token embedding $W_e$ 与 learned
position embedding $W_p$ 相加，经过多层 Transformer decoder；输出 softmax 又使用
$W_e^\top$（论文 Eq. 2，输入/输出权重共享）。

第二阶段把整段输入送入同一个模型，取最后位置的顶层 hidden state $h_l^m$，只加线性
层 $W_y$：$P(y\mid x^1,\ldots,x^m)=\mathrm{softmax}(h_l^mW_y)$（Eq. 3）。所有
Transformer 参数一起 fine-tune；论文说新增参数只有 $W_y$ 与新 delimiter token embeddings。
""")
    nb.code("""
demo_cfg = GPTConfig(vocab_size=17, block_size=8, n_layer=2, n_head=2, d_model=16,
                     d_ff=32, dropout=0.0, tie_weights=True)
demo = GPT(demo_cfg).eval()
demo_ids = torch.tensor([[1, 2, 3, 4, 5]])
hidden = demo.encode(demo_ids)
logits, _ = demo(demo_ids)
manual_logits = hidden @ demo.token_emb.weight.T

print("hidden:", tuple(hidden.shape), "logits:", tuple(logits.shape))
assert hidden.shape == (1, 5, 16)
assert demo.lm_head.weight.data_ptr() == demo.token_emb.weight.data_ptr()
assert torch.allclose(logits, manual_logits, atol=1e-6)
""")
    nb.exercise(
        "把 `demo_ids` 最后一个 token 改掉。运行前预测哪些 earlier hidden states 可以变化。",
        hint="这是单向 LM；最后 token 不能向左传播信息。",
        starter="""
changed_ids = demo_ids.clone()
changed_ids[0, -1] = 6
changed_hidden = demo.encode(changed_ids)
delta = (changed_hidden - hidden).norm(dim=-1)
print(delta)
assert torch.allclose(delta[:, :-1], torch.zeros_like(delta[:, :-1]), atol=1e-7)
assert delta[:, -1].item() > 0
""",
    )

    nb.md(r"""
---
## 3. GPT-1 到底有多大？

[论文 §4.1，printed p.4 / PDF p.5](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf#page=5)
的报告模型是：

| item | paper setting |
|---|---:|
| decoder blocks / state width / heads | 12 / 768 / 12 |
| feed-forward inner width | 3072 |
| maximum sequence | 512 tokens |
| tokenizer | BPE, 40,000 merges |
| position encoding | learned |
| activation / dropout | GELU / 0.1 |
| initialization | $\mathcal{N}(0, 0.02)$ |

预训练使用 BooksCorpus（超过 7,000 本未出版书、保留长连续文本），100 epochs，batch 64
个随机连续 512-token 序列；Adam 最大 LR $2.5\times10^{-4}$，前 2,000 updates 线性
warmup，再 cosine decay。论文报告 token-level perplexity 18.4。

官方 [`train.py` lines 129–136](https://github.com/openai/finetune-transformer-lm/blob/master/train.py#L129-L136)
还让一个 block 使用 **post-norm**：`LN(x + attention)`，再
`LN(n + MLP)`。本仓库通用 `GPT` 是便于与后续 GPT-2/3 共用的 pre-norm 教学实现。
因此本课忠实复现 causal decoder、learned positions、weight tying 和两阶段协议，**不把
toy backbone 称为逐层等同的 GPT-1**。
""")
    nb.code("""
paper = {"layers": 12, "d_model": 768, "heads": 12, "d_ff": 3072, "context": 512}
toy = {"layers": 2, "d_model": 48, "heads": 2, "d_ff": 96, "context": 10}

fig, ax = plt.subplots(figsize=(8, 3.5))
names = list(paper)
x = np.arange(len(names))
ax.bar(x - 0.2, [paper[n] for n in names], 0.4, label="paper", color=SERIES[0])
ax.bar(x + 0.2, [toy[n] for n in names], 0.4, label="this toy", color=SERIES[1])
ax.set_yscale("log")
ax.set_xticks(x, names)
ax.set_ylabel("size (log scale)")
ax.set_title("Mechanism reproduction, not scale reproduction")
ax.legend()
plt.tight_layout()

assert paper["d_ff"] == 4 * paper["d_model"]
assert toy["d_ff"] == 2 * toy["d_model"]  # explicit speed choice, not a paper claim
""")

    nb.md("""
---
## 4. 用 token 序列替代四套任务架构

[论文 §3.3 / Figure 1，printed pp.3–4](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf#page=4)
的关键工程思想叫 **task-specific input transformations**：结构化
输入先线性化，主干仍是同一个 LM。

| task | transformation |
|---|---|
| classification | `<s> text <e>` |
| entailment | `<s> premise $ hypothesis <e>` |
| similarity | 两个顺序都跑：`A $ B` 与 `B $ A`，末状态逐元素相加 |
| multiple-choice QA / reasoning | 每个候选独立跑：`context question $ answer_k`，再对候选 softmax |

所有变换都加入随机初始化的 start/end token；`$` 是 delimiter。这里的“task-agnostic”指
共享主干与很薄的输出头，不代表输入完全不做任务感知的整理。
""")
    nb.code("""
START, DELIM, END = "<start>", "<delimiter>", "<classify>"
premise = ["a", "bird", "can", "fly"]
hypothesis = ["an", "animal", "moves"]
entailment = [START, *premise, DELIM, *hypothesis, END]

context = ["mia", "opened", "an", "umbrella"]
question = ["what", "was", "the", "weather"]
answers = [["rainy"], ["sunny"]]
qa_choices = [[START, *context, *question, DELIM, *answer, END] for answer in answers]

print("entailment:", entailment)
print("QA option 1:", qa_choices[0])
print("QA option 2:", qa_choices[1])
assert qa_choices[0][:-2] == qa_choices[1][:-2]
""")
    nb.takeaway(
        "GPT-1 用输入 traversal 把不同任务变成 token 序列，而不是为每个任务重建深层网络。",
        "分类读取最后 token 的 hidden state；causal attention 使它能汇总左侧整段输入。",
    )

    nb.md(r"""
---
## 5. Fine-tuning 时为什么还保留 LM loss？

[论文 Eq. 5，printed p.3](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf#page=3)
最大化 $L_3(C)=L_2(C)+\lambda L_1(C)$；写成最小化的 loss 就是

$$\mathcal{J}=\mathcal{J}_{class}+\lambda\mathcal{J}_{LM}.$$

论文说 auxiliary LM objective 可改善泛化并加速收敛，fine-tuning 默认 $\lambda=0.5$。
但 Table 5 的证据更细：full model 的 8-task unweighted average 是 **74.7**，去掉 aux
反而是 **75.0**；作者据逐任务结果总结它帮助 NLI 与 QQP，整体趋势是大数据集更受益、
小数据集未必受益。因此不能把“aux 总会提高 accuracy”写成结论。
""")
    nb.code("""
torch.manual_seed(2)
demo_classifier = GPTSequenceClassifier(demo, num_classes=2).eval()
class_labels = torch.tensor([1])
demo_output = demo_classifier(demo_ids)
lm_targets = torch.full_like(demo_ids, -100)
lm_targets[:, :-1] = demo_ids[:, 1:]

total, class_loss, lm_loss = joint_transfer_loss(
    demo_output, class_labels, lm_targets, lm_weight=0.5
)
manual = F.cross_entropy(demo_output.class_logits, class_labels) + 0.5 * F.cross_entropy(
    demo_output.lm_logits.reshape(-1, demo_cfg.vocab_size), lm_targets.reshape(-1), ignore_index=-100
)
print(f"class={class_loss:.3f}, LM={lm_loss:.3f}, joint={total:.3f}")
assert torch.allclose(total, manual)
""")
    nb.exercise(
        "把 `lm_weight` 改成 0 与 1。哪一项仍会更新 classifier head？哪一项持续约束 backbone？",
        hint="LM logits 不经过分类头；classification logits 同时依赖 backbone 与 head。",
        starter="""
zero_total, zero_cls, zero_lm = joint_transfer_loss(
    demo_output, class_labels, lm_targets, lm_weight=0.0
)
one_total, one_cls, one_lm = joint_transfer_loss(
    demo_output, class_labels, lm_targets, lm_weight=1.0
)
print("lambda=0:", float(zero_total.detach()), "lambda=1:", float(one_total.detach()))
assert torch.allclose(zero_total, zero_cls)
assert torch.allclose(one_total, one_cls + one_lm)
""",
    )

    nb.md("""
---
## 6. 离线 mini-world：让未标注后果定义词义

每个未标注句子包含一个形容词以及 class-specific reaction：正向词后常见 `smiled /
cheered / applauded`，负向词后常见 `frowned / groaned / left`。预训练只预测下一个 token，
不读取 0/1 标签。下游句子去掉 reaction，只留下 4 个词的少量标签。
""")
    nb.code("""
positive_reactions = ["smiled", "cheered", "applauded"]
negative_reactions = ["frowned", "groaned", "left"]
nouns = ["movie", "film", "story"]
verbs = ["was", "seemed"]

def make_unlabelled(repeats=30, seed=41):
    rng = random.Random(seed)
    rows = []
    for _ in range(repeats):
        for adjective in positive_words:
            rows.append([START, "the", rng.choice(nouns), rng.choice(verbs), adjective,
                         "so", "viewers", rng.choice(positive_reactions), "<end>"])
        for adjective in negative_words:
            rows.append([START, "the", rng.choice(nouns), rng.choice(verbs), adjective,
                         "so", "viewers", rng.choice(negative_reactions), "<end>"])
    rng.shuffle(rows)
    return rows

unlabelled = make_unlabelled(6 if FAST else 40)
all_tokens = sorted({token for row in unlabelled for token in row} | {END})
stoi = {token: i for i, token in enumerate(all_tokens)}
itos = dict(enumerate(all_tokens))

pretrain_ids = torch.tensor([[stoi[t] for t in row] for row in unlabelled])
pretrain_x, pretrain_y = pretrain_ids[:, :-1], pretrain_ids[:, 1:]
print(f"{len(unlabelled)} unlabelled rows, vocab={len(stoi)}, LM shape={tuple(pretrain_x.shape)}")
assert pretrain_x.shape == pretrain_y.shape
assert all(word in stoi for word in held_out_words)
""")

    nb.md("""
---
## 7. Stage 1：只用 next-token loss 预训练

toy 是 2 layers、48 hidden、2 heads，远小于论文模型。每步抽一个 batch；同一个初始化
checkpoint 会被保存，供后续所有条件公平复用。
""")
    nb.code("""
cfg = GPTConfig(vocab_size=len(stoi), block_size=10, n_layer=2, n_head=2,
                d_model=48, d_ff=96, dropout=0.0, tie_weights=True)
torch.manual_seed(41)
pretrained = GPT(cfg)
initial_state = copy.deepcopy(pretrained.state_dict())
optimizer = torch.optim.AdamW(pretrained.parameters(), lr=2e-3, weight_decay=0.01)
pre_steps = 160 if FAST else 320
batch_size = min(64, len(pretrain_x))
generator = torch.Generator().manual_seed(4101)
pre_losses = []
started = time.perf_counter()

with torch.no_grad():
    _, initial_lm_loss = pretrained(pretrain_x, pretrain_y)
for step in range(pre_steps):
    ix = torch.randint(len(pretrain_x), (batch_size,), generator=generator)
    _, loss = pretrained(pretrain_x[ix], pretrain_y[ix])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(pretrained.parameters(), 1.0)
    optimizer.step()
    pre_losses.append(float(loss.detach()))
with torch.no_grad():
    _, final_lm_loss = pretrained(pretrain_x, pretrain_y)
pretrain_seconds = time.perf_counter() - started
pretrained_state = copy.deepcopy(pretrained.state_dict())

print(f"LM loss {initial_lm_loss:.3f} -> {final_lm_loss:.3f} in {pretrain_seconds:.2f}s")
assert final_lm_loss < initial_lm_loss
""")
    nb.code("""
fig, ax = plt.subplots(figsize=(8, 3.5))
smooth_n = min(10, len(pre_losses))
smooth = np.convolve(pre_losses, np.ones(smooth_n) / smooth_n, mode="valid")
ax.plot(np.arange(smooth_n - 1, len(pre_losses)), smooth, color=SERIES[0])
ax.axhline(math.log(len(stoi)), color=SERIES[1], linestyle="--", label="uniform log(V)")
ax.set_xlabel("pre-training step")
ax.set_ylabel("next-token cross entropy")
ax.set_title("Stage 1 uses no sentiment labels")
ax.legend()
plt.tight_layout()
""")
    nb.takeaway(
        "一个长度 T 的 LM 序列同时产生 T 个监督信号；这使原始文本成为可扩展的训练数据。",
        "保存的是 backbone 参数，不是某个 sentiment head；transfer 时才创建新线性层。",
    )

    nb.md("""
---
## 8. Stage 2：相同输入变换，四种 transfer 条件

分类输入固定为 `<start> the noun verb adjective <classify>`，所以最后位置能 causal 地读取
整句。比较：

1. **scratch**：随机 backbone + 监督 labels；对应论文 Table 5 的 `w/o pre-training` 思路。
2. **pretrained + frozen**：只训练 head；这是本课额外诊断，**不是论文 Table 5 行**。
3. **pretrained, λ=0**：全量 fine-tune，不保留 LM objective。
4. **pretrained, λ=0.5**：全量 fine-tune，使用论文默认 auxiliary LM weight。
""")
    nb.code("""
def classification_rows(words, repeats, seed):
    rng = random.Random(seed)
    rows, labels = [], []
    for _ in range(repeats):
        for word in words:
            rows.append([START, "the", rng.choice(nouns), rng.choice(verbs), word, END])
            labels.append(1 if word in positive_words else 0)
    order = list(range(len(rows)))
    rng.shuffle(order)
    x = torch.tensor([[stoi[t] for t in rows[i]] for i in order])
    y = torch.tensor([labels[i] for i in order])
    return x, y

train_x, train_labels = classification_rows(list(labelled_words), 10 if FAST else 30, 42)
test_x, test_labels = classification_rows(held_out_words, 4 if FAST else 12, 43)
print("labelled train:", tuple(train_x.shape), "held-out test:", tuple(test_x.shape))
assert set(train_x[:, -2].tolist()).isdisjoint(test_x[:, -2].tolist())
assert torch.all(train_x[:, -1] == stoi[END])
""")
    nb.code("""
def transfer_run(name, *, use_pretraining, freeze, lm_weight, seed=71):
    torch.manual_seed(seed)
    backbone = GPT(cfg)
    backbone.load_state_dict(pretrained_state if use_pretraining else initial_state)
    if freeze:
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
    model = GPTSequenceClassifier(backbone, num_classes=2)

    trainable_backbone = [p for p in backbone.parameters() if p.requires_grad]
    parameter_groups = [{"params": model.classifier.parameters(), "lr": 2e-3}]
    if trainable_backbone:
        # GPT-1 fine-tuned the pretrained Transformer with a much smaller peak
        # LR than pre-training.  Separate groups make that preservation pressure
        # explicit and avoid erasing the representation in a few toy updates.
        parameter_groups.append({"params": trainable_backbone, "lr": 2e-4})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0.01)
    steps = 140
    generator = torch.Generator().manual_seed(seed + 1000)
    history = []
    for step in range(steps):
        ix = torch.randint(len(train_x), (min(32, len(train_x)),), generator=generator)
        batch = train_x[ix]
        labels = train_labels[ix]
        output = model(batch)
        lm_targets = torch.full_like(batch, -100)
        lm_targets[:, :-1] = batch[:, 1:]
        total, class_loss, lm_loss = joint_transfer_loss(
            output,
            labels,
            lm_targets,
            lm_weight=lm_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        history.append((float(class_loss.detach()), float(lm_loss.detach())))

    model.eval()
    with torch.no_grad():
        train_acc = (model(train_x).class_logits.argmax(-1) == train_labels).float().mean().item()
        test_acc = (model(test_x).class_logits.argmax(-1) == test_labels).float().mean().item()
        _, retained_lm = backbone(pretrain_x, pretrain_y)
    return {"name": name, "train_acc": train_acc, "test_acc": test_acc,
            "lm_loss": float(retained_lm), "history": history}

conditions = [
    ("scratch", False, False, 0.0),
    ("pretrained + frozen", True, True, 0.0),
    ("pretrained, no aux", True, False, 0.0),
    ("pretrained + aux λ=.5", True, False, 0.5),
]
transfer_started = time.perf_counter()
results = [transfer_run(name, use_pretraining=pt, freeze=frozen, lm_weight=weight)
           for name, pt, frozen, weight in conditions]
transfer_seconds = time.perf_counter() - transfer_started

for result in results:
    print(f"{result['name']:>22}: train={result['train_acc']:.1%}  "
          f"held-out={result['test_acc']:.1%}  retained LM={result['lm_loss']:.3f}")
print(f"transfer ablations: {transfer_seconds:.2f}s")
assert all(0.0 <= result["test_acc"] <= 1.0 for result in results)
assert results[1]["test_acc"] > results[0]["test_acc"] + 0.05
assert max(results[2]["test_acc"], results[3]["test_acc"]) > results[0]["test_acc"] + 0.10
assert pretrain_seconds + transfer_seconds < 180
""")
    nb.code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
names = [r["name"] for r in results]
axes[0].barh(names, [r["test_acc"] for r in results], color=SERIES[:4])
axes[0].axvline(0.5, color=SERIES[4], linestyle="--", label="chance")
axes[0].set_xlim(0, 1)
axes[0].set_xlabel("accuracy on test-only adjectives")
axes[0].set_title("Transfer is measured, not assumed")
axes[0].legend()

axes[1].barh(names, [r["lm_loss"] for r in results], color=SERIES[:4])
axes[1].axvline(float(final_lm_loss), color=SERIES[4], linestyle="--", label="after pretrain")
axes[1].set_xlabel("LM loss after fine-tuning (lower is retained)")
axes[1].set_title("Auxiliary LM loss can resist forgetting")
axes[1].legend()
plt.tight_layout()
""")
    nb.exercise(
        "把 labelled adjectives 从 4 个减到 2 个，或把 λ 从 0.5 改成 2。先预测 held-out accuracy 与 retained LM loss 如何变化，再重跑四个条件。",
        hint="更大的 λ 更强调保留生成能力，但可能压过稀少的分类信号；这是真正的 multi-objective trade-off。",
        starter="""
# 例如先只改变一个变量：
# labelled_words = {"bright": 1, "dark": 0}
# conditions[-1] = ("pretrained + aux λ=2", True, False, 2.0)
print("Change one variable, rebuild train_x, then rerun the four conditions.")
""",
    )

    nb.md("""
---
## 9. 论文的 12 个任务与关键结果

论文 Table 1 的 **12 datasets** 是：5 个 NLI（SNLI、MultiNLI、QNLI、RTE、SciTail），
2 个 QA/reasoning（RACE、Story Cloze），3 个 similarity（MRPC、QQP、STS-B），2 个
classification（SST-2、CoLA）。MNLI matched/mismatched 是同一 dataset 的两个 split，
不能重复计数。官方研究页还展示 COPA，但它不在论文 Table 1 的这 12 个数据集中。

[论文 Tables 2–4，printed pp.5–6](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf#page=6)
中 `Finetuned Transformer LM (ours)` 的关键数字：

| family | dataset / metric | ours |
|---|---|---:|
| NLI | MNLI-m / MNLI-mm accuracy | 82.1 / 81.4 |
| NLI | SNLI / SciTail / QNLI / RTE accuracy | 89.9 / 88.3 / 88.1 / 56.0 |
| QA | Story Cloze / RACE overall accuracy | 86.5 / 59.0 |
| similarity | MRPC F1 / STS-B Pearson / QQP F1 | 82.3 / 82.0 / 70.3 |
| classification | CoLA Matthews / SST-2 accuracy | 45.4 / 91.3 |
| aggregate | GLUE | 72.8 |

论文总结 12 个数据集里 9 个达到当时 SOTA；失败也应保留：RTE 56.0 低于 61.7，MRPC
82.3 低于 86.0，SST-2 91.3 低于 93.2。Story Cloze 相对先前 77.6 提高 8.9 points，
RACE 相对 53.3 提高 5.7 points。
""")

    nb.md("""
---
## 10. Table 5：pre-training 的证据强，aux 的证据有条件

[论文 Table 5，printed p.7 / PDF p.8](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf#page=8)
在 8 个任务上的 ablation：

| condition | avg | CoLA | SST2 | MRPC | STSB | QQP | MNLI | QNLI | RTE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Transformer + aux LM (full) | 74.7 | 45.4 | 91.3 | 82.3 | 82.0 | 70.3 | 81.8 | 88.1 | 56.0 |
| without pre-training | 59.9 | 18.9 | 84.0 | 79.4 | 30.9 | 65.5 | 75.7 | 71.2 | 53.8 |
| without aux LM | 75.0 | 47.9 | 92.0 | 84.9 | 83.2 | 69.8 | 81.1 | 86.9 | 54.4 |
| LSTM + aux LM | 69.1 | 30.3 | 90.5 | 83.2 | 71.8 | 68.1 | 73.7 | 81.1 | 54.6 |

- 去掉 pre-training 平均下降 **14.8 points**，且每个列出的任务都下降。
- Transformer 换成单层 2048-unit LSTM 平均下降 **5.6 points**。
- aux LM 对 MNLI、QNLI、RTE、QQP 有帮助，但没有提高这张表的总体平均。
- Figure 2 逐层迁移实验显示，MultiNLI 完整迁移相对只迁移较少层最多多约 9 points。

我们的 synthetic held-out accuracy、frozen probe 与 forgetting 曲线回答更窄的问题，不能与
这些 benchmark 数字并排当作复现结果。
""")
    nb.takeaway(
        "GPT-1 的突破是协议：generative pre-training → task-aware token transformation → discriminative fine-tuning。",
        "主干是单向、masked-self-attention 的 decoder；分类读取最后位置，因为它能看到此前所有 token。",
        "论文对 pre-training 的 ablation 很强；对 auxiliary LM 的结果依任务规模而异，不支持“λ>0 总更好”。",
        "本课只复现迁移机制；BooksCorpus、BPE-40k、12×768 模型和 12 个真实 benchmark 均未复现。",
    )

    nb.save("notebooks/05_gpt/01_gpt1_pretraining.ipynb")


if __name__ == "__main__":
    build()

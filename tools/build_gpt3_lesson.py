"""Generate the GPT-3 in-context-learning paper lesson.

The lesson separates Brown et al.'s reported GPT-3 evidence from a small,
offline mechanism experiment.  It is a generator rather than hand-edited
notebook JSON, so the source stays reviewable and the emitted cells stay clean.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.nbuild import Notebook


def build() -> None:
    nb = Notebook(
        "GPT-3：学习发生在参数里，还是上下文里？",
        subtitle=r"""
**核心问题：** 不做梯度更新，只在 prompt 中多给几个例子，模型为什么可能学会当前
任务？

我们会训练一个 tiny GPT 跨许多随机的 modular-shift 任务，然后在**全程固定参数**的
条件下比较 0/1/few-shot。正确示范、错误示范和参数快照把“看起来会做”变成可证伪的
in-context learning 实验。

**主要来源：** Brown et al., [*Language Models are Few-Shot
Learners*](https://arxiv.org/abs/2005.14165), 2020；
[原论文 PDF](https://arxiv.org/pdf/2005.14165)；
[OpenAI 官方研究页](https://openai.com/index/language-models-are-few-shot-learners/)。

**前置知识：** decoder-only GPT、causal attention、next-token prediction。
""",
    )

    nb.md("""
---
## 0. 设置：离线、CPU、固定 seed

这里不下载 GPT-3 或数据。`LEARNAI_FAST=1` 仍走完训练、shots curve 与消融，只把
训练/评估 episode 数缩小。
""")
    nb.code("""
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from learnai.data.gpt_tasks import make_modular_shift_batch, modular_shift_accuracy
from learnai.env import setup
from learnai.nn import GPT, GPTConfig
from learnai.viz import SERIES, use_style

lesson_started = time.perf_counter()
ctx = setup(seed=2020, device="cpu")
torch.set_num_threads(1)
torch.set_printoptions(precision=3, sci_mode=False, linewidth=120)
use_style()
FAST = os.environ.get("LEARNAI_FAST", "0") == "1"
print("fast smoke path:", FAST)
""")

    nb.md(r"""
---
## 1. 三种 setting 的区别只在 prompt，不在 optimizer

论文 §2 / Figure 2.1 给出严格定义：

| setting | inference-time demonstrations | 还给什么 | test-time 梯度更新 |
|---|---:|---|---:|
| zero-shot | 0 | 自然语言任务说明 | 0 |
| one-shot | 1 | 自然语言任务说明 | 0 |
| few-shot | $K$ 个 | 任务说明可选/依任务而定 | 0 |
| fine-tuning | 通常数千到数十万训练例 | task dataset | **有** |

论文的 few-shot 是 $K$ 个完整的“context → completion”，再给 query context 让模型
补 completion；通常 $K=10\ldots100$，受 2048-token context 限制。Figure 2.1 明确说
0/1/few-shot 在测试时只有 forward pass。论文所谓 “learning curve” 因而是改变 prompt，
不是在每个测试任务上偷偷跑 optimizer。
""")
    nb.code("""
settings = {"zero-shot": 0, "one-shot": 1, "few-shot": 8}
for name, shots in settings.items():
    print(f"{name:>9}: {shots} demonstration(s), gradient updates = 0")
assert settings["zero-shot"] == 0
assert settings["one-shot"] == 1
assert settings["few-shot"] > 1
""")
    nb.takeaway(
        "shot 数是当前输入里可见的 demonstrations 数，不是 fine-tuning 样本数。",
        "in-context learning 发生在一次 forward pass 的激活中；模型参数在任务间不变。",
    )

    nb.md(r"""
---
## 2. 规模事实：175B、300B 与 2048 是三个不同量

最常见的误读是把它们混为一谈。原论文 Table 2.1 报告：最大的 GPT-3 有
**175.0B parameters**、96 层、$d_{model}=12{,}288$、96 heads；八个模型都训练
**300 billion tokens**；所有模型的最大上下文是 **2048 tokens**。

- 175B：模型参数容量；
- 300B：一次预训练过程消耗的 token 数，不是 context length，也不是数据集唯一 token 数；
- 2048：一次 forward 能条件化的最大序列长度，demonstrations 与 query 共享它。

Appendix D 的估算是 GPT-3 175B 训练约 **3.14e23 FLOPs / 3640 PF-days**；估算忽略
attention（作者称在所分析模型中通常低于总 compute 的 10%）。
""")
    nb.code("""
paper_scale = {
    "parameters": 175_000_000_000,
    "training_tokens": 300_000_000_000,
    "context_tokens": 2_048,
    "layers": 96,
    "d_model": 12_288,
    "heads": 96,
}
for key, value in paper_scale.items():
    print(f"{key:>16}: {value:,}")
assert paper_scale["training_tokens"] != paper_scale["context_tokens"]
""")

    nb.md(r"""
论文训练集混合（Table 2.2）也不是“300B 个互不重复 token”：filtered Common Crawl
410B tokens、WebText2 19B、Books1 12B、Books2 55B、Wikipedia 3B；采样权重不是按
语料大小比例，所以 Wikipedia 约 3.4 epochs，而 Common Crawl 约 0.44 epochs。
""")
    nb.takeaway(
        "参数量、总训练 token 和单次 context window 描述三个不同轴。",
        "few-shot 的 K 被 context budget 限制；例子更长，就只能放更少 demonstrations。",
    )

    nb.md(r"""
---
## 3. 一个能迫使模型读示范的任务：随机 modular shift episode

每个 episode 随机选一个 shift $s$，任务是

$$f_s(x)=(x+s)\bmod M.$$

prompt 形如 `x₁ y₁ x₂ y₂ ... query ?`。关键是每个 batch/episode 的 $s$ 都重新抽样；
只背一个固定映射无效。为了让 laptop 规模实验稳定，`query` 会重复某个 demonstration
里的 $x_i$：模型可通过“找到相同 key，再复制其 value”解决，而不必学会模加法。

这正是一个可控的 **induction / associative-recall** 显微镜。它证明 context 中的 pair
能改变固定权重模型的答案，但不证明模型推导了 $s$ 或掌握抽象 arithmetic rule；若把
query 换成未展示的 $x$，才会检验后者。

这只是 **mechanism-level analogue**：它隔离“固定权重如何利用当前 context”，不声称
tiny GPT 能复现自然语言能力或 GPT-3 的规模现象。

还有一个术语边界：这个符号任务没有自然语言 instruction token，因此下文曲线里的
“0-shot”只是操作性的 **K=0 demonstration baseline**；它不是论文严格定义的“仅给自然
语言任务说明”的 zero-shot。我们比较的是 context 中示范数量的因果作用，不比较两者
的绝对任务难度。
""")
    nb.code("""
MODULUS = 7
TRAIN_SHOTS = 4
MAX_SHOTS = 8
BATCH_SIZE = 32 if FAST else 64

demo_inputs, demo_targets = make_modular_shift_batch(
    batch_size=3,
    shots=TRAIN_SHOTS,
    modulus=MODULUS,
)
print("input shape:", tuple(demo_inputs.shape), "target shape:", tuple(demo_targets.shape))
print("first episode input :", demo_inputs[0].tolist())
print("first episode target:", demo_targets[0].tolist())

assert demo_inputs.ndim == demo_targets.ndim == 2
assert demo_inputs.shape == demo_targets.shape
assert demo_inputs.size(1) == 4 * TRAIN_SHOTS + 3
""")
    nb.exercise(
        "令 `shots_for_length` 为 0、1、4、8，画出或打印序列长度。为什么真实 GPT-3 的 few-shot K 不是固定常数？",
        hint="demonstration 数与每个例子的 token 长度共同消费有限 context window。",
        starter="""
shots_for_length = 4  # 改这里
x_probe, y_probe = make_modular_shift_batch(
    batch_size=2, shots=shots_for_length, modulus=MODULUS
)
print("shots:", shots_for_length, "sequence length:", x_probe.size(1))
assert x_probe.shape == y_probe.shape
""",
    )
    nb.takeaway(
        "跨随机 task 训练排除了只记住一个全局 shift 的捷径。",
        "query 重复已展示 key，所以成功的最小解释是 context lookup，不是模加法推理。",
        "demonstrations 是同一 causal 序列里的普通 token；attention 是传递 task 信息的通道。",
    )

    nb.md(r"""
---
## 4. Outer loop：训练一个 tiny meta-language-model

GPT-3 的 outer loop 是互联网文本 next-token prediction；这里把它缩成许多随机 shift
episodes。loss 只放在最后的 query answer；demonstration answers 是条件，不是假装可从
零信息预测的 targets。**训练阶段当然会更新权重**，因为我们必须先让 tiny model 获得使用示范的
能力。真正的检查是在下一节：训练完成后冻结快照，在新 episode 上只改变 prompt。
""")
    nb.code("""
config = GPTConfig(
    vocab_size=MODULUS + 3,
    block_size=4 * MAX_SHOTS + 3,
    n_layer=2,
    n_head=4,
    d_model=64,
    d_ff=128,
    dropout=0.0,
    tie_weights=False,
)
model = GPT(config).to(ctx.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
train_steps = 400 if FAST else 1200
loss_history = []

def answer_loss(logits, targets):
    # gpt_tasks 用负值标出不评分的位置；只在最终 query answer 上学习。
    selected = targets >= 0
    assert bool(selected.any())
    return F.cross_entropy(logits[selected], targets[selected])

model.train()
training_shots = [1, 2, 4, 8]
for step in range(train_steps):
    # Learned position embeddings beyond a fixed four-shot prompt would never
    # be trained.  Vary K in the outer loop so every evaluated context length is
    # represented while the hidden shift still changes on every episode.
    step_shots = training_shots[step % len(training_shots)]
    inputs, targets = make_modular_shift_batch(
        batch_size=BATCH_SIZE,
        shots=step_shots,
        modulus=MODULUS,
    )
    inputs, targets = inputs.to(ctx.device), targets.to(ctx.device)
    logits, _ = model(inputs)
    loss = answer_loss(logits, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    loss_history.append(float(loss.detach()))

print(f"parameters: {model.num_params():,}")
print(f"loss: {loss_history[0]:.3f} -> {loss_history[-1]:.3f} ({train_steps} steps)")
assert loss_history[-1] < loss_history[0]
""")
    nb.code("""
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(loss_history, color=SERIES[0], linewidth=1.2)
ax.set(xlabel="outer-loop optimizer step", ylabel="answer cross-entropy",
       title="Across-task training teaches the model how to use demonstrations")
plt.tight_layout()
""")
    nb.takeaway(
        "outer loop 更新共享参数，让模型学会一种跨 task 的推断策略。",
        "不能把 outer-loop 训练和 test-time ICL 混淆；后者必须用冻结参数证明。",
    )

    nb.md(r"""
---
## 5. 0/1/k-shot curve：改变 context，参数一个 bit 都不变

下面先 clone 全部参数，再用新随机 episode 测 $K=0,1,2,4,8$。评估函数只做
forward/argmax；循环后逐元素检查 state dict。若从 K=0 到“有示范”发生跃升而参数不动，
改变只能来自输入 context 及其激活。更多示范也会带来检索干扰，所以曲线不必单调。
""")
    nb.code("""
model.eval()
parameter_snapshot = {
    name: parameter.detach().cpu().clone()
    for name, parameter in model.named_parameters()
}
shots = [0, 1, 2, 4, 8]
eval_episodes = 256 if FAST else 1024
accuracies = [
    modular_shift_accuracy(
        model,
        shots=k,
        episodes=eval_episodes,
        modulus=MODULUS,
        device=ctx.device,
        generator=torch.Generator().manual_seed(20_200 + k),
    )
    for k in shots
]

for k, accuracy in zip(shots, accuracies):
    print(f"{k:>2}-shot accuracy: {accuracy:.1%}")

for name, parameter in model.named_parameters():
    assert torch.equal(parameter.detach().cpu(), parameter_snapshot[name])
assert accuracies[1] > accuracies[0] + 0.40
assert max(accuracies[2:]) > accuracies[0] + 0.20
""")
    nb.code("""
fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.plot(shots, accuracies, "o-", color=SERIES[1])
ax.axhline(1 / MODULUS, color=SERIES[7], linestyle="--", label="chance")
ax.set(xlabel="demonstrations in context", ylabel="accuracy", ylim=(0, 1.03),
       title="An in-context learning curve with frozen parameters")
ax.legend()
plt.tight_layout()
""")
    nb.exercise(
        "在运行前预测：把 `probe_shots` 从 0 改成 1、4、8，准确率应怎样变？它一定单调吗？",
        hint="更多例子通常有帮助，但有限模型、采样噪声与 context 干扰都可能造成局部不单调；论文也明确说 K 越大通常但不总是越好。",
        starter="""
probe_shots = 4
probe_accuracy = modular_shift_accuracy(
    model, shots=probe_shots, episodes=(64 if FAST else 512),
    modulus=MODULUS, device=ctx.device
)
print(probe_shots, "shot:", f"{probe_accuracy:.1%}")
assert 0.0 <= probe_accuracy <= 1.0
""",
    )
    nb.takeaway(
        "参数逐元素未变 + 有示范时 accuracy 跃升，是固定权重 ICL 的直接机制证据。",
        "这个 lookup toy 中 K=1 最容易；更多 key 带来干扰，所以 examples 增多不保证单调。",
    )

    nb.md(r"""
---
## 6. Wrong-demo / shuffled-label 消融：模型真的在读 pair 吗？

仅有 shots curve 仍可能存在捷径：也许模型只利用 query 的分布，或者只数了有几个 token。
更强的干预是保持输入数量/格式不变，给每个 demonstration label 加一个随机非零模偏移，
保证它变错，同时保持合法 token 范围与总体边际分布。

- 若模型忽略 demonstrations，正确与打乱 labels 应差不多；
- 若模型从 pair 推断 shift，错误 labels 应显著伤害 query accuracy；
- 这仍不证明模型用了与人相同的算法，只排除一类明显替代解释。
""")
    nb.code("""
k_ablate = 1
correct_demo_accuracy = modular_shift_accuracy(
    model,
    shots=k_ablate,
    episodes=eval_episodes,
    modulus=MODULUS,
    device=ctx.device,
    shuffle_labels=False,
    generator=torch.Generator().manual_seed(20_208),
)
wrong_demo_accuracy = modular_shift_accuracy(
    model,
    shots=k_ablate,
    episodes=eval_episodes,
    modulus=MODULUS,
    device=ctx.device,
    shuffle_labels=True,
    generator=torch.Generator().manual_seed(20_208),
)

print("correct demonstrations :", f"{correct_demo_accuracy:.1%}")
print("shuffled demo labels   :", f"{wrong_demo_accuracy:.1%}")
print("causal effect          :", f"{correct_demo_accuracy - wrong_demo_accuracy:+.1%}")

assert correct_demo_accuracy > wrong_demo_accuracy + 0.40
for name, parameter in model.named_parameters():
    assert torch.equal(parameter.detach().cpu(), parameter_snapshot[name])
""")
    nb.exercise(
        "设计一个更强对照：保持 label 合法且一致，但让 demonstrations 来自另一个 shift。你预测模型会跟示范 shift、真实 query shift，还是退回 chance？为什么该实验比随机 token 更有信息量？",
        hint="好的消融只改变被检验的因果信息，同时尽量保持长度、格式和 token 分布。",
        starter="""
# 用现有 shuffle_labels=True 作为起点；再阅读 gpt_tasks.py，尝试增加 coherent_wrong_shift。
baseline = wrong_demo_accuracy
print("current shuffled-label baseline:", f"{baseline:.1%}")
assert 0.0 <= baseline <= 1.0
""",
    )
    nb.takeaway(
        "同长度错误示范让性能下降，说明提升来自 demonstration 内容，不只是 prompt 变长。",
        "消融排除特定捷径，不自动揭示 Transformer 内部使用的完整算法。",
    )

    nb.md(r"""
---
## 7. 论文的关键结果：scale 有帮助，但 few-shot 并非处处赢

论文训练 8 个模型（125M → 175B），Figure 3.1 报告 validation loss 随训练 compute
近似 smooth power law；Figure 1.3 的 42 个 accuracy benchmarks 聚合显示 few-shot 随
模型大小上升得比 zero-shot 更快。作者谨慎写作“larger models **perhaps** are more
proficient meta-learners”，不是宣称只靠 scale 已解决通用学习。

几个必须成对阅读的原始数字：

| benchmark | zero-shot | one-shot | few-shot | 说明 |
|---|---:|---:|---:|---|
| LAMBADA accuracy | 76.2 | 72.5 | 86.4 | few-shot 强，但 one-shot 反而低于 zero-shot |
| TriviaQA accuracy | 64.3 | 68.0 | 71.2 | few-shot 超过同为 closed-book 的 fine-tuned baselines |
| WebQuestions accuracy | 14.4 | 25.3 | 41.5 | demonstrations 帮助很大，仍低于 44.7 baseline |
| Natural Questions accuracy | 14.6 | 23.0 | 29.9 | 低于 36.6 fine-tuned baseline |
| Winogrande accuracy | 70.2 | 73.2 | 77.7 | 人类报告值 94.0 |
| ARC-Challenge accuracy | 51.4 | 53.2 | 51.5 | few-shot 没有优于 one-shot |

SuperGLUE few-shot 用 32 examples/task、**无梯度更新**，平均 71.8；fine-tuned
BERT-Large 69.0，fine-tuned SOTA 89.0。WiC few-shot 49.4，约为 chance。论文的结论
是广泛但混合的结果，不是“few-shot 总比 zero-shot/one-shot 好”。
""")
    nb.code("""
paper_results = {
    "LAMBADA": [76.2, 72.5, 86.4],
    "TriviaQA": [64.3, 68.0, 71.2],
    "WebQuestions": [14.4, 25.3, 41.5],
    "NaturalQuestions": [14.6, 23.0, 29.9],
    "Winogrande": [70.2, 73.2, 77.7],
    "ARC-Challenge": [51.4, 53.2, 51.5],
}
print(f"{'task':>18} {'0-shot':>8} {'1-shot':>8} {'few-shot':>9}")
for task, values in paper_results.items():
    print(f"{task:>18} {values[0]:>8.1f} {values[1]:>8.1f} {values[2]:>9.1f}")

assert paper_results["LAMBADA"][1] < paper_results["LAMBADA"][0]
assert paper_results["ARC-Challenge"][2] < paper_results["ARC-Challenge"][1]
""")
    nb.takeaway(
        "scale 与平均 ICL 能力一起增长，但不同 benchmark 的 shot 曲线可能不单调。",
        "报告强结果时必须同时保留弱结果、baseline 的训练设定和 contamination 标记。",
    )

    nb.md(r"""
---
## 8. 局限与证据边界

论文 §4–6 自己列出的边界很重要：

1. **Contamination。** 训练集过滤有 bug，昂贵到无法重训；作者以 overlap/clean subset
   分析，给 PIQA 与 Winograd 加星号，并因几乎完全污染而不报告多项 Wikipedia LM
   benchmarks。LAMBADA 也有大量真实 overlap，clean subset 与 full 相差不到 0.5%。
2. **能力不均。** 长文会重复、失去连贯、矛盾或 non sequitur；common-sense physics、
   WiC/ANLI、一些 reading comprehension 较弱。
3. **目标/架构限制。** 仅自回归 next-token objective，没有 bidirectional/denoising；
   每个 token 等权且缺少 grounding，纯 scale 可能触及目标函数上限。
4. **“学习”含义未决。** §5 明说不能确定模型是在 inference 时从零学新任务，还是识别
   pretraining 中已有任务；不同任务可能位于这个连续谱不同位置。
5. **成本与风险。** 175B 推理昂贵不便；预训练样本效率低；输出不易解释/校准，继承
   web 数据的 bias，并带来误用、合成文本识别和能源问题。

本 notebook **只证明** tiny GPT 在训练分布内能以固定参数利用随机 shift demonstrations，
且错误 demonstrations 造成因果下降。它没有复现 175B scaling、自然语言 few-shot、
300B-token 预训练、真实 benchmark 或涌现；也不能由一个 seed 推断普适可靠性。
Toy 曲线的 K=0 也没有自然语言 instruction，因而不是论文严格意义的 zero-shot。
""")
    nb.code("""
elapsed = time.perf_counter() - lesson_started
print(f"mode={'FAST' if FAST else 'default'}, device=cpu, seed={ctx.seed}, elapsed={elapsed:.1f}s")
print("证据链：加入示范且参数不变时性能跃升；labels 打乱则性能下降；结论只覆盖这个 toy 分布。")
assert ctx.seed == 2020
assert elapsed < 180
""")
    nb.takeaway(
        "ICL 的最低证据标准：参数不变、context 改变、表现改变，并用内容匹配的消融排除捷径。",
        "GPT-3 论文展示的是 scale 下广泛但不均衡的 few-shot 行为，不是通用可靠性的证明。",
        "下一步应扫多个 seed、task family 与 model size，再问策略如何被 attention/MLP 实现。",
    )

    nb.save("notebooks/05_gpt/03_gpt3_in_context_learning.ipynb")


if __name__ == "__main__":
    build()

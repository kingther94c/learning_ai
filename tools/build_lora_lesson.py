"""Generate the LoRA paper lesson.

The committed notebook is derived from this reviewable script and stays free of
outputs.  The lesson uses only local synthetic data and the tested
``learnai.nn.LoRALinear`` implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.nbuild import Notebook


def build() -> None:
    nb = Notebook(
        "LoRA：只训练一个低秩的变化",
        subtitle=r"""
**核心问题：** 一个大模型有数十亿个权重，为什么下游任务有时只需学习一个很小的
低秩更新？

这节课从论文公式 (3) 出发，在 CPU 上造一个**真实秩为 3** 的迁移任务：冻结
$W_0$，只训练 $A,B$，固定同一个 $\alpha$ 扫描 rank，再用 SVD、checkpoint 和
merge 检验每个关键主张。

**主要来源：** Hu et al., [*LoRA: Low-Rank Adaptation of Large Language
Models*](https://openreview.net/forum?id=nZeVKeeFYf9), ICLR 2022；
[arXiv 正文](https://arxiv.org/html/2106.09685)；
[作者的官方 `loralib`](https://github.com/microsoft/LoRA)。

**前置知识：** 线性层、矩阵秩/SVD、反向传播；看过 Transformer 的 Q/K/V 会更有帮助。
""",
    )

    nb.md(r"""
---
## 0. 设置：一个完全离线、可重复的实验

不下载模型或数据。固定 seed、固定 CPU，也把线程数设为 1，使结果和运行时间更稳定。
`LEARNAI_FAST=1` 仍会走完相同实验，只减少优化步数。
""")
    nb.code("""
import io
import os
import time
from unittest.mock import patch

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from learnai.env import setup
from learnai.nn import LoRALinear, low_rank_approximation, relative_frobenius_error
from learnai.viz import SERIES, use_style

lesson_started = time.perf_counter()
ctx = setup(seed=2022, device="cpu")
torch.set_num_threads(1)
torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)
use_style()
FAST = os.environ.get("LEARNAI_FAST", "0") == "1"
print("fast smoke path:", FAST)
""")

    nb.md(r"""
---
## 1. 惊讶从这里开始：目标变了，但只沿三个方向变

先把“预训练模型”缩成一个 $16\times16$ 线性层。新任务的正确权重是

$$W_* = W_0 + \Delta W_*, \qquad \operatorname{rank}(\Delta W_*)=3.$$

我们用 QR 构造三个正交方向，奇异值固定为 $3,1.5,0.5$。这不是声称真实语言任务
一定秩为 3；它是一个可证伪的显微镜：rank 不够时应该留下误差，rank 达到 3 后才有
能力精确表示目标。
""")
    nb.code("""
d = 16
true_rank = 3

left, _ = torch.linalg.qr(torch.randn(d, true_rank), mode="reduced")
right, _ = torch.linalg.qr(torch.randn(d, true_rank), mode="reduced")
true_singular_values = torch.tensor([3.0, 1.5, 0.5])
delta_star = left @ torch.diag(true_singular_values) @ right.T

pretrained = nn.Linear(d, d, bias=False)
w0 = pretrained.weight.detach().clone()
x_train = torch.randn(512, d)
x_test = torch.randn(256, d)
y_train = F.linear(x_train, w0 + delta_star).detach()
y_test = F.linear(x_test, w0 + delta_star).detach()

base_mse = F.mse_loss(pretrained(x_test), y_test).item()
measured_rank = int(torch.linalg.matrix_rank(delta_star))
print("W0 shape:", tuple(w0.shape), "target delta rank:", measured_rank)
print("冻结 W0、不适配时的 test MSE:", f"{base_mse:.4f}")
assert measured_rank == true_rank
assert base_mse > 0.1
""")
    nb.takeaway(
        "LoRA 约束的是任务适配产生的变化 ΔW，不是声称预训练权重 W0 本身低秩。",
        "这个 toy task 把真实秩设为 3，因此 rank sweep 有一个事先知道的能力阈值。",
    )

    nb.md(r"""
---
## 2. 论文公式 (3)：$\Delta W=BA$

对 $W_0\in\mathbb{R}^{d_{out}\times d_{in}}$，LoRA 冻结 $W_0$，学习

$$
A\in\mathbb{R}^{r\times d_{in}},\quad
B\in\mathbb{R}^{d_{out}\times r},\quad
h=W_0x+\frac{\alpha}{r}BAx.
$$

所以一个线性层的可训练参数从 $d_{out}d_{in}$ 变为
$r(d_{in}+d_{out})$，且 $\operatorname{rank}(BA)\le r$。

论文 §4.1 用随机 Gaussian 初始化 $A$、把 $B$ 初始化为零；于是刚插入 LoRA 时
$BA=0$，预训练函数完全不变。缩放是 $\alpha/r$，论文还明确说明：比较不同 $r$ 时
$\alpha$ 保持常数。下面所有 rank 都固定 `alpha=3`。
""")
    nb.code("""
rank = 3
alpha = 3.0
lora = LoRALinear.from_linear(pretrained, rank=rank, alpha=alpha, dropout=0.0)

print("A:", tuple(lora.lora_A.shape), "B:", tuple(lora.lora_B.shape))
print("BA:", tuple((lora.lora_B @ lora.lora_A).shape))
print("scaling = alpha/r =", lora.scaling)
print("可训练参数:", lora.trainable_parameter_count, "vs full:", d * d)

assert lora.lora_A.shape == (rank, d)
assert lora.lora_B.shape == (d, rank)
assert lora.trainable_parameter_count == rank * (d + d)
assert torch.count_nonzero(lora.delta_weight()) == 0
assert torch.allclose(lora(x_test), pretrained(x_test), atol=0, rtol=0)
assert not lora.base.weight.requires_grad
assert lora.lora_A.requires_grad and lora.lora_B.requires_grad
""")
    nb.code("""
# 把公式逐项写开，并检查 library 路径没有藏别的计算。
with torch.no_grad():
    lora.lora_B.normal_(std=0.1)
manual = F.linear(x_test, w0) + (x_test @ lora.lora_A.T @ lora.lora_B.T) * (alpha / rank)
assert torch.allclose(lora(x_test), manual, atol=1e-6)
assert int(torch.linalg.matrix_rank(lora.delta_weight(), atol=1e-5)) <= rank
print("显式 W0x + (alpha/r)BAx 与 LoRALinear 一致。")
""")
    nb.exercise(
        "把 `r_try` 改成 1、2、4、8。先心算 LoRA 参数量，再运行；什么时候它反而不比完整矩阵省？",
        hint="比较 r(d_in+d_out) 与 d_in*d_out。方阵时分界大约是 r=d/2。",
        starter="""
r_try = 2  # 改这里
full_parameters = d * d
lora_parameters = r_try * (d + d)
print({"rank": r_try, "LoRA": lora_parameters, "full": full_parameters,
       "saving_factor": full_parameters / lora_parameters})
assert lora_parameters == r_try * (2 * d)
""",
    )
    nb.takeaway(
        "A 先把输入压到 r 维，B 再升回输出维；BA 的秩不可能超过 r。",
        "B=0 让 LoRA 插入时成为严格 no-op，W0 则始终冻结。",
        "rank sweep 必须固定同一个 alpha，才遵循原论文的比较设定。",
    )

    nb.md(r"""
---
## 3. 训练 rank=3：确认真正没动的是 $W_0$

先只训练已知足够的 rank=3。除了看 loss，还要保存训练前的 $W_0$ 并逐元素比较；
“optimizer 里没放它”不如直接测量可信。
""")
    nb.code("""
torch.manual_seed(103)
rank3_model = LoRALinear.from_linear(pretrained, rank=3, alpha=alpha, dropout=0.0)
w0_before = rank3_model.base.weight.detach().clone()
optimizer = torch.optim.Adam([rank3_model.lora_A, rank3_model.lora_B], lr=0.04)
rank3_steps = 60 if FAST else 900
rank3_history = []

for step in range(rank3_steps):
    optimizer.zero_grad()
    loss = F.mse_loss(rank3_model(x_train), y_train)
    loss.backward()
    optimizer.step()
    rank3_history.append(float(loss.detach()))

rank3_test_mse = F.mse_loss(rank3_model(x_test), y_test).item()
print(f"loss: {rank3_history[0]:.4f} -> {rank3_history[-1]:.6g}")
print("test MSE:", f"{rank3_test_mse:.6g}")
print("W0 最大变化:", float((rank3_model.base.weight - w0_before).abs().max()))

assert torch.equal(rank3_model.base.weight, w0_before)
assert rank3_history[-1] < rank3_history[0] * 0.02
assert rank3_test_mse < 0.02
""")
    nb.takeaway(
        "适配成功来自 A、B 学到的增量；冻结的 W0 在训练前后逐元素相同。",
        "参数少不等于优化一定更快；BA 是因子化参数化，能力与有限步训练要分开看。",
    )

    nb.md(r"""
---
## 4. 固定 $\alpha$ 扫 rank：能力阈值与优化结果分开看

先用截断 SVD 回答“rank-r **最多能表示多好**”，再用 Adam 回答“有限训练步里
**实际学到多好**”。把这两个问题混在一起，很容易把优化失败误判为 rank 不够。
""")
    nb.code("""
ranks = [1, 2, 3, 6]
svd_relative_errors = []
for r in ranks:
    approximation, singular_values = low_rank_approximation(delta_star, rank=r)
    svd_relative_errors.append(float(relative_frobenius_error(delta_star, approximation)))

print("target singular values:", singular_values[:6])
for r, error in zip(ranks, svd_relative_errors):
    print(f"best possible rank-{r} relative error = {error:.6f}")

assert svd_relative_errors[1] > 0.1       # rank 2 必须丢掉第三个方向
assert svd_relative_errors[2] < 1e-5      # rank 3 可精确表达 target
""")
    nb.code("""
def fit_lora(rank, steps):
    # 每个 rank 有固定、可重复的初始化；alpha 对所有 rank 都是同一个 3.0。
    torch.manual_seed(100 + rank)
    model = LoRALinear.from_linear(pretrained, rank=rank, alpha=alpha, dropout=0.0)
    opt = torch.optim.Adam([model.lora_A, model.lora_B], lr=0.04)
    history = []
    for _ in range(steps):
        opt.zero_grad()
        loss = F.mse_loss(model(x_train), y_train)
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
    test_mse = F.mse_loss(model(x_test), y_test).item()
    delta_error = float(relative_frobenius_error(delta_star, model.delta_weight()))
    return {"rank": rank, "model": model, "history": history,
            "test_mse": test_mse, "delta_error": delta_error}

sweep_steps = 60 if FAST else 900
rank_results = [fit_lora(r, sweep_steps) for r in ranks]

print(f"固定 alpha={alpha}；每个模型训练 {sweep_steps} 步")
print(f"{'rank':>5} {'params':>8} {'test MSE':>12} {'delta rel.err':>15}")
for result in rank_results:
    r = result["rank"]
    print(f"{r:>5} {r*(2*d):>8} {result['test_mse']:>12.6g} {result['delta_error']:>15.6f}")

test_by_rank = {result["rank"]: result["test_mse"] for result in rank_results}
assert test_by_rank[1] > test_by_rank[2] > test_by_rank[3]
assert test_by_rank[6] < test_by_rank[3]
""")
    nb.code("""
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].plot(ranks, svd_relative_errors, "o-", color=SERIES[0])
axes[0].axvline(true_rank, color=SERIES[3], linestyle="--", label="target rank = 3")
axes[0].set(yscale="log", xlabel="LoRA rank", ylabel="best relative error",
            title="表示能力：截断 SVD 的下界")
axes[0].legend()

axes[1].plot(ranks, [r["test_mse"] for r in rank_results], "o-", color=SERIES[1])
axes[1].axvline(true_rank, color=SERIES[3], linestyle="--")
axes[1].set(yscale="log", xlabel="LoRA rank", ylabel="test MSE",
            title="优化结果：同 alpha、同训练步数")
plt.tight_layout()
""")
    nb.exercise(
        "把 `probe_rank` 从 2 改成 3。SVD 误差为什么突然接近 0？再把最小奇异值 0.5 想象成 0.001，有限数据与噪声会怎样改变你选 rank 的判断？",
        hint="rank 是硬容量；奇异值大小决定丢掉一个方向的代价。真实任务没有写在答案上的 true_rank。",
        starter="""
probe_rank = 2  # 改成 3 再运行
probe, spectrum = low_rank_approximation(delta_star, probe_rank)
probe_error = float(relative_frobenius_error(delta_star, probe))
print("rank", probe_rank, "keeps", spectrum[:probe_rank].tolist(), "relative error", probe_error)
assert 0.0 <= probe_error <= 1.0
""",
    )
    nb.takeaway(
        "在这个受控任务中，rank<3 有不可消除的表示误差；rank≥3 才有精确表示能力。",
        "更大 rank 不保证真实任务更好：它增加参数，也可能学习噪声；rank 是要验证的超参数。",
    )

    nb.md(r"""
---
## 5. 原论文到底看到了什么：Q/V 与 rank 不是普适定律

论文 §7.1 的 Table 5 在 GPT-3 175B 的 96 层、约 18M 参数预算下比较把 LoRA 放在
哪些 attention 权重。原始数字如下（WikiSQL / MultiNLI validation accuracy）：
""")
    nb.code("""
table5 = [
    ("Wq", 8, 70.4, 91.0), ("Wk", 8, 70.0, 90.8),
    ("Wv", 8, 73.0, 91.0), ("Wo", 8, 73.2, 91.3),
    ("Wq,Wk", 4, 71.4, 91.3), ("Wq,Wv", 4, 73.7, 91.3),
    ("Wq,Wk,Wv,Wo", 2, 73.7, 91.7),
]
print(f"{'weights':>14} {'r':>3} {'WikiSQL':>9} {'MultiNLI':>10}")
for row in table5:
    print(f"{row[0]:>14} {row[1]:>3} {row[2]:>9.1f} {row[3]:>10.1f}")

gpt3_qv_parameters = 4 * 96 * 12_288 * 4  # Q,V 各有 A/B
assert gpt3_qv_parameters == 18_874_368
print("Q/V, r=4 的精确参数计数:", f"{gpt3_qv_parameters:,}",
      "=", f"{gpt3_qv_parameters*2/2**20:.1f} MiB at fp16")
""")
    nb.md(r"""
作者概括为“Q+V overall 最好”，但要读细：四个投影都适配在 WikiSQL 与 Q+V 打平，
在 MultiNLI 还高 0.4。**Q/V 是这个固定预算实验支持的实用默认值，不是数学定理。**

§7.2 Table 6 固定所选权重，扫 $r=1,2,4,8,64$：
""")
    nb.code("""
rank_columns = [1, 2, 4, 8, 64]
table6 = {
    ("WikiSQL", "Wq"): [68.8, 69.6, 70.5, 70.4, 70.0],
    ("WikiSQL", "Wq,Wv"): [73.4, 73.3, 73.7, 73.8, 73.5],
    ("WikiSQL", "all four"): [74.1, 73.7, 74.0, 74.0, 73.9],
    ("MultiNLI", "Wq"): [90.7, 90.9, 91.1, 90.7, 90.7],
    ("MultiNLI", "Wq,Wv"): [91.3, 91.4, 91.3, 91.6, 91.4],
    ("MultiNLI", "all four"): [91.2, 91.7, 91.7, 91.5, 91.4],
}
print("r columns:", rank_columns)
for key, scores in table6.items():
    print(f"{key[0]:>8} {key[1]:>9}:", scores)
assert table6[("WikiSQL", "Wq,Wv")][0] == 73.4
""")
    nb.md(r"""
作者的结论是 rank 1 已“competitive”，并明确提醒：不期待小 rank 对每个任务/数据集都
有效。Figure 3 比较第 48/96 层的 $r=8$ 与 $r=64$：Q、V 的首要方向共享一个
normalized similarity $>0.5$ 的一维子空间。Table 7 还报告 $r=4$ 的
$\Delta W_q$ 方向上，$\|U^TW_qV^T\|_F=0.32$、$\|\Delta W_q\|_F=6.91$，
作者称约 $6.91/0.32=21.5\times$ 的“amplification”。这些是 GPT-3 上的经验解释，
不是低秩更新对任意模型的证明。
""")
    nb.takeaway(
        "Q/V 是论文预算实验的强默认，不是每个架构、任务的唯一正确位置。",
        "论文发现 rank 1 已有竞争力，同时亲自写明小 rank 不会适用于所有任务。",
        "toy SVD 证明的是我们构造的任务；GPT-3 的 intrinsic-rank 说法来自经验测量。",
    )

    nb.md(r"""
---
## 6. Adapter-only checkpoint：省下的是“每个任务的副本”

LoRA checkpoint 只需保存 `lora_A`、`lora_B` 和配置；它**不能独立运行**，部署时仍须
持有同一个 $W_0$。论文 §4.2 的 GPT-3 例子是约 350GB → 35MB（约 10,000×）；
100 个任务约需 354GB，而不是约 35TB。下面用真实序列化字节验证缩减方向，不把玩具
比例冒充论文比例。
""")
    nb.code("""
checkpoint_base = nn.Linear(256, 256, bias=False)
checkpoint_source = LoRALinear.from_linear(
    checkpoint_base, rank=4, alpha=8.0, dropout=0.0
).eval()
with torch.no_grad():
    checkpoint_source.lora_B.normal_(std=0.02)

adapter_state = {
    name: tensor.detach().cpu().clone()
    for name, tensor in checkpoint_source.state_dict().items()
    if name in {"lora_A", "lora_B"}
}
payload = {
    "config": {"rank": 4, "alpha": 8.0, "in_features": 256, "out_features": 256},
    "state_dict": adapter_state,
}

def serialized_bytes(obj):
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.tell()

full_bytes = serialized_bytes(checkpoint_base.state_dict())
adapter_bytes = serialized_bytes(payload)
print("adapter keys:", sorted(adapter_state))
print(f"toy serialized: full={full_bytes:,} B, adapter={adapter_bytes:,} B, "
      f"ratio={full_bytes/adapter_bytes:.1f}x")
print("analytic parameter ratio:", 256 * 256 / (4 * (256 + 256)))

assert set(adapter_state) == {"lora_A", "lora_B"}
assert adapter_bytes < full_bytes
""")
    nb.code("""
# 先加载相同 W0，再叠加小 checkpoint；strict=False 是因为 checkpoint 故意没有 base.weight。
checkpoint_restored = LoRALinear.from_linear(
    checkpoint_base, rank=4, alpha=8.0, dropout=0.0
).eval()
load_result = checkpoint_restored.load_state_dict(adapter_state, strict=False)
probe_x = torch.randn(12, 256)

assert "base.weight" in load_result.missing_keys
assert not load_result.unexpected_keys
assert torch.allclose(checkpoint_restored(probe_x), checkpoint_source(probe_x), atol=1e-6)
print("相同 W0 + adapter-only state 恢复了相同输出。")
""")
    nb.exercise(
        "如果把 rank 从 4 加到 16，adapter 的理论参数量和缩减倍数各如何变化？先预测，再改 `rank_for_budget`。",
        hint="LoRA 参数量对 r 线性增长；完整矩阵参数量不变。",
        starter="""
rank_for_budget = 4  # 改成 16
adapter_parameter_count = rank_for_budget * (256 + 256)
ratio = (256 * 256) / adapter_parameter_count
print({"rank": rank_for_budget, "adapter_parameters": adapter_parameter_count,
       "full/adapter": ratio})
assert adapter_parameter_count > 0
""",
    )
    nb.takeaway(
        "LoRA checkpoint 小，是因为每个任务只存 A/B；基础模型 W0 仍然不可省。",
        "序列化文件还含 metadata，所以小矩阵的实际字节比纯参数公式多；应同时报告两者。",
    )

    nb.md(r"""
---
## 7. Merge：数值相同，推理只剩一个 Linear

部署时把缩放后的更新折进基础权重：

$$W_{merged}=W_0+\frac{\alpha}{r}BA.$$

论文 §4.1 写成 $W_0+BA$ 是沿用前文的缩放简写；实现不能漏掉 `alpha/r`。
这里在 `eval()` 后显式 merge，并同时检查输出、权重恢复和 `F.linear` 调用数。
`learnai` 为了把状态变化写清楚而使用显式 `merge()`；官方 `loralib` 默认在
`model.eval()` 时 merge、`model.train()` 时 unmerge。
""")
    nb.code("""
deploy = checkpoint_restored.eval()
deploy_input = torch.randn(8, 256)
base_before_merge = deploy.base.weight.detach().clone()

with patch.object(F, "linear", wraps=F.linear) as linear_spy:
    unmerged_output = deploy(deploy_input)
    unmerged_linear_calls = linear_spy.call_count

expected_merged_weight = base_before_merge + deploy.delta_weight()
deploy.merge()
with patch.object(F, "linear", wraps=F.linear) as linear_spy:
    merged_output = deploy(deploy_input)
    merged_linear_calls = linear_spy.call_count

print("F.linear calls: unmerged", unmerged_linear_calls, "-> merged", merged_linear_calls)
print("max output difference:", float((unmerged_output - merged_output).abs().max()))

assert torch.allclose(deploy.base.weight, expected_merged_weight, atol=1e-6)
assert torch.allclose(unmerged_output, merged_output, atol=1e-5)
assert unmerged_linear_calls == 3 and merged_linear_calls == 1

deploy.unmerge()
assert torch.allclose(deploy.base.weight, base_before_merge, atol=1e-6)
""")
    nb.md(r"""
调用数证明的是这份实现的**计算图结构**；它不是严谨的延迟 benchmark。论文的
“no additional inference latency”来自 merge 后与普通 fine-tuned Linear 完全相同的
结构，而不是我们在这台 CPU 上计时得出的普适速度数字。训练时应保持 unmerged；
不同样本若要在同一 batch 使用不同 adapter，也不能同时把它们都 merge 进一个 $W$。
""")
    nb.takeaway(
        "merge 前后输出在浮点容差内相同；merge 后从三次 Linear 调用降为普通层的一次。",
        "merge 消除额外推理路径，但会让同一 batch 动态混用不同 adapter 变得困难。",
    )

    nb.md(r"""
---
## 8. 证据边界与下一步

这节课**确实复现**了：$BA$ 的形状与参数计数、zero-init no-op、$W_0$ 冻结、已知
low-rank target 的容量阈值、SVD 谱、固定 alpha 的 rank sweep、adapter-only
checkpoint，以及 merge 的数值/结构等价。

它**没有复现** GPT-3 175B 的准确率、显存或吞吐。论文报告：Adam 训练显存
1.2TB → 350GB；Q/V、$r=4$ checkpoint 350GB → 35MB；吞吐
32.5 → 43.1 tokens/s/V100。那些数字需要原规模模型与硬件，不能从这个 toy CPU
实验外推。

还要记住一个工程对应：本仓库的 attention 使用 fused `qkv_proj`。若以后真的只给
Q/V 加 LoRA，必须拆开 Q/K/V，或实现官方 `MergedLinear(...,
enable_lora=[True, False, True])` 的分组语义；给整个 `3d` 输出加一个普通 LoRA
会连 K 一起改变，也不会分别约束 Q、V 的 rank。
""")
    nb.code("""
elapsed = time.perf_counter() - lesson_started
print(f"mode={'FAST' if FAST else 'default'}, device=cpu, seed={ctx.seed}, elapsed={elapsed:.2f}s")
print("最终可回答：什么被冻结？什么低秩？alpha/r 在哪里？rank 何时不够？merge 省掉什么？")
assert ctx.seed == 2022
assert elapsed < 180
""")
    nb.takeaway(
        "LoRA 的中心不是“模型低秩”，而是“任务所需的权重变化可能低秩”。",
        "选择 rank 与目标层是经验问题；用 baseline、消融、多个 seed 和明确证据边界来回答。",
        "小 checkpoint 与零额外 merged latency 是工程收益；它们不自动保证任务质量。",
    )

    nb.save("notebooks/06_frontier/01_lora.ipynb")


if __name__ == "__main__":
    build()

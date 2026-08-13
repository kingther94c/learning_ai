"""Generate the ResNet mechanism-and-evidence lesson.

The committed notebook is generated from Python so its structure is reviewable,
its cell IDs are stable, and all outputs remain cleared.  The lesson deliberately
separates a tiny identity-mapping optimisation microscope from the CIFAR/ImageNet
experiments reported in the original ResNet paper.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.nbuild import Notebook


def build() -> None:
    nb = Notebook(
        "ResNet：为什么更深的网络反而更难训练？",
        subtitle="""
**核心问题：** 如果更深的网络至少可以把新增层学成恒等映射，为什么它的训练误差反而会更高？

**你会动手验证：** residual mapping 的代数、零残差时的恒等路径与梯度、同参数量下的
plain/residual 优化坐标，以及原始两层 `3×3` ResNet block 的 shortcut 与 projection。

**原始论文：** He et al., [*Deep Residual Learning for Image Recognition*](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html),
CVPR 2016。文中的论文数字均来自[官方 PDF](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)，
并在相应位置标明表格或章节。

**证据边界先说清：** 本 notebook 会重画论文表格中的 ImageNet/CIFAR-10 数字，但不会下载
数据集或声称复现那些准确率；离线 toy 只是一台“近恒等映射优化显微镜”。
""",
    )

    nb.md("""
---
## 0. Setup

全程离线、固定种子并强制 CPU。默认路径只训练几个很小的线性网络；`LEARNAI_FAST=1`
保留相同实验结构但减少更新次数。
""")
    nb.code("""
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from learnai.env import setup
from learnai.nn import BasicResidualBlock, PlainLinearStack, ResidualLinearStack, TinyResNet
from learnai.viz import SERIES, use_style

ctx = setup(seed=7, device="cpu")
use_style()
torch.set_num_threads(1)
torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)
FAST = os.environ.get("LEARNAI_FAST", "0") == "1"
print("fast smoke path:", FAST)
""")

    nb.md("""
---
## 1. 先看反常现象：深度并不自动带来更低误差

下面是论文 **Table 2（ImageNet validation，10-crop top-1 error）** 的原始数字。
plain network 从 18 层加深到 34 层，错误率反而上升；残差网络则明显下降。

但请注意：**degradation problem 的定义不是“测试集过拟合”**。论文 §1、Figure 1 及
§4.1、Figure 4 的关键证据，是更深 plain net 的**训练误差也更高**。Table 2 只是与之
对应的验证集结果。论文还报告 plain nets 使用了 BatchNorm，前向信号和反向梯度范数正常；
因此作者明确把这个现象与经典 vanishing gradient 区分开。
""")
    nb.code("""
depths = np.array([18, 34])
imagenet_table2 = {
    "plain": np.array([27.94, 28.54]),
    "residual": np.array([27.88, 25.03]),
}

xpos = np.arange(len(depths))
fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.bar(xpos - 0.18, imagenet_table2["plain"], 0.36, label="plain", color=SERIES[1])
ax.bar(xpos + 0.18, imagenet_table2["residual"], 0.36, label="residual", color=SERIES[0])
ax.set_xticks(xpos, [f"{d} layers" for d in depths])
ax.set_ylabel("ImageNet top-1 error (%)")
ax.set_title("ResNet paper Table 2: adding layers has opposite effects")
ax.set_ylim(23.5, 29.5)
ax.legend()
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=2, fontsize=8)
plt.tight_layout()

plain_change = imagenet_table2["plain"][1] - imagenet_table2["plain"][0]
residual_change = imagenet_table2["residual"][1] - imagenet_table2["residual"][0]
assert np.isclose(plain_change, 0.60)
assert np.isclose(residual_change, -2.85)
print(f"18→34 layers: plain {plain_change:+.2f} points; residual {residual_change:+.2f} points")
""")
    nb.takeaway(
        "degradation 指更深 plain net 连训练误差都更高，不等同于过拟合。",
        "论文的 plain nets 已使用 BatchNorm；不能把这项结果简化成普通的梯度消失故事。",
        "验证集 Table 2 和训练曲线 Figure 4 是不同证据，解读时不要混为一谈。",
    )

    nb.md("""
---
## 2. 把要学的函数换一个坐标系

论文 §3.1 不直接让一组层拟合目标映射 $H(x)$，而是定义

$$F(x) := H(x)-x, \\qquad H(x)=F(x)+x.$$

函数表达能力没有凭空增加；变化的是**参数化坐标**。如果最优 $H$ 接近恒等映射，残差坐标
只需把 $F$ 推向零。下面先把卷积、BN、ReLU 都拿掉，让四维向量上的等式可以逐项打印。
""")
    nb.code("""
torch.manual_seed(3)
x_small = torch.tensor([[1.0, -2.0, 0.5, 3.0]])
F_matrix = torch.tensor([
    [0.0, 0.1, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.2, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, -0.1],
])
residual_part = x_small @ F_matrix.T
target_mapping = residual_part + x_small

print("x       =", x_small)
print("F(x)    =", residual_part)
print("F(x)+x  =", target_mapping)
assert torch.allclose(target_mapping - x_small, residual_part, atol=1e-7)
""")

    nb.md("""
论文 §1 给出一个构造性论证：一个更深模型原则上可以复制浅模型，再把新增层设为恒等映射，
所以它不应因为表达能力不足而有更高训练误差。残差参数化把这个构造写成“新增分支为零”。
""")
    nb.code("""
identity_stack = ResidualLinearStack(dimension=4, depth=12)
for layer in identity_stack.layers:
    nn.init.zeros_(layer.weight)

probe = torch.tensor([[1.0, -2.0, 0.5, 3.0]], requires_grad=True)
identity_output = identity_stack(probe)
identity_output.sum().backward()

print("output:", identity_output.detach())
print("d sum(output) / d input:", probe.grad)
assert torch.equal(identity_output, probe.detach())
assert torch.equal(probe.grad, torch.ones_like(probe))
""")
    nb.takeaway(
        "Residual learning 是 $H=F+x$ 的重参数化；shortcut 本身不是一套新的标签或损失。",
        "在线性显微镜里，零残差分支同时给出精确恒等前向路径和单位梯度路径。",
        "这个代数事实解释可用路径，却不单独证明真实 CNN 一定更容易优化。",
    )

    nb.exercise(
        "把第一个残差层的一项从 0 改成 0.5，先手算哪个输出坐标会变化，再运行检查。",
        hint="第二层仍为零，所以只有第一层的 `x + W x` 会留下痕迹。",
        starter="""
challenge = ResidualLinearStack(dimension=3, depth=2)
for layer in challenge.layers:
    nn.init.zeros_(layer.weight)
with torch.no_grad():
    challenge.layers[0].weight[0, 1] = 0.5

v = torch.tensor([[2.0, 4.0, -1.0]])
expected = v.clone()
expected[:, 0] += 0.5 * v[:, 1]
print("predicted:", expected, "actual:", challenge(v))
assert torch.equal(challenge(v), expected)
""",
    )

    nb.md("""
---
## 3. 同样的参数预算，不同的深度坐标

`PlainLinearStack` 复合 $W_L\\cdots W_1x$；`ResidualLinearStack` 复合
$(I+W_L)\\cdots(I+W_1)x$。每层都只有一个无 bias 的 $D\\times D$ 矩阵，因此两者的
可训练参数数目完全相同，shortcut 的加法不带参数。
""")
    nb.code("""
dimension, depth = 5, 8
torch.manual_seed(11)
plain_budget = PlainLinearStack(dimension, depth)
torch.manual_seed(11)
residual_budget = ResidualLinearStack(dimension, depth)

plain_parameters = sum(p.numel() for p in plain_budget.parameters())
residual_parameters = sum(p.numel() for p in residual_budget.parameters())
paired_weights_equal = all(
    torch.equal(a.weight, b.weight)
    for a, b in zip(plain_budget.layers, residual_budget.layers, strict=True)
)
print("parameters:", plain_parameters, residual_parameters)
print("identically initialised branch weights:", paired_weights_equal)
assert plain_parameters == residual_parameters == dimension * dimension * depth
assert paired_weights_equal
""")

    nb.md("""
---
## 4. 离线 toy：近恒等映射的优化显微镜

现在生成固定的 16 维数据，目标是

$$y=x+0.1\\tanh(Ax).$$

它刻意接近恒等映射。我们用相同初始分支权重、相同参数量、相同 SGD 设置，比较深度
2/8/16 的 plain 与 residual 线性栈。

**这不是论文 degradation 的复现。** 这里没有图像、卷积、BatchNorm 或分类损失；小权重
连乘还会让深 plain stack 的梯度变小，而原论文专门报告其 BN plain nets 的梯度范数正常。
这个 toy 只回答更窄的问题：“近恒等目标在 $W$ 与 $I+W$ 两种坐标下会怎样？”
""")
    nb.code("""
data_generator = torch.Generator().manual_seed(2025)
toy_x = torch.randn(512, 16, generator=data_generator)
teacher = torch.randn(16, 16, generator=data_generator) / 4
toy_y = toy_x + 0.1 * torch.tanh(toy_x @ teacher.T)

steps = 30 if FAST else 120
toy_depths = [2, 8, 16]
histories = {}
trained_models = {}
initial_gradients = {}
started = time.perf_counter()

for stack_depth in toy_depths:
    paired = {}
    initial_branch_weights = {}
    for name, stack_class in [("plain", PlainLinearStack), ("residual", ResidualLinearStack)]:
        torch.manual_seed(100 + stack_depth)  # same W_i for the pair
        model = stack_class(16, stack_depth, init_scale=0.05)
        paired[name] = model
        initial_branch_weights[name] = [layer.weight.detach().clone() for layer in model.layers]

        initial_loss = F.mse_loss(model(toy_x), toy_y)
        initial_loss.backward()
        initial_gradients[(stack_depth, name)] = model.layers[0].weight.grad.norm().item()
        model.zero_grad(set_to_none=True)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        losses = []
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(toy_x), toy_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        histories[(stack_depth, name)] = losses
        trained_models[(stack_depth, name)] = model

    assert all(
        torch.equal(a, b)
        for a, b in zip(
            initial_branch_weights["plain"], initial_branch_weights["residual"], strict=True
        )
    )
    assert sum(p.numel() for p in paired["plain"].parameters()) == sum(
        p.numel() for p in paired["residual"].parameters()
    )

elapsed = time.perf_counter() - started
assert all(np.isfinite(values).all() for values in histories.values())
print(f"{6 * steps} full-batch updates in {elapsed:.2f}s")
for d in toy_depths:
    print(
        f"depth {d:>2}: final MSE plain={histories[(d, 'plain')][-1]:.6f}, "
        f"residual={histories[(d, 'residual')][-1]:.6f}; "
        f"first-layer initial grad {initial_gradients[(d, 'plain')]:.2e} vs "
        f"{initial_gradients[(d, 'residual')]:.2e}"
    )
""")
    nb.code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for name, color in [("plain", SERIES[1]), ("residual", SERIES[0])]:
    axes[0].plot(histories[(16, name)], label=name, color=color)
axes[0].set_yscale("log")
axes[0].set_xlabel("SGD update")
axes[0].set_ylabel("MSE (log scale)")
axes[0].set_title("Toy depth 16: same W initialization")
axes[0].legend()

for name, color, marker in [("plain", SERIES[1], "o"), ("residual", SERIES[0], "s")]:
    final_losses = [histories[(d, name)][-1] for d in toy_depths]
    axes[1].plot(toy_depths, final_losses, marker=marker, label=name, color=color)
axes[1].set_yscale("log")
axes[1].set_xticks(toy_depths)
axes[1].set_xlabel("number of linear maps")
axes[1].set_ylabel("final MSE (log scale)")
axes[1].set_title(f"Near-identity toy after {steps} updates")
axes[1].legend()
plt.tight_layout()
""")
    nb.takeaway(
        "在这个固定 toy 中，residual 坐标从接近恒等映射的位置出发，plain 坐标从小矩阵连乘出发。",
        "曲线展示的是本次受控测量，不是“所有残差网络必然赢”的定理。",
        "toy 的梯度衰减是刻意暴露的机制，不能拿来替代论文中 BN CNN 的 degradation 证据。",
    )

    nb.exercise(
        "把训练好的 depth-16 residual 权重原样装进 plain 坐标，移除所有 `+x`，测量损失会怎样。先预测，再运行。",
        hint="权重是在 `(I+W)` 坐标中学到的；相同参数张量不代表相同函数。",
        starter="""
learned_residual = trained_models[(16, "residual")]
without_skips = PlainLinearStack(16, depth=16)
without_skips.load_state_dict(learned_residual.state_dict())
with torch.no_grad():
    residual_loss = F.mse_loss(learned_residual(toy_x), toy_y).item()
    ablated_loss = F.mse_loss(without_skips(toy_x), toy_y).item()
print(f"with shortcuts: {residual_loss:.6f}; same W without shortcuts: {ablated_loss:.6f}")
assert sum(p.numel() for p in learned_residual.parameters()) == sum(
    p.numel() for p in without_skips.parameters()
)
assert np.isfinite([residual_loss, ablated_loss]).all()
""",
    )

    nb.md("""
---
## 5. 回到论文的两层卷积 basic block

论文 §3.2、Figure 2 的原始 basic block 是：

```text
main:      3×3 conv → BN → ReLU → 3×3 conv → BN
shortcut:  identity（形状相同）或 projection（形状变化）
output:    ReLU(main + shortcut)
```

这就是 post-activation ResNet v1。下面不是凭模块名字相信它，而是手工重算两条分支。
""")
    nb.code("""
torch.manual_seed(19)
block = BasicResidualBlock(2, 2).eval()
feature = torch.randn(1, 2, 4, 4)

main = block.bn1(block.conv1(feature))
main = F.relu(main)
main = block.bn2(block.conv2(main))
shortcut = block.shortcut(feature)
manual_output = F.relu(main + shortcut)

print("input / main / shortcut / output:")
print(feature.shape, main.shape, shortcut.shape, manual_output.shape)
assert isinstance(block.shortcut, nn.Identity)
assert torch.allclose(block(feature), manual_output, atol=1e-7)
""")
    nb.code("""
zero_block = BasicResidualBlock(2, 2).eval()
with torch.no_grad():
    zero_block.conv1.weight.zero_()
    zero_block.conv2.weight.zero_()

positive_feature = torch.rand(1, 2, 3, 3)
signed_feature = torch.linspace(-1, 1, 18).reshape(1, 2, 3, 3)
assert torch.allclose(zero_block(positive_feature), positive_feature)
assert torch.allclose(zero_block(signed_feature), F.relu(signed_feature))
print("F(x)=0, non-negative activation -> exact identity")
print("F(x)=0, signed input          -> ReLU(x), because v1 activates after addition")
""")
    nb.takeaway(
        "形状相同时，identity shortcut 没有可训练参数，主分支学习的是要加到 residual stream 上的改变量。",
        "原始 ResNet v1 在相加后还有 ReLU；因此零主分支对非负 block activation 是精确恒等，对任意有符号输入则是 ReLU。",
    )

    nb.md("""
---
## 6. 什么时候 shortcut 不能直接是 x？

论文 Eq. (2) 写成 $y=F(x)+W_sx$。当空间尺寸减半、通道数增加时，两条分支必须先变成
相同形状才能逐元素相加。论文比较了：

- **A**：降采样并对新增通道补零，无额外参数；
- **B**：只在维度变化时使用 `1×1` projection；
- **C**：每个 shortcut 都使用 projection。

当前 `BasicResidualBlock` 在形状变化时实现的是 **B**，不是论文 CIFAR 实验所用的 A。
""")
    nb.code("""
identity_block = BasicResidualBlock(4, 4).eval()
projection_block = BasicResidualBlock(4, 8, stride=2).eval()
image = torch.randn(1, 4, 6, 6)

projected_main = projection_block.bn1(projection_block.conv1(image))
projected_main = F.relu(projected_main)
projected_main = projection_block.bn2(projection_block.conv2(projected_main))
projected_shortcut = projection_block.shortcut(image)

identity_shortcut_parameters = sum(p.numel() for p in identity_block.shortcut.parameters())
projection_shortcut_parameters = sum(p.numel() for p in projection_block.shortcut.parameters())
print("main shape:", projected_main.shape, "shortcut shape:", projected_shortcut.shape)
print("shortcut parameters: identity", identity_shortcut_parameters,
      "; projection", projection_shortcut_parameters)
assert projected_main.shape == projected_shortcut.shape == (1, 8, 3, 3)
assert identity_shortcut_parameters == 0
assert projection_shortcut_parameters > 0
""")

    nb.exercise(
        "手工完成 projection block 的最后一步，并与模块输出逐元素核对。",
        hint="两条分支已经算好；原论文 basic block 在相加后应用 ReLU。",
        starter="""
my_projection_output = F.relu(projected_main + projected_shortcut)
reference_projection_output = projection_block(image)
max_difference = (my_projection_output - reference_projection_output).abs().max().item()
print("maximum absolute difference:", max_difference)
assert torch.allclose(my_projection_output, reference_projection_output, atol=1e-7)
""",
    )

    nb.md("""
`TinyResNet` 用 CIFAR 风格 `3×3` stem；每进入下一 stage，stride 2 让高宽减半、通道翻倍；
global average pooling 再把任意空间尺寸压成分类向量。这里把 feature map 逐层打印出来。
""")
    nb.code("""
tiny = TinyResNet(
    num_classes=10,
    blocks_per_stage=(1, 1, 1),
    widths=(4, 8, 16),
).eval()
tiny_image = torch.randn(1, 3, 16, 16)

features = tiny.stem(tiny_image)
stage_shapes = [tuple(features.shape)]
for stage in tiny.stages:
    features = stage(features)
    stage_shapes.append(tuple(features.shape))
manual_logits = tiny.classifier(F.adaptive_avg_pool2d(features, 1).flatten(1))

print("stem then stages:", stage_shapes)
print("logits:", manual_logits.shape)
assert stage_shapes == [(1, 4, 16, 16), (1, 4, 16, 16), (1, 8, 8, 8), (1, 16, 4, 4)]
assert torch.allclose(manual_logits, tiny(tiny_image), atol=1e-7)
""")

    nb.md("""
---
## 7. Projection 有帮助，但不是解决 degradation 的必要条件

论文 **Table 3（ImageNet validation，10-crop）** 在同一个 ResNet-34 上比较 A/B/C。
三种 shortcut 都远好于 plain-34；B 比 A 略好，C 又只比 B 略好。论文据此强调 projection
不是解决 degradation 的必要条件，随后为节省复杂度使用 B。
""")
    nb.code("""
shortcut_names = ["plain-34", "ResNet-34 A", "ResNet-34 B", "ResNet-34 C"]
shortcut_top1 = np.array([28.54, 25.03, 24.52, 24.19])
shortcut_top5 = np.array([10.02, 7.76, 7.46, 7.40])

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
axes[0].bar(shortcut_names, shortcut_top1, color=SERIES[:4])
axes[0].set_ylabel("top-1 error (%)")
axes[0].set_title("Table 3: shortcut comparison")
axes[0].tick_params(axis="x", rotation=25)
axes[0].set_ylim(22, 30)
for container in axes[0].containers:
    axes[0].bar_label(container, fmt="%.2f", padding=2, fontsize=8)

axes[1].bar(shortcut_names, shortcut_top5, color=SERIES[:4])
axes[1].set_ylabel("top-5 error (%)")
axes[1].set_title("The conclusion is the same at top-5")
axes[1].tick_params(axis="x", rotation=25)
axes[1].set_ylim(6, 10.8)
for container in axes[1].containers:
    axes[1].bar_label(container, fmt="%.2f", padding=2, fontsize=8)
plt.tight_layout()

assert np.isclose(shortcut_top1[0] - shortcut_top1[1], 3.51)
assert np.isclose(shortcut_top1[2] - shortcut_top1[3], 0.33)
""")

    nb.md("""
同一张 Table 3 还给出更深的 bottleneck ResNets。50/101/152 层使用论文 Figure 5 的
`1×1 → 3×3 → 1×1` bottleneck，而不是上面实现的两层 basic block；所以这张图展示论文
尺度趋势，不代表 `TinyResNet` 具备相同架构或结果。
""")
    nb.code("""
paper_depth = np.array([34, 50, 101, 152])
paper_top1 = np.array([24.52, 22.85, 21.75, 21.43])  # 34 uses B; 50+ are bottlenecks
paper_top5 = np.array([7.46, 6.71, 6.05, 5.71])

fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.plot(paper_depth, paper_top1, marker="o", label="top-1", color=SERIES[0])
ax.plot(paper_depth, paper_top5, marker="s", label="top-5", color=SERIES[1])
for d, value in zip(paper_depth, paper_top1, strict=True):
    ax.annotate(f"{value:.2f}", (d, value), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
ax.set_xticks(paper_depth)
ax.set_xlabel("weighted-layer depth")
ax.set_ylabel("ImageNet validation error (%)")
ax.set_title("ResNet paper Table 3 (10-crop evaluation)")
ax.legend()
plt.tight_layout()
assert paper_top1[-1] == 21.43 and paper_top5[-1] == 5.71
""")
    nb.takeaway(
        "A（零填充）已经解决主要 degradation；projection 的小幅收益不等于它是残差学习的核心。",
        "50 层以上使用 bottleneck block；不能只改一个 `depth` 数字就把 basic-block toy 称为 ResNet-152。",
    )

    nb.md("""
---
## 8. CIFAR-10：更深有效，但也不是无限有效

论文 §4.2 的 CIFAR 网络使用 `6n+2` 层、stage 宽度 16/32/64，并全部采用 shortcut A。
下面重画 **Table 6** 的 ResNet 行：这是论文报告的 test error 和参数量，不是本地运行输出。
""")
    nb.code("""
cifar_depth = np.array([20, 32, 44, 56, 110, 1202])
cifar_parameters_m = np.array([0.27, 0.46, 0.66, 0.85, 1.7, 19.4])
cifar_test_error = np.array([8.75, 7.51, 7.17, 6.97, 6.43, 7.93])

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
axes[0].plot(cifar_depth, cifar_test_error, marker="o", color=SERIES[0])
axes[0].set_xscale("log")
axes[0].set_xticks(cifar_depth, [str(d) for d in cifar_depth])
axes[0].set_xlabel("depth (log scale)")
axes[0].set_ylabel("CIFAR-10 test error (%)")
axes[0].set_title("Paper Table 6: depth is not a monotone guarantee")
for d, error in zip(cifar_depth, cifar_test_error, strict=True):
    axes[0].annotate(f"{error:.2f}", (d, error), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)

axes[1].plot(cifar_depth, cifar_parameters_m, marker="s", color=SERIES[1])
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xticks(cifar_depth, [str(d) for d in cifar_depth])
axes[1].set_xlabel("depth (log scale)")
axes[1].set_ylabel("parameters (millions, log scale)")
axes[1].set_title("The 1202-layer model is also much larger")
plt.tight_layout()

assert np.all((cifar_depth - 2) % 6 == 0)
assert cifar_test_error[cifar_depth.tolist().index(1202)] > cifar_test_error[cifar_depth.tolist().index(110)]
""")
    nb.md("""
ResNet-110 的单次最佳误差为 6.43%；论文另报 5 次运行 `6.61 ± 0.16`。ResNet-1202 虽把
训练误差压到 0.1% 以下，test error 却回升到 7.93%；作者将其归因于过拟合，并指出此时
模型对这份小数据集“不必要地大”。所以论文支持的是“残差参数化让很深网络可优化”，
而不是“深度越大泛化必然越好”。
""")
    nb.takeaway(
        "CIFAR 20→110 的论文结果随深度改善，但 1202 层反例阻止我们把趋势外推成定律。",
        "训练优化与测试泛化是两条轴：残差网络可以把训练误差降得很低，同时仍然过拟合。",
        "架构选项也要记录：论文 CIFAR 用 A，本 notebook 的 `TinyResNet` 形状变化时用 B。",
    )

    nb.exercise(
        "只依据 Table 6 数组，找出 test error 最低的深度，并同时打印它与 1202 层的参数量比。",
        hint="不要假设数组最后一个就是最好；先用 `argmin` 找索引。",
        starter="""
best_index = int(cifar_test_error.argmin())
best_depth = int(cifar_depth[best_index])
parameter_ratio = cifar_parameters_m[-1] / cifar_parameters_m[best_index]
print(f"best reported row: ResNet-{best_depth}; ResNet-1202 has {parameter_ratio:.1f}× its parameters")
assert best_depth == 110
assert np.isclose(parameter_ratio, 19.4 / 1.7)
""",
    )

    nb.md("""
---
## 9. 我们复现了什么，又没有复现什么

**在本地真正运行并验证了：**

- $H(x)=F(x)+x$ 的代数，以及线性残差分支为零时的恒等前向与单位梯度；
- plain/residual 线性栈具有相同参数预算，但在固定近恒等 toy 上呈现不同优化轨迹；
- 原始 post-activation basic block 的 `conv-BN-ReLU-conv-BN + shortcut → ReLU`；
- identity 与 projection shortcut 的形状、参数量和逐元素相加；
- 小图像 ResNet 的 stage 降采样、通道扩张、global average pooling 和 logits 形状。

**没有复现：**

- ImageNet/CIFAR-10 训练、论文准确率、Figure 1/4 的 CNN training-error degradation；
- CIFAR 的零填充 option A、50+ 层 bottleneck、152 层模型或 ImageNet ensemble；
- toy 之外的泛化优势。图中的论文数字只是从 Tables 2/3/6 逐项重画。

### 原论文阅读地图

- **§1 + Figure 1**：degradation problem 与“深模型可构造浅模型解”的论证；
- **§3.1**：residual mapping $F=H-x$；
- **§3.2 + Eqs. (1–2) + Figure 2**：identity/projection shortcuts 与 basic block；
- **§3.3 + Table 1**：ImageNet stage、options A/B 和 bottleneck 架构；
- **§4.1 + Tables 2–3 + Figure 4**：ImageNet plain/residual、shortcut 与深度比较；
- **§4.2 + Table 6 + Figures 6–7**：CIFAR 深度、1202 层过拟合和 residual response。

入口：[CVPR 官方页面](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) ·
[官方 PDF](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)

如果你只能带走一句话：**ResNet 不是“让梯度神奇地不消失”的口号，而是把接近恒等的
映射改写为学习一个接近零的残差；论文证据仍来自受控的真实 CNN 实验。**
""")

    nb.save("notebooks/02_deep_learning/01_resnet.ipynb")


if __name__ == "__main__":
    build()

# NCSN: Noise Conditional Score Network

> 论文：*Generative Modeling by Estimating Gradients of the Data Distribution* (NeurIPS 2019)
> 作者：Yang Song, Stefano Ermon (Stanford AI Lab)

## 项目概述

NCSN 是一种基于**分数匹配（Score Matching）**的生成模型。核心思想是：不直接建模数据分布 $p(x)$，而是学习其**梯度（分数函数）** $\nabla_x \log p(x)$，然后通过朗之万动力学沿梯度方向采样来生成新数据。

关键创新在于引入**多尺度噪声条件**：对数据施加不同强度的高斯噪声，得到一系列扰动分布 $p_\sigma(x)$，模型同时学习所有噪声级别下的分数函数，采样时从大噪声逐步退火到小噪声（Annealed Langevin Dynamics）。

---

## 模型架构

核心模型为 **Conditional RefineNet**（`CondRefineNetDilated`），一个带膨胀卷积的 U-Net 变体，输入 $(x, \sigma)$，输出分数 $\nabla_x \log p_\sigma(x)$。

### 整体结构

```
输入: x [B, C, H, W]  +  sigma_label [B] (离散噪声级别索引)
  │
  begin_conv: Conv2d(C → ngf, 3×3)
  │
  Encoder (4 个残差阶段):
    res1: 2× ConditionalResidualBlock (ngf → ngf, 不下采样)
    res2: 2× ConditionalResidualBlock (ngf → 2·ngf, 下采样)
    res3: 2× ConditionalResidualBlock (2·ngf → 2·ngf, 膨胀卷积 d=2)
    res4: 2× ConditionalResidualBlock (2·ngf → 2·ngf, 膨胀卷积 d=4)
  │
  Decoder (4 个 RefineNet 块, 带 skip connection):
    refine1: CondRefineBlock([res4])
    refine2: CondRefineBlock([res3, refine1])    ← 融合 encoder skip
    refine3: CondRefineBlock([res2, refine2])    ← 通道 2·ngf → ngf
    refine4: CondRefineBlock([res1, refine3])
  │
  ConditionalInstanceNorm2dPlus → ELU → Conv2d(ngf → C, 3×3)
  │
输出: score [B, C, H, W]
```

### 核心参数

| 参数 | MNIST | CelebA / CIFAR-10 |
|---|---|---|
| `ngf`（基础通道数） | 64 | 128 |
| `num_classes`（sigma 离散级别数） | 10 | 10 |
| `sigma_begin` | 1.0 | 1.0 |
| `sigma_end` | 0.01 | 0.01 |
| 输入尺寸 | 28×28×1 | 32×32×3 |

---

## 训练过程

### 数据处理

1. 图像经过 `Resize` + `ToTensor()` 转到 $[0,1]$
2. **均匀反量化**：$x = x / 256 \times 255 + \text{rand} / 256$（消除离散像素值的退化）
3. 可选 logit 变换：$\log(x) - \log(1-x)$

### Sigma 调度

构建几何（对数均匀）分布的噪声级别序列：

$$\sigma_i = \exp\!\left(\text{linspace}(\ln \sigma_{\text{begin}},\; \ln \sigma_{\text{end}},\; L)\right)$$

其中 $L=10$，得到如 $[1.0, 0.60, 0.36, 0.22, 0.13, 0.08, 0.05, 0.03, 0.02, 0.01]$。

### 前向流程

对每个 batch 中的样本：

```
1. 随机采样 sigma 级别 label ∈ [0, L)
2. 加噪: x̃ = x + σ_label · ε,    ε ~ N(0, I)
3. 预测: s_θ = scorenet(x̃, label)
4. 计算目标: target = -(x̃ - x) / σ²_label    （高斯扰动的真实分数）
```

### 损失函数：Denoising Score Matching (DSM)

$$\mathcal{L} = \frac{1}{2} \left\| s_\theta(\tilde{x},\, \sigma) - \text{target} \right\|^2 \cdot \sigma^{\text{anneal\_power}}$$

其中 `anneal_power = 2.0`。乘以 $\sigma^2$ 是为了平衡不同噪声级别的损失量级——小 sigma 的分数天然更大（$1/\sigma$ 量级），不加权会使训练被小噪声主导。

---

## 推理过程：Annealed Langevin Dynamics

训练完成后，从纯噪声出发，利用学到的分数函数逐步去噪生成图像。

### 算法

```
初始化: x ~ U(0, 1)    （均匀随机噪声）

For σ in [σ₀=1.0, σ₁, ..., σ₉=0.01]:    （从大到小遍历所有 sigma）
    step_size = step_lr × (σ / σ_min)²    （自适应步长）
    For s = 1, 2, ..., 100:               （每个 sigma 跑 100 步 Langevin）
        ε ~ N(0, I)
        grad = scorenet(x, label=当前sigma索引)
        x = x + step_size × grad + √(2 × step_size) × ε
```

### 核心参数

| 参数 | 值 | 说明 |
|---|---|---|
| `step_lr` | 0.00002 | 基础步长 |
| `n_steps_each` | 100 | 每个 sigma 级别的 Langevin 步数 |
| 总更新次数 | 10 × 100 = 1000 | 全部 sigma 级别累计 |

### 直觉理解

- **大 sigma 阶段**：大步长，快速建立图像的粗略结构（全局布局）
- **小 sigma 阶段**：小步长，精细打磨图像细节（纹理、边缘）
- 更新公式 $x_{t+1} = x_t + \text{step} \cdot \nabla_x \log p_\sigma(x_t) + \sqrt{2 \cdot \text{step}} \cdot \varepsilon$ 是 Langevin SDE 的离散化，其平稳分布恰好是 $p_\sigma(x)$

---
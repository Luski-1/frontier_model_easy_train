# Score-Based Generative Modeling through Stochastic Differential Equations

## 项目概述

SDE核心思想：通过随机微分方程（SDE）将数据分布连续地变换为简单的噪声分布（前向过程），再通过求解反向 SDE 从噪声中生成样本。反向 SDE 的求解依赖于每个时间步的分数函数（score function），即对数边际概率密度的梯度 ∇log p_t(x)，该函数通过分数匹配（score matching）训练神经网络来估计。

SDE项目在 CIFAR-10 数据集上支持多种 SDE 类型（VE / VP / sub-VP）和多种模型架构（NCSN / NCSNv2 / NCSN++ / DDPM / DDPM++）。以下概述聚焦于两条连续训练路线：

| 训练路线 | SDE 类型 | 模型架构 | 配置文件 |
|:---------|:---------|:---------|:---------|
| VESDE | Variance Exploding SDE | NCSN++ | `configs/ve/cifar10_ncsnpp_continuous.py` |
| VPSDE | Variance Preserving SDE | DDPM++ | `configs/vp/cifar10_ddpmpp_continuous.py` |

> **说明**：DDPM++ 与 NCSN++ 共用同一个模型类 `NCSNpp`，通过不同的 config 参数（如 `fir`、`progressive_input`、`embedding_type`、`scale_by_sigma`）控制架构差异。因此 VPSDE 配置中的 `model.name` 也为 `'ncsnpp'`。

---

## 模型架构

两条训练路线均采用 **NCSN++ / DDPM++** 架构，本质是一个带时间条件注入的 **U-Net**，由编码器、中间层和解码器三部分组成。

### 整体结构

```
输入图像 (B, 3, 32, 32)
    │
    ├── 时间编码 (Fourier / Positional) → MLP → temb (B, 512)
    │
    ▼
conv3x3(3 → 128)
    │
    ▼
┌─────────── Encoder ───────────┐
│ Level 0: 4×ResBlock + Attn(16) │  32×32, ch=128
│   ↓ Downsample                  │
│ Level 1: 4×ResBlock + Attn(16) │  16×16, ch=256
│   ↓ Downsample                  │
│ Level 2: 4×ResBlock             │   8×8,  ch=256
│   ↓ Downsample                  │
│ Level 3: 4×ResBlock             │   4×4,  ch=256
└─────────────────────────────────┘
    │
    ▼
┌─────────── Middle ────────────┐
│ ResBlock → AttnBlock → ResBlock│  4×4, ch=256
└─────────────────────────────────┘
    │
    ▼
┌─────────── Decoder ───────────┐
│ Level 3: 5×ResBlock (+skip)    │   4×4,  ch=256
│   ↑ Upsample                   │
│ Level 2: 5×ResBlock + Attn(16) │   8×8,  ch=256
│   ↑ Upsample                   │
│ Level 1: 5×ResBlock + Attn(16) │  16×16, ch=256
│   ↑ Upsample                   │
│ Level 0: 5×ResBlock (+skip)    │  32×32, ch=128
└─────────────────────────────────┘
    │
    ▼
GroupNorm → Swish → conv3x3(128 → 3) → (可选) /σ
    │
    ▼
输出 (B, 3, 32, 32)  — 分数预测
```

### 特殊设计说明

- **时间编码**：
  - **Fourier（VE）**：对连续时间对应的 σ 取对数后，通过 Gaussian Fourier Projection 映射到 256 维向量，再经两层 MLP 扩展到 512 维。Fourier 特征使模型能更好地感知指数分布的噪声尺度。
  - **Positional（VP）**：使用 Transformer 风格的正弦位置编码，将连续时间 t×999 映射到 128 维，再经 MLP 扩展到 512 维。
- **渐进式输入分支（progressive_input=residual）**：仅 VESDE 启用。在编码器下采样时，对输入图像同步执行带 FIR 滤波的卷积下采样，与主干特征通过 skip_rescale 机制融合，为深层提供原始图像的低分辨率信息。

---

## 训练过程

### 数据处理

- **数据集**：CIFAR-10，图像尺寸 32×32，3 通道
- **预处理**：TFDS 加载 → 转 float32 → resize 到 32×32 → 随机水平翻转（训练时）
- **数据归一化**：
  - VESDE：`centered=False`，数据保持 [0, 1]，模型内部前向传播时执行 `x = 2x - 1` 转换到 [-1, 1]
  - VPSDE：`centered=True`，数据在送入模型前执行 `x = 2x - 1` 转换到 [-1, 1]
- **Batch size**：128

### 前向 SDE 定义

#### VESDE（Variance Exploding SDE）

前向过程不改变均值，仅持续叠加方差递增的噪声：

$$dx = \sigma(t) \sqrt{2 \ln \frac{\sigma_{\max}}{\sigma_{\min}}} \, dw$$

其中 $\sigma(t) = \sigma_{\min} \left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^t$，参数 $\sigma_{\min}=0.01$，$\sigma_{\max}=50$。

扰动核（解析解）：$x_t \sim \mathcal{N}(x_0, \sigma(t)^2 I)$

先验分布：$p_T(x) = \mathcal{N}(0, \sigma_{\max}^2 I)$（t=1 时 σ=σ_max 的高斯分布）

#### VPSDE（Variance Preserving SDE）

前向过程同时缩放均值和叠加噪声，保持方差有界：

$$dx = -\frac{1}{2}\beta(t) x \, dt + \sqrt{\beta(t)} \, dw$$

其中 $\beta(t) = \beta_{\min} + t(\beta_{\max} - \beta_{\min})$，参数 $\beta_{\min}=0.1$，$\beta_{\max}=20$。

扰动核（解析解）：$x_t \sim \mathcal{N}\left(x_0 e^{-\frac{1}{4}t^2(\beta_1-\beta_0) - \frac{1}{2}t\beta_0}, \left(1 - e^{-\frac{1}{2}t^2(\beta_1-\beta_0) - t\beta_0}\right) I\right)$

先验分布：$p_T(x) = \mathcal{N}(0, I)$（标准正态分布）

### 数据在模型中的流程

1. 从 U[eps, 1) 采样连续时间 t（eps=1e-5，避免 VESDE 在 t=0 不可导）
2. 采样噪声 z ~ N(0, I)
3. 通过扰动核构造加噪数据：$x_t = \text{mean}(x_0, t) + \text{std}(t) \cdot z$
4. 将 $x_t$ 和 t（VE 转换为 σ，VP 乘以 999）送入模型，得到模型预测
5. 计算去噪分数匹配损失

### 训练目标与损失函数

两条路线均使用**连续去噪分数匹配（DSM）**损失，但参数化方式不同：

**VESDE**（`reduce_mean=False`, `likelihood_weighting=False`, `scale_by_sigma=True`）：

模型输出已除以 σ，等效于直接预测 score。损失为：

$$\mathcal{L} = \frac{1}{B} \sum_{i} \frac{1}{2} \sum \left\| \text{score}(x_t, t) \cdot \sigma(t) + z \right\|^2$$

由于 score = model_output / σ，等效于 $\frac{1}{2} \sum \| \text{model\_output} + z \|^2$，即模型预测 $-z$。

**VPSDE**（`reduce_mean=True`, `likelihood_weighting=False`, `scale_by_sigma=False`）：

模型输出为原始预测值，在 `get_score_fn` 中通过 $-\text{output} / \text{std}$ 转换为 score。损失为：

$$\mathcal{L} = \frac{1}{B} \sum_{i} \text{mean} \left\| \text{score}(x_t, t) \cdot \text{std}(t) + z \right\|^2$$

由于 score = -model_output / std，等效于 $\text{mean} \| -\text{model\_output} + z \|^2$，即模型预测 $z$（噪声）。

---

## 推理过程

推理入口通过 `sampling.get_sampling_fn()` 根据 `config.sampling.method` 选择采样方法：

### 1. Predictor-Corrector (PC) 采样器

两条训练路线的默认采样方法（`sampling.method='pc'`）。

**核心思想**：在每个时间步，先用 Predictor 沿反向 SDE 推进一步（预测下一步），再用 Corrector 通过 MCMC 方法在当前时间步的分布上做若干步校正（改善样本质量）。

**VESDE 的 PC 配置**：
- Predictor: **Reverse Diffusion** — 使用 VESDE 特有的反向离散化公式：

  $$x_{i-1} = x_i + (\sigma_i^2 - \sigma_{i-1}^2) \cdot \text{score}(x_i, t_i) + \sqrt{\sigma_i^2 - \sigma_{i-1}^2} \cdot \varepsilon$$

- Corrector: **Langevin Dynamics** — 基于信噪比（SNR=0.16）自适应步长的朗之万动力学，每步执行 1 次校正：

  $$x \leftarrow x + \text{step\_size} \cdot \text{score}(x, t) + \sqrt{2 \cdot \text{step\_size}} \cdot \varepsilon$$

  步长由目标 SNR、score 范数和噪声范数自适应计算。

**VPSDE 的 PC 配置**：
- Predictor: **Euler-Maruyama** — 使用通用的反向 SDE 欧拉离散化：

  $$x_{i-1} = x_i + \text{drift}_{\text{reverse}} \cdot dt + \text{diffusion}_{\text{reverse}} \cdot \sqrt{|dt|} \cdot \varepsilon$$

  其中反向 SDE 的 drift = $f_{\text{forward}} - g_{\text{forward}}^2 \cdot \text{score}$，diffusion = $g_{\text{forward}}$。

- Corrector: **None** — 不使用校正器（纯 Predictor 采样）。

**采样流程**：
1. 从先验分布采样初始噪声：VE 为 $\mathcal{N}(0, \sigma_{\max}^2 I)$，VP 为 $\mathcal{N}(0, I)$
2. 从 T=1 到 eps（VE: 1e-5, VP: 1e-3）均匀划分 1000 个时间步
3. 每个时间步依次执行 Corrector → Predictor
4. 最终可选执行一步去噪（返回不含扩散噪声的 x_mean）
5. 反归一化输出图像

**总函数评估次数**：N × (n_steps + 1) = 1000 × 2 = 2000（VE），1000 × 1 = 1000（VP）

### 2. Probability Flow ODE 采样器

虽然两条训练路线的默认配置未使用 ODE 采样（`sampling.method='ode'`），但项目完整实现了该方法。

**核心思想**：对于任意前向 SDE，存在一个确定性的概率流 ODE，其边际分布与 SDE 完全一致。通过求解该 ODE 可以实现确定性的样本生成，同时支持精确似然计算。

**反向概率流 ODE**：

$$dx = \left[f(x, t) - \frac{1}{2} g(t)^2 \cdot \text{score}(x, t)\right] dt$$

与反向 SDE 的区别在于扩散项 $g(t)$ 被置零，漂移项中 score 的系数从 1 变为 1/2。

**实现方式**：
- 使用 SciPy 的黑盒 ODE 求解器 `scipy.integrate.solve_ivp`
- 方法：RK45（自适应步长的 Runge-Kutta 4/5 阶方法）
- 相对误差容限：rtol=1e-5
- 绝对误差容限：atol=1e-5
- 积分区间：(T, eps)，即从 t=1 到 t=eps
- 初始状态：从先验分布采样的噪声
- 可选最终去噪：使用 ReverseDiffusion Predictor 在 t=eps 处执行一步去噪

**与 PC 采样的对比**：
- ODE 采样是确定性的（给定相同初始噪声，结果唯一），PC 采样是随机的
- ODE 采样通过自适应步长控制误差，函数评估次数不固定
- ODE 采样质量通常略低于 PC 采样（论文中 FID(ODE) > FID(PC)），但可用于精确似然计算

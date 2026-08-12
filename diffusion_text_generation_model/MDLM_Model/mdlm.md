# MDLM (Masked Diffusion Language Models)

## 项目概述

MDLM 是一种**离散状态扩散语言模型**——与连续扩散（如 DDPM 在图像上的应用）不同，文本是离散 token 序列，扩散过程必须定义在离散空间上。MDLM 采用 **absorbing state diffusion（吸收态扩散）**：前向过程逐步把 token 替换为 `[MASK]`，反向过程学习从全 MASK 序列逐步"还原"出真实文本。

本项目提出了 **SUBStitution（SUBS）参数化**，将吸收态扩散的 ELBO 损失化简为**经典的掩码语言模型（MLM）损失**的混合形式。这使得训练**无需显式的时间步条件**

---

## 模型架构

### 核心架构

默认主干网络为 **DIT（Diffusion Transformer）**

**Noise Schedule**：默认 LogLinear（`σ(t) = -log(1 - (1-ε)t)`，使得 mask 概率 ≈ t），论文证明 a_t 的具体形式不影响 ELBO。

---

## 训练过程

### 数据处理

- **Tokenizer**：对 tokenizer 额外设置 BOS/EOS。若 tokenizer 缺少 MASK token，则在 vocab_size 末尾追加一个 mask_index。
- **打包（packing）**：将多段文本拼接后按 block_size（1024）分块，每块前后加 BOS/EOS，避免 padding 浪费。

### 数据在模型中的流程

1. 原始文本 → tokenizer → input_ids `[B, 1024]` + attention_mask
2. **前向加噪**：从 [ε, 1] 采样时间 t（默认反锯齿采样 antithetic_sampling=True，降低梯度方差）；通过 noise schedule 得到 σ(t)；计算 mask 概率 `1 - exp(-σ(t)) ≈ t`；以该概率把每个 token 替换为 MASK，得到 `x_t`。
3. **模型前向**：`x_t → Token Embedding → 12×DDiTBlock → logits`，再经 SUBS 参数化得到 `log p(x0 | x_t)`。
4. **损失计算**：取真实 x0 对应位置的 log 概率，乘以 `dsigma / expm1(sigma) = 1/t`（连续时间 SUBS 的 ELBO 简化结果），再按 attention_mask 聚合为 token 级 NLL。

### 训练目标与损失函数

**SUBS 参数化的核心简化（论文的关键洞见）**：

吸收态扩散的反向过程目标是预测 `p(x0 | x_t)`。作者提出两条规则化简：

- **RB1（Rule 1）**：对于 `x_t` 中**未被 mask 的位置**，模型输出固定为原 token（logit=0，其余=-∞），因为前向过程保证未 mask 的位置不会变化。
- **RB2（Rule 2）**：模型预测为 MASK 的 logit 永远设为 -∞，因为反向过程不会把已解码 token 变回 MASK。

在这两条规则下，连续时间 SUBS 的 ELBO 损失**恰好化为加权交叉熵**：

$$\mathcal{L}_{\text{SUBS}} = -\mathbb{E}_{t \sim U[\varepsilon,1]} \left[ \frac{1}{t} \cdot \log p_\theta(x_0 \mid x_t) \right]$$

即对 mask 位置做交叉熵，权重为 `1/t`（t 越小，被 mask 的位置越少，每个被 mask 位置的损失权重越大）。这与经典 MLM（如 BERT）只差一个 `1/t` 加权，因此可以说 **MDLM = 加权 MLM**。这也是"Simple and Effective"标题的由来。

其他参数化（作为基线）：
- **d3pm**：离散时间（T=1000）的 D3PM 损失，需计算变分下界 L_vb。

---

## 推理过程

### 采样器（sampling.predictor）

1. **`ddpm_cache`（默认，作者提出）**：利用 RB1/RB2 简化后的后验公式。核心思想：当某一步没有新 token 被"解码"（即采样结果与输入近似相同）时，**缓存上一步的 `p(x0)` 直接复用**，避免重复前向传播。这带来 **3-4× 加速**。
2. **`ddpm`**：D3PM 的 ancestral sampling，每步都重新前向。
3. **`analytic`**：SEDD 的 analytic sampler，基于 score matching。

### 采样流程

1. 初始化 `x` 为**全 MASK 序列** `[mask_index] × 1024`。
2. 生成时间点序列 `timesteps = linspace(1, eps, num_steps+1)`，步长 `dt = (1-eps)/num_steps`，默认 num_steps=128（推理命令中可设 1000）。
3. 从 t=1（几乎全 MASK）逐步降到 t=eps（几乎全还原）：
   - 计算当前 σ(t)、下一步 σ(t-dt)
   - 模型前向得到 `p(x0 | x_t)`
   - 依据后验公式 `p(x_s | x_t)` 采样下一步 `x_s`：
     - 被解码为真实 token 的概率正比于 `(a_s - a_t) · p(x0)`
     - 保持 MASK 的概率为 `1 - a_s`（其中 `a_t = exp(-σ(t))`）
   - 对未 mask 位置保持原值，仅对 mask 位置采样
4. 可选 `noise_removal`：最后一步 t→0 时再前向一次，argmax 得到最终 token。
5. 用 tokenizer.batch_decode 解码为文本。

### 半自回归（SAR）生成

MDLM 可生成任意长度文本：把上一次采样的后半段作为"已知前缀"拼到新的全 MASK 序列前面，再迭代采样。

---

## 过程中的时间步t抽样

### 1.均匀采样（uniform_sampling）

从均匀分布采样t

### 2. 反锯齿采样（antithetic_sampling）

从均匀分布采样t后，进行缩小降低方差。再把 [0,1) 分成 n 个等长小区间，加上缩小后的t，得到每个区间内各采一个点的效果。方差比纯随机低 O(n²) 倍。

### 8. 重要性采样（importance_sampling）

SUBS 损失有 `1/t` 因子，t 很小时权重爆炸，梯度方差大。重要性采样把 t 的采样分布从 U[ε,1] 改为 `q(t) = 1/(t · ln(1/ε))`（密度与 1/t 成正比），使 `t · q(t) = 常数`，消去 1/t 因子。实现上用逆变换法：`t = ε · ε^(-u)`，其中 `u ~ U[ε,1]`。这是重要性采样在扩散模型中的具体应用，理解它对掌握方差缩减很有帮助。


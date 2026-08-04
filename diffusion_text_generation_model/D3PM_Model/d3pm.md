# D3PM

## 核心思想

把扩散过程定义在**离散状态空间**上（像素取有限整数类别），用**离散转移矩阵** $Q_t$ 替代连续高斯核，从而避开高斯扩散模型在数值、类别数据上的不适用问题。

## 模型架构

- **噪声调度**：余弦调度 `alpha_bar = cos((s+0.008)/1.008 · π/2)`，`beta_t = min(1 - ᾱ_t/ᾱ_{t-1}, 0.999)`，共 1000 步。
- **单步转移矩阵** $Q_t$：`uniform` 类型，对角线 `1 - (K-1)/K·β`，其余 `β/K`（K=N 为离散类别数，行和=1）。
- **累积转移矩阵** $\bar{Q}_t$：由 $Q_t$ 连乘得到，`register_buffer` 预计算存表。
- **前向加噪** `q_sample`：取 `log( x₀·Q̄_t + eps )`，叠加 Gumbel 噪声后 `argmax`，得到离散 $x_t$。
- **后验 logits** `q_posterior_logits`：实现论文式 (3)，`log( x_t·Q_tᵀ ⊙ x₀·Q̄_{t-1} )`；`t==1` 时退化为 `x₀` 的 one-hot logit（对应 $L_0$ 项）。
- **VB 损失** `vb`：对真实后验与预测后验做 KL 散度。
- **采样** `p_sample` / `sample` / `sample_with_image_sequence`：反向逐步 Gumbel-max 采样；最后一步（`t==1`）不注入 Gumbel 随机性，直接取 argmax 保真。

## 训练过程

### 数据处理

- 预处理：`transforms.Compose([ToTensor(), Pad(2)])`——把 28×28 补零到 32×32，便于与 ConvTranspose 的逐级 ×2 下采样对齐（32→16→8→4→2）。

### 离散化与前向流程

1. 取一个 batch `(x, cond)`，`x∈[0,1]` 连续值，`cond∈{0..9}` 标签。
2. **离散化**：`x = (x * (N-1)).round().long().clamp(0, N-1)`，N=2 时即把像素二值化为 {0,1}。
3. `d3pm.forward(x, cond)`：
   - 随机采 `t ∈ [1, n_T)`（1000）。
   - `q_sample`：用 $\bar{Q}_t$ 对 $x_0$ 取行，log + Gumbel-max 抽样得 $x_t$（整数）。
   - `model_predict`：`DummyX0Model(x_t, t, cond)` 预测 $\hat{x}_0$ 的 logits（未 softmax），形状 `(B,C,H,W,N)`。
   - `q_posterior_logits`：分别用**真实 $x_0$** 和**预测 $\hat{x}_0$** 计算后验 logits。
   - `vb`：两者做 KL 散度（VB 损失）。
   - `ce_loss = CrossEntropyLoss，辅助监督直接预测 $x_0$ 类别。
   - 总损失 `loss = hybrid_loss_coeff * vb_loss + ce_loss`。

### 推理流程

1. `d3pm.eval()`，进入 `torch.no_grad()`。
2. 构造类别标签 `cond = torch.arange(0,4) % 10`（生成 4 张图）。
3. 构造整数噪声 `init_noise = torch.randint(0, N, (4,1,32,32))`（从均匀分布采样的整数 $x_T$）。
4. `sample_with_image_sequence` 反向遍历 `t = 1000→1`，每步：
   - `model_predict` 预测 $\hat{x}_0$ logits。
   - `q_posterior_logits` 算出 $q(x_{t-1}|x_t,\hat{x}_0)$ 的 logits。
   - 叠加 Gumbel 噪声（`t==1` 时不加，保证最终保真），`argmax` 采样得 $x_{t-1}$。

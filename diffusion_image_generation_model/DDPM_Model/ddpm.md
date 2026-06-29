# Reference 项目概述：DDPM（去噪扩散概率模型）

本项目是 DDPM（Denoising Diffusion Probabilistic Models）的一个精简参考实现，基于 CIFAR-10 数据集训练，包含完整的模型定义、训练脚本和推理脚本。

---

## 模型架构

模型核心为 **UNet 降噪网络**，由编码器-解码器 + 跳跃连接构成，关键组件如下：

### 时间嵌入（Time Embedding）

采用类似 Transformer 的 **正弦位置编码 + 三层 MLP**：

```
PositionalEmbedding(128) → Linear(128→512) → SiLU → Linear(512→512)
```

- 将离散时间步 `t` 映射为连续向量，作为 bias 注入到每个 ResBlock 中

### 编码器（Encoder）

- **通道基数**: `base_channels = 128`，通道倍增 `(1, 2, 2, 2)`
- 共 4 级，每级含 2 个 ResBlock，第 1 级启用自注意力
- 通过步长为 2 的卷积下采样（除最后一级）

### 解码器（Decoder）

- 与编码器对称，通过 `torch.cat` 拼接跳跃连接特征
- 每级含 3 个 ResBlock，通过 nearest 插值 + 卷积上采样
- 输出层: `GroupNorm → SiLU → Conv2d(128→3)`

---

## 训练流程

### 数据预处理

```
ToTensor()          # [0, 255] → [0, 1]
RescaleChannels()   # [0, 1] → [-1, 1]
```

### 前向扩散（加噪过程）

在 T=1000 步内逐步向数据添加高斯噪声：

```
x_t = √(ᾱ_t) · x_0 + √(1 - ᾱ_t) · ε,   ε ~ N(0, I)
```

其中 `ᾱ_t = ∏(1 - β_s)` 为累积乘积，噪声调度支持：
- **线性调度**: `β_t` 从 1e-4 线性增长到 0.02
- **余弦调度**: 更平滑的衰减曲线，偏移量 s=0.008

### 训练循环

```python
for iteration in range(800_000):
    x, y = next(train_loader)        # batch_size=128
    loss = diffusion(x)              # 随机采样时间步 t，计算损失
    loss.backward()
    optimizer.step()                 # Adam, lr=2e-4
    diffusion.update_ema()           # EMA 更新
```

### EMA（指数滑动平均）

- 衰减率 0.9999，5000 步后开始生效
- 推理时使用 EMA 权重，输出更稳定

---

## 推理流程

### 迭代去噪

从纯高斯噪声出发，逆向执行 T 步去噪：

```python
x = torch.randn(batch_size, 3, 32, 32)   # T=999，纯噪声
for t in range(999, -1, -1):
    x = remove_noise(x, t)               # 预测并去除噪声
    if t > 0:
        x += σ_t · z                     # 添加随机扰动（t=0 时不加）
```

### 单步去噪公式

```
x_{t-1} = (1/√α_t) · [x_t - (β_t/√(1-ᾱ_t)) · ε_pred] + σ_t · z
```

- `ε_pred`: 模型预测的噪声
- `σ_t = √β_t`: 随机采样噪声（t>0 时添加）

### 输出后处理

```
image = ((samples + 1) / 2).clip(0, 1)   # [-1, 1] → [0, 1]
```

支持按类别条件采样和无条件采样两种模式。

---

## 损失函数

采用**简化目标**：模型预测噪声 ε（而非直接预测 x_0），最小化预测噪声与真实噪声之间的差异。

```python
noise = torch.randn_like(x)                          # 真实噪声 ε
perturbed_x = perturb_x(x, t, noise)                 # 加噪图像 x_t
estimated_noise = model(perturbed_x, t, y)            # 预测噪声 ε̂

# 默认使用 L2 损失
loss = F.mse_loss(estimated_noise, noise)             # ||ε̂ - ε||²

# 也支持 L1 损失
loss = F.l1_loss(estimated_noise, noise)              # |ε̂ - ε|
```

- **随机采样时间步**: 每个 batch 中 `t ~ Uniform(0, T)`
- 该简化目标等价于最小化变分下界（ELBO），避免了复杂的 KL 散度计算

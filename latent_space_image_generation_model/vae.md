# VAE 人脸生成项目概述

## 项目简介

基于卷积 VAE（Variational Autoencoder）的人脸图像学习与生成项目，训练数据集为 CelebA 人脸数据集，目标是让模型学会将图像压缩到低维隐空间，并能从隐空间重建/生成图像。

---

## 模型架构（ConvVAE）

整体结构：**编码器 → 重参数化 → 解码器**，编码器与解码器对称。

### 基础模块

| 模块 | 说明 |
|------|------|
| `ResnetBlock` | 核心构建块，GroupNorm + SiLU + Conv，带跳跃连接 |
| `AttentionBlock` | 缩放点积自注意力，在低分辨率（如 32/16/8）启用 |
| `Downsample` | stride=2 卷积下采样 |
| `Upsample` | 最近邻插值 + 卷积上采样 |

### 编码器

```
输入图像 (3×H×W)
  → conv_in（初始卷积）
  → 多阶段下采样：[ResnetBlock × N → AttentionBlock（可选）→ Downsample] × stages
  → 中间层：ResnetBlock → AttentionBlock → ResnetBlock
  → Flatten / GlobalAvgPool
  → fc_mu（均值）+ fc_scale → std（经 Softplus 约束，防止方差爆炸）
  → 输出：mu, logvar
```

**关键点：**
- 方差用 `Softplus(raw_scale) + 1e-4` 约束，避免训练初期方差爆炸

### 解码器

```
隐向量 z (latent_dim)
  → fc_decode（线性投影）
  → reshape 为空间特征图 (C × final_res × final_res)
  → 中间层：ResnetBlock → AttentionBlock → ResnetBlock
  → 多阶段上采样：[ResnetBlock × (N+1) → Upsample] × stages（逆序）
  → GroupNorm + SiLU + conv_out
  → tanh 输出 [-1, 1]
```

---

## 训练流程

### 数据准备

- 数据集：CelebA（`img_align_celeba`）
- 预处理：Resize → CenterCrop → ToTensor → Normalize `[0,1] → [-1,1]`
- 划分：99% 训练，1% 评估

### 损失函数

$$\mathcal{L} = \text{ReconLoss} + \beta \cdot \text{KL}$$

- **重建损失**：MSE（像素级均方误差）
- **KL 散度**：$-0.5 \sum (1 + \log\sigma^2 - \mu^2 - \sigma^2)$，量化隐空间与标准正态的距离
- **KL 权重 β**：默认 1.0，支持线性 Warm-up Anneal（从 0 渐增至目标值）

### 训练配置（典型值）

| 参数 | 值 |
|------|----|
| 图像尺寸 | 128×128 |
| 隐空间维度 | 256 |
| 最终下采样分辨率 | 4×4 |
| 优化器 | AdamW（lr=2e-4，β₁=0.9，β₂=0.999） |
| 学习率调度 | Constant with Warmup（warmup_ratio=0.05） |
| 混合精度 | BF16 |
| 梯度裁剪 | max_norm=1.0 |

### 评估可视化

每次 evaluate 保存三类图像：
1. **重建对比**：真实图像 vs 重建图像并排对比
2. **先验采样**：从 N(0,I) 随机采样 z → decode 生成新图像
3. **隐空间插值**：在两个随机 z 之间线性插值，观察语义平滑过渡

---

## 推理流程

### 图像重建（encode → decode）

```
输入图像 → encode() → (mu, logvar)
  → reparameterize: z = mu + eps * std
  → decode(z / scaling_factor) → 重建图像 [-1,1] → 反归一化 [0,1]
```

### 随机生成（先验采样）

```
z ~ N(0, I)  [shape: (N, latent_dim)]
  → decode(z) → 生成图像
```

### 隐空间插值

```
z1, z2 ~ N(0, I)
  → z_interp = α·z1 + (1-α)·z2  [α 从 0 到 1]
  → decode(z_interp) → 平滑过渡序列
```

---

# CFG：Classifier-Free Guidance

## 项目概述

基于 **DDPM（Denoising Diffusion Probabilistic Model）**的模型架构，引入 **Classifier-Free Guidance (CFG)**

## 模型架构

在DDPM（可以参考DDPM项目）的模型架构基础上，增加分类标签的传递参数，使得模型可以同时接受xt、t、label信息

## 训练过程

### 数据处理

- 数据集：ImageNet-256，通过 `torchvision.datasets.ImageFolder` 加载
- 预处理流程：转 RGB → Resize → CenterCrop → ToTensor([0,1]) → Normalize([-1,1])
- `HFTrainerImageDataset` 包装 ImageFolder，返回 `{"pixel_values": tensor, "labels": long}` 字典格式以适配 Trainer

### 训练期间

1. 从 DataLoader 获取一个 batch 的图像 `x_start` 和类别标签 `label`
2. 随机采样时间步 `t ~ Uniform(0, T)`
3. **CFG 训练策略**：以概率 `cfg_threshold`（默认 0.1）将标签替换为 `class_num`（即"无条件"标签），实现无条件模型的联合训练
4. 前向加噪：`x_noisy = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε`，其中 `ε ~ N(0, I)`
5. UNet 接收 `(x_noisy, t, label)` 预测噪声 `ε̂`
6. 计算 MSE 损失：`L = MSE(ε̂, ε)`

## 推理过程

### 推理流程

1. **初始化**：从标准正态分布采样纯噪声 `x_T ~ N(0, I)`，形状为 `(num_sample_images, 3, image_size, image_size)`
2. **类别构造**：随机生成类别标签，每 4 张图共享同一类别（便于网格展示）
3. **反向去噪循环**（`t = T-1 → 0`）：
   - 用 **EMA 模型** 预测条件噪声：`ε_cond = EMA_model(x_t, t, label)`
   - 用 **EMA 模型** 预测无条件噪声：`ε_uncond = EMA_model(x_t, t, unconditional_label)`
   - **CFG 合成**：`ε̂ = (1 + w) × ε_cond - w × ε_uncond`
   - 计算均值：`μ = (1/√α_t) × [x_t - (1-α_t)/√(1-ᾱ_t) × ε̂]`
   - 计算方差：`σ² = β_t × (1-ᾱ_{t-1}) / (1-ᾱ_t)`
   - 采样：`x_{t-1} = μ + σ × z`（最后一步 t=0 直接返回 μ）
4. **后处理**：反归一化 `(x+1)/2 → [0,1]`，clamp 到 [0,1]，保存为网格图像

### CFG 的核心思想

训练时以一定概率丢弃条件信息（设为"无条件"标签），使同一个模型同时学会条件生成和无条件生成。推理时通过外推方式放大条件信号的影响：`ε̂ = (1+w)·ε_cond - w·ε_uncond`，`w` 越大，生成结果越贴合类别条件，但多样性可能下降。

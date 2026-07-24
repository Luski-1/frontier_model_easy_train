# NICE — Non-linear Independent Components Estimation

## 项目概述

NICE 是一种基于流模型（Flow-based Model）的生成模型，通过学习数据到隐空间的可逆变换。核心思想：将观测数据 x 经过一系列可逆且雅可比行列式易算的变换映射到隐变量 z，假设 z 服从标准高斯分布，则可通过逆变换从 z 采样生成新样本。

## 模型架构

NICE 模型由三个核心组件串联组成：

1. **耦合层（Coupling Layer）**：4 层 additive coupling 层，交替使用奇偶 mask，每层的变换规则为 `z1 = x1, z2 = x2 + m(x1)`。mask 按奇偶交替将输入维度分成两部分，保证每层至少一半维度保持不变，使得雅可比矩阵为对角阵、行列式恒为 1，log_det 透传为 0。
2. **缩放层（Scaling Layer）**：位于所有耦合层之后，对每个维度施加独立缩放 `y = exp(s) * z`。
3. **先验分布（Prior）**：标准高斯 N(0, I)，用于计算隐变量 z 的 log probability。

## 训练过程

- **数据处理**：MNIST 图像转为 [0,1] 张量 → 加均匀随机噪声 /256（离散→连续化）→ clamp 到 [0,1] → logit 变换映射到 (-∞, +∞)（与高斯先验对应，降低学习难度）
- **数据流**：输入 x (batch, 784) → 4 层耦合层依次变换 → 缩放层 → 隐变量 z → 高斯先验计算 log_prob → 加 log_det_jacobian 得到 log_likelihood
- **训练目标**：最大化 log likelihood，即最小化 `-mean(log_likelihood)`。损失等价于负对数似然 NLL

## 推理过程

- **推理流程**：采样 z (batch, 784) → 缩放层逆变换 → 4 层耦合层逆变换（按相反顺序）

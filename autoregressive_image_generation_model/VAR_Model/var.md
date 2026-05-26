# VAR 项目概要

> Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction
> NeurIPS 2024 Best Paper

## 1. 模型架构

VAR 采用**两阶段架构**：预训练的 VQVAE 将图像编码为多尺度离散 token，VAR Transformer 以自回归方式逐尺度生成这些 token。

### 1.1 VQVAE（预训练冻结）

- **Encoder**：标准 U-Net 下采样结构，5 层 (`ch_mult=(1,1,2,2,4)`)，下采样率 16×，将 256×256 图像编码为 `[B, 32, 16, 16]` 的连续特征
- **Decoder**：对称的上采样结构，将 `[B, 32, 16, 16]` 还原为 `[B, 3, 256, 256]` 图像
- **VectorQuantizer2**：核心量化器，4096 词表，32 维码本向量

### 1.2 VAR Transformer（训练目标）

- **DiT-style 架构**：depth 层 AdaLN-self-attention block
- **默认配置 (d16)**：depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4
- **10 个生成尺度**：`patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)`，总 token 数 L=1²+2²+…+16²=666
- **核心子模块**：
  - **SelfAttention**：L2-norm attention（Q/K 归一化 + 可学缩放系数），支持 FlashAttention / KV Cache
  - **AdaLNSelfAttn**：每个 block 从类别条件向量推导 6 个 AdaLN 参数 (γ₁, γ₂, scale₁, scale₂, shift₁, shift₂)，控制 LayerNorm + attention + FFN
  - **AdaLNBeforeHead**：输出头前的 AdaLN（2 参数：scale + shift）
  - **FFN**：标准两层 MLP（GELU 激活），支持 fused 算子
- **Attention Mask**：同尺度内双向可见，大尺度可看小尺度全部 token（非逐 token 的因果掩码）
- **Embedding 体系**：
  - `class_emb`：类别条件 [1001, C]（+1 为 CFG 无条件标签）
  - `word_embed`：token embedding [Cvae→C]
  - `lvl_embed`：层级编码 [10, C]（区分不同尺度）
  - `pos_1LC`：绝对位置编码 [1, L, C]
  - `pos_start`：起始信号 [1, 1, C]

## 2. 训练数据与编码过程

### 2.1 数据组织

- **数据集**：ImageNet-1K（1000 类）
- **预处理**：
  - 训练：Resize(1.125×reso) → RandomCrop → ToTensor → [0,1]→[-1,1]（可选 RandomHorizontalFlip）
  - 验证：Resize → CenterCrop → ToTensor → 归一化

### 2.2 编码过程（图像 → 多尺度 token）

VQVAE 的多尺度量化采用**残差量化**策略，从粗到细逐尺度编码：

```
1. 图像 → Encoder → quant_conv → 连续特征 f [B, 32, 16, 16]
2. 初始化: f_rest = f (残差), f_hat = 0 (重建)
3. 逐尺度循环 (si=0..9, pn=1..16):
   a. 将 f_rest 下采样到 pn×pn (area 插值，最后一尺度不插值)
   b. 最近邻查找码本 → 离散 token ID idx [B, pn²]
   c. ID → embedding → 上采样回 16×16 (bicubic，最后一尺度不插值)
   d. 经 Phi 卷积（quant_resi=0.5 控制: 0.5×原始 + 0.5×卷积）
   e. f_hat += h (累加重建), f_rest -= h (更新残差)
4. 返回各尺度 token ID 列表: [[B,1], [B,4], [B,9], ..., [B,256]]
```

### 2.3 VAR 训练输入构建（idxBl_to_var_input）

将 ground truth token ID 转为 teacher-forcing 输入：

```
1. 初始化 f_hat = 0 [B, 32, 16, 16]
2. 逐尺度 (si=0..8):
   a. 当前尺度 ID → embedding → 上采样到 16×16 → Phi 卷积 → 累加到 f_hat
   b. 将 f_hat 下采样到下一尺度 pn_next×pn_next (area) → [B, pn_next², 32]
3. 拼接所有尺度的输入 → x_BLCv_wo_first_l [B, L-1, 32]
   (不含第一个尺度的 1×1 token，它由 SOS 替代)
```

### 2.4 训练目标

VAR 的训练目标是**逐尺度预测下一个尺度的全部 token**：

- 输入：类别标签 + 截止当前尺度已重建的 token 序列（teacher forcing）
- 输出：所有尺度 token 的 4096-way 分类 logits `[B, L, 4096]`
- 每个尺度的 token 可以看到更小尺度全部 token（通过 attention mask），但看不到同尺度或更大尺度的 token

## 3. 训练损失函数

**CrossEntropyLoss**（带可选 label smoothing，默认 0）：

$$\text{loss} = \frac{1}{L} \sum_{i=1}^{L} \text{CE}(\text{logits}_i, \text{gt\_idx}_i)$$

- **全阶段训练**（默认）：每个 token 的损失权重均等（1/L）
- **渐进式训练**（可选，pg>0）：新加入尺度的损失权重通过 warmup 系数从 0.01 渐增到 1

**CFG 训练**：10% 概率丢弃类别标签（替换为 `num_classes=1000`），为推理时 Classifier-Free Guidance 提供条件

**优化器**：AdamW (β₁=0.9, β₂=0.95, fused)，梯度裁剪=2，线性 warmup + 线性衰减 LR 调度

**学习率**：`tlr = tblr × (glb_batch_size / 256)`，默认 `1e-4 × 768/256 = 3e-4`

## 4. 推理与解码过程

### 4.1 VAR 自回归推理（autoregressive_infer_cfg）

```
1. 初始化: f_hat = 0 [B, 32, 16, 16], 构建条件向量 (CFG 双 Batch)
2. 开启所有 block 的 KV Cache
3. 逐尺度循环 (si=0..9):
   a. 上一尺度输出 → word_embed + lvl_embed + pos_embed → next_token_map
   b. Transformer 前向传播（用 KV Cache，仅处理当前尺度 token）
   c. 计算 logits [2B, pn², 4096]
   d. CFG: logits = (1+t)×logits_cond - t×logits_uncond (t = cfg × ratio，细尺度更强)
   e. top-k/top-p 采样 → idx_Bl [B, pn²]
   f. ID → embedding → 调整形状为 [B, 32, pn, pn]
   g. get_next_autoregressive_input:
      - embedding 上采样到 16×16 → Phi 卷积 → 累加到 f_hat
      - f_hat 下采样到下一尺度 → 作为下一尺度的输入
4. 关闭 KV Cache
5. f_hat → post_quant_conv → Decoder → 图像 [B, 3, 256, 256]，[-1,1]→[0,1]
```

### 4.2 解码过程（token → 图像）

```
f_hat [B, 32, 16, 16] → post_quant_conv → Decoder (5层上采样) → 图像 [B, 3, 256, 256]
```

Decoder 是标准 U-Net 上采样结构：conv_in → mid_block → 逐层上采样（ResnetBlock + 可选 AttnBlock + Upsample2x）→ norm_out → conv_out → SiLU → `[B, 3, H, W]`

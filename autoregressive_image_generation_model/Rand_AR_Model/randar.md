# RandAR 项目文档

RandAR（Random-order Autoregressive）是一种**随机顺序自回归**图像生成模型。与传统光栅扫描逐 token 生成不同，RandAR 以随机顺序生成图像 token，并通过**位置指令向量**显式指定每个 token 的空间位置，从而支持推理时的**并行解码**（88步完成256 token，而非256步逐个生成）。

---

## 模型架构与核心组件

整体为 **Decoder-only GPT**（LLaMA-style Transformer），输入序列组织为：

```
[Class Token, Instruct₀, x₀, Instruct₁, x₁, ..., Instruct₂₅₅, x₂₅₅]
```

位置指令向量（Instruct）与图像 token（x）**交替插入**，每个 Instruct 告诉模型"下一步要预测图像中哪个位置的 token"。

核心组件：

- **Label Embedder**：将 ImageNet 类别标签嵌入为向量，训练时 10% 概率替换为空类（支持 CFG）
- **Token Embedding**：将 VQVAE 离散 token（0–16383）嵌入为向量
- **Position Instruction Embedding**：**RandAR 的核心创新**——一个可学习向量 `pos_instruct_embeddings`，repeat 后通过 2D ROPE 编码变成指向不同空间位置的指令向量。因为生成顺序是随机的，不能像传统 AR 模型靠序列位置隐含空间位置，必须额外用指令向量显式告知
- **Transformer Blocks**：RMSNorm → Attention（2D ROPE）→ SwiGLU FFN → DropPath，标准 LLaMA block
- **Output Layer**：线性层 → vocab_size=16384，**零初始化**
- **2D ROPE**：将 head_dim 分成两半分别编码 x/y 两个空间维度，Class Token 的 ROPE 设为全零

模型配置：XL (dim=1280, 36层, 20头, 0.7B) / L (dim=1024, 24层, 16头, 0.3B)，block_size=256。

---

## 训练数据组织与编码

**编码过程**：训练前用 VQVAE tokenizer 将 ImageNet 图像预编码为离散 token codes。

```
256×256 RGB → Encoder → 16×16×256 → quant_conv → 16×16×8 → VectorQuantizer → 256 个离散 indices (0-16383)
```

VQVAE（`vq_ds16_c2i.pt`）结构：Encoder（U-Net, 通道倍数[1,1,2,2,4], 4次下采样）→ quant_conv(256→8) → VectorQuantizer(codebook=16384, L2归一化+最近邻查找) → post_quant_conv(8→256) → Decoder（对称U-Net上采样回256×256）。

数据增强支持 adm（水平翻转）或 ten-crop，每个样本保存为 `.npy` 文件 `(aug_num, 256)`，按类别分目录存放。

**训练数据加载**：`INatLatentDataset` 直接加载 `.npy` latent codes，每步随机选择一种增强版本，返回 `(latents, label, index)`。

---

## 训练目标与损失函数

**训练目标**：给定类别标签和随机顺序的图像 token 序列，预测每个位置应生成的 token ID。

**损失函数**：交叉熵损失（Cross Entropy）。

```python
# 训练前向：
# 1. 每个样本独立 shuffle token 顺序 (torch.randperm)
# 2. 按 token_order 重排 idx 和 targets
# 3. 构建序列 h = [Class, Instruct₀, x₀, Instruct₁, x₁, ...]（交替插入）
# 4. Transformer 前向 → logits
# 5. 取图像 token 位置的 logits: token_logits = logits[:, cls_token_num::2]
# 6. loss = CE(token_logits, shuffled_targets)
```

Teacher-forcing 训练，每步随机顺序不同，使模型学会在任意顺序下生成图像。

---

## 推理与解码过程

**解码过程**：生成的 256 个 token indices 通过 VQVAE 解码回图像。

```
256 indices → get_codebook_entry(码本查表) → 16×16×8 量化特征 → post_quant_conv → 16×16×256 → Decoder上采样 → 256×256 RGB → 反归一化(results*127.5+128, clamp[0,255])
```

**推理流程**（`generate()` 方法）：

1. 生成随机 token_order（每个样本独立 shuffle）
2. 准备位置指令向量 + 2D ROPE 编码
3. 开启 KV Cache，CFG 时 batch 翻倍为 [cond, uncond]
4. **并行解码循环**：

```
Step0: 输入[Class, Query₀] → 预测1 token → img₀
Step1: 输入[img₀, Query₁] → 预测1 token → img₁  
Step2: 输入[img₁, Query₂, Query₃] → 预测2 token → img₂,img₃
...  余弦调度逐步增加并行度，88步完成256 token
```

import torch
from torch import nn as nn
from torch.nn import functional as F


def sample_with_top_k_top_p_(
        logits_BlV: torch.Tensor,
        top_k: int = 0,
        top_p: float = 0.0,
        rng=None,
        num_samples=1
) -> torch.Tensor:  # return idx, shaped (B, l, num_samples)
    # ===================== 1. 解析输入张量形状 =====================
    B, l, V = logits_BlV.shape  # B=批次, l=序列长度, V=码本大小(4096)

    # ===================== 2. Top-K 采样（保留概率最高的 K 个 token） =====================
    if top_k > 0:
        # 步骤1：取每个位置 概率最大的 top_k 个 logit 值
        # topk(...) 返回 (值, 索引)，我们只需要值 [B, l, top_k]
        topk_vals = logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0]
        # 步骤2：求这 top_k 个值的**最小值**，作为筛选阈值 [B, l, 1]
        threshold = topk_vals.amin(dim=-1, keepdim=True)
        # 步骤3：标记所有 小于阈值 的 logit（这些是低概率token）
        idx_to_remove = logits_BlV < threshold
        # 步骤4：原地将低概率token的logit置为 -∞（softmax后概率=0，不会被采样）
        logits_BlV.masked_fill_(idx_to_remove, -torch.inf)

    # ===================== 3. Top-P (核采样)（保留累积概率≥P的最小token集合） =====================
    if top_p > 0:
        # 步骤1：将logits按**概率从低到高**排序（descending=False），返回 排序后logits + 原始索引
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        # 步骤2：对排序后的logits做softmax，得到概率分布，再**累积求和**（从低概率到高概率累加）
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum_(dim=-1)
        # 步骤3：标记 累积概率 ≤ (1-top_p) 的token（低概率，需要过滤）
        # 例：top_p=0.9 → 保留累积概率≥90%的token，过滤前10%的低概率token
        sorted_idx_to_remove = cumulative_probs <= (1 - top_p)
        # 步骤4：强制保留最后1个token（避免所有token都被过滤，导致采样报错）
        sorted_idx_to_remove[..., -1:] = False
        # 步骤5：将「排序后的mask」映射回「原始logits的位置」（核心：scatter）
        scatter_mask = sorted_idx_to_remove.scatter(
            sorted_idx.ndim - 1,  # 在最后一维（词表维）散射
            sorted_idx,  # 排序后的原始索引
            sorted_idx_to_remove  # 排序后的mask
        )
        # 步骤6：原地将过滤掉的token logit置为 -∞
        logits_BlV.masked_fill_(scatter_mask, -torch.inf)

    # ===================== 4. 最终随机采样（从过滤后的logits中选token） =====================
    # 采样规则：num_samples为负 → 不重复采样；为正 → 可重复采样
    replacement = num_samples >= 0
    num_samples = abs(num_samples)  # 取绝对值，统一采样数量

    # 关键：torch.multinomial 只支持2D张量，所以把 [B,l,V] 展平为 [B*l, V]
    # 1. logits做softmax转概率  2. 展平  3. 多项式采样
    sampled_idx = torch.multinomial(
        logits_BlV.softmax(dim=-1).view(-1, V),
        num_samples=num_samples,  # 每个位置采样1个token
        replacement=replacement,  # 是否可重复采样
        generator=rng  # 随机数生成器（保证可复现）
    )

    # 把展平的 [B*l, num_samples] 恢复为原始形状 [B, l, num_samples]
    return sampled_idx.view(B, l, num_samples)

def gumbel_softmax_with_rng(logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1, rng: torch.Generator = None) -> torch.Tensor:
    """
    Gumbel 分布是用来描述"一系列随机变量中最大值"的分布: G = -log(-log(U)),  U ~ Uniform(0, 1)
    属于偏右的分布（峰值偏左，长尾向右），均值≈0.577，方差≈1.64

    如果想从概率分布 (p₁, p₂, ..., p_K) 中按概率随机采样一个类别，可以通过对每个概率分布增加Gumbel分布的随机值，然后取最大值，等价于随机采样
    sample = argmax_i [ log(p_i) + G_i ]

    但是argmax非连续，可以通过使用softmax中/tau，tau越小，softmax结果越尖锐
    y_soft = softmax( (log(p_i) + G_i) / tau )
    """
    
    if rng is None:
        return F.gumbel_softmax(logits=logits, tau=tau, hard=hard, eps=eps, dim=dim)
    
    # torch.empty_like(logits) — 创建和 logits 同形状的空 tensor
    # .exponential_(generator=rng) — 用 rng 生成一个均匀分布 U(0,1) 的随机数，计算X=−ln(U)，那么X就服从 Exp(1) 分布（即.exponential_），用X填充
    # .log() — 取 log → log(X) = log(-log(U))
    # 外面加负号 - → -log(-log(U)) = 标准 Gumbel(0,1)
    gumbels = (-torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_(generator=rng).log())
    # 加gumbels噪声 和 除tau
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)
    
    if hard:
        index = y_soft.max(dim, keepdim=True)[1] # [0]代表values， [1]代表取下标，index维度是[B, l, 1]
        # scatter例子，根据index的最后维度的数字，对零矩阵具体位置设置为1，即one_hot
        # index = [[2], [0]]     # 形状 [1, 2, 1]

        # y_hard 初始: zeros [1, 2, 4]
        # [[0, 0, 0, 0],
        #  [0, 0, 0, 0]]

        # scatter_(dim=-1, index, 1.0):
        # 位置[0,0,2] → 1.0  (index[0,0,0]=2)
        # 位置[0,1,0] → 1.0  (index[0,1,0]=0)

        # y_hard:
        # [[0, 0, 1, 0],       # token 0 → class 2
        #  [1, 0, 0, 0]]       # token 1 → class 0
        # → 形状 [1, 2, 4]     # one-hot encoding!
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0) 
        ret = y_hard - y_soft.detach() + y_soft # Straight-Through Estimator (STE) 
    else:
        ret = y_soft
    return ret


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    # 1. 推理模式 / 不丢弃：直接返回原张量
    if drop_prob == 0. or not training: return x
    # 2. 计算保留概率
    keep_prob = 1 - drop_prob
    # 3. 生成掩码形状：仅 Batch 维度独立，其余维度全为 1
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    # 4. 生成 0/1 随机掩码（1=保留，0=丢弃）
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    # 5. 数值归一化：保证训练/推理的数值期望一致
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    # 6. 输入张量 × 掩码 = 最终结果
    return x * random_tensor


class DropPath(nn.Module):  # taken from timm
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
    def extra_repr(self):
        return f'(drop_prob=...)'

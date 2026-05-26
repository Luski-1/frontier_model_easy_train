"""
精简版 VAR 项目 VAR 模型模块
合并原文件：models/var.py + models/basic_var.py + models/helpers.py(采样+DropPath部分)
改动：去掉 VARHF，去掉 flash_attn/xformers 多分支只保留标准实现(torch scaled_dot_product_attention)，
      去掉分布式代码，去掉 fused_mlp/fused_add_norm，去掉 prog_si 分支，去掉 torch.compile
"""

import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from new_args import PATCH_NUMS, NUM_CLASSES
from new_vqvae import VQVAE, VectorQuantizer2


# ===================== helpers.py 的 DropPath + 采样函数 =====================

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
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'(drop_prob=...)'


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


# ===================== basic_var.py 的核心组件 =====================

def slow_attn(query, key, value, scale: float, attn_mask=None, dropout_p=0.0):
    """标准 attention 实现（使用 PyTorch scaled_dot_product_attention）"""
    # 使用 PyTorch 内置的 scaled_dot_product_attention
    return F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=attn_mask,
        dropout_p=dropout_p if dropout_p > 0 else 0.0,
        scale=scale
    )


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features  # 1024
        hidden_features = hidden_features or in_features  # 1024 * 4
        # 维度[1024, 1024*4]
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')  # x * cdf(x); cdf累积分布函数 =  1/2 * [1 + erf(x/sqrt(2))]; erf误差函数，可使用tanh快速模拟
        # 维度[1024*4, 1024]
        self.fc2 = nn.Linear(hidden_features, out_features)
        # 默认0
        self.drop = nn.Dropout(drop, inplace=True) if drop > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class SelfAttention(nn.Module):
    def __init__(
            self, block_idx, embed_dim=768, num_heads=12,
            attn_drop=0., proj_drop=0., attn_l2_norm=False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        # self.num_heads = 16
        # self.head_dim = 1024 / 16 = 64
        self.block_idx, self.num_heads, self.head_dim = block_idx, num_heads, embed_dim // num_heads
        # True
        self.attn_l2_norm = attn_l2_norm

        if self.attn_l2_norm:
            # 自主缩放Attention 公式是softmax(s * norm(Q) @ norm(K^T) / scale) V
            # scale = 1 ，存在科学系的Q的多头独立的缩放系数
            self.scale = 1
            # Q的多头独立的缩放系数[1, H, 1, 1], ln(4)≈1.386
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(),
                                               requires_grad=True)
            # 限制缩放系数最大不超过 100
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            # 常规Attention：softmax(QK^T / sqrt(head_dim)) V
            # 再缩小1/4，猜测目的是训练初期进一步降低softmax的方差
            self.scale = 0.25 / math.sqrt(self.head_dim)

        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        # 自定义偏置：Q和V有可学习bias，K固定为0
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        # W_O
        self.proj = nn.Linear(embed_dim, embed_dim)
        # W_O 的dropout概率 默认0
        self.proj_drop = nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        # Attention 的dropout概率
        self.attn_drop: float = attn_drop

        # KV cache，仅当推理阶段开启
        self.caching, self.cached_k, self.cached_v = False, None, None

    def kv_caching(self, enable: bool):
        # 开启kv cache
        self.caching, self.cached_k, self.cached_v = enable, None, None

    def forward(self, x, attn_bias):
        """
        Args:
            x: [B, L, C]
            attn_bias: attention mask，推理时为None（因为kv cache）
        """
        B, L, C = x.shape
        qkv = F.linear(input=x,
                       weight=self.mat_qkv.weight,
                       bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))).view(B, L, 3, self.num_heads,
                                                                                          self.head_dim)

        # 统一使用 BHLc 格式（适配 scaled_dot_product_attention）
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)  # [3, B, L, H, c] > [B, H, L, c]
        dim_cat = 2

        if self.attn_l2_norm:
            # 1. 缩放系数裁剪后 exp 还原
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            # 2. Q 归一化 + 缩放
            q = F.normalize(q, dim=-1).mul(scale_mul)
            # 3. K 只归一化
            k = F.normalize(k, dim=-1)

        # 开启KV cache
        if self.caching:
            if self.cached_k is None:
                self.cached_k = k
                self.cached_v = v
            else:
                # 存放入kv cache，并更新当前kv
                k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat)
                v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)

        dropout_p = self.attn_drop if self.training else 0.0

        # 使用 PyTorch 内置的 scaled_dot_product_attention（标准实现）
        oup = slow_attn(query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias,
                        dropout_p=dropout_p).transpose(1, 2).reshape(B, L, C)

        return self.proj_drop(self.proj(oup))

    def extra_repr(self) -> str:
        return f'using_sdpa=True, attn_l2_norm={self.attn_l2_norm}'


class AdaLNSelfAttn(nn.Module):
    """每个 block 有自己的 AdaLN（shared_aln=False 是默认配置，不保留 shared_aln=True 分支）"""
    def __init__(
            self, block_idx, last_drop_p, embed_dim, cond_dim, norm_layer,
            num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., attn_l2_norm=False,
    ):
        super().__init__()
        self.block_idx, self.last_drop_p, self.C = block_idx, last_drop_p, embed_dim
        self.C, self.D = embed_dim, cond_dim
        # 第一层不开启层(深度)dropout
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = SelfAttention(block_idx=block_idx,
                                  embed_dim=embed_dim,
                                  num_heads=num_heads,
                                  attn_drop=attn_drop,
                                  proj_drop=drop,
                                  attn_l2_norm=attn_l2_norm)
        self.ffn = FFN(in_features=embed_dim,
                       hidden_features=round(embed_dim * mlp_ratio),
                       drop=drop)
        # ln_wo_grad = layerNorm，但取消缩放和偏移，由adaLN接管缩放和偏移
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        lin = nn.Linear(cond_dim, 6 * embed_dim)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin)

    def forward(self, x, cond_BD, attn_bias):
        # C: token embedding dim
        # D: condition embedding dim
        # cond_BD: condition embedding 分类信息
        # attn_bias: attention mask

        # [B,1,C] @ [C,6C] > [B,1,6C] > [B,1,6,C] > 6[B,1,C]
        gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)
        x = x + self.drop_path( # 层(深度)dropout，为数不多的非0 dropout，比attention和FFN里dropout更靠后
            self.attn(
                self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1), # 内部缩放+偏移，adaln
                attn_bias=attn_bias
            ).mul_(gamma1)  # 外部缩放
        )
        x = x + self.drop_path(
            self.ffn(
                self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2)).mul(gamma2)
        )
        return x


class AdaLNBeforeHead(nn.Module):
    def __init__(self, C, D, norm_layer):
        super().__init__()
        self.C, self.D = C, D  # 1024 1024
        self.ln_wo_grad = norm_layer(C, elementwise_affine=False)  # layerNorm 关闭仿射变换
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(D, 2 * C))

    def forward(self, x_BLC: torch.Tensor, cond_BD: torch.Tensor):
        # [B,D] > [B,2C] > [B,1,2,C] > 2[B,1,C]
        scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.C).unbind(2)
        return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add_(shift)


# ===================== var.py 的 VAR 主模型 =====================

class VAR(nn.Module):
    def __init__(
            self,
            vae_local: VQVAE,
            num_classes=NUM_CLASSES,
            depth=16,
            embed_dim=1024,
            num_heads=16,
            mlp_ratio=4.,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_eps=1e-6,
            cond_drop_rate=0.1,
            attn_l2_norm=False,
            patch_nums=PATCH_NUMS,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0
        # self.Cvae = 32, VAE的latent维度
        # self.V = 4096 VAE词表
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        # self.depth = 16
        # self.C = 1024,  token embedding dim
        # self.D = 1024, condition embedding dim
        # self.num_heads = 16
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        # self.cond_drop_rate = 0.1 CFG概率
        self.cond_drop_rate = cond_drop_rate
        # self.patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        self.patch_nums: Tuple[int] = patch_nums
        # self.L(length) = sum(1, 4, 9, 16, 25, 36, 64, 100, 169, 256) 计算预测的总token数
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        # self_first_l = 1 第1个尺寸/阶段/stride需要预测的token数
        self.first_l = self.patch_nums[0] ** 2
        # self.begin_ends = [(0,1), (1, 5)...] 每stride的区间下标
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur + pn ** 2))
            cur += pn ** 2
        # self.num_stages_minus_1 = 9 用于推理阶段计算当前阶段在全阶段的占比
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        # 随机生成器
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.rng = torch.Generator(device=device)

        # input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        # [32, 1024]
        self.word_embed = nn.Linear(self.Cvae, self.C)

        # 计算初始化方差
        # 目的是让总方差=1，每一个token的输入包含3部分：token embedding、stride/level embedding、 absolute position embedding，因此需要/3
        init_std = math.sqrt(1 / self.C / 3)
        # 类别数量 1000
        self.num_classes = num_classes
        # 维度[1, 1000]的0.001 用于推理阶段没有显式传入分类标签时进行抽样
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32,
                                       device=device)
        # [1001, 1024] 类别Embedding，+1代表新增'无分类'
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        # [1, 1, 1024] 起始token
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)

        # absolute position embedding
        # self.pos_1LC的维度 [1, L(length), C]
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn * pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1) # 维度[1, L(length), C]
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)

        # level embedding
        # self.lvl_embed 维度[10(level,即stride阶段), 1024]
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate  # 层(深度)dropout 0.1 * depth / 24 ≈ 0.066
        # dpr：层(深度)dropout的线性递增
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # DIT风格的adaLN（每个block有自己的ada_lin）
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D,
                block_idx=block_idx,
                embed_dim=self.C,
                norm_layer=norm_layer,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,    # 4 控制FFN的中间维度的倍数
                drop=drop_rate, # 0 linear 的drouput概率，例如W_O, FFN
                attn_drop=attn_drop_rate,   # 0 Attention 的dropout概率
                drop_path=dpr[block_idx],
                last_drop_p=0 if block_idx == 0 else dpr[block_idx - 1],    # 整个流程没使用到
                attn_l2_norm=attn_l2_norm,
            )
            for block_idx in range(depth)
        ])

        print(
            f'\n[constructor]  ==== VAR config ==== \n'
            f'    embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )

        # 获得Attention mask，使得相同尺寸/stride内token互相看见，大尺寸可以看见小尺寸
        # Attention mask不会应用到推理阶段，因为开启了KV cache
        d: torch.Tensor = torch.cat(
            [torch.full((pn * pn,), i) for i, pn in enumerate(self.patch_nums)]
        ).view(1, self.L, 1)
        dT = d.transpose(1, 2)
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        # d [1, L, 1] 和 dT [1, 1, L] 会自动广播为 [1, L, L]，参考以下mask矩阵
        # 0 - -
        # - 0 0
        # - 0 0
        # 维度[1, 1, L, L]
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())

        # classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)  # 预测头，预测稀疏token，4096

    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                   cond_BD: Optional[torch.Tensor]):
        if isinstance(h_or_h_and_residual, tuple):
            h = h_or_h_and_residual[1] + self.blocks[-1].drop_path(h_or_h_and_residual[0])
        else:
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()

    @torch.no_grad()
    def autoregressive_infer_cfg(
            self,
            B: int,
            label_B: Optional[Union[int, torch.LongTensor]],
            g_seed: Optional[int] = None,
            cfg=1.5,
            top_k=0,
            top_p=0.0,
            more_smooth=False,
    ) -> torch.Tensor:
        """
        only used for inference, on autoregressive mode
        返回生成的图像 [B, 3, H, W]，范围 [0, 1]
        """
        # ========== 1.1 初始化随机数生成器 ==========
        if g_seed is None:
            rng = None
        else:
            self.rng.manual_seed(g_seed)
            rng = self.rng

        # ========== 1.2 处理生成标签 ==========
        if label_B is None:
            # 标签为 None：随机采样类别（均匀分布）
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            # 标签为 int：指定类别（所有样本都生成这个类别）
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B,
                                 device=self.lvl_1L.device)

        # ========== 2.1 构建 CFG 双 Batch 条件向量 ==========
        # 前 B 个：条件生成；后 B 个：无条件生成
        sos = cond_BD = self.class_emb(
            torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0)
        )

        # ========== 3.1 预计算层级编码 + 绝对位置编码 ==========
        # lvl_pos 形状：[1, L, C]（L=length，C=1024）
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        # ========== 3.2 初始化起始输入 ==========
        # 分类标签信息 + 生成开始信号+ 层级编码 + 绝对编码
        # next_token_map 形状：[2*B, first_l, C]（first_l=1，对应阶段 0的要生成的token数量）
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) \
                         + self.pos_start.expand(2 * B, self.first_l, -1) \
                         + lvl_pos[:, :self.first_l]

        # ========== 3.3 初始化重建特征图 ==========
        cur_L = 0   # 当前已生成的 Token 长度
        # f_hat 形状：[B, Cvae, H, W]（H=W=16，对应encoder输出的latent的维度，也是decoder输入的维度）
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])

        # ========== 4.1 开启 KV Cache ==========
        for b in self.blocks:
            b.attn.kv_caching(True)
        # ========== 5.1 遍历每个尺度（从粗到细） ==========
        for si, pn in enumerate(self.patch_nums):
            ratio = si / self.num_stages_minus_1    # 当前阶段在全阶段的占比，用于计算抽样的温度
            cur_L += pn * pn    # 更新当前已生成的 Token 长度

            x = next_token_map
            for b in self.blocks:
                # 注意：attn_bias=None，因为推理时用 KV Cache，不需要因果掩码
                x = b(x=x, cond_BD=cond_BD, attn_bias=None)

            logits_BlV = self.get_logits(x, cond_BD)    # logits_BlV 形状：[2*B, cur_L, V]（V=4096，码本大小）

            # ========== CFG ==========
            # - logits_BlV[:B]：条件生成的 logits
            # - logits_BlV[B:]：无条件生成的 logits
            # - 公式：logits = (1+t)*logits_cond - t*logits_uncond
            # - 效果：放大条件信号，抑制无条件噪声，生成更贴合类别的图像
            t = cfg * ratio
            logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]

            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]

            if not more_smooth:
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)
            else:
                # gum_t 从0.27 > 0.05
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                # hard控制返回的是硬标签还是软标签
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1,
                                                  rng=rng) @ \
                         self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            # h_BChw 维度[B, Cvae, pn, pn]
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)

            # f_hat是重建信息的累计，next_token_map是下一个尺寸的输入
            # f_hat [B,Cvae(32),H(16),W(16)]
            # next_token_map [B, Cvae(32), pn+1, pn+1)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums),
                                                                                           f_hat, h_BChw)
            
            # 非最后阶段/尺寸/stride才需要整理输入信息
            if si != self.num_stages_minus_1:
                # [B, Cvae(32), pn+1, pn+1] > [B, Cvae(32), pn+1*pn+1], [B,pn+1*pn+1,C)]
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                # [B, pn+1*pn+1, Cvae] > [B, pn+1*pn+1, C(1024)] + 对应长度的层级编码 + 绝对位置编码
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:,
                                                                   cur_L:cur_L + self.patch_nums[si + 1] ** 2]
                # [2B, pn+1*pn+1, C]
                next_token_map = next_token_map.repeat(2, 1, 1)

        for b in self.blocks: b.attn.kv_caching(False)
        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)  # de-normalize, from [-1, 1] to [0, 1]

    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:
        """
        :param label_B: 分类标签，维度[B]
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :return: logits BLV, V is vocab_size
        """
        B = x_BLCv_wo_first_l.shape[0]
        with torch.cuda.amp.autocast(enabled=False):
            # CFG训练的概率
            label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes,
                                  label_B)
            # 构建起始sos embedding
            # 先获取分类向量
            sos = cond_BD = self.class_emb(label_B)
            # 再扩增到first level/stride的token数量的维度，再加上生成开始信号 pos_start embedding
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)

            # x_BLCv_wo_first_l [B, L - 1, 32] > [B, L - 1, 1024]
            # x_BLC [B, L, 1024]
            x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
            # 加上level/stride embedding和absolute position embedding
            x_BLC += self.lvl_embed(self.lvl_1L.expand(B, -1)) + self.pos_1LC

        # 获取对应的mask
        attn_bias = self.attn_bias_for_masking

        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype

        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD = cond_BD.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, cond_BD=cond_BD, attn_bias=attn_bias)
        #  归一化+输出分类，传入的是原始分类信息cond_BD
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)
        return x_BLC

    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02,
                     conv_std_or_gain=0.02):
        # 基于He初始化的变体，保证输入输出方差一致。
        # 输入的embedding，除了token embdding，还有level embedding和absulute embedding，所以/3
        # sqrt(1/(3*1024)) ≈ 0.018，接近手动设置的 0.02
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5

        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()  # padding 位置强制为 0，避免干扰
            elif isinstance(m, (
                    nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm,
                    nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                # 归一化层初始化为恒等变换：y = 1*x + 0，保证训练初期网络行为稳定
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, (
                    nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0:
                    nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else:
                    nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()

        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)  # 输出头权重* 0.02，保证初始预测接近均匀分布
                self.head.bias.data.zero_()

        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()

        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            # 方差选择2 * depth的原因：每一层都会增加方差，并且每一层有attn和ffn
            # W_O
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            # FFN
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2 * self.C:].mul_(init_adaln)  # 后 2C 部分（shift/scale）
                sab.ada_lin[-1].weight.data[:2 * self.C].mul_(init_adaln_gamma)  # 前 2C 部分（gamma）
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()

    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'


# ===================== 构建 VQ-VAE + VAR 双模型 =====================

def build_vae_var(
        device,
        patch_nums=PATCH_NUMS,
        V=4096,
        Cvae=32,
        ch=160,
        share_quant_resi=4,
        num_classes=NUM_CLASSES,
        depth=16,
        attn_l2_norm=True,
        init_adaln=0.5,
        init_adaln_gamma=1e-5,
        init_head=0.02,
        init_std=-1,
) -> Tuple[VQVAE, VAR]:
    # 自动计算VAR模型的核心超参
    heads = depth
    width = depth * 64
    dpr = 0.1 * depth / 24  # Drop Path衰减率

    # 禁用PyTorch默认参数初始化（加速模型构建）
    for clz in (nn.Linear, nn.LayerNorm, nn.BatchNorm2d, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d,
                nn.ConvTranspose2d):
        setattr(clz, 'reset_parameters', lambda self: None)

    # 步骤1：构建 VQ-VAE
    vae_local = VQVAE(
        vocab_size=V,
        z_channels=Cvae,
        ch=ch,
        test_mode=True,
        share_quant_resi=share_quant_resi,
        v_patch_nums=patch_nums
    ).to(device)

    # 步骤2：构建 VAR
    var = VAR(
        vae_local=vae_local,
        num_classes=num_classes,
        depth=depth,
        embed_dim=width,
        num_heads=heads,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=dpr,
        norm_eps=1e-6,
        cond_drop_rate=0.1,
        attn_l2_norm=attn_l2_norm,
        patch_nums=patch_nums,
    ).to(device)

    # 步骤3：手动初始化VAR权重
    var.init_weights(
        init_adaln=init_adaln,
        init_adaln_gamma=init_adaln_gamma,
        init_head=init_head,
        init_std=init_std
    )

    return vae_local, var
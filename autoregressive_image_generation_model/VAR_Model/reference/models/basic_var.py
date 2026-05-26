import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.helpers import DropPath, drop_path

# this file only provides the 3 blocks used in VAR transformer
__all__ = ['FFN', 'AdaLNSelfAttn', 'AdaLNBeforeHead']

# automatically import fused operators
dropout_add_layer_norm = fused_mlp_func = memory_efficient_attention = flash_attn_func = None
try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
    from flash_attn.ops.fused_dense import fused_mlp_func
except ImportError:
    pass
# automatically import faster attention implementations
try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    pass
try:
    from flash_attn import flash_attn_func  # qkv: BLHc, ret: BLHcq
except ImportError:
    pass
try:
    from torch.nn.functional import scaled_dot_product_attention as slow_attn  # q, k, v: BHLc
except ImportError:
    def slow_attn(query, key, value, scale: float, attn_mask=None, dropout_p=0.0):
        attn = query.mul(scale) @ key.transpose(-2, -1)  # BHLc @ BHcL => BHLL
        if attn_mask is not None: 
            attn.add_(attn_mask)
        if dropout_p > 0:
            return F.dropout(
                attn.softmax(dim=-1), 
                p=dropout_p, 
                inplace=True) @ value
        else:
            return attn.softmax(dim=-1) @ value

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., fused_if_available=True):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if fused_if_available else None
        out_features = out_features or in_features  # 1024
        hidden_features = hidden_features or in_features  # 1024 * 4
        # 维度[1024, 1024*4]
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(
            approximate='tanh')  # x * cdf(x); cdf累积分布函数 =  1/2 * [1 + erf(x/sqrt(2))]; erf误差函数，可使用tanh快速模拟
        # 维度[1024*4, 1024]
        self.fc2 = nn.Linear(hidden_features, out_features)
        # 默认0
        self.drop = nn.Dropout(drop, inplace=True) if drop > 0 else nn.Identity()

    def forward(self, x):
        if self.fused_mlp_func is not None:
            return self.drop(self.fused_mlp_func(
                x=x, weight1=self.fc1.weight, weight2=self.fc2.weight, bias1=self.fc1.bias, bias2=self.fc2.bias,
                activation='gelu_approx', save_pre_act=self.training, return_residual=False, checkpoint_lvl=0,
                heuristic=0, process_group=None,
            ))
        else:
            return self.drop(self.fc2(self.act(self.fc1(x))))

    def extra_repr(self) -> str:
        return f'fused_mlp_func={self.fused_mlp_func is not None}'


class SelfAttention(nn.Module):
    def __init__(
            self, block_idx, embed_dim=768, num_heads=12,
            attn_drop=0., proj_drop=0., attn_l2_norm=False, flash_if_available=True,
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
            # scale = 1 ，存在科学系的Q的多头独立的缩放系数，因此scale=1，让模型自主学习s即可
            self.scale = 1
            # Q的多头独立的缩放系数[1, H, 1, 1], ln(4)≈1.386
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(),
                                               requires_grad=True)
            # 限制缩放系数最大不超过 100, ln(100)≈4.6
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            # 常规Attention：softmax(QK^T / sqrt(head_dim)) V
            # TODO 理论上是1/sqrt(head_dim)，现在再缩小1/4，猜测目的是训练初期进一步降低softmax的方差，使得注意力分数更加平滑？
            self.scale = 0.25 / math.sqrt(self.head_dim)

        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        # 自定义偏置：Q和V有可学习bias，K固定为0
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        # W_O
        self.proj = nn.Linear(embed_dim, embed_dim)
        # W_O 的dropout概率 默认0
        self.proj_drop = nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()  # proj_drop=0
        # Attention 的dropout概率
        self.attn_drop: float = attn_drop  # 0
        self.using_flash = flash_if_available and flash_attn_func is not None
        self.using_xform = flash_if_available and memory_efficient_attention is not None

        # only used during inference
        # KV cache，仅当推理阶段开启
        self.caching, self.cached_k, self.cached_v = False, None, None

    def kv_caching(self, enable: bool):
        # 开启kv cache
        self.caching, self.cached_k, self.cached_v = enable, None, None

    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(self, x, attn_bias):
        """
        Args:
            x: 
            attn_bias: 就是attn_mask，名字有点误导
        """

        B, L, C = x.shape  # L是当前尺寸/stride的总token数(length)，相同分块内是双向attn
        # qkv: 维度[B, L, 3, H, c]
        qkv = F.linear(input=x, 
                       weight=self.mat_qkv.weight, 
                       bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))).view(B, L, 3, self.num_heads, self.head_dim)
        main_type = qkv.dtype
        
        # 仅限推理时使用using_flash
        using_flash = self.using_flash and attn_bias is None and qkv.dtype != torch.float32
        if using_flash or self.using_xform:
            q, k, v = qkv.unbind(dim=2)
            dim_cat = 1  # q or k or v: BLHc
        else:
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)  # [B, L, 3, H, c] > [3, B, H, L, c] > [B, H, L, c]
            dim_cat = 2  # q or k or v: BHLc

        if self.attn_l2_norm:
            # 1. 每个注意力头独立的缩放系数经过裁剪后，再exp还原
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            if using_flash or self.using_xform:
                scale_mul = scale_mul.transpose(1, 2)  # 1H11 to 11H1
            # 2. Q 归一化 + 缩放
            q = F.normalize(q, dim=-1).mul(scale_mul)
            # 3. K 只归一化，不缩放
            k = F.normalize(k, dim=-1)
        # 开启KV cache
        if self.caching:
            if self.cached_k is None:
                self.cached_k = k
                self.cached_v = v
            else:
                # 存放入kv cache，并更新当前kv
                k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat);
                v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)
        # attn_drop = 0
        dropout_p = self.attn_drop if self.training else 0.0
        if using_flash:
            oup = flash_attn_func(q.to(dtype=main_type), k.to(dtype=main_type), v.to(dtype=main_type),
                                  dropout_p=dropout_p, softmax_scale=self.scale).view(B, L, C)
        elif self.using_xform:
            oup = memory_efficient_attention(q.to(dtype=main_type), k.to(dtype=main_type), v.to(dtype=main_type),
                                             attn_bias=None if attn_bias is None else attn_bias.to(
                                                 dtype=main_type).expand(B, self.num_heads, -1, -1), p=dropout_p,
                                             scale=self.scale).view(B, L, C)
        else:
            # attention计算的默认实现工程，默认不开启attention dropout
            oup = slow_attn(query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias,
                            dropout_p=dropout_p).transpose(1, 2).reshape(B, L, C)

        return self.proj_drop(self.proj(oup))
        # attn = (q @ k.transpose(-2, -1)).add_(attn_bias + self.local_rpb())  # BHLc @ BHcL => BHLL
        # attn = self.attn_drop(attn.softmax(dim=-1))
        # oup = (attn @ v).transpose_(1, 2).reshape(B, L, -1)     # BHLL @ BHLc = BHLc => BLHc => BLC

    def extra_repr(self) -> str:
        return f'using_flash={self.using_flash}, using_xform={self.using_xform}, attn_l2_norm={self.attn_l2_norm}'


class AdaLNSelfAttn(nn.Module):
    def __init__(
            self, block_idx, last_drop_p, embed_dim, cond_dim, shared_aln: bool, norm_layer,
            num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., attn_l2_norm=False,
            flash_if_available=False, fused_if_available=True,
    ):
        super(AdaLNSelfAttn, self).__init__()
        # self.last_drop_p: 0 if block_idx == 0 else dpr[block_idx - 1] TODO
        # self.C: 1024 
        self.block_idx, self.last_drop_p, self.C = block_idx, last_drop_p, embed_dim
        # self.C: 1024 token embedding channel
        # self.D: 1024 condition embedding channel
        self.C, self.D = embed_dim, cond_dim
        # 第一层不开启层(深度)dropout
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = SelfAttention(block_idx=block_idx,
                                  embed_dim=embed_dim,  # 1024
                                  num_heads=num_heads,  # 16
                                  attn_drop=attn_drop,  # 0
                                  proj_drop=drop,  # 0
                                  attn_l2_norm=attn_l2_norm,  # True
                                  flash_if_available=flash_if_available)
        self.ffn = FFN(in_features=embed_dim,  # 1024
                       hidden_features=round(embed_dim * mlp_ratio),  # 1024*4
                       drop=drop,  # 0
                       fused_if_available=fused_if_available)
        # self.ln_wo_grad = layerNorm，但取消缩放和偏移，由adaLN接管缩放和偏移
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.shared_aln = shared_aln  # 大多数情况下shared_aln=False，当训练512*512时即开启共享adaln
        if self.shared_aln:
            # 设置为共享参数，减少参数量
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim ** 0.5)
        else:
            # 设置为矩阵
            lin = nn.Linear(cond_dim, 6 * embed_dim)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin)

        self.fused_add_norm_fn = None

    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(self, x, cond_BD, attn_bias):  
        # C: token embedding dim
        # D: condition embedding dim
        # cond_BD: condition embedding 分类信息
        # attn_bias: attention mask

        if self.shared_aln:
            # 直接拆分
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(
                2)  # 116C + B16C =unbind(2)=> 6 B1C
        else:
            # [B,1,C] @ [C,6C] > [B,1,6C] > [B,1,6,C] > 6[B,1,C]
            gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)
        x = x + self.drop_path( # 层(深度)dropout，为数不多的非0 dropout，比attention和FFN里dropout更靠后
            self.attn(
                self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1), # 内部缩放+偏移，adaln
                attn_bias=attn_bias
            ).mul_(gamma1) # 外部缩放
        )
        x = x + self.drop_path(
            self.ffn(
                self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2)).mul(gamma2)
        )  # this mul(gamma2) cannot be in-placed when FusedMLP is used
        return x

    def extra_repr(self) -> str:
        return f'shared_aln={self.shared_aln}'


class AdaLNBeforeHead(nn.Module):
    def __init__(self, C, D, norm_layer):  # C: token embedding dim, D: condition embedding dim
        super().__init__()
        self.C, self.D = C, D # 1024 1024
        self.ln_wo_grad = norm_layer(C, elementwise_affine=False) # layerNorm 关闭仿射变换
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(D, 2 * C))

    def forward(self, x_BLC: torch.Tensor, cond_BD: torch.Tensor):
        # [B,D] > [B,2C] > [B,1,2,C] > 2[B,1,C] 
        scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.C).unbind(2)
        return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add_(shift)

"""
精简版 VAR 项目 VQVAE 模块
合并原文件：models/vqvae.py + models/basic_vae.py + models/quant.py + models/helpers.py(Phi部分)
改动：去掉分布式 quantizer allreduce，去掉 prog_si 分支，去掉 idxBl_to_img/embed_to_img/img_to_reconstructed_img/eini，
      去掉 PhiNonShared/PhiShared，只保留 PhiPartiallyShared(share_quant_resi=4 默认)
"""

import math
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from new_args import PATCH_NUMS, SHARE_QUANT_RESI


# ===================== basic_vae.py 组件 =====================

def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample2x(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))  # 插值上采样


class Downsample2x(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, pad=(0, 1, 0, 1), mode='constant', value=0))


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, dropout):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity()
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # 保证输入输出的通道一致
        if self.in_channels != self.out_channels:
            self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x), inplace=True))
        h = self.conv2(self.dropout(F.silu(self.norm2(h), inplace=True)))
        return self.nin_shortcut(x) + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.C = in_channels

        self.norm = Normalize(in_channels)
        self.qkv = torch.nn.Conv2d(in_channels, 3 * in_channels, kernel_size=1, stride=1, padding=0)
        self.w_ratio = int(in_channels) ** (-0.5)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        qkv = self.qkv(self.norm(x))
        B, _, H, W = qkv.shape  # should be B,3C,H,W
        C = self.C
        q, k, v = qkv.reshape(B, 3, C, H, W).unbind(1)

        # compute attention
        q = q.view(B, C, H * W).contiguous()
        q = q.permute(0, 2, 1).contiguous()     # B,HW,C
        k = k.view(B, C, H * W).contiguous()    # B,C,HW
        w = torch.bmm(q, k).mul_(self.w_ratio)  # B,HW,HW    w[B,i,j]=sum_c q[B,i,C]k[B,C,j]
        w = F.softmax(w, dim=2)

        # attend to values
        v = v.view(B, C, H * W).contiguous()
        w = w.permute(0, 2, 1).contiguous()  # B,HW,HW (first HW of k, second of q)
        h = torch.bmm(v, w)  # B, C, HW = sum_i v[B,C,i] w[B,i,j]
        h = h.view(B, C, H, W).contiguous()

        return x + self.proj_out(h)


def make_attn(in_channels, using_sa=True):
    return AttnBlock(in_channels) if using_sa else nn.Identity()


class Encoder(nn.Module):
    def __init__(
        self, *, ch=128, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
        dropout=0.0, in_channels=3,
        z_channels, double_z=False, using_sa=True, using_mid_sa=True,
    ):
        super().__init__()
        self.ch = ch    # 160
        self.num_resolutions = len(ch_mult)     # 5     (1,1,2,2,4)
        self.downsample_ratio = 2 ** (self.num_resolutions - 1)     # 下采样率=2^4，最后一层不下采样
        self.num_res_blocks = num_res_blocks    # 2
        self.in_channels = in_channels  # 3

        # 初始卷积
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)  # (1,1,1,2,2,4)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]     # 入维度(1,1,1,2,2,4)
            block_out = ch * ch_mult[i_level]   # 出维度(1,1,2,2,4)
            for i_block in range(self.num_res_blocks):
                # 增加残差块
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_sa:    # 仅最后一层执行attention
                    attn.append(make_attn(block_in, using_sa=True))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:     # 最后一层不执行下采样
                down.downsample = Downsample2x(block_in)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, (2 * z_channels if double_z else z_channels), kernel_size=3,
                                         stride=1, padding=1)

    def forward(self, x):
        # downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)    # 前4层：res_blocks>res_blocks>downsample
                if len(self.down[i_level].attn) > 0:        # 第5层：res_blocks>attn>res_blocks>attn>downsample
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        # middle
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))

        # end
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
        return h


class Decoder(nn.Module):
    def __init__(
        self, *, ch=128, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
        dropout=0.0, in_channels=3,
        z_channels, using_sa=True, using_mid_sa=True,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):  # 多一个残差块
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample2x(block_in)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, in_channels, kernel_size=3, stride=1, padding=1)  # 最后转换为C=3

    def forward(self, z):
        # z to block_in
        # middle
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(self.conv_in(z))))

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
        return h


# ===================== helpers.py 的 Phi 部分 =====================

class Phi(nn.Conv2d):
    """
    核心作用：不同阶段的stride，都要上采样到最大尺寸（原始分辨率），根据quant_resi调整原始特征与卷积比例
    """
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        # 保持尺度不变
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks // 2)
        self.resi_ratio = abs(quant_resi)

    def forward(self, h_BChw):
        """
        核心公式：输出 = (1-残差比例)×原始特征 + 残差比例×卷积后特征
        """
        return h_BChw.mul(1 - self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)


class PhiPartiallyShared(nn.Module):
    """
    代码实现上与PhiNonShared几乎没区别，因为继承的父类不一样，取数方法不一样而已，但是如何取数的逻辑是完全一样
    """
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K,
                                                                                            1 - 1 / 2 / K, K)

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]

    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'


# ===================== quant.py 的 VectorQuantizer2 =====================

class VectorQuantizer2(nn.Module):
    # VQGAN originally use beta=1.0, never tried 0.25; SD seems using 0.25
    def __init__(
            self,
            vocab_size,
            Cvae,
            using_znorm,
            beta: float = 0.25,
            v_patch_nums=None,
            quant_resi=0.5,
            share_quant_resi=4,  # share_quant_resi: args.qsr
    ):
        super().__init__()
        self.vocab_size: int = vocab_size  # 词表 4096
        self.Cvae: int = Cvae  # 隐向量z 通道数 32
        self.using_znorm: bool = using_znorm  # 计算最近邻前是否归一化 False
        self.v_patch_nums: Tuple[int] = v_patch_nums  # VAR生成图像的不同阶段的stride

        self.quant_resi_ratio = quant_resi  # 不同阶段的stride上采样到最大尺寸前，控制原始特征与卷积的比例 0.5

        # phi是stride上采样到最大尺寸前，对原始特征进行卷积的操作器；其中quant_resi控制原始特征与卷积的比例
        # share_quant_resi=4，代表所有stride共享4个phi
        self.quant_resi = PhiPartiallyShared(
            nn.ModuleList([
                Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()
                for _ in range(share_quant_resi)
            ])
        )
        # 维度[10, 4096]，使用ema方式统计不同阶段stride对应的词表使用情况
        self.register_buffer('ema_vocab_hit_SV',
                             torch.full((len(self.v_patch_nums), self.vocab_size), fill_value=0.0))
        self.record_hit = 0

        self.beta: float = beta  # 控制Loss_code中|sg(encoder) - code|^2 + β|encoder - sg(code)|^2中编码器与码本之间的互相靠近速度
        self.embedding = nn.Embedding(self.vocab_size, self.Cvae)  # 量化器的码表，维度[4096, 32]

    def extra_repr(self) -> str:
        return f'{self.v_patch_nums}, znorm={self.using_znorm}, beta={self.beta}  |  S={len(self.v_patch_nums)}, quant_resi={self.quant_resi_ratio}'

    # ===================== `forward` is only used in VAE training =====================
    def forward(self, f_BChw: torch.Tensor, ret_usages=False) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        """
        目的：希望获得能够接近完美地代替encoder输出的连续向量的离散向量
        常规方法：把离散向量与目标向量之间的残差，作为下一步的目标向量，多步拟合残差。后续把所有离散向量合并≈目标向量（连续）。例子为1维向量
            Continuous vector(target vector) - Discrete vector 1(token 1) = residual vector 1
            residual vector 1 - Discrete vector 2(token 2) = residual vector 2
            ...
            residual vector N-1 - Discrete vector N(token N) = residual vector N ≈ 0
            那么Discrete vector 1 + .... + Discrete vector N ≈ Continuous vector(target vector) 完美代替✅️
        VAR方法：把离散矩阵(经过下采样再上采样)与目标矩阵之间的残差，作为下一步的目标矩阵，多步拟合残差，后续把所有离散矩阵通过上采样后合并≈目标矩阵（连续）。例子为2维图像
            Continuous vector(target vector)(16*16) -  [target vector > downsample 1*1 > Discrete vector 1(token 1 * 1) > upsample 16*16] = residual vector 1(16*16)
            residual vector 1(16*16) - [residual vector 1 > downsample 2*2 > Discrete vector 2(token 2 * 2) > upsample 16*16]  = residual vector 2(16*16)
            ...
            residual vector 8(16*16) - [residual vector 8 > downsample 13*13 > Discrete vector 9(token 13 * 13) > upsample 16*16]  = residual vector 9(16*16)
            residual vector 9(16*16) - [residual vector 9 >  Discrete vector 10(token 16 * 16)]  = residual vector 10 ≈ 0
            那么Discrete vector 1 + .... + Discrete vector N ≈ Continuous vector(target vector) 完美代替✅️
        VAR方法对应的原因：
            1)如果仅存在1步残差拟合（即使用16*16的离散向量去拟合16*16连续向量）则效果不佳 
            2)如果按照常规方法使用原始尺寸的多步残差拟合（即使用N次16*16的离散向量）则速度很慢
        """
        dtype = f_BChw.dtype
        if dtype != torch.float32: f_BChw = f_BChw.float()
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()

        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        with (torch.cuda.amp.autocast(enabled=False)):  # 禁止开启混合精度
            mean_vq_loss: torch.Tensor = 0.0
            vocab_hit_V = torch.zeros(self.vocab_size, dtype=torch.float, device=f_BChw.device)
            SN = len(self.v_patch_nums)  # SN=10
            for si, pn in enumerate(self.v_patch_nums):
                # 1. 下采样到指定尺寸/stride，使用该尺寸获取码表最接近的token
                if self.using_znorm:  # 默认False
                    # 如果等于最后stride，那么直接转换维度即可，因为encoder已经固定是下采样2^4。图像原始分辨率为256，256/16=16=最后stride

                    # 如果不等于最后stride，执行F.interpolate 区域插值下采样，把特征图强制缩放到(pn, pn)固定尺寸；
                    # mode = 'area'：区域平均插值（下采样专用）→ 对像素区域取平均值，无锯齿、更平滑；
                    # 维度变化：(B, C, H, W) → (B, C, pn, pn) → (B, pn, pn, C) → (B*pn*pn, C)
                    rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (
                            si != SN - 1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                    # 归一化
                    rest_NC = F.normalize(rest_NC, dim=-1)
                    # self.embedding.weight.data，取embedding的权重的张量（不计算梯度），并归一化
                    # [B*pn*pn, C]  @  [C, 4096] → [B*pn*pn, 4096] →  [B*pn*pn]，获得最相似的码表(离散token)的id
                    idx_N = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
                else:
                    rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (
                            si != SN - 1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                    # 快速计算欧式距离平方：||x-y||² = ||x||² + ||y||² - 2xy
                    d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(
                        self.embedding.weight.data.square(), dim=1, keepdim=False)
                    d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)
                    idx_N = torch.argmin(d_no_grad, dim=1)

                hit_V = idx_N.bincount(minlength=self.vocab_size).float()

                # 2. 转换维度
                idx_Bhw = idx_N.view(B, pn, pn)
                # 3. id 转换为 embedding > 上采样（最后stride不执行）
                # [B, pn, pn] > [B, pn, pn, C] > [B, C, pn, pn] > [B, C, H, W]
                # bicubic：双三次插值，即取目标位置周围 4×4 共 16 个原始像素，按远近加权算出新像素值
                # contiguous：调整内存使得连续
                h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W),
                                       mode='bicubic').contiguous() if (si != SN - 1) else self.embedding(
                    idx_Bhw).permute(0, 3, 1, 2).contiguous()
                # 4. 卷积
                h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
                # 5. 累加   所有stride的上采样结果进行累加，当累加结束时f_hat就是target vector
                f_hat = f_hat + h_BChw
                # 6. 计算残差   target vector累减所有stride的上采样结果，每一步相减的结果就是残差，就是下一步的target vector
                f_rest -= h_BChw
                # 7. ema统计（简化版，去掉分布式 allreduce）
                if self.training:
                    if self.record_hit == 0:
                        self.ema_vocab_hit_SV[si].copy_(hit_V)
                    elif self.record_hit < 100:
                        self.ema_vocab_hit_SV[si].mul_(0.9).add_(hit_V.mul(0.1))
                    else:
                        self.ema_vocab_hit_SV[si].mul_(0.99).add_(hit_V.mul(0.01))
                    self.record_hit += 1
                vocab_hit_V.add_(hit_V)
                # 8. 码表损失：|sg(encoder) - code|^2 + β|encoder - sg(code)|^2，VQGAN是1，VQVAE是0.25
                # 目的是让离散码表靠拢encoder输出的连续向量的速度 >= encoder输出的连续向量靠拢离散码表的速度
                mean_vq_loss += F.mse_loss(f_hat.data, f_BChw).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)

            mean_vq_loss *= 1. / SN
            # 9. Straight-Through Estimator (STE)
            f_hat = (f_hat.data - f_no_grad).add_(f_BChw)

        # 计算利用率
        margin = (f_BChw.numel() / f_BChw.shape[1]) / self.vocab_size * 0.08
        if ret_usages:
            usages = [(self.ema_vocab_hit_SV[si] >= margin).float().mean().item() * 100 for si, pn in
                      enumerate(self.v_patch_nums)]
        else:
            usages = None
        return f_hat, usages, mean_vq_loss

    # ===================== `forward` is only used in VAE training =====================

    def f_to_idxBl_or_fhat(self,
                           f_BChw: torch.Tensor,
                           to_fhat: bool,
                           v_patch_nums=None) -> List[Union[torch.Tensor, torch.LongTensor]]:
        """
        将encoder向量转换为离散token ID 或者离散token embedding
        整体思路参考forward方法即可
        """
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        f_hat_or_idx_Bl: List[torch.Tensor] = []

        v_patch_nums = v_patch_nums or self.v_patch_nums
        patch_hws = [(pn, pn) if isinstance(pn, int) else (pn[0], pn[1]) for pn in v_patch_nums]
        assert patch_hws[-1][0] == H and patch_hws[-1][1] == W, f'{patch_hws[-1]=} != ({H=}, {W=})'

        SN = len(patch_hws)  # 10

        for si, (ph, pw) in enumerate(patch_hws):
            # ========== 下采样到当前尺度 ==========
            z_NC = F.interpolate(f_rest, size=(ph, pw),
                                 mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (
                    si != SN - 1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
            # ========== 最近邻查找码本索引 ==========
            if self.using_znorm:
                z_NC = F.normalize(z_NC, dim=-1)
                idx_N = torch.argmax(z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(
                    self.embedding.weight.data.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(z_NC, self.embedding.weight.data.T, alpha=-2, beta=1)
                idx_N = torch.argmin(d_no_grad, dim=1)
            # ========== 维度转换 2D ID图 ==========
            idx_Bhw = idx_N.view(B, ph, pw)
            # ========== ID转嵌入向量 + 上采样回原始尺寸 ==========
            # [B,ph,pw] > [B,ph,pw,Cvae] > [B,Cvae,ph,pw] > [B,Cvae,h,w]
            h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2),
                                   size=(H, W),
                                   mode='bicubic').contiguous() if (si != SN - 1) else self.embedding(idx_Bhw).permute(
                0, 3, 1, 2).contiguous()
            # ========== 通过卷积（quant_resi） ==========
            h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
            f_hat.add_(h_BChw)
            f_rest.sub_(h_BChw)
            # 如果to_fhat为False，返回ID图即可[B, ph*pw]
            # 如果to_fhat为True，返回离散化向量[B,Cvae,h,w]
            f_hat_or_idx_Bl.append(f_hat.clone() if to_fhat else idx_N.reshape(B, ph * pw))
        return f_hat_or_idx_Bl

    # ===================== idxBl_to_var_input: only used in VAR training =====================
    def idxBl_to_var_input(self, gt_ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
        """
        将离散的id 2位图转换为VAR输入
        gt_ms_idx_Bl: [[B,1], [B,4], [B,9]...] 离散token ID的列表
        """
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.Cvae
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)

        # 初始化重建特征图
        # 重建的离散向量，维度[B, Cvae, 16, 16]
        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, H, W, dtype=torch.float32)
        pn_next: int = self.v_patch_nums[0]

        # 循环处理前 SN-1 (8)个尺度（因为最后一个尺度不需要生成输入）

        # VQVAE的离散token，是把隐向量z作为初始target vector，随后下采样(尺寸逐渐增大)到stride_i*stride_i，再上采样到z的h*w维度得到discrete vector
        # 下一步的target vector = target vector - discrete vector，即有目标的残差，随后遍历所有阶段/stride/尺寸

        # VAR的不同阶段的n个 token embedding，是将之前所有阶段的离散token ID转换为离散token embedding，并且各自经过上采样(固定维度H,W，即z的H,W)+卷积
        # 随后累加得到重建的离散向量，再下采样到下一个阶段/stride/尺寸，得到的向量，就是下一阶段的输入n个 token
        # 即每个阶段的输入n个token embedding是包含了截止之前阶段已经重建好的图像信息，去预测当前阶段应该重建什么
        # 可看出返回数据时没有1*1阶段
        for si in range(SN - 1):
            # ========== 当前尺度 Token 转嵌入 + 形状调整 ==========
            # 步骤1：Token 索引 → 嵌入向量
            # 步骤2：transpose_ + view 调整形状为 [B, C, pn_next, pn_next]
            h_BChw = F.interpolate(
                self.embedding(gt_ms_idx_Bl[si])
                .transpose_(1, 2)
                .view(B, C, pn_next, pn_next),
                size=(H, W),
                mode='bicubic'
            )
            # ========== 经过卷积 ==========
            # 累计重建的离散向量
            f_hat.add_(self.quant_resi[si / (SN - 1)](h_BChw))
            # ========== 更新下一个尺度的分块数 ==========
            pn_next = self.v_patch_nums[si + 1]
            next_scales.append(
                F.interpolate(
                    f_hat,  # 当前累加的重建特征 [B, Cvae, 16, 16]
                    size=(pn_next, pn_next),    # 下采样到下一个尺度
                    mode='area'
                )
                .view(B, C, -1)
                .transpose(1, 2)
            )
        # ========== 拼接所有尺度的输入特征 ==========
        # 把 next_scales 里的 [B, pn*pn, C] 拼接成 [B, L-1, C]
        # L-1 = 总Token数 - 第一个尺度的1个Token
        return torch.cat(next_scales, dim=1) if len(next_scales) else None

    # ===================== get_next_autoregressive_input: only used in VAR inference =====================
    def get_next_autoregressive_input(self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor) -> Tuple[
        Optional[torch.Tensor], torch.Tensor]:
        """
        主要参考idxBl_to_var_input
        """
        HW = self.v_patch_nums[-1]
        if si != SN - 1:
            # 先上采样到最大尺寸，再通过phi
            h = self.quant_resi[si / (SN - 1)](
                F.interpolate(h_BChw, size=(HW, HW), mode='bicubic'))
            f_hat.add_(h)
            return f_hat, F.interpolate(f_hat, size=(self.v_patch_nums[si + 1], self.v_patch_nums[si + 1]),
                                        mode='area')
        else:
            h = self.quant_resi[si / (SN - 1)](h_BChw)
            f_hat.add_(h)
            return f_hat, f_hat


# ===================== vqvae.py 的 VQVAE =====================

class VQVAE(nn.Module):
    def __init__(
            self,
            vocab_size=4096,  # 词表 4096
            z_channels=32,  # latent的通道数
            ch=128,  # 初始卷积的通道数
            dropout=0.0,
            beta=0.25,  # 控制VQVAE的训练过程中，encoder编码向量向量化器码表靠近的速度
            using_znorm=False,  # 计算最近邻时是否归一化
            quant_conv_ks=3,  # 卷积的kernal
            quant_resi=0.5,  # 不同阶段的stride上采样到最大尺寸前，控制原始特征与卷积的比例 0.5
            share_quant_resi=4,  # 所有stride共享多少个phi
            v_patch_nums=PATCH_NUMS,  # VAR生成图像的不同阶段的stride
            test_mode=True,  # 默认为测试模式（VAR模型训练时VAE冻结）
    ):
        super().__init__()
        self.test_mode = test_mode
        self.V, self.Cvae = vocab_size, z_channels  # V: 词表大小(4096), Cvae: latent通道数(32)

        # ===================== 1. 配置 Encoder/Decoder 超参数 =====================
        ddconfig = dict(
            dropout=dropout,
            ch=ch,  # 160
            z_channels=z_channels,  # 32
            in_channels=3,
            ch_mult=(1, 1, 2, 2, 4),  # 不同层的通道数的倍数关系
            num_res_blocks=2,
            using_sa=True,
            using_mid_sa=True,
        )
        ddconfig.pop('double_z', None)  # 只有 KL-VAE 才用 double_z

        # ===================== 2. 构建 Encoder 和 Decoder =====================
        self.encoder = Encoder(double_z=False, **ddconfig)
        self.decoder = Decoder(**ddconfig)

        # ===================== 3. 计算下采样率 =====================
        self.vocab_size = vocab_size
        self.downsample = 2 ** (len(ddconfig['ch_mult']) - 1)

        # ===================== 4. 构建核心量化器 =====================
        self.quantize: VectorQuantizer2 = VectorQuantizer2(
            vocab_size=vocab_size,
            Cvae=self.Cvae,
            using_znorm=using_znorm,
            beta=beta,
            v_patch_nums=v_patch_nums,
            quant_resi=quant_resi,
            share_quant_resi=share_quant_resi,
        )

        # ===================== 5. 构建前后置卷积 =====================
        self.quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1,
                                          padding=quant_conv_ks // 2)
        self.post_quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1,
                                               padding=quant_conv_ks // 2)

        # ===================== 6. 测试模式配置 =====================
        if self.test_mode:
            self.eval()
            [p.requires_grad_(False) for p in self.parameters()]  # 【关键】冻结所有参数，不训练

    def forward(self, inp, ret_usages=False):  # -> rec_B3HW, idx_N, loss
        # 流程：图像 -> Encoder -> quant_conv -> 量化器 -> post_quant_conv -> Decoder -> 重建图像
        f_hat, usages, vq_loss = self.quantize(self.quant_conv(self.encoder(inp)), ret_usages=ret_usages)
        # decoder给出图像解码，后续可以用于计算MSE loss, lipid loss, GAN loss，并且结合vq_loss
        return self.decoder(self.post_quant_conv(f_hat)), usages, vq_loss

    def fhat_to_img(self, f_hat: torch.Tensor):
        # 直接把量化后的特征通过 Decoder 还原成图像
        return self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1)

    def img_to_idxBl(self, inp_img_no_grad: torch.Tensor) -> List[torch.LongTensor]:
        """
        将图像经过encoder编码，随后通过量化器获得离散token id
        """
        f = self.quant_conv(self.encoder(inp_img_no_grad))
        return self.quantize.f_to_idxBl_or_fhat(f, to_fhat=False)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if 'quantize.ema_vocab_hit_SV' in state_dict and state_dict['quantize.ema_vocab_hit_SV'].shape[0] != \
                self.quantize.ema_vocab_hit_SV.shape[0]:
            state_dict['quantize.ema_vocab_hit_SV'] = self.quantize.ema_vocab_hit_SV
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)
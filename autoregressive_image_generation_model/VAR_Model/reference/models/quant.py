from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import distributed as tdist, nn as nn
from torch.nn import functional as F

import dist

# this file only provides the VectorQuantizer2 used in VQVAE
__all__ = ['VectorQuantizer2', ]


class VectorQuantizer2(nn.Module):
    # VQGAN originally use beta=1.0, never tried 0.25; SD seems using 0.25
    def __init__(
            self,
            vocab_size,
            Cvae,
            using_znorm,
            beta: float = 0.25,
            default_qresi_counts=0,
            v_patch_nums=None,
            quant_resi=0.5,
            share_quant_resi=4,  # share_quant_resi: args.qsr
    ):
        super().__init__()
        self.vocab_size: int = vocab_size  # 词表 4096
        self.Cvae: int = Cvae  # 隐向量z 通道数 32
        self.using_znorm: bool = using_znorm  # 计算最近邻前是否归一化 False
        self.v_patch_nums: Tuple[int] = v_patch_nums  # VAR生成图像的不同阶段的stride(1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

        self.quant_resi_ratio = quant_resi  # 不同阶段的stride上采样到最大尺寸前，控制原始特征与卷积的比例 0.5

        # phi是stride上采样到最大尺寸前，对原始特征进行卷积的操作器；其中quant_resi控制原始特征与卷积的比例
        # share_quant_resi=0，代表每个stride有独立的phi，即有独立的卷积参数
        if share_quant_resi == 0:  # non-shared: \phi_{1 to K} for K scales
            self.quant_resi = PhiNonShared(
                [(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(default_qresi_counts or len(self.v_patch_nums))])  # 0 or 10
        # share_quant_resi=1，代表所有stride共享相同的1个phi
        elif share_quant_resi == 1:  # fully shared: only a single \phi for K scales
            self.quant_resi = PhiShared(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
        # share_quant_resi=x，代表所有stride共享x个phi
        else:  # partially shared: \phi_{1 to share_quant_resi} for K scales
            self.quant_resi = PhiPartiallyShared(
                nn.ModuleList([(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(share_quant_resi)]))
        # 维度[10, 4096]，使用ema方式统计不同阶段stride对应的词表使用情况
        self.register_buffer('ema_vocab_hit_SV', torch.full((len(self.v_patch_nums), self.vocab_size), fill_value=0.0))
        self.record_hit = 0

        self.beta: float = beta  # 控制Loss_code中|sg(encoder) - code|^2 + β|encoder - sg(code)|^2中编码器与码本之间的互相靠近速度，VQGAN是1，VQVAE是0.25
        self.embedding = nn.Embedding(self.vocab_size, self.Cvae)  # 量化器的码表，维度[4096, 32]

        # only used for progressive training of VAR (not supported yet, will be tested and supported in the future)
        # 渐进式训练相关参数
        self.prog_si = -1  # progressive training: not supported yet, prog_si always -1

    def eini(self, eini):
        """
        好像没使用到
        """
        if eini > 0:
            nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0:
            self.embedding.weight.data.uniform_(-abs(eini) / self.vocab_size, abs(eini) / self.vocab_size)

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

        :param f_BChw: feature of encoder latent，维度是BChw
        :param ret_usages:
        :return:
        """
        dtype = f_BChw.dtype
        if dtype != torch.float32: f_BChw = f_BChw.float()
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()

        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        with (torch.cuda.amp.autocast(enabled=False)): # 禁止开启混合精度
            mean_vq_loss: torch.Tensor = 0.0
            vocab_hit_V = torch.zeros(self.vocab_size, dtype=torch.float, device=f_BChw.device)  # 4096 用于计算词表使用情况
            SN = len(self.v_patch_nums)  # (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)  SN=10
            for si, pn in enumerate(self.v_patch_nums):  # si 0 > 9  pn 1 > 16
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
                    d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
                    # 获取距离最近的码表(离散token)的id
                    idx_N = torch.argmin(d_no_grad, dim=1)

                hit_V = idx_N.bincount(minlength=self.vocab_size).float()  # 统计每个码本被使用的次数
                handler = None
                if self.training:
                    if dist.initialized(): 
                        handler = tdist.all_reduce(hit_V, async_op=True)

                # 2. 转换维度
                # [B*pn*pn] > [B, pn, pn]
                idx_Bhw = idx_N.view(B, pn, pn)
                # 3. id 转换为 embedding > 转换维度 > 上采样（最后stride不执行）
                # [B, pn, pn] > [B, pn, pn, C] > [B, C, pn, pn] > [B, C, H, W]
                # bicubic：双三次插值，即取目标位置周围 4×4 共 16 个原始像素，按远近加权算出新像素值
                # contiguous：调整内存使得连续
                h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W),
                                       mode='bicubic').contiguous() if (si != SN - 1) else self.embedding(
                    idx_Bhw).permute(0, 3, 1, 2).contiguous()
                # 4. 卷积
                h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)  # 传入当前第几步，根据计算规则获得对应的Phi，通过phi按照quant_resi_ratio控制原特征通过和卷积的比例
                # 5. 所有stride的上采样结果进行累加，当累加结束时f_hat就是target vector
                f_hat = f_hat + h_BChw  # 累加当前stride的上采样结果
                # 6. 计算残差
                f_rest -= h_BChw  # target vector累减所有stride的上采样结果，每一步相减的结果就是残差，就是下一步的target vector
                # 7. ema方式统计各离散code的命中次数
                if self.training and dist.initialized():
                    handler.wait()
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
            # 9. Straight-Through Estimator (STE)：梯度直接回传给编码器，避免离散取值导致梯度中断
            f_hat = (f_hat.data - f_no_grad).add_(f_BChw)
        # 计算利用率阈值
        margin = tdist.get_world_size() * (f_BChw.numel() / f_BChw.shape[1]) / self.vocab_size * 0.08
        # margin = pn*pn / 100
        if ret_usages:
            # 如果超过阈值，就统计
            usages = [(self.ema_vocab_hit_SV[si] >= margin).float().mean().item() * 100 for si, pn in
                      enumerate(self.v_patch_nums)]
        else:
            usages = None
        # 10. 返回STE的离散向量，码本使用率，mean_vq_loss
        return f_hat, usages, mean_vq_loss

    # ===================== `forward` is only used in VAE training =====================

    def embed_to_fhat(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale=True, last_one=False) -> Union[
        List[torch.Tensor], torch.Tensor]:
        ls_f_hat_BChw = []
        B = ms_h_BChw[0].shape[0]
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        if all_to_max_scale:
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, H, W, dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums):  # from small to large
                h_BChw = ms_h_BChw[si]
                if si < len(self.v_patch_nums) - 1:
                    h_BChw = F.interpolate(h_BChw, size=(H, W), mode='bicubic')
                h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
                f_hat.add_(h_BChw)
                if last_one:
                    ls_f_hat_BChw = f_hat
                else:
                    ls_f_hat_BChw.append(f_hat.clone())
        else:
            # WARNING: this is not the case in VQ-VAE training or inference (we'll interpolate every token map to the max H W, like above)
            # WARNING: this should only be used for experimental purpose
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, self.v_patch_nums[0], self.v_patch_nums[0],
                                           dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums):  # from small to large
                f_hat = F.interpolate(f_hat, size=(pn, pn), mode='bicubic')
                h_BChw = self.quant_resi[si / (SN - 1)](ms_h_BChw[si])
                f_hat.add_(h_BChw)
                if last_one:
                    ls_f_hat_BChw = f_hat
                else:
                    ls_f_hat_BChw.append(f_hat)

        return ls_f_hat_BChw

    def f_to_idxBl_or_fhat(self,
                           f_BChw: torch.Tensor,
                           to_fhat: bool,
                           v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[
        Union[torch.Tensor, torch.LongTensor]]:  # z_BChw is the feature from inp_img_no_grad
        """
        将encoder向量转换为离散token ID 或者离散token embedding  
        整体思路参考forward方法即可      
        """
        
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        f_hat_or_idx_Bl: List[torch.Tensor] = []

        patch_hws = [(pn, pn) if isinstance(pn, int) else (pn[0], pn[1]) for pn in (v_patch_nums or self.v_patch_nums)]  # [(1,1), (2,2), (3,3)...]
        assert patch_hws[-1][0] == H and patch_hws[-1][1] == W, f'{patch_hws[-1]=} != ({H=}, {W=})'

        SN = len(patch_hws)  # 10
        
        for si, (ph, pw) in enumerate(patch_hws):  # from small to large
            # 如果不开启阶段式训练，self.prog_si=-1
            # 如果开启阶段式训练，self.prog_si>0，可以假设为默认值4，如果si＞self.prog_si 代表超出允许的阶段/尺寸/stride
            if 0 <= self.prog_si < si:
                break  # progressive training: not supported yet, prog_si always -1
            # ========== 下采样到当前尺度 ==========
            # [B,Cvae,h,w] > [B,Cvae,ph,pw] > [B,ph,pw,Cvae] > [B*ph*pw,Cvae]
            z_NC = F.interpolate(f_rest, size=(ph, pw), 
                                 mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN - 1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
            # ========== 最近邻查找码本索引 ==========
            if self.using_znorm:
                z_NC = F.normalize(z_NC, dim=-1)
                idx_N = torch.argmax(z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(
                    self.embedding.weight.data.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(z_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
                idx_N = torch.argmin(d_no_grad, dim=1)
            # ========== 维度转换 2D ID图 ==========
            # [B,ph,pw]
            idx_Bhw = idx_N.view(B, ph, pw)
            # ========== ID转嵌入向量 + 上采样回原始尺寸 ==========
            # [B,ph,pw] > [B,ph,pw,Cvae] > [B,Cvae,ph,pw] > [B,Cvae,h,w]
            h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), 
                                   size=(H, W),
                                   mode='bicubic').contiguous() if (si != SN - 1) else self.embedding(idx_Bhw).permute(
                0, 3, 1, 2).contiguous()
            # ========== 通过卷积（quant_resi） ==========
            h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
            f_hat.add_(h_BChw) # 更新encoder vector的离散化
            f_rest.sub_(h_BChw) # 更新残差
            # 如果to_fhat为False，返回ID图即可[B, ph*pw]
            # 如果to_fhat为True，返回离散化向量[B,Cvae,h,w]
            f_hat_or_idx_Bl.append(f_hat.clone() if to_fhat else idx_N.reshape(B, ph * pw))
        # 返回
        return f_hat_or_idx_Bl

    # ===================== idxBl_to_var_input: only used in VAR training, for getting teacher-forcing input =====================
    def idxBl_to_var_input(self, gt_ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
        """
        将离散的id 2位图转换为VAR输入
        gt_ms_idx_Bl: [[B,1], [B,4], [B,9]...] 离散token ID的列表
        """
        # ========== 1.1 初始化结果列表（存每个尺度的输入特征） ==========
        next_scales = []
        # ========== 1.2 解析核心维度 ==========
        B = gt_ms_idx_Bl[0].shape[0]  # Batch 大小
        C = self.Cvae  # VQVAE 特征通道数（默认 32）
        H = W = self.v_patch_nums[-1]  # 最后一个（最大）尺度的尺寸（默认 16）
        SN = len(self.v_patch_nums)  # 总尺度数（默认 10）

        # ========== 1.3 初始化重建特征图（累加各尺度量化特征） ==========
        # 重建的离散向量，维度[B, Cvae, 16, 16]
        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, H, W, dtype=torch.float32)  # 
        # ========== 1.4 初始化下一个尺度的分块数 ==========
        pn_next: int = self.v_patch_nums[0]  # 从第一个尺度（1x1）开始
        # 循环处理前 SN-1 (8)个尺度（因为最后一个尺度不需要生成输入）

        # VQVAE的离散token，是把隐向量z作为初始target vector，随后下采样(尺寸逐渐增大)到stride_i*stride_i，再上采样到z的h*w维度得到discrete vector
        # 下一步的target vector = target vector - discrete vector，即有目标的残差，随后遍历所有阶段/stride/尺寸

        # VAR的不同阶段的n个 token embedding，是将之前所有阶段的离散token ID转换为离散token embedding，并且各自经过上采样(固定维度H,W，即z的H,W)+卷积
        # 随后累加得到重建的离散向量，再下采样到下一个阶段/stride/尺寸，得到的向量，就是下一阶段的输入n个 token
        # 即每个阶段的输入n个token embedding是包含了截止之前阶段已经重建好的图像信息，去预测当前阶段应该重建什么
        # 可看出返回数据时没有1*1阶段
        for si in range(SN - 1):
            # 第0个阶段或者超出允许的阶段，就提前退出；默认是全阶段训练，永远不会提前退出
            if self.prog_si == 0 or (
                    0 <= self.prog_si - 1 < si): break  # progressive training: not supported yet, prog_si always -1
            # ========== 2.1.1 当前尺度 Token 转嵌入 + 形状调整 ==========
            # 步骤1：Token 索引 → 嵌入向量
            # 步骤2：transpose_ + view 调整形状为 [B, C, pn_next, pn_next]
            h_BChw = F.interpolate(
                self.embedding(gt_ms_idx_Bl[si])  # [B, pn*pn, Cvae]
                .transpose_(1, 2)  # [B, Cvae, pn*pn]
                .view(B, C, pn_next, pn_next),  # [B, Cvae, pn_next, pn_next]
                size=(H, W),  # 上采样到最大尺度 (16,16)
                mode='bicubic'  # 双三次插值，保证细节
            )
            # ========== 2.1.2 经过卷积 ==========
            # 累计重建的离散向量
            f_hat.add_(self.quant_resi[si / (SN - 1)](h_BChw))
            # ========== 2.2.1 更新下一个尺度的分块数 ==========
            pn_next = self.v_patch_nums[si + 1]
            next_scales.append(
                F.interpolate(
                    f_hat,  # 当前累加的重建特征 [B, Cvae, 16, 16]
                    size=(pn_next, pn_next),  # 下采样到下一个尺度
                    mode='area'
                )
                .view(B, C, -1)  # [B, C, pn_next*pn_next]
                .transpose(1, 2)  # [B, pn_next*pn_next, C]
            )
        # ========== 3.1 拼接所有尺度的输入特征 ==========
        # 把 next_scales 里的 [B, pn*pn, C] 拼接成 [B, L-1, C]
        # L-1 = 总Token数 - 第一个尺度的1个Token
        return torch.cat(next_scales, dim=1) if len(next_scales) else None  # cat BlCs to BLC, this should be float32

    # ===================== get_next_autoregressive_input: only used in VAR inference, for getting next step's input =====================
    def get_next_autoregressive_input(self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor) -> Tuple[
        Optional[torch.Tensor], torch.Tensor]:  # only used in VAR inference
        """
        主要参考idxBl_to_var_input
        :param si: 当前阶段的下标
        :param SN: 阶段的总数
        :param f_hat: 已重建的信息
        :param h_BChw: 当前var预测的token，已经通过量化器把token id 转换为token embedding [B, Cvae, pn, pn]
        :return:
        """
        HW = self.v_patch_nums[-1]
        if si != SN - 1:
            # 先上采样到最大尺寸（encoder的latent尺寸），再通过phi进行特征向量的直流和卷积
            h = self.quant_resi[si / (SN - 1)](
                F.interpolate(h_BChw, size=(HW, HW), mode='bicubic'))  # conv after upsample
            # 已重建信息的累加
            f_hat.add_(h)
            # 返回已重建信息
            # 返回已重建信息下采样到，下一个尺寸/阶段/stride的维度
            return f_hat, F.interpolate(f_hat, size=(self.v_patch_nums[si + 1], self.v_patch_nums[si + 1]), mode='area')
        else:
            # 直接通过phi进行特征向量的直流和卷积即可
            h = self.quant_resi[si / (SN - 1)](h_BChw)
            f_hat.add_(h)
            return f_hat, f_hat


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
        :param h_BChw:
        :return:
        """
        return h_BChw.mul(1 - self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)


class PhiShared(nn.Module):
    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi: Phi = qresi

    def __getitem__(self, _) -> Phi:
        return self.qresi


class PhiPartiallyShared(nn.Module):
    """
    代码实现上与PhiNonShared几乎没区别，因为继承的父类不一样，取数方法不一样而已，但是如何取数的逻辑是完全一样
    """
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]

    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'


class PhiNonShared(nn.ModuleList):
    """
    生成刻度尺，随后根据不同stride与刻度尺的距离，取出最近的phi（卷积）
    """
    def __init__(self, qresi: List):
        super().__init__(qresi)
        # self.qresi = qresi
        K = len(qresi)
        # 若K = 4：1 / (3 * 4) = 1 / 12≈0.083，1 - 1 / 12≈0.916 → ticks = [0.083, 0.333, 0.583, 0.833]
        # 若K = 2：1 / (2 * 2) = 0.25，1 - 0.25 = 0.75 → ticks = [0.25, 0.75]
        self.ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        # np.abs(self.ticks - at_from_0_to_1)：计算与不同刻度的绝对距离
        # np.argmin：找出绝对距离最小的下标
        # .item()：转换为原生python整数
        # super().__getitem__：根据下标找出对应的phi
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())

    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'

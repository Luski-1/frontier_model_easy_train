"""
References:
- VectorQuantizer2: https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L110
- GumbelQuantize: https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L213
- VQVAE (VQModel): https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/models/autoencoder.py#L14
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .basic_vae import Decoder, Encoder
from .quant import VectorQuantizer2


class VQVAE(nn.Module):
    def __init__(
            self,
            vocab_size=4096, # 词表 4096
            z_channels=32, # latent的通道数
            ch=128, # 初始卷积的通道数
            dropout=0.0,
            beta=0.25,  # 控制VQVAE的训练过程中，encoder编码向量向量化器码表靠近的速度
            using_znorm=False,  # 计算最近邻时是否归一化
            quant_conv_ks=3,  # 卷积的kernal
            quant_resi=0.5,  # 不同阶段的stride上采样到最大尺寸前，控制原始特征与卷积的比例 0.5
            share_quant_resi=4,  # 所有stride共享多少个phi
            default_qresi_counts=0,  # 默认多少个phi
            v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # VAR生成图像的不同阶段的stride (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
            test_mode=True,  # 默认为测试模式（VAR模型训练时VAE冻结）
    ):
        super().__init__()
        self.test_mode = test_mode
        self.V, self.Cvae = vocab_size, z_channels  # V: 词表大小(4096), Cvae: latent通道数(32)

        # ===================== 1. 配置 Encoder/Decoder 超参数 =====================
        # 直接来自 Stable Diffusion/Latent Diffusion 的 vq-f16 配置
        ddconfig = dict(
            dropout=dropout,  # 0.0
            ch=ch,  # 160
            z_channels=z_channels,  # 32
            in_channels=3,
            ch_mult=(1, 1, 2, 2, 4), # 不同层的通道数的倍数关系
            num_res_blocks=2, # 层内残差块数量
            using_sa=True, # 是否执行Attention
            using_mid_sa=True,  # mid层是否执行Attention
        )
        ddconfig.pop('double_z', None)  # 只有 KL-VAE 才用 double_z（即输出均值和方差），VQ-VAE 不用

        # ===================== 2. 构建 Encoder 和 Decoder =====================
        # 来自 basic_vae.py，标准的U-net
        self.encoder = Encoder(double_z=False, **ddconfig)  # 对图像编码 → 连续特征，C=32
        self.decoder = Decoder(**ddconfig)  # 将连续特征解码 → 图像， C=3

        # ===================== 3. 计算下采样率 =====================
        # ch_mult 长度为 5，所以下采样率是 2^(5-1) = 16
        # 即：256x256 图像 → 16x16 特征图
        self.vocab_size = vocab_size
        self.downsample = 2 ** (len(ddconfig['ch_mult']) - 1)

        # ===================== 4. 构建核心量化器 (VectorQuantizer2) =====================
        # 来自 quant.py，这是 VQ-VAE 的灵魂：连续特征 → 离散 Token
        self.quantize: VectorQuantizer2 = VectorQuantizer2(
            vocab_size=vocab_size,  # 词表 4096
            Cvae=self.Cvae,  # encoder的latent通道 32
            using_znorm=using_znorm,  # 计算最近邻前是否归一化 False
            beta=beta,  # 控制VQVAE的训练过程中，encoder编码向量向量化器码表靠近的速度 0.25
            default_qresi_counts=default_qresi_counts,  # 默认多少个phi 0
            v_patch_nums=v_patch_nums,  # VAR生成图像的不同阶段的stride(1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
            quant_resi=quant_resi,  # 不同阶段的stride上采样到最大尺寸前，控制原始特征与卷积的比例 0.5
            share_quant_resi=share_quant_resi,  # 所有stride共享多少个phi 4
        )

        # ===================== 5. 构建前后置卷积 =====================
        # quant_conv: Encoder 输出 → 量化器输入
        self.quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks // 2)
        # post_quant_conv: 量化器输出 → Decoder 输入
        self.post_quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks // 2)

        # ===================== 6. 测试模式配置 =====================
        if self.test_mode:
            self.eval()  # 切换到评估模式（关闭 Dropout 等）
            [p.requires_grad_(False) for p in self.parameters()]  # 【关键】冻结所有参数，不训练

    # ===================== `forward` is only used in VAE training =====================
    def forward(self, inp, ret_usages=False):  # -> rec_B3HW, idx_N, loss
        # 流程：图像 -> Encoder -> quant_conv -> 量化器 -> post_quant_conv -> Decoder -> 重建图像
        f_hat, usages, vq_loss = self.quantize(self.quant_conv(self.encoder(inp)), ret_usages=ret_usages)
        # decoder给出图像解码，后续可以用于计算MSE loss, lipid loss, GAN loss，并且结合vq_loss
        return self.decoder(self.post_quant_conv(f_hat)), usages, vq_loss

    # ===================== `forward` is only used in VAE training =====================

    def fhat_to_img(self, f_hat: torch.Tensor):
        # 直接把量化后的特征通过 Decoder 还原成图像
        return self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1)

    def img_to_idxBl(self,
                     inp_img_no_grad: torch.Tensor,
                     v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[
        torch.LongTensor]:  # return List[Bl]
        """
        将图像经过encoder编码，随后通过量化器获得离散token id
        """

        # 步骤 1：图像[B,3,H,W] -> Encoder vector[B,Cvae,h,w] -> quant_conv_vector[B,Cvae,h,w]
        f = self.quant_conv(self.encoder(inp_img_no_grad))

        # 步骤 2：调用 量化器的方法，连续特征 -> 离散 Token 索引
        # to_fhat=False：只需要索引，不需要重建特征
        return self.quantize.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums)

    def idxBl_to_img(self, ms_idx_Bl: List[torch.Tensor], same_shape: bool, last_one=False) -> Union[
        List[torch.Tensor], torch.Tensor]:
        """
        没使用到
        """
        B = ms_idx_Bl[0].shape[0]
        ms_h_BChw = []


        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            pn = round(l ** 0.5)


            ms_h_BChw.append(
                self.quantize.embedding(idx_Bl)
                .transpose(1, 2)
                .view(B, self.Cvae, pn, pn)
            )

        # 步骤 2：调用 embed_to_img 完成后续解码
        return self.embed_to_img(ms_h_BChw=ms_h_BChw, all_to_max_scale=same_shape, last_one=last_one)

    def embed_to_img(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale: bool, last_one=False) -> Union[
        List[torch.Tensor], torch.Tensor]:
        """
        没使用到
        """
        if last_one:
            return self.decoder(self.post_quant_conv(
                self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=True))).clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1) for f_hat in
                    self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=False)]

    def img_to_reconstructed_img(self, x, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
                                 last_one=False) -> List[torch.Tensor]:
        """
        没使用到
        """
        f = self.quant_conv(self.encoder(x))
        ls_f_hat_BChw = self.quantize.f_to_idxBl_or_fhat(
            f,
            to_fhat=True,
            v_patch_nums=v_patch_nums
        )
        if last_one:
            return self.decoder(self.post_quant_conv(ls_f_hat_BChw[-1])).clamp_(-1, 1)
        else:
            return [
                self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1)
                for f_hat in ls_f_hat_BChw
            ]

    def load_state_dict(self, state_dict: Dict[str, Any], strict=True, assign=False):
        if 'quantize.ema_vocab_hit_SV' in state_dict and state_dict['quantize.ema_vocab_hit_SV'].shape[0] != \
                self.quantize.ema_vocab_hit_SV.shape[0]:
            state_dict['quantize.ema_vocab_hit_SV'] = self.quantize.ema_vocab_hit_SV
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)

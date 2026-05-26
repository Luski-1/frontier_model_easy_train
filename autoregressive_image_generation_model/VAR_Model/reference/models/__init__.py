from typing import Tuple
import torch.nn as nn

# 导入项目自定义的4个核心组件
from .quant import VectorQuantizer2  # 向量量化层（VQ-VAE的核心）
from .var import VAR  # 主模型：视觉自回归生成模型（训练对象）
from .vqvae import VQVAE  # 预训练模型：矢量量化变分自编码器（图像转token）


# ===================== 核心函数：构建VQ-VAE + VAR 双模型 =====================
# 函数作用：一次性创建【预训练VQVAE编码器】和【VAR生成模型】，并返回
# 返回值类型：固定为 (VQVAE模型实例, VAR模型实例)
def build_vae_var(
        # ===================== 1. 共享参数（两个模型都会用到的核心参数） =====================
        device,  # 模型运行的设备（cuda:0/cpu，分布式的当前GPU）
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # VAR生成图像的不同阶段的stride

        # ===================== 2. VQVAE 专属参数（预训练模型固定配置） =====================
        V=4096,  # 词表大小：VQ-VAE的视觉词汇量（4096个视觉token）
        Cvae=32,  # VQVAE输出的latent的通道数
        ch=160,  # VQVAE基础的卷积通道数
        share_quant_resi=4,  # 控制phi的共享程度 4 

        # ===================== 3. VAR 专属参数（生成模型的结构/训练配置） =====================
        num_classes=1000,  # 分类类别数（ImageNet-1K固定为1000）
        depth=16,  # Transformer编码器的深度（16层）
        shared_aln=False,  # 是否使用共享自适应层归一化
        attn_l2_norm=True,  # 注意力是否做L2归一化（提升训练稳定性）
        flash_if_available=True,  # 自动使用Flash Attention（加速注意力计算）
        fused_if_available=True,  # 自动使用融合算子（加速LayerNorm/MLP）
        init_adaln=0.5,  # adaLN的shift/scale的初始缩放
        init_adaln_gamma=1e-5,  # adaLN的gamma的初始缩放
        init_head=0.02,  # 输出头（分类头）的初始缩放
        init_std=-1,  # 权重初始化标准差：<0代表自动计算
) -> Tuple[VQVAE, VAR]:  # 类型注解：固定返回(VQVAE, VAR)

    # ===================== 自动计算VAR模型的核心超参（基于depth自动推导，无需手动指定） =====================
    heads = depth  # 注意力头数 = 模型深度
    width = depth * 64  # 模型特征维度（宽度）= 16*64=1024
    dpr = 0.1 * depth / 24  # Drop Path衰减率：深度越深，衰减率越高（16层→0.0667），用于减轻不同层的过拟合，类似dropout的层数（深度）的应用

    # ===================== 关键优化：禁用PyTorch默认参数初始化（加速模型构建） =====================
    # 遍历所有常用神经网络层
    for clz in (nn.Linear, nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d,
                nn.ConvTranspose2d):
        # 重写这些层的 reset_parameters 方法（默认初始化方法）
        setattr(clz, 'reset_parameters', lambda self: None)

    # ===================== 步骤1：构建 VQ-VAE 模型（预训练，只用于推理，不训练） =====================
    vae_local = VQVAE(
        vocab_size=V,  # 词表 4096
        z_channels=Cvae,  # 隐向量z的通道数 32
        ch=ch,  # 初始卷积的通道数 160
        test_mode=True, # 非训练模式
        share_quant_resi=share_quant_resi,  # 控制phi的共享程度 4 
        v_patch_nums=patch_nums  # VAR生成图像的不同阶段的stride (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    ).to(device)

    # ===================== 步骤2：构建 VAR 模型（核心训练对象，Transformer生成模型） =====================
    var_wo_ddp = VAR(
        vae_local=vae_local,  # 传入VQVAE，用于把图像转成视觉token
        num_classes=num_classes,  # 类别数 1000
        depth=depth,  # 16
        embed_dim=width,  # 1024
        num_heads=heads,  # 16
        drop_rate=0., # W_O和FFN的dropout概率
        attn_drop_rate=0., # Attention 的dropout概率
        drop_path_rate=dpr,  # 层（深度）dropout， 0.1 * depth / 24 ≈0.066
        norm_eps=1e-6,  # LayerNorm的eps（防止除0）
        shared_aln=shared_aln,  # 是否共享adaLN，仅当训练的图像分辨率≥512时开启 False
        cond_drop_rate=0.1,  # CFG的概率
        attn_l2_norm=attn_l2_norm,  # attntion前是否对QK归一化 True
        patch_nums=patch_nums,  # VAR生成图像的不同阶段的stride (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        flash_if_available=flash_if_available, # 开启flash加速算子
        fused_if_available=fused_if_available # 开启加速算子
    ).to(device)

    # ===================== 步骤3：手动初始化VAR权重（自定义初始化，比默认初始化更稳定） =====================
    var_wo_ddp.init_weights(
        init_adaln=init_adaln,  # 0.5
        init_adaln_gamma=init_adaln_gamma,  # 1e-5
        init_head=init_head,  # 0.02
        init_std=init_std  # -1
    )

    # ===================== 返回两个模型 =====================
    # vae_local：预训练VQVAE（图像→token）
    # var_wo_ddp：VAR原生模型（未包装DDP，后续会被分布式包装）
    return vae_local, var_wo_ddp

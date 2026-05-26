"""
精简版 VAR 项目参数配置模块
原文件：utils/arg_util.py
改动：Tap → argparse，去掉分布式参数，去掉 pg 渐进训练参数，去掉多分辨率参数，硬编码 256 分支
"""

import argparse
import os
import random
import sys

import numpy as np
import torch


# ===================== 固定配置（256×256 分支） =====================
PATCH_NUMS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)    # VAR生成图像的不同阶段的stride
PATCH_SIZE = 16                                        # VAE的下采样倍率
RESOS = tuple(pn * PATCH_SIZE for pn in PATCH_NUMS)    # 不同生成阶段的分辨率
DATA_LOAD_RESO = max(RESOS)                            # 训练图片的最大分辨率 = 256
SHARE_QUANT_RESI = 4                                   # 所有stride共享多少个phi
NUM_CLASSES = 1000                                     # ImageNet 类别数


def parse_args():
    parser = argparse.ArgumentParser(description='VAR (Visual AutoRegressive) Training')

    # ===================== 数据 =====================
    parser.add_argument('--data_path', type=str, default='/path/to/imagenet',
                        help='ImageNet 数据集根目录路径')
    parser.add_argument('--hflip', type=bool, default=False,
                        help='是否开启随机水平翻转')
    
    # ===================== VAE 模型 =====================
    parser.add_argument('--vae_ckpt', type=str, default='/path/to/vae_checkpoint',
                        help='vae 模型权重根目录路径')

    # ===================== VAR 模型 =====================
    parser.add_argument('--depth', type=int, default=16,
                        help='VAR depth 【模型层数，同时影响drop_path_rate[0.1 * depth/24]】')
    parser.add_argument('--ini', type=float, default=-1,
                        help='模型参数初始化的方差，-1代表根据参数维度进行计算')
    parser.add_argument('--hd', type=float, default=0.02,
                        help='输出头权重的缩放系数，目的是训练开始时预测相对均匀')
    parser.add_argument('--aln', type=float, default=0.5,
                        help='AdaLN shift/scale的初始缩放，目的是训练开始时接近0')
    parser.add_argument('--alng', type=float, default=1e-5,
                        help='AdaLN gamma 初始缩放接近0')
    parser.add_argument('--anorm', type=bool, default=True,
                        help='Attention模块中，在计算注意力得分前，Q/K是否开启归一化')

    # ===================== 训练 =====================
    parser.add_argument('--bs', type=int, default=32,
                        help='单卡 batch size')
    parser.add_argument('--ep', type=int, default=250,
                        help='训练 epoch 数')
    parser.add_argument('--fp16', type=int, default=0,
                        help='混合精度，0=关闭, 1=fp16, 2=bf16')
    parser.add_argument('--tblr', type=float, default=1e-4,
                        help='train base learning rate，最终的基础学习率是tblr × (bs/256)')
    parser.add_argument('--twd', type=float, default=0.05,
                        help='train weight decay，训练正则化的权重衰减')
    parser.add_argument('--tclip', type=float, default=2.,
                        help='梯度裁剪阈值，<=0不裁剪')
    parser.add_argument('--ls', type=float, default=0.0,
                        help='标签平滑，让不是target label的类别也有微弱概率，避免过拟合')

    # ===================== 学习率调度 =====================
    parser.add_argument('--sche', type=str, default='lin0',
                        help='lr调度策略（仅保留 lin0 和 cos）')
    parser.add_argument('--wp0', type=float, default=0.005,
                        help='warmup期间起始lr比例')
    parser.add_argument('--wpe', type=float, default=0.01,
                        help='warmup期间最终lr比例')

    # ===================== 其他 =====================
    parser.add_argument('--seed', type=int, default=None,
                        help='基础随机种子')
    parser.add_argument('--workers', type=int, default=4,
                        help='dataloader线程数')
    parser.add_argument('--tf32', type=bool, default=True,
                        help='是否使用TensorFloat32加速')
    parser.add_argument('--output_dir', type=str, default='./local_output',
                        help='输出目录路径')

    args = parser.parse_args()

    # ===================== 自动计算参数 =====================
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.patch_nums = PATCH_NUMS
    args.patch_size = PATCH_SIZE
    args.resos = RESOS
    args.data_load_reso = DATA_LOAD_RESO

    # 计算学习率：lr = 基础lr × (batch_size / 256)
    args.tlr = args.tblr * args.bs / 256
    # 权重衰减的最终值 = 传入值 或 权重衰减的初始值
    args.twde = args.twde if hasattr(args, 'twde') and args.twde else args.twd
    # warmup epoch 默认为总epoch的 1/50
    args.wp = args.ep * 1 / 50

    # 模型深度自动推导的超参数
    args.heads = args.depth
    args.embed_dim = args.depth * 64
    args.drop_path_rate = 0.1 * args.depth / 24

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def seed_everything(seed: int, benchmark: bool = True):
    """设置随机种子，保证可复现"""
    torch.backends.cudnn.enabled = True
    # 不开启渐进式训练时，所有批次的训练数据长度一致，可以开启cudnn benchmark（加速卷积）
    torch.backends.cudnn.benchmark = benchmark

    if seed is None:
        torch.backends.cudnn.deterministic = False
    else:
        # 固定种子，保证可复现
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


def set_tf32(tf32: bool):
    """开启卷积/矩阵乘法的TF32加速"""
    if torch.cuda.is_available():
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high' if tf32 else 'highest')
        print(f'[tf32] cudnn: {torch.backends.cudnn.allow_tf32}, matmul: {torch.backends.cuda.matmul.allow_tf32}')
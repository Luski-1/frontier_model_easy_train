from transformers import PretrainedConfig
from dataclasses import dataclass
from typing import Tuple


@dataclass
class VAEConfig(PretrainedConfig):
    model_type = "vae"  # 注册模型

    # 图像参数
    in_channels: int = 3   # 图像通道数，MNIST（黑白图）为1，CELEBA（RGB图）为3
    image_size: int = 512  # 输入图像的尺寸

    # 隐空间参数
    latent_dim: int = 64    # 隐空间通道数

    # 架构参数
    ch: int = 64
    target_final_res: int = 8  # 最终下采样的尺寸
    dropout: float = 0.0 # 假设dropout是0.5，训练时其余非失活的神经元会自动* 1 / dropout
    use_gap: bool = False  # mu与logvar的方式，默认采取flatten而不是average pool
    channel_mult: Tuple = (1, 2, 4, 8)  # 每个阶段的channel，是基础通道的倍数，对应ch, ch*2, ch*4, ch*8
    num_res_blocks: int = 2  # 每个阶段的残差块数量
    attention_resolutions: Tuple = (32, 16, 8)  # 下采样到达指定分辨率时开启Attention（32*32和16x16和8x8）
    num_groups: int = 32  # GroupNorm的组数，GroupNorm比BatchNorm更适合小batch

    # 损失参数
    scaling_factor: float = 1.0  # 隐空间均值缩放因子；如果是SD模型，建议0.18215；如果是VAE模型，建议1.0
    kld_weight: float = 1.0  # 控制KL散度损失的权重，VAE默认是1.0
    recon_loss_type: str = "mse"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
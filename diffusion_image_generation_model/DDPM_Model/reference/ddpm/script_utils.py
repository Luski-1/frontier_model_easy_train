import argparse
import torchvision
import torch.nn.functional as F

from .unet import UNet
from .diffusion import (
    GaussianDiffusion,
    generate_linear_schedule,
    generate_cosine_schedule,
)


def cycle(dl):
    """
    https://github.com/lucidrains/denoising-diffusion-pytorch/
    """
    while True:
        for data in dl:
            yield data

def get_transform():
    class RescaleChannels(object):
        def __call__(self, sample):
            return 2 * sample - 1

    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),      # [0, 255] 离散 > [0, 1] 连续
        RescaleChannels(),                      # [0, 1] > [-1, 1]
    ])


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def add_dict_to_argparser(parser, default_dict):
    """
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/script_util.py
    """
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def diffusion_defaults():
    defaults = dict(
        num_timesteps=1000,             # 时间步总数
        schedule="linear",              # 时间步调度
        loss_type="l2",                 # 损失函数
        use_labels=False,               # 使用分类信息

        base_channels=128,              # 模型基础Channel
        channel_mults=(1, 2, 2, 2),     # 基础Channel倍数
        num_res_blocks=2,               # 残差块数量
        time_emb_dim=128 * 4,           # 时间步向量维度
        norm="gn",                      # 归一化方式
        dropout=0.1,                    # dropout比例
        activation="silu",              # 激活函数方式
        attention_resolutions=(1,),     # 第几层使用Attention

        ema_decay=0.9999,               # 指数移动平均的更新比例
        ema_update_rate=1,              # 指数移动平均的更新间隔
    )

    return defaults


def get_diffusion_from_args(args):
    activations = {
        "relu": F.relu, #   relu = max(0, x)
        "mish": F.mish, #   
        "silu": F.silu, #   silu = x * sigmoide(x)
    }

    model = UNet(
        img_channels=3,                                     # 图像channel

        base_channels=args.base_channels,                   # 模型基础Channel
        channel_mults=args.channel_mults,                   # 基础Channel倍数
        time_emb_dim=args.time_emb_dim,                     # 时间步向量维度
        norm=args.norm,                                     # 归一化方式
        dropout=args.dropout,                               # dropout比例
        activation=activations[args.activation],            # 激活函数方式
        attention_resolutions=args.attention_resolutions,   # 底基层使用Attention

        num_classes=None if not args.use_labels else 10,    # 分类信息
        initial_pad=0,                                      # 填充长度
    )

    if args.schedule == "cosine":
        betas = generate_cosine_schedule(args.num_timesteps)    # 1000
    else:
        betas = generate_linear_schedule(
            args.num_timesteps,
            args.schedule_low * 1000 / args.num_timesteps,  # 0.02
            args.schedule_high * 1000 / args.num_timesteps, # 0.0001
        )

    diffusion = GaussianDiffusion(
        model, (32, 32), 3, 10,
        betas,
        ema_decay=args.ema_decay,
        ema_update_rate=args.ema_update_rate,
        ema_start=2000,
        loss_type=args.loss_type,
    )

    return diffusion
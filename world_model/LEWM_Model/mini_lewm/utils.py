import json
import os
from pathlib import Path

import yaml


def load_config(path):
    """从 YAML 文件加载配置，确保数值字段为正确的类型。"""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # 确保常见数值字段为 float/int（某些 PyYAML 版本会将科学记数法解析为 str）
    _NUMERIC_KEYS = {
        "lr", "weight_decay", "gradient_clip_val",
        "train_split", "sigreg_weight", "warmup"
    }
    for key in _NUMERIC_KEYS:
        if key in cfg and isinstance(cfg[key], str):
            try:
                cfg[key] = float(cfg[key])
            except ValueError:
                pass
    return cfg


def create_optimizer(model, cfg):
    """显式创建优化器。

    去掉 spt.Module 的 dict 配置和 regex 模块匹配，
    只创建一个 AdamW 优化器用于全部参数。
    """
    import torch
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )


def create_scheduler(optimizer, cfg, max_epochs, steps_per_epoch):
    """显式创建学习率调度器。

    线性 warmup + cosine annealing。
    """
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    total_steps = max_epochs * steps_per_epoch

    # 原项目 LinearWarmupCosineAnnealingLR 默认 warmup_start_steps = int(0.01 * estimated_stepping_batches)
    warmup_steps = max(int(cfg["warmup"] * total_steps), 1)

    # warmup 阶段: lr 从 0.01*lr 线性增加到 lr（原项目 start_factor=0.01）
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                       total_iters=warmup_steps)
    # cosine 阶段: lr 从 cfg["lr"] 衰减到 0
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                              milestones=[warmup_steps])
    return scheduler


def save_checkpoint(model, optimizer, scheduler, epoch, save_dir, cfg, action_dim):
    """保存 checkpoint。

    保存两个文件：
    - weights.pt: 原始 state_dict（torch.save(model.state_dict())）
    - config.json: 纯超参数 JSON，用于 load_model 重建模型

    Parameters
    ----------
    action_dim : int
        原始 action 维度，运行时从数据集获取（pusht=2）
    """
    import torch

    os.makedirs(save_dir, exist_ok=True)

    # 1. 保存 state_dict
    torch.save(model.state_dict(), os.path.join(save_dir, "weights.pt"))

    # 2. 保存纯超参数（不需要 _target_，模型结构由 build_model 固定）
    model_config = generate_model_config(cfg, action_dim)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(model_config, f, indent=2)

    print(f"Checkpoint saved: {save_dir}/weights.pt (epoch {epoch})")


def generate_model_config(cfg, action_dim):
    """从训练配置中提取模型超参数，保存为纯 JSON。

    模型结构由 build_model() 固定，config.json 只需存储超参数，
    不需要 _target_ 等 hydra 格式。

    Parameters
    ----------
    cfg : dict
        训练配置（train.yaml 的内容）
    action_dim : int
        原始 action 维度（pusht=2），运行时从数据集获取
    """
    return {
        "encoder_size": cfg["encoder_size"],
        "encoder_patch_size": cfg["encoder_patch_size"],
        "img_size": cfg["img_size"],
        "embed_dim": cfg["embed_dim"],
        "history_size": cfg["history_size"],
        "frameskip": cfg["frameskip"],
        "action_dim": action_dim,
        "predictor_depth": cfg["predictor_depth"],
        "predictor_heads": cfg["predictor_heads"],
        "predictor_mlp_dim": cfg["predictor_mlp_dim"],
        "predictor_dim_head": cfg["predictor_dim_head"],
        "predictor_dropout": cfg["predictor_dropout"],
    }


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """加载 checkpoint，恢复训练状态（用于断点续训，非 eval 用途）。"""
    import torch

    ckpt = torch.load(path, map_location="cpu")
    # 兼容两种格式：包装格式 或 原始 state_dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        return ckpt["epoch"]
    else:
        # 原始 state_dict 格式（load_pretrained 格式）
        model.load_state_dict(ckpt)
        return 0


class AverageMeter:
    """简单的滑动平均工具，替代 Lightning 的 log_dict。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0.0


# =====================================================================
# build_model / load_model — 从 checkpoint 重建模型
#
# build_model: 根据超参数字典直接构建 JEPA 模型，无需 _target_ 或
#   importlib 动态导入。模型结构固定，只需传入超参数。
# load_model: 从 checkpoint 目录读取 config.json → build_model →
#   加载 weights.pt，返回可推理的模型。
# =====================================================================

def build_model(cfg):
    """根据超参数字典构建 JEPA 模型。

    替代原来的 hydra _target_ / instantiate() 机制。
    mini_lewm 只有一种模型架构（JEPA），直接硬编码构建逻辑，
    不需要动态导入。

    Parameters
    ----------
    cfg : dict
        包含以下键的超参数字典（即 config.json 的内容）：
        encoder_size, encoder_patch_size, img_size, embed_dim,
        history_size, frameskip, action_dim, predictor_depth,
        predictor_heads, predictor_mlp_dim, predictor_dim_head,
        predictor_dropout

    Returns
    -------
    model : mini_lewm.model.JEPA
    """
    import torch.nn as nn
    from model import (
        JEPA, ARPredictor, Embedder, MLP, create_encoder,
    )

    encoder = create_encoder(
        size=cfg["encoder_size"],
        patch_size=cfg["encoder_patch_size"],
        image_size=cfg["img_size"],
        pretrained=False,
    )
    predictor = ARPredictor(
        num_frames=cfg["history_size"],
        input_dim=cfg["embed_dim"],
        hidden_dim=cfg["embed_dim"],
        output_dim=cfg["embed_dim"],
        depth=cfg["predictor_depth"],
        heads=cfg["predictor_heads"],
        mlp_dim=cfg["predictor_mlp_dim"],
        dim_head=cfg["predictor_dim_head"],
        dropout=cfg["predictor_dropout"],
        emb_dropout=0.0,
    )
    action_encoder = Embedder(
        input_dim=cfg["frameskip"] * cfg["action_dim"],
        emb_dim=cfg["embed_dim"],
    )
    projector = MLP(
        input_dim=cfg["embed_dim"],
        output_dim=cfg["embed_dim"],
        hidden_dim=2048,
        norm_fn=nn.BatchNorm1d,
    )
    pred_proj = MLP(
        input_dim=cfg["embed_dim"],
        output_dim=cfg["embed_dim"],
        hidden_dim=2048,
        norm_fn=nn.BatchNorm1d,
    )

    return JEPA(encoder, predictor, action_encoder, projector, pred_proj)


def load_model(ckpt_path, extra_args=None):
    """从 checkpoint 加载模型。

    checkpoint 路径可以是：
    - .pt 文件路径 → 读同目录下的 config.json
    - 目录路径 → 在目录中找唯一的 .pt 文件

    config.json 是纯超参数 JSON（无 _target_），通过 build_model() 构建模型。

    Parameters
    ----------
    ckpt_path : str
        checkpoint 文件路径或目录路径
    extra_args : dict, optional
        额外的配置覆盖（与原 load_pretrained 兼容）

    Returns
    -------
    model : torch.nn.Module
    """
    import torch

    ckpt_path = Path(ckpt_path)

    # 解析 .pt 文件路径
    if ckpt_path.is_dir():
        pt_files = list(ckpt_path.glob("*.pt"))
        if len(pt_files) == 0:
            raise FileNotFoundError(f"No .pt file found in {ckpt_path}")
        if len(pt_files) > 1:
            raise ValueError(
                f"Ambiguous: multiple .pt files in {ckpt_path}: "
                f"{[f.name for f in pt_files]}. Specify the file directly."
            )
        checkpoint_path = pt_files[0]
    else:
        checkpoint_path = ckpt_path

    # 读 config.json（纯超参数格式）
    config_path = checkpoint_path.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found at {config_path}")
    with open(config_path) as f:
        config = json.load(f)

    # 额外参数覆盖
    if extra_args is not None:
        config.update(extra_args)

    # 构建模型 + 加载权重
    model = build_model(config)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model

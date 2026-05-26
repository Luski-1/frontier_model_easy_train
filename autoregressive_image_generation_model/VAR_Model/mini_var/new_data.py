"""
精简版 VAR 项目数据加载模块
原文件：utils/data.py
改动：去掉分布式 sampler，使用简单 DataLoader，固定 256×256 分辨率
"""

import os.path as osp

import PIL.Image as PImage
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms
from torch.utils.data import DataLoader

from new_args import DATA_LOAD_RESO


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


def print_aug(transform, label):
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')


def build_dataset(
    data_path: str,    # 数据集根目录路径（必填，如ImageNet的根路径）
    final_reso: int = DATA_LOAD_RESO,   # 模型最终接收的图像分辨率（256）
    hflip=False,       # 是否开启随机水平翻转（训练增强，默认关闭）
    mid_reso=1.125,    # 中间缩放比例：先把图片缩放到 1.125*最终分辨率，再裁剪
):
    """
    核心功能：
    1. 定义训练/验证的图像预处理/增强流水线
    2. 加载ImageNet格式的数据集（train/val文件夹结构）
    3. 返回 类别数、训练集DataLoader、验证集DataLoader
    """

    # ===================== 第一步：计算中间缩放分辨率 =====================
    # 计算中间尺寸：先将图片短边缩放到 mid_reso（1.125倍最终分辨率），为后续裁剪做准备
    mid_reso = round(mid_reso * final_reso)

    # ===================== 第二步：定义图像预处理/增强流水线 =====================
    # 定义【训练集】增强列表
    train_aug = [
        # 1. 缩放：将图片**短边**缩放到 mid_reso，使用高质量LANCZOS插值算法
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
        # 2. 随机裁剪：从缩放后的图片中，随机裁剪出 (final_reso, final_reso) 大小的区域
        transforms.RandomCrop((final_reso, final_reso)),
        # 3. 转张量：将PIL图片(H,W,C) → PyTorch张量(C,H,W)，像素值归一化到 [0, 1]
        transforms.ToTensor(),
        # 4. 归一化：将像素值从 [0, 1] → [-1, 1]（适配VAE模型的输入要求）
        normalize_01_into_pm1,
    ]

    # 定义【验证集】增强列表（无随机增强，保证评估公平性）
    val_aug = [
        # 1. 缩放：和训练集一致
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
        # 2. 中心裁剪：从缩放后的图片**正中心**裁剪（无随机性）
        transforms.CenterCrop((final_reso, final_reso)),
        # 3. 转张量 + 归一化：和训练集完全一致
        transforms.ToTensor(),
        normalize_01_into_pm1,
    ]

    # 如果开启水平翻转，在训练增强的**最前面**插入随机水平翻转
    if hflip:
        train_aug.insert(0, transforms.RandomHorizontalFlip())

    # 将增强列表组合为 Compose
    train_aug = transforms.Compose(train_aug)  # 训练集完整增强流水线
    val_aug = transforms.Compose(val_aug)      # 验证集完整预处理流水线

    # ===================== 第三步：加载数据集 =====================
    # 构建【训练集】：使用PyTorch官方DatasetFolder，根据文件夹自行分类的数据加载器
    # root：数据集训练集路径 → data_path/train
    # loader：自定义的pil_loader（保证图片为RGB格式）
    # extensions：只加载图片文件（jpg/png等，由IMG_EXTENSIONS定义）
    train_set = DatasetFolder(
        root=osp.join(data_path, 'train'),
        loader=pil_loader,
        extensions=IMG_EXTENSIONS,
        transform=train_aug
    )

    # 构建【验证集】
    val_set = DatasetFolder(
        root=osp.join(data_path, 'val'),
        loader=pil_loader,
        extensions=IMG_EXTENSIONS,
        transform=val_aug
    )

    # ===================== 第四步：配置参数 + 打印日志 =====================
    # 固定类别数=1000（专为ImageNet-1K数据集设计）
    num_classes = 1000

    # 打印数据集核心信息
    print(f'[Dataset] {len(train_set)=}, {len(val_set)=}, {num_classes=}')
    print_aug(train_aug, '[train]')
    print_aug(val_aug, '[val]')

    # ===================== 第五步：构建 DataLoader =====================
    return num_classes, train_set, val_set


def build_dataloaders(args):
    """
    构建训练和验证的 DataLoader（单卡，无分布式）
    """
    num_classes, train_set, val_set = build_dataset(
        args.data_path, final_reso=args.data_load_reso, hflip=args.hflip
    )

    ld_train = DataLoader(
        dataset=train_set,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    ld_val = DataLoader(
        dataset=val_set,
        batch_size=args.bs,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    return num_classes, ld_train, ld_val
# =====================================================================
# dataset.py — HDF5 数据集加载
#
# 自写版本，替代 swm.data.load_dataset + spt.data.transforms。
# 数据结构与原版 stable_worldmodel HDF5 文件格式完全一致：
#   - ep_len:      每回合的总帧数 [B]
#   - ep_offset:   每回合在总数据中的起始索引（绝对位置） [B]
#   - pixels:      RGB 图像 [N, H, W, 3]
#   - action:      动作 [N, 2]（dim 1 代表动作的 x, y 轴）
#   - proprio:     机器人本体状态 [N, 4]（可选）
#   - state:       环境状态 [N, 7]（可选）
#
# +------------+------------------------------------------------------+
# | dataset参数名 | 含义 |
# +------------+------------------------------------------------------+
# | num_steps | 最终想要多少个连续的数据点（模型输入的序列长度） |
# +------------+------------------------------------------------------+
# | frameskip | 从原始数据里，每隔几帧取1个数据点（跳帧降采样，降低数据量） |
# +------------+------------------------------------------------------+
# | span | 为了取到这些点，原始数据必须占多少连续帧（总跨度 / 最小长度） |
# +------------+------------------------------------------------------+
# =====================================================================

import logging
from pathlib import Path

import hdf5plugin  # noqa: F401  注册 HDF5 压缩滤镜，与原项目一致
import h5py
import numpy as np
import PIL.Image
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import tv_tensors
from torchvision.transforms import v2

# ImageNet 归一化参数
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@torch.jit.unused
def to_image(
    input: torch.Tensor | PIL.Image.Image | np.ndarray,
) -> tv_tensors.Image:
    """将输入转换为 tv_tensors.Image（CHW 格式）。

    与原项目 spt.transforms.to_image 完全一致。
    """
    if isinstance(input, np.ndarray):
        output = torch.from_numpy(np.atleast_3d(input)).transpose(-3, -1).contiguous()
    elif isinstance(input, PIL.Image.Image):
        output = torchvision.transforms.functional.pil_to_tensor(input)
    elif isinstance(input, torch.Tensor):
        output = input
    else:
        raise TypeError(
            f"Input can either be a pure Tensor, a numpy array, or a PIL image, but got {type(input)} instead."
        )
    return tv_tensors.Image(output)


def get_transform(img_size=224):
    """图像预处理 pipeline。

    to_image > ToDtype(scale=True) > Normalize(ImageNet) > Resize

    使用 v2.Compose 包裹 v2 transform，与原项目一致。
    """
    return v2.Compose([
        to_image,                                    # numpy/tensor -> tv_tensors.Image
        v2.ToDtype(torch.float32, scale=True),        # uint8[0,255] -> float32[0,1]
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),  # ImageNet 归一化
        v2.Resize(img_size),                     # resize 到指定尺寸
    ])


class PushTDataset(Dataset):
    """从 HDF5 文件加载 PushT 专家演示数据。

    数据布局：HDF5 文件中按 episode 分段存储，每个 episode 包含连续的 frames。
    每条数据样本包含 num_steps 个时间步的连续观测。

    Parameters
    ----------
    h5_path : str
        HDF5 文件路径
    num_steps : int
        每条样本包含的时间步数（= history_size + num_preds）
    frameskip : int
        从原始数据中每隔多少帧取一个数据点（降采样）
    keys_to_load : list[str]
        要加载的数据列（如 ['pixels', 'action', 'proprio', 'state']）
    keys_to_cache : list[str]
        完全加载到内存的列（低维数据，如 action/proprio/state）
    transform : callable or None
        对像素样本施加的图像变换函数
    """

    def __init__(
        self,
        h5_path: str,
        num_steps: int = 4,
        frameskip: int = 5,
        keys_to_load: list[str] | None = None,
        keys_to_cache: list[str] | None = None,
        transform=None,
        normalizers: dict[str, 'ZScoreNormalizer'] | None = None,
    ):
        self.h5_path = Path(h5_path)
        self.frameskip = frameskip       # 5 每5帧抽1帧
        self.num_steps = num_steps       # 4 每次预测需要多少步
        self.span = num_steps * frameskip  # 20 每次预测需要多少帧
        self.transform = transform       # None
        self.normalizers = normalizers or {}  # {col: ZScoreNormalizer}

        self._cache: dict[str, np.ndarray] = {}
        self.h5_file: h5py.File | None = None

        with self._open_h5() as f:
            # ep_len 每回合的总帧数 [B]
            # ep_offset 每回合在总数据中的起始索引（绝对位置） [B]
            lengths, offsets = f['ep_len'][:], f['ep_offset'][:]
            # [pixels, action, proprio, state]
            # pixels RGB图像 [B,224,224,3]
            # action [B,2] dim 1是代表动作的x,y轴
            # proprio 可能是机器人本体状态 [B,4]
            # state 环境状态 [B,7]
            self._keys = keys_to_load or [
                k for k in f.keys() if k not in ('ep_len', 'ep_offset')
            ]
            # [action, proprio, state] — 缓存低维数据到内存
            for key in keys_to_cache or []:
                self._cache[key] = f[key][:]
                logging.info(f"Cached '{key}' from '{self.h5_path}'")

        self.lengths = lengths
        self.offsets = offsets

        # 计算每个回合，若span长度要求，遍历递加起始坐标（在当前回合内的相对位置）；
        # 例子：回合0有22帧，得到(0,0),(0,1),(0,2)
        self.clip_indices = [
            (ep, start)
            for ep, length in enumerate(lengths)
            if length >= self.span
            for start in range(length - self.span + 1)
        ]

    @property
    def column_names(self) -> list[str]:
        return self._keys

    def _open_h5(self) -> h5py.File:
        """打开 HDF5 文件。

        r 只读模式
        swmr 读取/写入互不堵塞模式
        读缓存 256MB
        """
        return h5py.File(
            self.h5_path, 'r', swmr=True, rdcc_nbytes=256 * 1024 * 1024
        )

    def _open(self) -> None:
        """确保 self.h5_file 不为空（惰性打开，支持 DataLoader 多进程）。"""
        if self.h5_file is None:
            self.h5_file = self._open_h5()

    def __getstate__(self) -> dict:
        """pickle 序列化时关闭文件句柄（DataLoader 多进程需要）。"""
        state = self.__dict__.copy()
        state['h5_file'] = None
        return state

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        """加载指定回合的一段数据。

        由 __getitem__ 调用。
        ep_idx: 回合下标
        start: 回合内的相对起始位置
        end: 回合内的相对结束位置
        """
        # 确保self.h5_file不为空
        self._open()
        # 回合的起始索引（绝对位置）+回合的相对索引（相对位置） > 特定片段的真实起始位置和真实结束位置
        g_start, g_end = (
            self.offsets[ep_idx] + start,
            self.offsets[ep_idx] + end,
        )
        steps = {}
        # self._keys = [pixels, action, proprio, state]
        # self._cache = [action, proprio, state]
        for col in self._keys:
            src = self._cache if col in self._cache else self.h5_file
            data = src[col][g_start:g_end]
            if col != 'action':
                # 常规字段每5帧抽1帧 > 动作字段保留20帧
                data = data[:: self.frameskip]
            # 字符串相关类型数据的处理
            if data.dtype == np.object_ or data.dtype.kind in ('S', 'U'):
                val = data[0] if len(data) > 0 else b''
                steps[col] = val.decode() if isinstance(val, bytes) else val
            else:
                # 数值相关类型数据的处理
                steps[col] = torch.from_numpy(data)
                # 符合该条件基本就是图像类型数据
                if data.ndim == 4 and data.shape[-1] in (1, 3):
                    # [B,H,W,C] > [B,C,H,W]
                    steps[col] = steps[col].permute(0, 3, 1, 2)

        return steps

    def __len__(self) -> int:
        return len(self.clip_indices)

    def __getitem__(self, idx: int) -> dict:
        """返回 dict: {'pixels': (T, C, H, W), 'action': (T, D), ...}"""
        ep_idx, start = self.clip_indices[idx]
        steps = self._load_slice(ep_idx, start, start + self.span)

        # 如果有 normalizers，对对应字段应用 z-score 归一化（在 reshape 之前）
        for col, normalizer in self.normalizers.items():
            if col in steps:
                steps[col] = normalizer(steps[col])

        if 'action' in steps:
            # [20,2] > [4,10] 即调整后每一步包含连续5帧的动作
            steps['action'] = steps['action'].reshape(self.num_steps, -1)

        # 如果有 transform，对像素应用变换
        if self.transform is not None and 'pixels' in steps:
            # transform 处理每帧图像，shape: (T, C, H, W) -> 逐帧变换
            transformed = []
            for i in range(steps['pixels'].size(0)):
                transformed.append(self.transform(steps['pixels'][i]))
            steps['pixels'] = torch.stack(transformed)

        return steps

    def get_dim(self, col: str) -> int:
        """获取指定列的特征维度。"""
        if col in self._cache:
            data = self._cache[col]
        else:
            self._open()
            data = self.h5_file[col][:]
        return np.prod(data.shape[1:]).item() if data.ndim > 1 else 1

    def get_col_data(self, col: str) -> np.ndarray:
        """获取指定列的全部数据（用于计算归一化参数）。"""
        if col in self._cache:
            return self._cache[col]
        self._open()
        return self.h5_file[col][:]


class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def compute_normalizers(dataset: PushTDataset, cols_to_normalize: list[str]):
    """根据数据集计算各列的 z-score 归一化参数，返回归一化函数列表。

    替代 get_column_normalizer。对每列计算 mean/std，返回 ZScoreNormalizer。

    Parameters
    ----------
    dataset : PushTDataset
    cols_to_normalize : list[str]
        需要归一化的列名（排除 pixels，因为像素通过图像 transform 归一化）

    Returns
    -------
    dict : {col_name: ZScoreNormalizer}
    """
    normalizers = {}
    for col in cols_to_normalize:
        col_data = dataset.get_col_data(col)
        data = torch.from_numpy(np.array(col_data))
        data = data[~torch.isnan(data).any(dim=1)]  # 排除含nan的行
        mean = data.mean(0, keepdim=True).clone()
        std = data.std(0, keepdim=True).clone()
        normalizers[col] = ZScoreNormalizer(mean, std)
    return normalizers


def make_dataloaders(
    h5_path: str,
    frameskip: int,
    history_size: int,
    num_preds: int,
    img_size: int,
    batch_size: int,
    num_workers: int,
    train_split: float,
    seed: int,
    keys_to_load: list[str],
    keys_to_cache: list[str]
):
    """创建训练集和验证集的 DataLoader。

    封装了整个数据加载流程：
    1. 创建 PushTDataset
    2. 计算归一化参数（action/proprio/state 用 z-score，pixels 用 ImageNet 归一化）
    3. 划分训练/验证集
    4. 创建 DataLoader

    Returns
    -------
    train_loader, val_loader, action_dim
    """
    num_steps = history_size + num_preds

    # 创建数据集（不加载像素 transform，后面在 __getitem__ 中逐帧变换）
    transform = get_transform(img_size)
    dataset = PushTDataset(
        h5_path=h5_path,
        num_steps=num_steps,
        frameskip=frameskip,
        keys_to_load=keys_to_load,
        keys_to_cache=keys_to_cache,       # action 维度低，缓存到内存
        transform=transform,
    )

    # 获取 action 维度（reshape 前的原始维度）
    action_dim = dataset.get_dim('action')

    # 计算并应用 z-score 归一化（action）
    normalizers = compute_normalizers(dataset, ['action'])
    dataset.normalizers = normalizers

    # 划分数据集
    rnd_gen = torch.Generator().manual_seed(seed)
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size], generator=rnd_gen
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=rnd_gen,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, action_dim

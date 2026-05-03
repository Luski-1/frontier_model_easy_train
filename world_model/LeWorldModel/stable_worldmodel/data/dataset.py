"""Dataset classes for episode-based reinforcement learning data."""

import logging
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

import h5py # # 核心：读写 HDF5 文件的标准库
import hdf5plugin  # noqa: F401 HDF5 压缩插件（自动注册解码器，忽略未使用警告）
import numpy as np
import torch
from PIL import Image

from stable_worldmodel.data.utils import get_cache_dir


class Dataset:
    """Base class for episode-based datasets.

    Args:
        lengths: Array of episode lengths.
        offsets: Array of episode start offsets in the data.
        frameskip: Number of frames to skip between samples.
        num_steps: Number of steps per sample.
        transform: Optional transform to apply to loaded data.
    """

    def __init__(
        self,
        lengths: np.ndarray,
        offsets: np.ndarray,
        frameskip: int = 1,
        num_steps: int = 1,
        transform: Callable[[dict], dict] | None = None,
    ) -> None:
        self.lengths = lengths
        self.offsets = offsets
        self.frameskip = frameskip
        self.num_steps = num_steps
        self.span = num_steps * frameskip
        self.transform = transform
        self.clip_indices = \
        [
            (ep, start)  # 最终存储的：第ep个回合，从start帧开始采样
            # 遍历所有回合：ep=回合索引，length=这个回合的总帧数
            for ep, length in enumerate(lengths)
            # 过滤：只有回合长度 ≥ 需要的总帧数，才能采样（否则会越界）
            if length >= self.span
            # 生成这个回合内，所有合法的起始帧
            for start in range(length - self.span + 1)
        ]

    @property
    def column_names(self) -> list[str]:
        raise NotImplementedError

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.clip_indices)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, start = self.clip_indices[idx]
        # 使用的是具体子类的clip_indices方法
        steps = self._load_slice(ep_idx, start, start + self.span)
        # 关键，这里处理action使得20,2->4,10，即20帧浓缩为4点，每点各5帧
        if 'action' in steps:
            steps['action'] = steps['action'].reshape(self.num_steps, -1)
        return steps

    def load_chunk(
        self, episodes_idx: np.ndarray, start: np.ndarray, end: np.ndarray
    ) -> list[dict]:
        chunk = []
        for ep, s, e in zip(episodes_idx, start, end):
            steps = self._load_slice(ep, s, e)
            if 'action' in steps:
                steps['action'] = steps['action'].reshape(
                    (e - s) // self.frameskip, -1
                )
            chunk.append(steps)
        return chunk

    def load_episode(self, episode_idx: int) -> dict:
        """Load full episode by index."""
        return self._load_slice(episode_idx, 0, self.lengths[episode_idx])

    def get_col_data(self, col: str) -> np.ndarray:
        raise NotImplementedError

    def get_dim(self, col: str) -> int:
        raise NotImplementedError

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        raise NotImplementedError

    def merge_col(
        self,
        source: list[str] | str,
        target: str,
        dim: int = -1,
    ) -> None:
        raise NotImplementedError


class HDF5Dataset(Dataset):
    """Dataset loading from HDF5 file.

    Reads data from a single .h5 file containing all episode data.
    Uses SWMR mode for robust reading while writing.

    Args:
        name: Name of the dataset (filename without extension).
        frameskip: Number of frames to skip between samples.
        num_steps: Number of steps per sample sequence.
        transform: Optional data transform callable.
        keys_to_load: Specific keys to load (defaults to all except metadata).
        keys_to_cache: Keys to load entirely into memory for faster access.
        cache_dir: Directory containing the dataset file.
    """

    def __init__(
        self,
        name: str,
        frameskip: int = 1,
        num_steps: int = 1,
        transform: Callable[[dict], dict] | None = None,
        keys_to_load: list[str] | None = None,
        keys_to_cache: list[str] | None = None,
        keys_to_merge: dict[str, list[str] | str] | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        # 获取数据的目录
        datasets_dir = get_cache_dir(cache_dir, sub_folder='datasets')
        # 拼接数据的完整路径
        self.h5_path = Path(datasets_dir, f'{name}.h5')
        # HDF5 文件句柄（初始为 None，延迟加载/复用）
        self.h5_file: h5py.File | None = None
        self._cache: dict[str, np.ndarray] = {}

        with h5py.File(self.h5_path, 'r') as f:
            # 读取两个核心元数据：
            # ep_len：每个回合的长度（步数）
            # ep_offset：每个回合在数据数组中的起始索引
            lengths, offsets = f['ep_len'][:], f['ep_offset'][:]
            # 如果用户指定 keys_to_load → 用指定的
            # 否则 → 加载所有键，排除元数据 ep_len/ep_offset
            self._keys = keys_to_load or [
                k for k in f.keys() if k not in ('ep_len', 'ep_offset')
            ]
            # 缓存指定的键到内存：
            # 把 keys_to_cache 里的所有数据，一次性读入 self._cache 字典
            for key in keys_to_cache or []:
                self._cache[key] = f[key][:]
                logging.info(f"Cached '{key}' from '{self.h5_path}'")

        super().__init__(lengths, offsets, frameskip, num_steps, transform)

        if keys_to_merge:
            for target, source in keys_to_merge.items():
                self.merge_col(source, target)

    @property
    def column_names(self) -> list[str]:
        return self._keys

    def _open(self) -> None:
        if self.h5_file is None:
            self.h5_file = h5py.File(
                self.h5_path, 'r', swmr=True, rdcc_nbytes=256 * 1024 * 1024
            )

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        self._open()
        # 获取指定回合的指定区间
        g_start, g_end = (
            self.offsets[ep_idx] + start,
            self.offsets[ep_idx] + end,
        )
        steps = {}
        for col in self._keys:
            src = self._cache if col in self._cache else self.h5_file
            data = src[col][g_start:g_end]
            # 跳过动作列，其他列按 frameskip 跳过
            if col != 'action':
                data = data[:: self.frameskip]
            # np.object_ 是「对象类型数组」，在此项目中可能是数据中的变长元数据，比如回合 ID、任务名称、环境配置等
            # 'S'：bytestring 字节串（Python 的bytes类型，HDF5 存字符串的默认格式）
            # 'U'：Unicode 字符串
            if data.dtype == np.object_ or data.dtype.kind in ('S', 'U'):
                val = data[0] if len(data) > 0 else b''
                # 如果是bytes类型，就用decode()转成 UTF-8 格式的str；如果已经是str类型，就直接使用
                steps[col] = val.decode() if isinstance(val, bytes) else val
            else:
            # pusht.yaml配置为例：frameskip=5、num_steps=4、end-start=20（总跨度），转换后的结果：
            # action：numpy 数组(20, 2) → 转成shape=(20, 2)的torch.float32张量
            # proprio/state：跳帧后 numpy 数组(4, D) → 转成shape=(4, D)的torch.float32张量
            # pixels：跳帧后 numpy 数组(4, 224, 224, 3) → 转成shape=(4, 224, 224, 3)的torch.uint8张量
                steps[col] = torch.from_numpy(data)
            # 4 个维度的原始顺序是 [T, H, W, C]：
            # - T：时间步 / 帧数（你的num_steps=4，就是 4 帧）
            # - H：图像高度（你的img_size=224）
            # - W：图像宽度（224）
            # - C：通道数

            # 数组最后一维（也就是通道数 C）只能是 1 或 3：
            # - 1 = 灰度图，3=RGB 彩色图，是 CV 领域唯二的标准图像格式
                if data.ndim == 4 and data.shape[-1] in (1, 3):
                    steps[col] = steps[col].permute(0, 3, 1, 2)

        return self.transform(steps) if self.transform else steps

    def _get_col(self, col: str) -> np.ndarray:
        if col in self._cache:
            return self._cache[col]
        self._open()
        return self.h5_file[col][:]

    def get_col_data(self, col: str) -> np.ndarray:
        return self._get_col(col)

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        self._open()
        return {col: self.h5_file[col][row_idx] for col in self._keys}

    def merge_col(
        self,
        source: list[str] | str,
        target: str,
        dim: int = -1,
    ) -> None:
        self._open()

        if isinstance(source, str):
            source = [k for k in self.h5_file.keys() if re.match(source, k)]

        merged = np.concatenate([self._get_col(s) for s in source], axis=dim)
        self._cache[target] = merged
        if target not in self._keys:
            self._keys.append(target)
        logging.info(f"Merged columns {source} into '{target}' and cached it")

    def get_dim(self, col: str) -> int:
        data = self.get_col_data(col)
        return np.prod(data.shape[1:]).item() if data.ndim > 1 else 1

from typing import Any, Union

import numpy as np
import PIL.Image
import torch
import torchvision

from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode


class Transform(v2.Transform):
    """Base transform class extending torchvision v2.Transform with nested data handling."""

    def single_nested_get(self, v, name):
        # 如果路径是空字符串，直接返回原值
        if name == "":
            return v
        # 把路径按 . 拆分（支持嵌套：如 "obs.pixels" → ["obs", "pixels"]）
        i = name.split(".")
        # 如果第一个节点是数字（列表索引，如 "0.image"），转成整数
        if i[0].isnumeric():
            i[0] = int(i[0])
        # 递归：取第一层的值，再处理剩余路径
        return self.single_nested_get(v[i[0]], ".".join(i[1:]))

    def nested_get(self, v, name):
        # name = "pixels"，不是列表/元组，跳过if分支
        if type(name) in [list, tuple]:
            return [self.single_nested_get(v, n) for n in name]
        # 直接执行：调用 single_nested_get，参数 v=x，name="pixels"
        return self.single_nested_get(v, name)

    def single_nested_get(self, v, name):
        # v = 输入字典x，name = "pixels"
        # 1. name不是空字符串，跳过
        # 递归调用到此处，name=""，即直接返回
        if name == "":
            return v
        # 2. 按.拆分路径："pixels"无. → i = ["pixels"]
        i = name.split(".")
        # 3. i[0] = "pixels" 不是数字，不转换
        if i[0].isnumeric():
            i[0] = int(i[0])
        # 4. 递归调用：
        # v[i[0]] = x["pixels"] → 原始numpy图像数组
        # ".".join(i[1:]) → 空字符串""
        return self.single_nested_get(v[i[0]], ".".join(i[1:]))

    def nested_set(self, original, value, name):
        # 如果name是列表/元组，批量赋值
        if type(name) in [list, tuple]:
            assert type(value) in [list, tuple]
            assert len(value) == len(name)
            return [self.single_nested_set(original, v, n) for v, n in zip(value, name)]
        # 否则，处理单个路径
        return self.single_nested_set(original, value, name)

    def get_name(self, x):
        base = self.name
        assert "_" not in base
        if base not in x:
            return base
        ctr = 0
        while f"{base}_{ctr}" in base:
            ctr += 1
        return f"{base}_{ctr}"

    @property
    def name(self):
        return self.__class__.__name__


@torch.jit.unused
def to_image(
        input: Union[torch.Tensor, PIL.Image.Image, np.ndarray],
) -> tv_tensors.Image:
    """See :class:`~torchvision.transforms.v2.ToImage` for details."""
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


class ToImage(Transform):
    """
    Convert input to image tensor with optional normalization.
    把数据集中的原始图像（numpy 数组 / PIL 图 / 张量） → 转换成 PyTorch 标准图像张量，并自动做数据类型转换 + 数值缩放 + 归一化
    """

    def __init__(
            self,
            dtype=torch.float32,
            scale=True,
            mean=None,
            std=None,
            source: str = "image",
            target: str = "image",
    ):
        super().__init__()
        t = [to_image,  # 统一把 numpy/PIL/ 张量 → tv_tensors.Image（Torchvision 标准图像类型）
             v2.ToDtype(dtype, scale=scale)]  # 转 float32 + 自动缩放（uint8(0-255) → float32(0-1)）
        if mean is not None and std is not None:
            t.append(v2.Normalize(mean=mean, std=std))
        self.t = v2.Compose(t)
        self.source = source
        self.target = target

    def __call__(self, x):
        # 对x递归取值 → 预处理 → 对x递归存值
        # self.source = "pixel"，非列表
        # x = {
        #     "pixels": np.array([[[0, 0, 0], ...]]),  # 原始图像
        #     "action": np.array([0.1, 0.2]),
        #     "proprio": np.array([0.5, 0.3])
        # }
        self.nested_set(x, self.t(self.nested_get(x, self.source)), self.target)
        return x


class WrapTorchTransform(Transform, v2.Lambda):
    """Applies a lambda callable to target key and store it in source."""

    def __init__(self, transform, source: str = "image", target: str = "image"):
        super().__init__(transform)
        self.source = source
        self.target = target

    def __call__(self, x) -> Any:
        self.nested_set(
            x, super().__call__(self.nested_get(x, self.source)), self.target
        )
        return x


class Resize(Transform, v2.Resize):
    """Resize image to specified size."""

    def __init__(
            self,
            size,
            interpolation=2,
            max_size=None,
            antialias=True,
            source="image",
            target="image",
    ) -> None:
        super().__init__(size, interpolation, max_size, antialias)
        self.source = source
        self.target = target

    def __call__(self, x):
        self.nested_set(
            x, self.transform(self.nested_get(x, self.source), []), self.target
        )
        return x


class Compose(v2.Transform):
    """Compose multiple transforms together in sequence."""

    def __init__(self, *args):
        super().__init__()
        self.args = args

    def __call__(self, sample):
        for a in self.args:
            sample = a(sample)
        return sample

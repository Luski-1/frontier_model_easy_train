"""Layers used for up-sampling or down-sampling images.

Many functions are ported from https://github.com/NVlabs/stylegan2.
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from op import upfirdn2d


# Function ported from StyleGAN2
def get_weight(module,
               shape,
               weight_var='weight',
               kernel_init=None):
  """Get/create weight tensor for a convolution or fully-connected layer."""

  return module.param(weight_var, kernel_init, shape)


class Conv2d(nn.Module):
  """Conv2d layer with optimal upsampling and downsampling (StyleGAN2)."""

  def __init__(self, in_ch, out_ch, kernel, up=False, down=False,
               resample_kernel=(1, 3, 3, 1),
               use_bias=True,
               kernel_init=None):
    # kernel=3
    # resample_kernel=(1, 3, 3, 1)
    super().__init__()
    assert not (up and down)
    assert kernel >= 1 and kernel % 2 == 1
    self.weight = nn.Parameter(torch.zeros(out_ch, in_ch, kernel, kernel))  # 卷积核[out_ch, in_ch, 3, 3]
    if kernel_init is not None:
      self.weight.data = kernel_init(self.weight.data.shape)  # 还是使用默认scale=1
    if use_bias:
      self.bias = nn.Parameter(torch.zeros(out_ch))

    self.up = up
    self.down = down
    self.resample_kernel = resample_kernel  # (1, 3, 3, 1)
    self.kernel = kernel  # 3
    self.use_bias = use_bias  # True

  def forward(self, x):
    if self.up:
      # 上采样+卷积
      x = upsample_conv_2d(x, self.weight, k=self.resample_kernel)
    elif self.down:
      # 卷积+下采样
      x = conv_downsample_2d(x, self.weight, k=self.resample_kernel)
    else:
      x = F.conv2d(x, self.weight, stride=1, padding=self.kernel // 2)

    if self.use_bias:
      x = x + self.bias.reshape(1, -1, 1, 1)

    return x


def naive_upsample_2d(x, factor=2):
  _N, C, H, W = x.shape
  # 转换维度，使得每个元素扩增为factor * factor的区间
  x = torch.reshape(x, (-1, C, H, 1, W, 1))
  x = x.repeat(1, 1, 1, factor, 1, factor)
  return torch.reshape(x, (-1, C, H * factor, W * factor))


def naive_downsample_2d(x, factor=2):
  _N, C, H, W = x.shape
  # 转换维度，使得factor * factor的区间进行求均值
  x = torch.reshape(x, (-1, C, H // factor, factor, W // factor, factor))
  return torch.mean(x, dim=(3, 5))


def upsample_conv_2d(x, w, k=None, factor=2, gain=1):
  """Fused `upsample_2d()` followed by `tf.nn.conv2d()`.

     Padding is performed only once at the beginning, not between the
     operations.
     The fused op is considerably more efficient than performing the same
     calculation
     using standard TensorFlow ops. It supports gradients of arbitrary order.
     Args:
       x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
         C]`.
       w:            Weight tensor of the shape `[filterH, filterW, inChannels,
         outChannels]`. Grouped convolution can be performed by `inChannels =
         x.shape[0] // numGroups`.
       k:            FIR filter of the shape `[firH, firW]` or `[firN]`
         (separable). The default is `[1] * factor`, which corresponds to
         nearest-neighbor upsampling.
       factor:       Integer upsampling factor (default: 2).
       gain:         Scaling factor for signal magnitude (default: 1.0).

     Returns:
       Tensor of the shape `[N, C, H * factor, W * factor]` or
       `[N, H * factor, W * factor, C]`, and same datatype as `x`.
  """
  # k=(1,3,3,1)
  assert isinstance(factor, int) and factor >= 1

  # Check weight shape.
  assert len(w.shape) == 4
  convH = w.shape[2]  # 3
  convW = w.shape[3]  # 3
  inC = w.shape[1]
  outC = w.shape[0]

  assert convW == convH

  # 想理解下面的代码，需要先理解F.conv_transpose2d（本人感觉目前主流都是使用torch.nn.functional.interpolate进行插值上采样）
  # F.conv_transpose2d有两种理解方式
  # a)
  # 对x（2*2）的每个元素，单独逐元素相乘kernal（假设3*3），那么每个元素变成3*3大小
  # 随后根据stride进行排布，例如处于[0,0]的元素要霸占[0-2, 0-2]位置，处于[0,1]的的元素要霸占[0-2,1-3]位置，隔了stride步。
  # 同理[1,0]的元素要霸占[1-3,0-2]位置，[1,1]的元素要霸占[1-3,1-3]位置
  # 重叠位置的数值进行相加即可
  # b)
  # 在每个元素之间插入stride-1个0
  # 旋转卷积核180°
  # 两边各做kernal_size - 1 - padding的填充0向量，随后进行stride固定=1的普通卷积
  # PS：padding在F.conv_transpose2d的参数中是代表要裁剪两边各多少元素，output_padding是仅在最后结果的右边/下边填充多少0元素
  # 还要理解卷积和转置卷积的分辨率计算公式
  # 卷积 = (input_size + 2 * padding - kernal_size) / 2
  # 转置卷积 = (input_size - 1) * stride - 2 * padding + kernal_size  + output_padding


  # Setup filter kernel.
  if k is None:
    k = [1] * factor  # 如果没提供k，那么就是[1, 1]
  # array([[1/sum, 3/sum, 3/sum, 1/sum],
  #        [3/sum, 9/sum, 9/sum, 3/sum],
  #        [3/sum, 9/sum, 9/sum, 3/sum],
  #        [1/sum, 3/sum, 3/sum, 1/sum]])  * 4

  # 按照b)思路，那么每个元素的下/右/右下都是0，需要放大4倍才能恢复信号强度
  k = _setup_kernel(k) * (gain * (factor ** 2))
  # p = (4 - 2) - (3 - 1) = 2 - 2 = 0，不清楚为什么这样计算，需要自行查看upfirdn2d代码逻辑
  p = (k.shape[0] - factor) - (convW - 1)
  # 设置步长
  stride = (factor, factor)

  # stride = [1, 1, factor, factor] 这里有问题，好像是tensorflow的写法
  # 计算上采样的尺寸 = ((3 - 1) * 2 + 3, (3 - 1) * 2 + 3) = (7, 7)
  output_shape = ((_shape(x, 2) - 1) * factor + convH, (_shape(x, 3) - 1) * factor + convW)
  # 计算output_padding的尺寸（意义不大，因为上方的上采样尺寸就没加上output_padding） = (7 - (3 - 1) * 2 - 3), 7 - (3 - 1) * 2 - 3) = (0, 0)
  output_padding = (output_shape[0] - (_shape(x, 2) - 1) * stride[0] - convH,
                    output_shape[1] - (_shape(x, 3) - 1) * stride[1] - convW)
  assert output_padding[0] >= 0 and output_padding[1] >= 0
  # 计算分组数，通过x的channel // 卷积核的channel
  num_groups = _shape(x, 1) // inC

  # w 维度 [num_groups, outC, inC, 3, 3]
  w = torch.reshape(w, (num_groups, -1, inC, convH, convW))
  # -1代表颠倒，参考b)思路，那么conv_transpose2d就应该自带翻转的效果，那么只能猜测作者是想获得上采样+w做普通卷积的效果
  w = w[..., ::-1, ::-1].permute(0, 2, 1, 3, 4)
  # 得到最终inC
  w = torch.reshape(w, (num_groups * inC, -1, convH, convW))
  # 上采样
  x = F.conv_transpose2d(x, w, stride=stride, output_padding=output_padding, padding=0)
  # 里面就不详细人工注释了，应该就是对x进行滤波卷积，提高图像平滑性
  return upfirdn2d(x, torch.tensor(k, device=x.device),
                   pad=((p + 1) // 2 + factor - 1, p // 2 + 1))


def conv_downsample_2d(x, w, k=None, factor=2, gain=1):
  """Fused `tf.nn.conv2d()` followed by `downsample_2d()`.

    Padding is performed only once at the beginning, not between the operations.
    The fused op is considerably more efficient than performing the same
    calculation
    using standard TensorFlow ops. It supports gradients of arbitrary order.
    Args:
        x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
          C]`.
        w:            Weight tensor of the shape `[filterH, filterW, inChannels,
          outChannels]`. Grouped convolution can be performed by `inChannels =
          x.shape[0] // numGroups`.
        k:            FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to
          average pooling.
        factor:       Integer downsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).

    Returns:
        Tensor of the shape `[N, C, H // factor, W // factor]` or
        `[N, H // factor, W // factor, C]`, and same datatype as `x`.
  """

  assert isinstance(factor, int) and factor >= 1
  _outC, _inC, convH, convW = w.shape
  assert convW == convH
  if k is None:
    k = [1] * factor  # 如果没提供k，那么就是[1, 1]
  k = _setup_kernel(k) * gain # 因为先滤波卷积，所以不需要放大信号幅度
  p = (k.shape[0] - factor) + (convW - 1) # (4 - 2) + (3 - 1) = 4 不清楚为什么这样计算，需要自行查看upfirdn2d代码逻辑
  s = [factor, factor]  # 设置补偿[2, 2]
  # 里面就不详细人工注释了，应该就是对x进行滤波卷积，提高图像平滑性
  x = upfirdn2d(x, torch.tensor(k, device=x.device),
                pad=((p + 1) // 2, p // 2))
  # 进行下采样卷积
  return F.conv2d(x, w, stride=s, padding=0)


def _setup_kernel(k):
  k = np.asarray(k, dtype=np.float32) # 得到np.array([1,3,3,1])
  if k.ndim == 1:
    # array([[1, 3, 3, 1],
    #        [3, 9, 9, 3],
    #        [3, 9, 9, 3],
    #        [1, 3, 3, 1]])
    k = np.outer(k, k)
  k /= np.sum(k)  # 归一化
  assert k.ndim == 2
  assert k.shape[0] == k.shape[1]
  return k


def _shape(x, dim):
  return x.shape[dim]


def upsample_2d(x, k=None, factor=2, gain=1):
  r"""Upsample a batch of 2D images with the given filter.

    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]`
    and upsamples each image with the given filter. The filter is normalized so
    that
    if the input pixels are constant, they will be scaled by the specified
    `gain`.
    Pixels outside the image are assumed to be zero, and the filter is padded
    with
    zeros so that its shape is a multiple of the upsampling factor.
    Args:
        x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
          C]`.
        k:            FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to
          nearest-neighbor upsampling.
        factor:       Integer upsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).

    Returns:
        Tensor of the shape `[N, C, H * factor, W * factor]`
  """
  assert isinstance(factor, int) and factor >= 1
  if k is None:
    k = [1] * factor
  k = _setup_kernel(k) * (gain * (factor ** 2))
  p = k.shape[0] - factor
  return upfirdn2d(x, torch.tensor(k, device=x.device),
                   up=factor, pad=((p + 1) // 2 + factor - 1, p // 2))


def downsample_2d(x, k=None, factor=2, gain=1):
  r"""Downsample a batch of 2D images with the given filter.

    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]`
    and downsamples each image with the given filter. The filter is normalized
    so that
    if the input pixels are constant, they will be scaled by the specified
    `gain`.
    Pixels outside the image are assumed to be zero, and the filter is padded
    with
    zeros so that its shape is a multiple of the downsampling factor.
    Args:
        x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
          C]`.
        k:            FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to
          average pooling.
        factor:       Integer downsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).

    Returns:
        Tensor of the shape `[N, C, H // factor, W // factor]`
  """

  assert isinstance(factor, int) and factor >= 1
  if k is None:
    k = [1] * factor    # 如果没提供k，那么就是[1, 1]
  k = _setup_kernel(k) * gain
  p = k.shape[0] - factor # 4 - 2 = 2 不清楚为什么这样计算，需要自行查看upfirdn2d代码逻辑
  return upfirdn2d(x, torch.tensor(k, device=x.device),
                   down=factor, pad=((p + 1) // 2, p // 2))

# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Layers for defining NCSN++.
"""
from . import layers
from . import up_or_down_sampling
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

conv1x1 = layers.ddpm_conv1x1
conv3x3 = layers.ddpm_conv3x3
NIN = layers.NIN
default_init = layers.default_init


class GaussianFourierProjection(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""
  # 相比transformer的手动设计频率的位置向量，这种方法能够得到各种各样频率的位置向量

  def __init__(self, embedding_size=256, scale=1.0):
    super().__init__()
    # embedding_size = 128
    # 获得128维，服从N(0, scale^2)分布的参数
    self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

  def forward(self, x):
    # x是经过Log处理的
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    # 构建[sin, cos]
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Combine(nn.Module):
  """Combine information from skip connections."""

  def __init__(self, dim1, dim2, method='cat'):
    super().__init__()
    self.Conv_0 = conv1x1(dim1, dim2)
    self.method = method
  # 控制Unet架构的encoder模块的每个输出结果如何对接到decoder模块的对应输入
  def forward(self, x, y):
    h = self.Conv_0(x)
    if self.method == 'cat':
      return torch.cat([h, y], dim=1)
    elif self.method == 'sum':
      return h + y
    else:
      raise ValueError(f'Method {self.method} not recognized.')


class AttnBlockpp(nn.Module):
  """Channel-wise self-attention block. Modified from DDPM."""

  def __init__(self, channels, skip_rescale=False, init_scale=0.):
    super().__init__()
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels,
                                  eps=1e-6)
    self.NIN_0 = NIN(channels, channels)  # 采用默认的指定init_scale=0.1，就是[channels,channels]的矩阵
    self.NIN_1 = NIN(channels, channels)
    self.NIN_2 = NIN(channels, channels)
    self.NIN_3 = NIN(channels, channels, init_scale=init_scale) # 指定init_scale=0.0=>init_scale=1e-10，目的是让残差分支训练初期≈0使得主干不受影响，后续慢慢修改
    self.skip_rescale = skip_rescale

  def forward(self, x):
    B, C, H, W = x.shape
    h = self.GroupNorm_0(x)
    q = self.NIN_0(h)
    k = self.NIN_1(h)
    v = self.NIN_2(h)
    # 二维矩阵间的计算score，hw和ij理解为tokens，c理解为token的hidden_dim，最后除sqrt(hidden_dim)进行点积缩放
    w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5))
    w = torch.reshape(w, (B, H, W, H * W))  # [B,H,W,HW]
    w = F.softmax(w, dim=-1)  # 即理解为每个像素点与所有像素点的注意力得分
    w = torch.reshape(w, (B, H, W, H, W))
    h = torch.einsum('bhwij,bcij->bchw', w, v)  # 就是正常的Scoree@V
    h = self.NIN_3(h)
    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)  # 除sqrt(2)，是默认残差分支的方差约等于主干的方差，进行缩放避免方差累积越来越大


class Upsample(nn.Module):
  def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False,
               fir_kernel=(1, 3, 3, 1)):
    # with_conv=True
    # fir=True
    # fir_kernel=(1, 3, 3, 1)
    super().__init__()
    out_ch = out_ch if out_ch else in_ch    # 如果没指定输出维度，那么默认与输入维度一致
    if not fir:
      if with_conv:
        # 普通的，分辨率不变的卷积
        self.Conv_0 = conv3x3(in_ch, out_ch)
    else:
      if with_conv:
        # 用于设置上采样或者下采样的模块
        self.Conv2d_0 = up_or_down_sampling.Conv2d(in_ch, out_ch,
                                                 kernel=3, up=True,
                                                 resample_kernel=fir_kernel,
                                                 use_bias=True,
                                                 kernel_init=default_init())
    self.fir = fir
    self.with_conv = with_conv
    self.fir_kernel = fir_kernel
    self.out_ch = out_ch

  def forward(self, x):
    B, C, H, W = x.shape
    # 如果不开启滤波
    if not self.fir:
      # 直接插值上采样
      h = F.interpolate(x, (H * 2, W * 2), 'nearest')
      # 如果开启初始化卷积
      if self.with_conv:
        h = self.Conv_0(h)
    else:
      # 如果不开启初始化卷积
      if not self.with_conv:
        # 直接滤波上采样
        h = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
      else:
        # 上采样（转置卷积）+滤波卷积混合
        h = self.Conv2d_0(x)

    return h


class Downsample(nn.Module):
  def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False,
               fir_kernel=(1, 3, 3, 1)):
      # with_conv=True
    # fir=True
    # fir_kernel=(1, 3, 3, 1)
    super().__init__()
    out_ch = out_ch if out_ch else in_ch # 如果没指定输出维度，那么默认与输入维度一致
    if not fir:
      if with_conv:
        # 普通的，分辨率下降的卷积
        self.Conv_0 = conv3x3(in_ch, out_ch, stride=2, padding=0)
    else:
      if with_conv:
        self.Conv2d_0 = up_or_down_sampling.Conv2d(in_ch, out_ch,
                                                 kernel=3, down=True,
                                                 resample_kernel=fir_kernel,
                                                 use_bias=True,
                                                 kernel_init=default_init())
    self.fir = fir                # True
    self.fir_kernel = fir_kernel  # (1,3,3,1)
    self.with_conv = with_conv    # True
    self.out_ch = out_ch

  def forward(self, x):
    B, C, H, W = x.shape
    # 如果不开启滤波
    if not self.fir:
      # 如果开启初始化卷积
      if self.with_conv:
        # 补右边和下边1个像素的0，进行卷积，达到缩小2倍
        x = F.pad(x, (0, 1, 0, 1))
        x = self.Conv_0(x)
      else:
        # 不开启初始化卷积的话，直接平均池化缩小2倍即可
        x = F.avg_pool2d(x, 2, stride=2)
    else:
      # 如果不开启初始化卷积
      if not self.with_conv:
        # 直接滤波下采样
        x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
      else:
        # 滤波卷积+下采样（卷积）
        x = self.Conv2d_0(x)

    return x


class ResnetBlockDDPMpp(nn.Module):
  """ResBlock adapted from DDPM."""

  def __init__(self, act, in_ch, out_ch=None, temb_dim=None, conv_shortcut=False,
               dropout=0.1, skip_rescale=False, init_scale=0.):
    super().__init__()
    out_ch = out_ch if out_ch else in_ch
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
    self.Conv_0 = conv3x3(in_ch, out_ch)
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(temb_dim, out_ch)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
      nn.init.zeros_(self.Dense_0.bias)
    self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
    self.Dropout_0 = nn.Dropout(dropout)
    self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
    if in_ch != out_ch:
      if conv_shortcut:
        self.Conv_2 = conv3x3(in_ch, out_ch)
      else:
        self.NIN_0 = NIN(in_ch, out_ch)

    self.skip_rescale = skip_rescale
    self.act = act
    self.out_ch = out_ch
    self.conv_shortcut = conv_shortcut

  def forward(self, x, temb=None):
    h = self.act(self.GroupNorm_0(x))
    h = self.Conv_0(h)
    if temb is not None:
      h += self.Dense_0(self.act(temb))[:, :, None, None]
    h = self.act(self.GroupNorm_1(h))
    h = self.Dropout_0(h)
    h = self.Conv_1(h)
    if x.shape[1] != self.out_ch:
      if self.conv_shortcut:
        x = self.Conv_2(x)
      else:
        x = self.NIN_0(x)
    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)


class ResnetBlockBigGANpp(nn.Module):
  def __init__(self, act, in_ch, out_ch=None, temb_dim=None, up=False, down=False,
               dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1),
               skip_rescale=True, init_scale=0.):
    super().__init__()

    out_ch = out_ch if out_ch else in_ch      # 如果没有指定out_ch就默认=in_ch
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6) # 设置GroupNorm
    self.up = up      # 待定
    self.down = down  # 待定
    self.fir = fir    # True
    self.fir_kernel = fir_kernel  # (1,3,3,1)

    self.Conv_0 = conv3x3(in_ch, out_ch)  # 不改分辨率的3*3卷积
    if temb_dim is not None:    # 128 * 4
      self.Dense_0 = nn.Linear(temb_dim, out_ch)  
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)  # 默认init_scale=1.0的初始化权重
      nn.init.zeros_(self.Dense_0.bias)

    self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6) # 设置GroupNorm
    self.Dropout_0 = nn.Dropout(dropout)  # 0.1
    self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)  # 不改分辨率的3*3卷积，但init_scale=0的初始化权重
    if in_ch != out_ch or up or down:     # channel对齐
      self.Conv_2 = conv1x1(in_ch, out_ch)

    self.skip_rescale = skip_rescale  # True
    self.act = act        # swish
    self.in_ch = in_ch
    self.out_ch = out_ch

  def forward(self, x, temb=None):
    h = self.act(self.GroupNorm_0(x)) # 归一化 > 激活

    if self.up:
      # 如果开启滤波
      if self.fir:
        # 残差分支进行滤波卷积上采样
        h = up_or_down_sampling.upsample_2d(h, self.fir_kernel, factor=2)
        # 主干进行滤波卷积上采样
        x = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
      else:
        # 从残差分支进行直接复制上采样
        h = up_or_down_sampling.naive_upsample_2d(h, factor=2)
        # 主干进行直接复制上采样
        x = up_or_down_sampling.naive_upsample_2d(x, factor=2)
    elif self.down:
      # 如果开启滤波
      if self.fir:
        # 残差分支进行滤波卷积下采样
        h = up_or_down_sampling.downsample_2d(h, self.fir_kernel, factor=2)
        # 主干进行滤波卷积下采样
        x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
      # 如果不开启滤波
      else:
        # 从残差分支进行均值池化下采样
        h = up_or_down_sampling.naive_downsample_2d(h, factor=2)
        # 主干进行均值池化下采样
        x = up_or_down_sampling.naive_downsample_2d(x, factor=2)

    # 共享路线，包含非下采样和非上采样
    # 1.卷积
    h = self.Conv_0(h)
    # 2.增加时间信息
    if temb is not None:
      # 激活 > 卷积
      h += self.Dense_0(self.act(temb))[:, :, None, None]
    # 3. 再次归一化 > 激活
    h = self.act(self.GroupNorm_1(h))
    # 4. dropout
    h = self.Dropout_0(h)
    # 5. 卷积（残差分支），训练早期≈0
    h = self.Conv_1(h)
    # 6. 对齐channel
    if self.in_ch != self.out_ch or self.up or self.down:
      x = self.Conv_2(x)
    # 是否开启缩放方差？
    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)

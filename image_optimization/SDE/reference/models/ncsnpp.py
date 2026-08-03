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

from . import utils, layers, layerspp, normalization
import torch.nn as nn
import functools
import torch
import numpy as np

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


@utils.register_model(name='ncsnpp')
class NCSNpp(nn.Module):
  """NCSN++ model"""

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.act = act = get_act(config)                # NCSN++ 是swish激活函数
    self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))

    self.nf = nf = config.model.nf      # channel的基础通道数 128
    ch_mult = config.model.ch_mult      # channel的倍数 (1, 2, 2, 2)
    self.num_res_blocks = num_res_blocks = config.model.num_res_blocks        # 残差块数量 4 
    self.attn_resolutions = attn_resolutions = config.model.attn_resolutions  # 进行Attention的分辨率要求 (16,)
    dropout = config.model.dropout                          # dropout=0.1
    resamp_with_conv = config.model.resamp_with_conv        # True 上采样或下采样时是否开启卷积
    self.num_resolutions = num_resolutions = len(ch_mult)   # 网络层数 4
    self.all_resolutions = all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)] # 分辨率的所有情况 [32, 16, 8, 4]

    self.conditional = conditional = config.model.conditional  # 控制时间t是否注入模型 True 该参数意义不大，肯定得要注入啊
    fir = config.model.fir                      # 开启滤波 True
    fir_kernel = config.model.fir_kernel        # 滤波核 [1, 3, 3, 1]
    self.skip_rescale = skip_rescale = config.model.skip_rescale  # True 主干+残差后，是否缩放来降低方差
    self.resblock_type = resblock_type = config.model.resblock_type.lower() # biggan 控制选择什么架构的resnet block
    self.progressive = progressive = config.model.progressive.lower() # "none"
    self.progressive_input = progressive_input = config.model.progressive_input.lower() # "residual" 控制分支选择什么架构 | DDPM++是"none"
    self.embedding_type = embedding_type = config.model.embedding_type.lower()  # "fourier" 控制位置/时间编码的方式 | DDPM++是"positional"
    init_scale = config.model.init_scale  # 0.0 残差分支的相关参数的权重初始化的方差，使得训练初期残差分支不影响主干
    assert progressive in ['none', 'output_skip', 'residual']
    assert progressive_input in ['none', 'input_skip', 'residual']
    assert embedding_type in ['fourier', 'positional']
    combine_method = config.model.progressive_combine.lower() # 控制Unet架构的encoder模块的每个输出结果如何对接到decoder模块的对应输入 "sum"
    combiner = functools.partial(Combine, method=combine_method)

    modules = []
    # timestep/noise_level embedding; only for continuous training
    if embedding_type == 'fourier':
      # Gaussian Fourier features embeddings.
      assert config.training.continuous, "Fourier features are only used for continuous training."

      modules.append(layerspp.GaussianFourierProjection(
        embedding_size=nf, scale=config.model.fourier_scale   # 16
      ))
      embed_dim = 2 * nf  # 128 * 2

    elif embedding_type == 'positional':
      embed_dim = nf      # embedding维度128

    else:
      raise ValueError(f'embedding type {embedding_type} unknown.')

    # 如果开启时间注入
    if conditional:
      modules.append(nn.Linear(embed_dim, nf * 4))  # 128 > 512
      modules[-1].weight.data = default_initializer()(modules[-1].weight.shape) # 初始化权重，使得y=XWT的方差与X维持一致
      nn.init.zeros_(modules[-1].bias)
      modules.append(nn.Linear(nf * 4, nf * 4)) # 512 > 512
      modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      nn.init.zeros_(modules[-1].bias)

    AttnBlock = functools.partial(layerspp.AttnBlockpp,       # 设置Attention模块
                                  init_scale=init_scale,      # 0.0
                                  skip_rescale=skip_rescale)  # True

    Upsample = functools.partial(layerspp.Upsample,
                                 with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)
    # NCSN默认为"none"跳过
    if progressive == 'output_skip':
      self.pyramid_upsample = layerspp.Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
    elif progressive == 'residual':
      pyramid_upsample = functools.partial(layerspp.Upsample,
                                           fir=fir, fir_kernel=fir_kernel, with_conv=True)
      

    Downsample = functools.partial(layerspp.Downsample,
                                   with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)
    

    # NCSN默认为"residual"  | DDPM默认为"none"，即DDPM没有分支
    if progressive_input == 'input_skip':
      self.pyramid_downsample = layerspp.Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
    elif progressive_input == 'residual':
      pyramid_downsample = functools.partial(layerspp.Downsample,
                                             fir=fir, fir_kernel=fir_kernel, with_conv=True)
      
    # NCSN默认"biggan"
    if resblock_type == 'ddpm':
      ResnetBlock = functools.partial(ResnetBlockDDPM,
                                      act=act,
                                      dropout=dropout,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=nf * 4)
    elif resblock_type == 'biggan':
      ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                      act=act,                  # swish
                                      dropout=dropout,          # 0.1
                                      fir=fir,                  # True
                                      fir_kernel=fir_kernel,    # (1,3,3,1)
                                      init_scale=init_scale,    # 0.0
                                      skip_rescale=skip_rescale,# True
                                      temb_dim=nf * 4)          # 128 * 4

    else:
      raise ValueError(f'resblock type {resblock_type} unrecognized.')

    # Downsampling block
    channels = config.data.num_channels # 3 原始数据的channel

    # NCSN默认为"residual"
    if progressive_input != 'none':
      input_pyramid_ch = channels # 3
  
    # 不改分辨率的卷积，将原始图像的channel转换为模型基础channel
    modules.append(conv3x3(channels, nf))
    # 记录首次的输出channel
    hs_c = [nf]

    # 索引  产生来源                          值    所在分辨率
    # [0]   conv3x3(3→128) 输出              128   32×32
    # [1]   Level 0 ResBlock #0              128   32×32
    # [2]   Level 0 ResBlock #1              128   32×32
    # [3]   Level 0 ResBlock #2              128   32×32
    # [4]   Level 0 ResBlock #3              128   32×32
    # [5]   Level 0 down ResBlock            128   16×16   ← 下采样后
    # [6]   Level 1 ResBlock #0              256   16×16
    # [7]   Level 1 ResBlock #1              256   16×16
    # [8]   Level 1 ResBlock #2              256   16×16
    # [9]   Level 1 ResBlock #3              256   16×16
    # [10]  Level 1 down ResBlock            256   8×8     ← 下采样后
    # [11]  Level 2 ResBlock #0              256   8×8
    # [12]  Level 2 ResBlock #1              256   8×8
    # [13]  Level 2 ResBlock #2              256   8×8
    # [14]  Level 2 ResBlock #3              256   8×8
    # [15]  Level 2 down ResBlock            256   4×4     ← 下采样后
    # [16]  Level 3 ResBlock #0              256   4×4
    # [17]  Level 3 ResBlock #1              256   4×4
    # [18]  Level 3 ResBlock #2              256   4×4
    # [19]  Level 3 ResBlock #3              256   4×4
    in_ch = nf
    for i_level in range(num_resolutions):  # 遍历网络层数
      for i_block in range(num_res_blocks): # 遍历残差块的数量
        out_ch = nf * ch_mult[i_level]  # 倍率 * 基础channel = out_channel
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch)) # 保存残差块module
        in_ch = out_ch  # 在相同倍率内，上一个残差块的输出channel = 下一个残差块的输入channel
        # 如果当前分辨率属于要开启的Attention的分辨率：保存Attention module
        if all_resolutions[i_level] in attn_resolutions:
          modules.append(AttnBlock(channels=in_ch))
        # 记录本次的输出channel
        hs_c.append(in_ch)

      # 遍历完所有残差块后，如果当前不是最后的channel倍率，进行下采样
      if i_level != num_resolutions - 1:

        if resblock_type == 'ddpm':
          modules.append(Downsample(in_ch=in_ch))
        else:
          # 进入该逻辑：利用残差块作为下采样
          modules.append(ResnetBlock(down=True, in_ch=in_ch))

        if progressive_input == 'input_skip':
          modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
          if combine_method == 'cat':
            in_ch *= 2
        # 进入该逻辑：分支的下采样，输入channel是该网络层数的第一次in_ch
        elif progressive_input == 'residual':
          modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
          input_pyramid_ch = in_ch
        # 记录本次的输出channel
        hs_c.append(in_ch)

    in_ch = hs_c[-1]
    modules.append(ResnetBlock(in_ch=in_ch))  # 残差块
    modules.append(AttnBlock(channels=in_ch)) # Attention
    modules.append(ResnetBlock(in_ch=in_ch))  # 残差块

    pyramid_ch = 0
    # Upsampling block

    # Upsampling Level 3 (分辨率 4×4)，5 个 pop：
    #   pop [19] = Level 3 ResBlock #3       (4×4)  ✓ 同分辨率
    #   pop [18] = Level 3 ResBlock #2       (4×4)  ✓
    #   pop [17] = Level 3 ResBlock #1       (4×4)  ✓
    #   pop [16] = Level 3 ResBlock #0       (4×4)  ✓
    #   pop [15] = Level 2 down ResBlock     (4×4)  ✓ 同分辨率！down后就在4×4了

    # Upsampling Level 2 (分辨率 8×8)，5 个 pop：
    #   pop [14] = Level 2 ResBlock #3       (8×8)  ✓
    #   pop [13] = Level 2 ResBlock #2       (8×8)  ✓
    #   pop [12] = Level 2 ResBlock #1       (8×8)  ✓
    #   pop [11] = Level 2 ResBlock #0       (8×8)  ✓
    #   pop [10] = Level 1 down ResBlock     (8×8)  ✓ 同分辨率！down后就在8×8了

    # Upsampling Level 1 (分辨率 16×16)，5 个 pop：
    #   pop [9]  = Level 1 ResBlock #3       (16×16) ✓
    #   pop [8]  = Level 1 ResBlock #2       (16×16) ✓
    #   pop [7]  = Level 1 ResBlock #1       (16×16) ✓
    #   pop [6]  = Level 1 ResBlock #0       (16×16) ✓
    #   pop [5]  = Level 0 down ResBlock     (16×16) ✓ 同分辨率！down后就在16×16了

    # Upsampling Level 0 (分辨率 32×32)，5 个 pop：
    #   pop [4]  = Level 0 ResBlock #3       (32×32) ✓
    #   pop [3]  = Level 0 ResBlock #2       (32×32) ✓
    #   pop [2]  = Level 0 ResBlock #1       (32×32) ✓
    #   pop [1]  = Level 0 ResBlock #0       (32×32) ✓
    #   pop [0]  = conv3x3 输出              (32×32) ✓ 同分辨率！    
    for i_level in reversed(range(num_resolutions)):          # 反向遍历网络层数
      for i_block in range(num_res_blocks + 1):               # 遍历残差块的数量（+1）
        out_ch = nf * ch_mult[i_level]                        # 倍率 * 基础channel = out_channel
        modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(),  # 保存残差块module，in_channel=上一个module的输出channel + 对应encoder架构的module的输出channel
                                   out_ch=out_ch))
        in_ch = out_ch                                        # 在相同倍率内，上一个残差块的输出channel = 下一个残差块的输入channel的一部分

      if all_resolutions[i_level] in attn_resolutions:        # 如果当前分辨率属于要开启的Attention的分辨率：保存Attention module
        modules.append(AttnBlock(channels=in_ch))

      # NCSN默认"none"跳过
      if progressive != 'none':
        if i_level == num_resolutions - 1:
          if progressive == 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
            pyramid_ch = channels
          elif progressive == 'residual':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, in_ch, bias=True))
            pyramid_ch = in_ch
          else:
            raise ValueError(f'{progressive} is not a valid name.')
        else:
          if progressive == 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, channels, bias=True, init_scale=init_scale))
            pyramid_ch = channels
          elif progressive == 'residual':
            modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
            pyramid_ch = in_ch
          else:
            raise ValueError(f'{progressive} is not a valid name')
      
      # 遍历完所有残差块后，如果当前不是首个的channel倍率，进行上采样
      if i_level != 0:
        if resblock_type == 'ddpm':
          modules.append(Upsample(in_ch=in_ch))
        # 进入该逻辑：利用残差块作为上采样
        else:
          modules.append(ResnetBlock(in_ch=in_ch, up=True))

    assert not hs_c 

    # NCSN默认"none"
    if progressive != 'output_skip':
      # 最终的分组归一化
      modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                  num_channels=in_ch, eps=1e-6))
      # 最终的卷积，不改变分辨率，把模型架构的channel转换为原始数据的channel，并且权重归一化的init_scale≈0
      modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

    self.all_modules = nn.ModuleList(modules)

  def forward(self, x, time_cond):
    # timestep/noise_level embedding; only for continuous training
    modules = self.all_modules
    m_idx = 0
    # NCSN默认"fourier"
    if self.embedding_type == 'fourier':
      # Gaussian Fourier features embeddings.
      # 此时time_cond是浮点数
      used_sigmas = time_cond
      # 获取位置/时间编码
      temb = modules[m_idx](torch.log(used_sigmas))
      m_idx += 1
    # DDPM默认"positional"
    elif self.embedding_type == 'positional':
      # Sinusoidal positional embeddings.
      timesteps = time_cond
      used_sigmas = self.sigmas[time_cond.long()] # 实际上DDPM执行这段代码没有任何意义
      # 获取位置/时间编码，timesteps可以接受连续的浮点数
      temb = layers.get_timestep_embedding(timesteps, self.nf)  # 128

    else:
      raise ValueError(f'embedding type {self.embedding_type} unknown.')

    # 开启时间注入：True
    if self.conditional:
      temb = modules[m_idx](temb)
      m_idx += 1
      temb = modules[m_idx](self.act(temb))
      m_idx += 1
    else:
      temb = None

    # 开启数据归一化：not False
    if not self.config.data.centered:
      # If input data is in [0, 1]
      x = 2 * x - 1.

    # Downsampling block
    input_pyramid = None
    # 是否开启分支
    if self.progressive_input != 'none':
      input_pyramid = x

    # 将原始数据channel => 模型基础channel
    hs = [modules[m_idx](x)]
    m_idx += 1
    for i_level in range(self.num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        h = modules[m_idx](hs[-1], temb)
        m_idx += 1
        # 是否执行Attention？
        if h.shape[-1] in self.attn_resolutions:
          h = modules[m_idx](h)
          m_idx += 1

        hs.append(h)

      # 如果不是最后一层
      if i_level != self.num_resolutions - 1:
        if self.resblock_type == 'ddpm':
          h = modules[m_idx](hs[-1])
          m_idx += 1
        # NCSN默认"biggan"方式的下采样
        else:
          h = modules[m_idx](hs[-1], temb)
          m_idx += 1

        if self.progressive_input == 'input_skip':
          input_pyramid = self.pyramid_downsample(input_pyramid)
          h = modules[m_idx](input_pyramid, h)
          m_idx += 1
        # NCSN默认"residual"，即开启分支下采样
        elif self.progressive_input == 'residual':
          input_pyramid = modules[m_idx](input_pyramid)
          m_idx += 1
          # 是否开启缩放来减少方差？
          if self.skip_rescale:
            input_pyramid = (input_pyramid + h) / np.sqrt(2.)
          else:
            input_pyramid = input_pyramid + h
          # 替换
          h = input_pyramid

        hs.append(h)
    # 中间层
    h = hs[-1]
    h = modules[m_idx](h, temb)
    m_idx += 1
    h = modules[m_idx](h)   # Attention不需要时间信息
    m_idx += 1
    h = modules[m_idx](h, temb)
    m_idx += 1

    pyramid = None

    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb) # 拼接上一个模块的输入 + 对应encoder架构的module的输出channel
        m_idx += 1
      # 是否开启Attention
      if h.shape[-1] in self.attn_resolutions:
        h = modules[m_idx](h)
        m_idx += 1

      if self.progressive != 'none':
        if i_level == self.num_resolutions - 1:
          if self.progressive == 'output_skip':
            pyramid = self.act(modules[m_idx](h))
            m_idx += 1
            pyramid = modules[m_idx](pyramid)
            m_idx += 1
          elif self.progressive == 'residual':
            pyramid = self.act(modules[m_idx](h))
            m_idx += 1
            pyramid = modules[m_idx](pyramid)
            m_idx += 1
          else:
            raise ValueError(f'{self.progressive} is not a valid name.')
        else:
          if self.progressive == 'output_skip':
            pyramid = self.pyramid_upsample(pyramid)
            pyramid_h = self.act(modules[m_idx](h))
            m_idx += 1
            pyramid_h = modules[m_idx](pyramid_h)
            m_idx += 1
            pyramid = pyramid + pyramid_h
          elif self.progressive == 'residual':
            pyramid = modules[m_idx](pyramid)
            m_idx += 1
            if self.skip_rescale:
              pyramid = (pyramid + h) / np.sqrt(2.)
            else:
              pyramid = pyramid + h
            h = pyramid
          else:
            raise ValueError(f'{self.progressive} is not a valid name')
      # 如果不是首层
      if i_level != 0:
        if self.resblock_type == 'ddpm':
          h = modules[m_idx](h)
          m_idx += 1
        # NCSN默认"biggan"
        else:
          h = modules[m_idx](h, temb)
          m_idx += 1

    assert not hs

    if self.progressive == 'output_skip':
      h = pyramid
    # NCSN默认"none"
    else:
      # 归一化 > 激活
      h = self.act(modules[m_idx](h))
      m_idx += 1
      # 将模型channel转换原始数据channel
      h = modules[m_idx](h)
      m_idx += 1

    assert m_idx == len(modules)
      # | score * σ + z | ^ 2 
      # 除σ的原因是，如果模型直接学习score，score标签与σ^2挂钩，那么损失的期望与σ^2挂钩导致波动大，所以NCSN初始项目在loss的外部*σ来抵消loss波动
    if self.config.model.scale_by_sigma:
      used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
      h = h / used_sigmas

    return h

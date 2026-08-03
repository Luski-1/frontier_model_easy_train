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

# Lint as: python3
"""Training NCSN++ on CIFAR-10 with VE SDE."""
from configs.default_cifar10_configs import get_default_configs


def get_config():
  config = get_default_configs()                  # 获取cifar数据集的默认参数
  # training
  training = config.training
  training.sde = 'vesde'
  training.continuous = True

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'                    # predict-correct
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'langevin'

  # model
  model = config.model
  model.name = 'ncsnpp'
  model.scale_by_sigma = True
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128                        # channel的基本数量
  model.ch_mult = (1, 2, 2, 2)          # channel的倍数
  model.num_res_blocks = 4              # 残差块数量
  model.attn_resolutions = (16,)        # 进行Attention的分辨率要求
  model.resamp_with_conv = True         # 上采样或下采样时是否开启卷积
  model.conditional = True              # 开启时间信息注入模型
  model.fir = True                      # 上采样或下采样时否开启滤波
  model.fir_kernel = [1, 3, 3, 1]       # 滤波核
  model.skip_rescale = True             # 主干与残差相加后，是否缩放？
  model.resblock_type = 'biggan'        # 控制选择什么架构的resnet block
  model.progressive = 'none'
  model.progressive_input = 'residual'  # 控制分支选择什么架构
  model.embedding_type = 'fourier'      # 控制位置/时间编码的方式
  model.progressive_combine = 'sum'     # 控制Unet架构的encoder模块的每个输出结果如何对接到decoder模块的对应输入
  model.attention_type = 'ddpm'
  model.init_scale = 0.                 # 控制残差分支的相关参数的权重初始化的方差，尽量使得训练初期≈0，不影响主干
  model.fourier_scale = 16              # 采用高斯分布抽样作为位置编码的不同频率，scale用于放缩频率
  model.conv_size = 3

  return config

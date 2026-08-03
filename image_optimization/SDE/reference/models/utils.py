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

"""All functions and modules related to model definition.
"""

import torch
import sde_lib
import numpy as np


_MODELS = {}


def register_model(cls=None, *, name=None):
  """A decorator for registering model classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _MODELS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _MODELS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_model(name):
  return _MODELS[name]


def get_sigmas(config):
  """
  核心设计：在对数空间均匀采样 → 指数空间指数增长的噪声序列（等比序列）

  Args:
      config: 配置字典对象
          - config.model.sigma_max: 最大噪声标准差 (t=T时的噪声强度)
          - config.model.sigma_min: 最小噪声标准差 (t=0时的噪声强度)
          - config.model.num_scales: 噪声尺度的总数量 (离散化步数)

  Returns:
      sigmas序列顺序：从 sigma_max 降序到 sigma_min
  """
  # --------------------------
  # 1. np.log(config.model.sigma_max): 对最大噪声取对数
  # 2. np.log(config.model.sigma_min): 对最小噪声取对数
  # 3. np.linspace(a, b, N): 在对数空间 [log(sigma_max), log(sigma_min)] 均匀采样 N 个点
  # 4. np.exp(...): 指数化回到原始空间 → 得到指数衰减的噪声序列
  sigmas = np.exp(
    np.linspace(np.log(config.model.sigma_max), np.log(config.model.sigma_min), config.model.num_scales))

  return sigmas


def get_ddpm_params(config):
  """Get betas and alphas --- parameters used in the original DDPM paper."""
  num_diffusion_timesteps = 1000
  # parameters need to be adapted if number of time steps differs from 1000
  beta_start = config.model.beta_min / config.model.num_scales
  beta_end = config.model.beta_max / config.model.num_scales
  betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

  alphas = 1. - betas
  alphas_cumprod = np.cumprod(alphas, axis=0)
  sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
  sqrt_1m_alphas_cumprod = np.sqrt(1. - alphas_cumprod)

  return {
    'betas': betas,
    'alphas': alphas,
    'alphas_cumprod': alphas_cumprod,
    'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
    'sqrt_1m_alphas_cumprod': sqrt_1m_alphas_cumprod,
    'beta_min': beta_start * (num_diffusion_timesteps - 1),
    'beta_max': beta_end * (num_diffusion_timesteps - 1),
    'num_diffusion_timesteps': num_diffusion_timesteps
  }


def create_model(config):
  """Create the score model."""
  model_name = config.model.name                # NCSN++和DDPM++，但实际上DDPM++的模型架构与NCSN++基本一致（项目中仅实现了NCSN++整体架构），仅有部分参数控制不同的变化，因此DDPM++的配置脚本中的model.name是'NCSNPP'
  # 在run_lib.py中存在from models import ddpm, ncsnv2, ncsnpp
  # ncsnpp有装饰器register_model装饰，因此ncsnpp类被Import时，自动在_MODELS进行注册
  score_model = get_model(model_name)(config)   
  score_model = score_model.to(config.device)
  score_model = torch.nn.DataParallel(score_model)  # DDP分布式包装
  return score_model


def get_model_fn(model, train=False):
  """Create a function to give the output of the score-based model.

  Args:
    model: The score model.
    train: `True` for training and `False` for evaluation.

  Returns:
    A model function.
  """

  def model_fn(x, labels):
    """Compute the output of the score-based model.

    Args:
      x: A mini-batch of input data.
      labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
        for different models.

    Returns:
      A tuple of (model output, new mutable states)
    """
    if not train:
      model.eval()
      return model(x, labels)
    else:
      model.train()
      return model(x, labels)

  return model_fn


def get_score_fn(sde, model, train=False, continuous=False):
  """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A score model.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.

  Returns:
    A score function.
  """
  model_fn = get_model_fn(model, train=train) # 灵活调整train或eval模式的调用模型的函数
  # DDPM分支
  if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
    def score_fn(x, t):
      # Scale neural network output by standard deviation and flip sign
      if continuous or isinstance(sde, sde_lib.subVPSDE):
        # 时间放大999倍，是为了放大位置编码的频率，但实际上还是连续的浮点数
        labels = t * 999
        score = model_fn(x, labels)
        std = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VP-trained models, t=0 corresponds to the lowest noise level
        labels = t * (sde.N - 1)
        score = model_fn(x, labels)
        std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]
      # 除std，后续在get_sde_loss_fn.loss_fn会乘std变回原样，最终训练目标还是噪声ε，会更加稳定，避免首σ影响
      # 但是这个项目是SDE，是通过score来去噪获得图像的，所以根据原始的DDPM的公式，除std=score
      score = -score / std[:, None, None, None]
      return score
  # NCSN分支
  elif isinstance(sde, sde_lib.VESDE):
    def score_fn(x, t):
      # 开启连续
      if continuous:
        labels = sde.marginal_prob(torch.zeros_like(x), t)[1] # 获取σ，也可以理解为时间t
      else:
        # For VE-trained models, t=0 corresponds to the highest noise level
        labels = sde.T - t
        labels *= sde.N - 1
        labels = torch.round(labels).long()

      score = model_fn(x, labels) # 得到模型预测的score
      return score

  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  return score_fn


def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))
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
# pytype: skip-file
"""Various sampling methods."""
import functools

import torch
import numpy as np
import abc

from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
import sde_lib
from models import utils as mutils

_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
  """A decorator for registering predictor classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _PREDICTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _PREDICTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def register_corrector(cls=None, *, name=None):
  """A decorator for registering corrector classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _CORRECTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _CORRECTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_predictor(name):
  return _PREDICTORS[name]


def get_corrector(name):
  return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps):
  """Create a sampling function.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """

  sampler_name = config.sampling.method
  # Probability flow ODE sampling with black-box ODE solvers
  if sampler_name.lower() == 'ode':
    sampling_fn = get_ode_sampler(sde=sde,                                # 获取调用ODE求解的函数，传参是model
                                  shape=shape,
                                  inverse_scaler=inverse_scaler,
                                  denoise=config.sampling.noise_removal,  # 是否开启最后一步的噪声去除？True
                                  eps=eps,                                # NCSN: 1e-5    |   DDPM: 1e-3
                                  device=config.device)
  # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
  elif sampler_name.lower() == 'pc':
    # get_predictor方法和get_corrector方法也是使用装饰器来动态注册
    predictor = get_predictor(config.sampling.predictor.lower())  # NCSN:reverse_diffusion    | DDPM:euler_maruyama
    corrector = get_corrector(config.sampling.corrector.lower())  # NCSN:langevin             | DDPM:none
    sampling_fn = get_pc_sampler(sde=sde,                         # 获取调用PC求解的函数，传参是model
                                 shape=shape,
                                 predictor=predictor,
                                 corrector=corrector,
                                 inverse_scaler=inverse_scaler,
                                 snr=config.sampling.snr,                 # 0.16
                                 n_steps=config.sampling.n_steps_each,    # 1 每次执行多少次C
                                 probability_flow=config.sampling.probability_flow, # 是否使用ODE求解？False
                                 continuous=config.training.continuous,   # True
                                 denoise=config.sampling.noise_removal,   # 是否开启最后一步的噪声去除？True
                                 eps=eps,                                 # NCSN: 1e-5  |   DDPM: 1e-3
                                 device=config.device)
  else:
    raise ValueError(f"Sampler name {sampler_name} unknown.")

  return sampling_fn


class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    self.rsde = sde.reverse(score_fn, probability_flow)  # 调用SDE类的reverse，获得RSDE对象，但是NCSN/DDPM的denoise_update_fn决定probability_flow=False即使用SDE方法
    self.score_fn = score_fn

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__()
    self.sde = sde
    self.score_fn = score_fn
    self.snr = snr
    self.n_steps = n_steps

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the corrector.

    Args:
      x: A PyTorch tensor representing the current state
      t: A PyTorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, t):
    # 反向离散的方法是常规的RSDE，而不是DDPM特有的反向SDE离散方法，求解算法使用欧拉法
    # 反向求解的ODE：dx = [f(前向) - 1/2 * g(前向)^2 * score] * dt
    # 因此在反向ODE中，f(反向) = [f(前向) - 1/2 * g(前向)^2 * score]，g(反向) = 0

    # 反向SDE，时间步为负
    dt = -1. / self.rsde.N
    # 抽样噪声
    z = torch.randn_like(x)
    # 获取反向SDE的f和g
    drift, diffusion = self.rsde.sde(x, t)
    # x <= x - f * dt + g * dw
    x_mean = x + drift * dt
    x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
    return x, x_mean


@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow) # NCSN/DDPM的denoise_update_fn: probability_flow=False  | NCSN的PC: probability_flow=False

  def update_fn(self, x, t):
    # self.rsde来自父类Predictor，其中调用SDE父类的reverse，获得RSDE对象
    # 调用NCSN或DDPM特有的反向SDE离散方法，获得f和g的部分结果，再通过rsde得到F和G，F已经包含f*dt，G已经包含g*dt
    # x <= x - F + G * z
    f, G = self.rsde.discretize(x, t)
    z = torch.randn_like(x)
    x_mean = x - f  # 添加漂移项
    x = x_mean + G[:, None, None, None] * z # 添加扩散项
    return x, x_mean


@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
  """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
    assert not probability_flow, "Probability flow not supported by ancestral sampling"

  def vesde_update_fn(self, x, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).long()
    sigma = sde.discrete_sigmas[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
    score = self.score_fn(x, t)
    x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[:, None, None, None]
    std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
    noise = torch.randn_like(x)
    x = x_mean + std[:, None, None, None] * noise
    return x, x_mean

  def vpsde_update_fn(self, x, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).long()
    beta = sde.discrete_betas.to(t.device)[timestep]
    score = self.score_fn(x, t)
    x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]
    noise = torch.randn_like(x)
    x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
    return x, x_mean

  def update_fn(self, x, t):
    if isinstance(self.sde, sde_lib.VESDE):
      return self.vesde_update_fn(x, t)
    elif isinstance(self.sde, sde_lib.VPSDE):
      return self.vpsde_update_fn(x, t)


@register_predictor(name='none')
class NonePredictor(Predictor):
  """An empty predictor that does nothing."""

  def __init__(self, sde, score_fn, probability_flow=False):
    pass

  def update_fn(self, x, t):
    return x, x


@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      # NCSN分支
      alpha = torch.ones_like(t)

    for i in range(n_steps):
      grad = score_fn(x, t)   # 获得score
      noise = torch.randn_like(x) # 获得高斯分布的噪声
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()  # 求出score的信号强度
      noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean() # 求出噪声的信号强度
      step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha    # 求出步长
      # 调用朗之万动力学退火：xi <= xi + step * score + sqrt(2 * step) * ε | ε ~ N(0, I)
      # 其中step在一开始的NCSN项目中，是当前sigma/最小sigma作为自动步长
      x_mean = x + step_size[:, None, None, None] * grad
      x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

    return x, x_mean


@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
  """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

  We include this corrector only for completeness. It was not directly used in our paper.
  """

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    std = self.sde.marginal_prob(x, t)[1]

    for i in range(n_steps):
      grad = score_fn(x, t)
      noise = torch.randn_like(x)
      step_size = (target_snr * std) ** 2 * 2 * alpha
      x_mean = x + step_size[:, None, None, None] * grad
      x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

    return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
  """An empty corrector that does nothing."""

  def __init__(self, sde, score_fn, snr, n_steps):
    pass

  def update_fn(self, x, t):
    return x, x


def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous):
  """A wrapper that configures and returns the update function of predictors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)  # 获得调用模型去预测score的函数，入参是x和t
  if predictor is None:
    # Corrector-only sampler
    predictor_obj = NonePredictor(sde, score_fn, probability_flow)
  else:
    predictor_obj = predictor(sde, score_fn, probability_flow)    # NCSN：创建predictor对象reverse_diffusion    | DDPM:euler_maruyama
  return predictor_obj.update_fn(x, t)                            # 调用predictor对象


def shared_corrector_update_fn(x, t, sde, model, corrector, continuous, snr, n_steps):
  """A wrapper tha configures and returns the update function of correctors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  if corrector is None:
    # Predictor-only sampler
    corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
  else:
    corrector_obj = corrector(sde, score_fn, snr, n_steps)  # NCSN：创建corrector对象langevin   | DDPM: none
  return corrector_obj.update_fn(x, t)                      # 调用corrector对象


def get_pc_sampler(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda'):
  """Create a Predictor-Corrector (PC) sampler.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
  # Create predictor & corrector update functions
  predictor_update_fn = functools.partial(shared_predictor_update_fn,       # 获得调用predictor的函数，入参是x和t
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,    # False
                                          continuous=continuous)                # True
  corrector_update_fn = functools.partial(shared_corrector_update_fn,       # 获得调用corrector的函数，入参是x和t
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,            # 0.16
                                          n_steps=n_steps)    # 1

  def pc_sampler(model):
    """ The PC sampler funciton.

    Args:
      model: A score model.
    Returns:
      Samples, number of function evaluations.
    """
    with torch.no_grad():
      # 获得时刻1的噪声图像
      x = sde.prior_sampling(shape).to(device)
      # 获得NCSN区间[1, 1e-5]或DDPM区间[1, 1e-3]，间隔1000的等差时间
      timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

      for i in range(sde.N):
        t = timesteps[i]
        vec_t = torch.ones(shape[0], device=t.device) * t # 调整时间维度
        # 先P后C或者先C后P，问题不大
        x, x_mean = corrector_update_fn(x, vec_t, model=model)
        x, x_mean = predictor_update_fn(x, vec_t, model=model)
      # denoise=True，因为已经使用朗之万动力学矫正，就没必要再执行额外去噪了
      return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)

  return pc_sampler


def get_ode_sampler(sde, shape, inverse_scaler,
                    denoise=False, rtol=1e-5, atol=1e-5,
                    method='RK45', eps=1e-3, device='cuda'):
  """Probability flow ODE sampler with the black-box ODE solver.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    denoise: If `True`, add one-step denoising to final samples.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """

  def denoise_update_fn(model, x):
    score_fn = get_score_fn(sde, model, train=False, continuous=True) # 获得调用模型去预测score的函数，入参是x和t
    # 获取执行反向SDE的对象
    predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
    vec_eps = torch.ones(x.shape[0], device=x.device) * eps # eps此时为时间，调整时间的维度
    _, x = predictor_obj.update_fn(x, vec_eps)  # 执行NCSN特有反向SDE的离散方法，_是包含扩散项的最终结果，x是剔除扩散项的最终结果
    return x

  def drift_fn(model, x, t):
    """Get the drift function of the reverse-time SDE."""
    score_fn = get_score_fn(sde, model, train=False, continuous=True) # 获得调用模型去预测score的函数，入参是x和t
    rsde = sde.reverse(score_fn, probability_flow=True)   # reverse是SDE类的公有方法，获得RSDE对象 | probability_flow用于控制是否为ODE
    return rsde.sde(x, t)[0]  # 仅需要获取反向SDE中f即可

  def ode_sampler(model, z=None):
    """The probability flow ODE sampler with black-box ODE solver.

    Args:
      model: A score model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      if z is None:
        # 给出时刻1的噪声
        x = sde.prior_sampling(shape).to(device)
      else:
        x = z

      def ode_func(t, x):
        # 重新调整x维度
        x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
        # 重新调整t维度
        vec_t = torch.ones(shape[0], device=x.device) * t
        # 获得反向求解ODE的漂移项f
        drift = drift_fn(model, x, vec_t)
        # 又拍平为一维
        return to_flattened_numpy(drift)

      # Black-box ODE solver for the probability flow ODE
      # ode_func参数是ODE求解的函数f，即dx/dt = f(x, t)
      # (sde.T, eps)是求解时间区间，[1, 1E-5] 或 [1, 1E-3]
      # to_flattened_numpy(x)是初始状态的x
      # RK45是具体数值求解方法
      # rtol=1e-5，是相对误差忍耐，即|error| > rtol * |x|则缩小求解步长
      # atol=1e-5，是绝对误差忍耐，即|error| > atol则缩小求解步长
      # |error|具体数值求解方法中使用高阶求解与低阶求解的结果差异，例如欧拉法是一阶，heun法是二阶
      solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                     rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev # 求解的步数
      x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32) # 最终时刻的x

      # 是否执行最后一步去噪
      if denoise:
        x = denoise_update_fn(model, x)

      # 反归一化
      x = inverse_scaler(x)
      return x, nfe

  return ode_sampler

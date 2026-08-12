import abc

import torch
import torch.nn as nn

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def get_noise(config, dtype=torch.float32):
  if config.noise.type == 'geometric':
    return GeometricNoise(config.noise.sigma_min,
                          config.noise.sigma_max)
  elif config.noise.type == 'loglinear':
    return LogLinearNoise()
  elif config.noise.type == 'cosine':
    return CosineNoise()
  elif config.noise.type == 'cosinesqr':
    return CosineSqrNoise()
  elif config.noise.type == 'linear':
    return Linear(config.noise.sigma_min,
                  config.noise.sigma_max,
                  dtype)
  else:
    raise ValueError(f'{config.noise.type} is not a valid noise')


def binary_discretization(z):
  z_hard = torch.sign(z)
  z_soft = z / torch.norm(z, dim=-1, keepdim=True)
  return z_soft + (z_hard - z_soft).detach()


class Noise(abc.ABC, nn.Module):
  """
  Baseline forward method to get the total + rate of noise at a timestep
  """
  def forward(self, t):
    # Assume time goes from 0 to 1
    return self.total_noise(t), self.rate_noise(t)
  
  @abc.abstractmethod
  def rate_noise(self, t):
    """
    Rate of change of noise ie g(t)
    """
    pass

  @abc.abstractmethod
  def total_noise(self, t):
    """
    Total noise ie \int_0^t g(t) dt + g(0)
    """
    pass


class CosineNoise(Noise):
  def __init__(self, eps=1e-3):
    super().__init__()
    self.eps = eps

  def rate_noise(self, t):
    cos = (1 - self.eps) * torch.cos(t * torch.pi / 2)
    sin = (1 - self.eps) * torch.sin(t * torch.pi / 2)
    scale = torch.pi / 2
    return scale * sin / (cos + self.eps)

  def total_noise(self, t):
    cos = torch.cos(t * torch.pi / 2)
    return - torch.log(self.eps + (1 - self.eps) * cos)


class CosineSqrNoise(Noise):
  def __init__(self, eps=1e-3):
    super().__init__()
    self.eps = eps

  def rate_noise(self, t):
    cos = (1 - self.eps) * (
      torch.cos(t * torch.pi / 2) ** 2)
    sin = (1 - self.eps) * torch.sin(t * torch.pi)
    scale = torch.pi / 2
    return scale * sin / (cos + self.eps)

  def total_noise(self, t):
    cos = torch.cos(t * torch.pi / 2) ** 2
    return - torch.log(self.eps + (1 - self.eps) * cos)


class Linear(Noise):
  def __init__(self, sigma_min=0, sigma_max=10, dtype=torch.float32):
    super().__init__()
    self.sigma_min = torch.tensor(sigma_min, dtype=dtype)
    self.sigma_max = torch.tensor(sigma_max, dtype=dtype)

  def rate_noise(self, t):
    return self.sigma_max - self.sigma_min

  def total_noise(self, t):
    return self.sigma_min + t * (self.sigma_max - self.sigma_min)

  def importance_sampling_transformation(self, t):
    f_T = torch.log1p(- torch.exp(- self.sigma_max))
    f_0 = torch.log1p(- torch.exp(- self.sigma_min))
    sigma_t = - torch.log1p(- torch.exp(t * f_T + (1 - t) * f_0))
    return (sigma_t - self.sigma_min) / (
      self.sigma_max - self.sigma_min)


class GeometricNoise(Noise):
  def __init__(self, sigma_min=1e-3, sigma_max=1):
    super().__init__()
    self.sigmas = 1.0 * torch.tensor([sigma_min, sigma_max])

  def rate_noise(self, t):
    return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t * (
      self.sigmas[1].log() - self.sigmas[0].log())

  def total_noise(self, t):
    return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t


class LogLinearNoise(Noise):
  """Log Linear noise schedule.
  
  Built such that 1 - 1/e^(n(t)) interpolates between 0 and
  ~1 when t varies from 0 to 1. Total noise is
  -log(1 - (1 - eps) * t), so the sigma will be
  (1 - eps) * t.
  """

  """
  文献中作者认为a_t是什么形式都不影响ELBO，作者设定a_t = e^(-σ(t))
  这个类就是求出σ(t)=-log(1-t)，即log-linear，此时a_t = 1 - t
  可知a_t是前向加噪过程中维持token不变的概率，那么1 - a_t = 1 - 1 + t = t就是前向加噪过程中变成mask的概率
  """
  def __init__(self, eps=1e-3):
    super().__init__()
    self.eps = eps
    # 以下这两个仅仅用于重要性采样，与常规的均匀采样时间t无关
    self.sigma_max = self.total_noise(torch.tensor(1.0))            # -log(1 - (1 - eps) * 1) ≈ 6.9
    self.sigma_min = self.eps + self.total_noise(torch.tensor(0.0)) # eps - log(1 - (1 - eps) * 0) ≈ 0.001

  def rate_noise(self, t):
    # 这是log(at)的求导
    # dlogX = 1/X 把X=1 - (1 - eps) * t代入就可以得到以下导数
    return (1 - self.eps) / (1 - (1 - self.eps) * t)

  def total_noise(self, t):
    """
    -log1p(x) = -log(1 + x)
    -log(1 - (1 - eps) * t)
    增加eps的原因是避免t=1时出现log0，基本可以理解为-log(1 - t)即可
    """

    return -torch.log1p(-(1 - self.eps) * t)

  def importance_sampling_transformation(self, t):
    """
    当t为log-linear时，MDLM的Loss是1/t * logP< π(x_0 | x_t), true_x_0 >，即1/t * 交叉熵。1/t导致当t抽样到比较小，该数据的损失权重极其夸张【可以参考NCSN的损失前也会乘sigma避免损失与sigma成正比】

    Loss = E_t~U[ε, 1] [f(t) * 1/t * 1 * dt] | 其中1就是t的均匀分布 | 
    通过重要性采样=> ∫ε>1 f(t) / ( 1 / (t * q(t) ) * q(t) * dt 使得从均匀分布U抽样变成从q(t)分布抽样 | 如果t * q(t) = 常数，那么f(t)的损失权重不再与t挂钩
    设q(t) = 1 / (t * ln(1/ε)) 那么t * q(t) = 1/ln(1/ε)符合常数要求 | q(t)不是乱设置的，必须满足q(t)【作为PDF】的积分=1，q(t)的CDF是单调递增并且在取值ε处=0，在取值1处=1 | 可以自行积分验证

    任何分布抽样的t -> 传入其分布的CDF -> 输出的值满足均匀分布 | 符合常理
    那么t~q(t) -> CDF_q  -> U 则可以通过逆变换 U -> CDF_q逆函数 -> t~q(t)
    CDF = ∫ε>t q(s) * ds = 1/ln(1/ε) * ∫ε>t 1/s * ds = 1/ln(1/ε) * ln(s)|ε>t = ln(t/ε) / ln(1/ε)
    设u~U[ε, 1]则u = ln(t/ε) / ln(1/ε)，则u * ln(1/ε) = ln(t/ε)，则ln( (1/ε)^u ) = ln(t/ε)，则(1/ε)^u = t/ε，则t = ε * (1/ε)^u，则t = ε * ε^(-u)，则t = ε^(1 - u)，则log(t) = (1 - u) * log(ε)

    以下代码本可以更加直接:
    log_t_new = (1 - t) * torch.log(self.eps)   # = (1-u)·log(ε)
    return torch.exp(log_t_new)                 # = ε^(1-u)
    但是为了满足更多中a_t的设计，导致代码比较复杂
    """

    # MASK概率    = 1 - e^(-sigma)                   # 在外部通过获得sigma即可得到a_t和1 - a_t的值
    #             = 1 - e^(-(-log(1-(1-eps)*t)))     # 把 sigma 代入
    #             = 1 - e^(log(1-(1-eps)*t))         # 负负抵消
    #             = 1 - (1-(1-eps)*t)                # e^(log x) = x
    #             = (1-eps)*t                        # 化简
    #             ≈ t                                # 得到加噪为MASK的概率

    # 所以1 - e^(-sigma)就是为了得到加噪为MASK的概率
    # p_T = 1 - exp(log(1 - (1 - eps) * (t=1))) = 1 - 1 + (1 - eps) * (t=1) = (1 - eps) * (t=1)
    # f_T = log(p_T) = log( (1 - eps) * (t=1) ) = log(0.999) ≈ log(1)

    # p_0 = 1 - exp(eps + log(1 - (1 - eps) * (t=0))) = 1 - exp(-eps)
    # f_0 = log(p_0) = log(1 - exp(-eps)) ≈ log(1 - 0.999) = log(0.001) 

    # log(1 - e^(-sigma))是为了得到MASK的概率对数    | f_T = log(p_T) | f_0 = log(p_0)

    f_T = torch.log1p(- torch.exp(- self.sigma_max))  # f_T = log1p(-exp(-sigma_max)) = log(1 - e^(-6.908)) = log(1 - 0.001) = log(0.999) ≈ -0.001
    f_0 = torch.log1p(- torch.exp(- self.sigma_min))  # f_0 = log1p(-exp(-sigma_min)) = log(1 - e^(-0.001)) = log(1 - 0.999) = log(0.001) ≈ -6.908

    # 通过t * f_T + (1 - t) * f_0线性插值，得到对数的插值。实际上= (1 - u) * log(ε)，因为f_T约等于0
    # exp( (1 - u) * log(ε) ) = t，在log-linear情况下既是时间t又是概率
    # -log(1 - t)就是σ(t)的定义
    sigma_t = - torch.log1p(- torch.exp(t * f_T + (1 - t) * f_0))

    # 反向推出变换后的t
    # t = - expm1(-sigma_t) / (1-eps) = (1 - e^(-sigma_t)) / (1-eps) = move_chance / (1-eps) 因为 move_chance=(1-eps)*t 
    t = - torch.expm1(- sigma_t) / (1 - self.eps)
    return t

"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc
import torch
import numpy as np


class SDE(abc.ABC):
  """SDE abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self, N):
    """Construct an SDE.

    Args:
      N: number of discretization time steps.
    """
    super().__init__()
    self.N = N

  @property
  @abc.abstractmethod
  def T(self):
    """End time of the SDE."""
    pass

  @abc.abstractmethod
  def sde(self, x, t):
    pass

  @abc.abstractmethod
  def marginal_prob(self, x, t):
    """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
    pass

  @abc.abstractmethod
  def prior_sampling(self, shape):
    """Generate one sample from the prior distribution, $p_T(x)$."""
    pass

  @abc.abstractmethod
  def prior_logp(self, z):
    """Compute log-density of the prior distribution.

    Useful for computing the log-likelihood via probability flow ODE.

    Args:
      z: latent code
    Returns:
      log probability density
    """
    pass

  def discretize(self, x, t):
    """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

    Useful for reverse diffusion sampling and probabiliy flow sampling.
    Defaults to Euler-Maruyama discretization.

    Args:
      x: a torch tensor
      t: a torch float representing the time step (from 0 to `self.T`)

    Returns:
      f, G
    """
    dt = 1 / self.N
    drift, diffusion = self.sde(x, t)
    f = drift * dt
    G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
    return f, G

  def reverse(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N        # NCSN: 1000    | DDPM: 1000
    T = self.T        # NCSN: 1       | DDPM: 1
    sde_fn = self.sde # 获取前向过程中f和g的函数，NCSN是dx = 0 + σmin * (σmax/σmin)^t * sqrt( 2 * log(σmax/σmin) ) * dw
    discretize_fn = self.discretize # 获取范向过程中f和g的函数，NCSN是x_i-1 = x_i + (σ(i)^2 - σ(i - 1)^2) * score  + sqrt( (σ(i)^2 - σ(i - 1)^2) ) * ε | 其中 ε ~ N(0, I)

    # Build the class for reverse-time SDE.
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N  # NCSN的ODE: 1000
        self.probability_flow = probability_flow    # NCSN或DDPM的ODE迭代求解：True  | NCSN的denoise_update_fn：False

      @property
      def T(self):
        return T  # NCSN：1

      def sde(self, x, t):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        # 反向求解的ODE：dx = [f(前向) - 1/2 * g(前向)^2 * score] * dt
        # 因此在反向ODE中，f(反向) = [f(前向) - 1/2 * g(前向)^2 * score]，g(反向) = 0

        # 获取NCSN的前向SDE：drift=0 | diffusion = σmin * (σmax/σmin)^t * sqrt( 2 * log(σmax/σmin) )
        # 获取DDPM的前向SDE：drift=-1/2 * ( β_bar_min + t * (β_bar_max - β_bar_min)) * x | diffusion=sqrt(β_bar_min + t * (β_bar_max - β_bar_min))

        # 反向求解的SDE: dx = [f(前向) - g(前向)^2 * score] * dt + g(前向) * dw_bar
        # 因此在反向SDE中，f(反向) = [f(前向) - g(前向)^2 * score], g(反向) = g(前向)
        drift, diffusion = sde_fn(x, t)
        # score_fn是调用模型获得score的函数，传参是x和t
        score = score_fn(x, t)
        # 参考上述公式
        drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        # 参考上述公式
        diffusion = 0. if self.probability_flow else diffusion
        return drift, diffusion

      def discretize(self, x, t):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        # VESDE的discretize方法返回结果：f=0 | G = sqrt( σ(i)^2 - σ(i - 1)^2 )
        # NCSN分支denoise_update_fn方法中probability_flow=False，即反向SDE的NCSN专有离散方法：x_i-1 = x_i + ( σ(i)^2 - σ(i - 1)^2 ) * score + sqrt( σ(i)^2 - σ(i - 1)^2 ) * ε | 其中 ε ~ N(0, I)
        # ref_f = 0 - (σ(i)^2 - σ(i - 1)^2) * score
        # ref_G = sqrt( σ(i)^2 - σ(i - 1)^2 )

        # VPSDE的discretize方法返货结果：f=sqrt(1 - β_i_t) * x_i - x_i  | G=sqrt(β_i_t)
        # DDPM分支denoise_update_fn方法中probability_flow=False，即反向SDE的NCSN专有离散方法：x_i-1 = (2 - sqrt(1 - β_i_t)) * x_i + β_i_t * score + sqrt(β_i_t) * ε | 其中 ε ~ N(0, I)
        # ref_f = sqrt(1 - β_i_t) * x_i - x_i - β_i_t * score
        # ref_G = sqrt(β_i_t)
        f, G = discretize_fn(x, t)
        rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G

    return RSDE()


class VPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct a Variance Preserving SDE.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min  # 0.1
    self.beta_1 = beta_max  # 20
    self.N = N              # 1000
    self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N) # β_bar / N = β_i
    self.alphas = 1. - self.discrete_betas  # 1 - β_i
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    """
    DDPM前向过程的SDE公式: dx = -1/2 * β(t) * x(t) * dt【漂移项】 + sqrt(β(t)) * dw【扩散项】
    对应的离散化公式: x_t - x_t-1 = -1/2 * β(t) * x(t) * δt + sqrt(β(t)) * sqrt(δt) * ε，ε~N(0,I)
    
    β(t)具体形式为: β_bar_min + t * (β_bar_max - β_bar_min)

    具体为：dx = -1/2 * ( β_bar_min + t * (β_bar_max - β_bar_min)) * x * dt + sqrt(β_bar_min + t * (β_bar_max - β_bar_min)) * dw
    """
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None, None] * x
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion

  def marginal_prob(self, x, t):
    """
    DDPM前向过程的扰动核，理解为前向过程SDE的解析解，那么等于通过传入t给解析解可以直接获得t时刻的x_t具体形式

    DDPM的扰动核: N(x_0 * e^(-1/2 * ∫[0 > t]β(s)ds), I - I * e^(-∫[0 > t]β(s)ds) )

    NCSN的扰动核具体公式: N(x_0 * e^(-1/4 * t^2 * (β_bar_max - β_bar_min) - 1/2 * t * β_bar_min), I - I * e^(-1/2 * t^2 * (β_bar_max - β_bar_min) - t * β_bar_min))
    """
    # -1/4 * t^2 * (β_bar_max - β_bar_min) - 1/2 * t * β_bar_min
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    # x_0 * e^(-1/4 * t^2 * (β_bar_max - β_bar_min) - 1/2 * t * β_bar_min)
    mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
    # sqrt(1 - e^(-1/2 * t^2 * (β_bar_max - β_bar_min) - t * β_bar_min))
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

  def prior_sampling(self, shape):
    # t = 1，从原始的DDPM项目，加噪最终结果就是服从高斯正态分布
    # 即使按照扰动核的方式进行计算，最终结果也是非常接近高斯正态分布
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
    return logps

  def discretize(self, x, t):
    """
    反向过程的SDE公式: dx = [f(x,t) - g(t)^2 * dlogPt(x)/dx]dt + g(t)dw【反向维纳过程】

    DDPM特有的反向精确离散化:  x_i-1 = (2 - sqrt(1 - β_i_t)) * x_i + β_i_t * score + sqrt(β_i_t) * ε | 其中 ε ~ N(0, I)
    """

    # [0, 1] * 999 / 1获得整数
    timestep = (t * (self.N - 1) / self.T).long()
    # 根据整数去获得β_i_t
    beta = self.discrete_betas.to(x.device)[timestep]
    # 根据整数去获得1 - β_i_t
    alpha = self.alphas.to(x.device)[timestep]
    # sqrt(β_i_t)
    sqrt_beta = torch.sqrt(beta)
    # f = sqrt(1 - β_i_t) * x - x
    f = torch.sqrt(alpha)[:, None, None, None] * x - x
    # G = sqrt(β_i_t)
    G = sqrt_beta
    return f, G


class subVPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct the sub-VP SDE that excels at likelihoods.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None, None] * x
    discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
    diffusion = torch.sqrt(beta_t * discount)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff)[:, None, None, None] * x
    std = 1 - torch.exp(2. * log_mean_coeff)
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.


class VESDE(SDE):
  def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
    """Construct a Variance Exploding SDE.

    Args:
      sigma_min: smallest sigma.
      sigma_max: largest sigma.
      N: number of discretization steps
    """
    super().__init__(N)
    self.sigma_min = sigma_min      # 0.01
    self.sigma_max = sigma_max      # 50
    # 1. np.log(config.model.sigma_max): 对最大噪声取对数
    # 2. np.log(config.model.sigma_min): 对最小噪声取对数
    # 3. np.linspace(a, b, N): 在对数空间 [log(sigma_max), log(sigma_min)] 均匀采样 N 个点
    # 4. np.exp(...): 指数化回到原始空间 → 得到指数衰减的噪声序列
    self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
    self.N = N    # 1000

  @property
  def T(self):
    # 返回最终T
    return 1

  def sde(self, x, t):
    """
    NCSN前向过程的SDE公式: dx = 0【漂移项】 + sqrt(d[σ(t)^2]/dt) * dw【扩散项】
    对应的离散化公式: x_t - x_t-1 = sqrt(σ(t)^2 - σ(t - 1)^2) * ε，ε~N(0,I)
    
    σ(t)具体形式为: σmin * (σmax/σmin)^t

    利用a^x的导数=a^x * log(a)，推导除dx = 0 + σmin * (σmax/σmin)^t * sqrt( 2 * log(σmax/σmin) ) * dw
    """
    sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    drift = torch.zeros_like(x)
    diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                device=t.device))
    return drift, diffusion

  def marginal_prob(self, x, t):
    """
    NCSN前向过程的扰动核，理解为前向过程SDE的解析解，那么等于通过传入t给解析解可以直接获得t时刻的x_t具体形式

    NCSN的扰动核: N(x_0, [σ(t)^2 - σ(0)^2] * I)，当σ(0)=0时，可直观地对应的NCSN原来的加噪重采样公式x_0 + σ(t)，即(x_0, σ(t)^2)

    NCSN的扰动核具体公式: N(x_0, σmin^2 * (σmax/σmin)^(2t) * I)
    """
    std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    mean = x
    return mean, std

  def prior_sampling(self, shape):
    # t = 1，那么self.sigma_min * (self.sigma_max / self.sigma_min) ** 1 = self.sigma_max
    return torch.randn(*shape) * self.sigma_max

  def prior_logp(self, z):
    """
    假设z是N维的标准高斯分布，假设维度独立，那么对数概率密度logP(z) = N * (-1 / 2) * log(2π * σ^2) - (x - 0)^ 2 / (2 * σ^2)
    
    暂时不知道用来干什么
    """
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_max ** 2)

  def discretize(self, x, t):
    """
    反向过程的SDE公式: dx = [f(x,t) - g(t)^2 * dlogPt(x)/dx]dt + g(t)dw【反向维纳过程】

    NCSN特有的反向精确离散化: x_i-1 = x_i + (σ(i)^2 - σ(i - 1)^2) * score  + sqrt( (σ(i)^2 - σ(i - 1)^2) ) * ε | 其中 ε ~ N(0, I)
    """
    timestep = (t * (self.N - 1) / self.T).long() # 得到[0,1,2,...,999]离散时间步
    sigma = self.discrete_sigmas.to(t.device)[timestep] # 获取对应σ
    # 获取上一个时间步的σ
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                 self.discrete_sigmas[timestep - 1].to(t.device))
    # f = 0
    f = torch.zeros_like(x)
    # G = sqrt( σ(i)^2 - σ(i - 1)^2 )
    G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
    return f, G
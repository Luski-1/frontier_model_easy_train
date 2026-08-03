import torch
import numpy as np

class VESDE:
    def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
        """Construct a Variance Exploding SDE.

        Args:
            sigma_min: smallest sigma.
            sigma_max: largest sigma.
            N: number of discretization steps
        """
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

    def get_forward_f_and_g(self, x, t):
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

    def get_special_reverse_discretize_parameters(self, x, t):
        """
        反向过程的SDE公式: dx = [f(x,t) - g(t)^2 * dlogPt(x)/dx]dt + g(t)dw【反向维纳过程】

        NCSN特有的反向精确离散化: x_i-1 = x_i + (σ(i)^2 - σ(i - 1)^2) * score  + sqrt( (σ(i)^2 - σ(i - 1)^2) ) * ε | 其中 ε ~ N(0, I)
        """
        timestep = (t * (self.N - 1) / self.T).long() # 得到[0,1,2,...,999]离散时间步
        discrete_sigmas = self.discrete_sigmas.to(t.device)
        sigma = discrete_sigmas[timestep] # 获取对应σ
        # 获取上一个时间步的σ
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                        discrete_sigmas[timestep - 1])
        # f = 0
        f = torch.zeros_like(x)
        # G = sqrt( σ(i)^2 - σ(i - 1)^2 )
        G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
        return f, G
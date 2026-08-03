import torch

class VPSDE:
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        """Construct a Variance Preserving SDE.

        Args:
            beta_min: value of beta(0)
            beta_max: value of beta(1)
            N: number of discretization steps
        """
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

    def get_forward_f_and_g(self, x, t):
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

    def get_special_reverse_discretize_parameters(self, x, t):
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
import torch
import torch.nn as nn


class LogLinearNoise(nn.Module):
    """Log Linear noise schedule.
    
    Built such that 1 - 1/e^(n(t)) interpolates between 0 and
    ~1 when t varies from 0 to 1. Total noise is
    -log(1 - (1 - eps) * t), so the sigma will be
    (1 - eps) * t.
    """

    """
    文献中作者认为a_t是什么形式都不影响ELBO，作者设定a_t = e^(-σ(t))
    其中一种形式就是σ(t)=-log(1-t)，即log-linear，此时a_t = 1 - t
    可知a_t是前向加噪过程中维持token不变的概率，那么1 - a_t = 1 - 1 + t = t就是前向加噪过程中变成mask的概率
    """
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def rate_noise(self, t):
        # 这是log(a_t)的求导
        # dlogX = 1/X 把X=1 - (1 - eps) * t代入就可以得到以下导数
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t):
        """
        -log1p(x) = -log(1 + x)
        -log(1 - (1 - eps) * t)
        增加eps的原因是避免t=1时出现log0，基本可以理解为-log(1 - t)即可
        """

        return -torch.log1p(-(1 - self.eps) * t)
  
    def forward(self, t):
        # Assume time goes from 0 to 1
        return self.total_noise(t), self.rate_noise(t)

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
        """

        return torch.exp((1 - t) * torch.log(self.eps))

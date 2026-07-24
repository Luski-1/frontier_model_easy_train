import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Uniform
from config import cfg



class CouplingLayer(nn.Module):
    """
    Implementation of the additive coupling layer from section 3.2 of the NICE
    paper.
    """

    def __init__(self, data_dim, hidden_dim, mask, num_layers=4):
        super().__init__()

        assert data_dim % 2 == 0

        self.mask = mask

        modules = [nn.Linear(data_dim, hidden_dim), nn.LeakyReLU(0.2)]  # [784, 1000]
        for _ in range(num_layers - 2):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.LeakyReLU(0.2))
        modules.append(nn.Linear(hidden_dim, data_dim)) # [1000, 784]

        self.m = nn.Sequential(*modules)

    def forward(self, x, logdet, invert=False):
        """
        z1 = x1             此时1可以理解为前半段，与下方假设涉及的1是不一样的含义
        z2 = x2 + m(x1)     同理2

        教学视频中，x1和x2分别截取前半段和后半段，此时对应的jacobian矩阵很明显就是对角阵，行列式必然是1
        然而实际代码中，x1和x2是相邻分隔，那么得到的jacobian矩阵还是对角阵吗？

        本人先不妨做个最简单的假设，m就是sum函数，并且设x为4维
        y0 = x0 + 0 + 0                 + 0 的原因是y1和y3中sum的维度就是4维，但是通过* (1.0 - self.mask)在y0的维度变成0，可以看到return中y1+y2，即各维度相加
        y1 = x1 + sum(x0, 0, x2, 0)
        y2 = x0 + 0 + 0                 + 0 的原因是y1和y3中sum的维度就是4维，但是通过* (1.0 - self.mask)在y0的维度变成0，可以看到return中y1+y2，即各维度相加
        y3 = x3 + sum(x0, 0, x2, 0) 

        jacobian矩阵如下，其中g是代表任意数，即关于m函数的导数
        虽然看起来不像对角阵，但是相信聪明的读者能迅速看出来实际上就是对角阵（通过列变换），所以行列式还是1，符合要求
               x0    x1    x2    x3
             ┌─────┬─────┬─────┬─────┐
         y0  │ 1.0 │ 0.0 │ 0.0 │ 0.0 │
             ├─────┼─────┼─────┼─────┤
         y1  │ g   │ 1.0 │ g   │ 0.0 │
             ├─────┼─────┼─────┼─────┤
         y2  │ 0.0 │ 0.0 │ 1.0 │ 0.0 │
             ├─────┼─────┼─────┼─────┤
         y3  │ g   │ 0.0 │ g   │ 1.0 │
             └─────┴─────┴─────┴─────┘
        """

        if not invert:
            x1, x2 = self.mask * x, (1.0 - self.mask) * x       # 使用mask区分x1区域和x2区域，不需要拆分维度
            y1, y2 = x1, x2 + (self.m(x1) * (1.0 - self.mask))  # * (1.0 - self.mask)，是为了让m(x1)输出值在非x2区域必须=0，避免有对应导数
            return y1 + y2, logdet  # logdet直接透传，因为0

        # 反向时，x1=y1，x2=y2-m(x1=y1)
        y1, y2 = self.mask * x, (1.0 - self.mask) * x
        x1, x2 = y1, y2 - (self.m(y1) * (1.0 - self.mask))
        return x1 + x2, logdet


class ScalingLayer(nn.Module):
    """
    Implementation of the scaling layer from section 3.3 of the NICE paper.
    """

    def __init__(self, data_dim):
        super().__init__()
        # 关键：零初始化，初始 scale=exp(0)=1
        self.log_scale_vector = nn.Parameter(torch.zeros(1, data_dim))

    def forward(self, x, logdet, invert=False):
        # 限制 log_scale 范围，防止 exp(50) 这种爆炸值
        log_scale = 10 * torch.tanh(self.log_scale_vector)

        log_det_jacobian = torch.sum(log_scale) # 直接相加就是雅可比矩阵
        
        if invert:
            return torch.exp(-log_scale) * x, logdet - log_det_jacobian # exp(-s) = 1 / exp(s) 正向时乘s，反向时就除s
        
        return torch.exp(log_scale) * x, logdet + log_det_jacobian  # exp后才是缩放值


# class LogisticDistribution(Distribution):
#     def __init__(self):
#         super().__init__()

#     def log_prob(self, x):
#         return -(F.softplus(x) + F.softplus(-x))

#     def sample(self, size):
#         if cfg["USE_CUDA"]:
#             z = Uniform(
#                 torch.cuda.FloatTensor([0.0]), torch.cuda.FloatTensor([1.0])
#             ).sample(size)
#         else:
#             z = Uniform(torch.FloatTensor([0.0]), torch.FloatTensor([1.0])).sample(size)

#         return torch.log(z) - torch.log(1.0 - z)
    

class GaussDistribution(Distribution):
    
    def __init__(self):
        super().__init__()
        # 预计算归一化常数
        self.log_sqrt_2pi = -0.5 * np.log(2 * np.pi)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        标准高斯分布 N(0,I) 的逐元素 log probability
        p(x_i) = 1 / √(2σπ) * exp(-(x_i - 0)^2 / 2σ^2) 其中σ=1
        log p(x_i) = -0.5 * x_i^2 - 0.5*log(2π)
        
        Args:
            value: [batch_size, data_dim]
        Returns:
            [batch_size, data_dim] 的 log probability（逐元素）
        """
        # return self.log_sqrt_2pi  -0.5 * (value ** 2) 
        return -0.5 * (value ** 2)  # 实际上丢弃log_sqrt_2pi也可以，因为常数不影响
    
    def sample(self, sample_shape) -> torch.Tensor:
        """
        从 N(0,I) 采样
        
        Args:
            sample_shape: 采样形状，如 [batch_size, data_dim]
        """
        if cfg["USE_CUDA"]:
            return torch.randn(sample_shape).cuda()
        else:
            return torch.randn(sample_shape)


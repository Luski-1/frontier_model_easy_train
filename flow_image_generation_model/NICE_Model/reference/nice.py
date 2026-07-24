import numpy as np
import torch
import torch.nn as nn

from config import cfg
from modules import GaussDistribution, CouplingLayer, ScalingLayer


class NICE(nn.Module):
    def __init__(self, data_dim, num_coupling_layers=3):
        super().__init__()

        self.data_dim = data_dim

        # 根据层数的奇偶性，交叉替换
        masks = [
            self._get_mask(data_dim, orientation=(i % 2 == 0))
            for i in range(num_coupling_layers)
        ]

        self.coupling_layers = nn.ModuleList(           # 交叉耦合层
            [
                CouplingLayer(
                    data_dim=data_dim,                  # 784
                    hidden_dim=cfg["NUM_HIDDEN_UNITS"], # 1000
                    mask=masks[i],
                    num_layers=cfg["NUM_NET_LAYERS"],   # 6
                )
                for i in range(num_coupling_layers)     # 4
            ]
        )

        self.scaling_layer = ScalingLayer(data_dim=data_dim)    # 缩放层

        self.prior = GaussDistribution()

    def forward(self, x, invert=False):
        if not invert:
            z, log_det_jacobian = self.f(x) # 获得变换后的z，以及雅可比矩阵
            log_likelihood = torch.sum(self.prior.log_prob(z), dim=1) + log_det_jacobian  # 传入z，希望z的概率越大，+ 雅可比矩阵得到的结果=P_data(x_true)，肯定是越高越好啊
            return z, log_likelihood

        return self.f_inverse(x)

    def f(self, x):
        # 正向函数
        z = x
        log_det_jacobian = 0
        for _, coupling_layer in enumerate(self.coupling_layers):       # 遍历每一次变换
            z, log_det_jacobian = coupling_layer(z, log_det_jacobian)   
        z, log_det_jacobian = self.scaling_layer(z, log_det_jacobian)   # 最后是缩放
        return z, log_det_jacobian

    def f_inverse(self, z):
        # 反向逆函数
        x = z
        x, _ = self.scaling_layer(x, 0, invert=True)        # 反向时，雅可比矩阵没用
        for _, coupling_layer in reversed(list(enumerate(self.coupling_layers))):
            x, _ = coupling_layer(x, 0, invert=True)
        return x

    def sample(self, num_samples):
        z = self.prior.sample([num_samples, self.data_dim]).view(
            num_samples, self.data_dim
        )   # 分布中抽样
        return self.f_inverse(z)

    def _get_mask(self, dim, orientation=True):
        # 通过步长%2来设置MASK，根据层数颠倒MASK
        mask = np.zeros(dim)
        mask[::2] = 1.0         # 如果是奇数层，偶数下标为1，奇数下标为0
        if orientation:
            mask = 1.0 - mask   # 如果是偶数层，那么偶数下标为0，奇数下标为1
        mask = torch.tensor(mask)
        if cfg["USE_CUDA"]:
            mask = mask.cuda()
        return mask.float()

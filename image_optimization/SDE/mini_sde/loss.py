from sde.vpsde import VPSDE
from sde.vesde import VESDE
from typing import Union
import torch.nn as nn
import torch


def calculate_loss(model: nn.Module, x: torch.tensor, sde: Union[VPSDE, VESDE], eps=1e-5):

    t = torch.rand(x.shape[0], device=x.device) * (sde.T - eps) + eps # 均匀分布U[0,1)中抽样 * (1 - 1e-5) + 1e-5 得到[1e-5, 1)，因为VESDE在0是不可导，因此避免取0
    z = torch.randn_like(x) # 高斯分布抽样

    mean, std = sde.marginal_prob(x, t) # 获取扰动核的mean和std
    perturbed_data = mean + std[:, None, None, None] * z  # 重采样

    labels = t * 999 # 时间放大999倍，是为了放大位置编码的频率，但实际上还是连续的浮点数
    pred_noise = model(perturbed_data, labels)

    if isinstance(sde, VPSDE):
        losses = torch.square(-pred_noise + z)
    else:
        losses = torch.square(pred_noise + z)  
    losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)
    return loss
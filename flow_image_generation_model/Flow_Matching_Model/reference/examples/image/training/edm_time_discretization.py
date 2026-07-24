# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
"""This is an ad-hoc sampling schedule that was proposed in https://arxiv.org/abs/2206.00364 it works very well for cifar 10 so we added its implementation here. It did not yield an improvement on ImageNet."""
import torch


def get_time_discretization(nfes: int, rho=7):
    step_indices = torch.arange(nfes, dtype=torch.float64)  # [0, 1, 2, ..., nfes-1]
    sigma_min = 0.002
    sigma_max = 80.0
    # 对应的公式：( σ_max^(1/ρ) + i / (N-1) * (σ_min^(1/ρ) - σ_max^(1/ρ)) )^ρ，其中i是列表step_indices，递减性质
    sigma_vec = (
        sigma_max ** (1 / rho)
        + step_indices / (nfes - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho    # {80^(1/7) + [0, 1/nfes-1, 2/nfes-1, ..., 1] * (0.002^(1/7) - 80^(1/7))} ^ 7，首位是80
    sigma_vec = torch.cat([sigma_vec, torch.zeros_like(sigma_vec[:1])]) # 末尾+0
    time_vec = (sigma_vec / (1 + sigma_vec)).squeeze()  # 递增列表，从1/81到1/1
    t_samples = 1.0 - torch.clip(time_vec, min=0.0, max=1.0)    # 递减列表 80/81到0
    return t_samples

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import torch

from torch import Tensor


def get_grad_norm_(parameters, norm_type: float = 2.0) -> Tensor:
    if isinstance(parameters, Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return Tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == torch.inf:  # 如果是无穷大
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)    # 直接返回最大梯度即可
    else:
        total_norm = torch.norm(                                                        # 合并所有梯度，获取L2距离
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters] # 获取每个梯度的L2距离
            ),
            norm_type,
        )
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)        # 对loss放大
        if update_grad:
            if clip_grad is not None:           # 如果需要裁剪梯度
                assert parameters is not None
                self._scaler.unscale_(          # 先缩小梯度为正常
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)    # 裁剪梯度，并返回梯度的L2距离（范数），用于后续打印
            else:
                self._scaler.unscale_(optimizer)    # 先缩小梯度为正常
                norm = get_grad_norm_(parameters)   # 手动计算梯度的L2距离（范数）
            self._scaler.step(optimizer)            # 更新梯度权重，如果梯度上溢就跳过本轮更新
            self._scaler.update()                   # 自动调整缩放倍率，如果梯度上溢就减小缩放倍率，否则就缓慢增大
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

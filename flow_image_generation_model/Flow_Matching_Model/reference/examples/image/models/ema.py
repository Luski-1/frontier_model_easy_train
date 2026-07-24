# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import logging
from typing import List

import torch
from torch.nn import Module, Parameter, ParameterList

logger = logging.getLogger(__name__)


class EMA(Module):
    def __init__(self, model: Module, decay: float = 0.999):
        super().__init__()
        self.model = model      # 真正的训练模型
        self.decay = decay

        # Put this in a buffer so that it gets included in the state dict
        self.register_buffer("num_updates", torch.tensor(0))
        # 复制并且冻结参数，EMA模型的权重
        self.shadow_params: ParameterList = ParameterList(
            [
                Parameter(p.clone().detach(), requires_grad=False)
                for p in model.parameters()
                if p.requires_grad
            ]
        )
        # 临时保存的权重
        self.backup_params: List[torch.Tensor] = []

    def train(self, mode: bool) -> None:
        # 如果目标模式和当前已经一致，直接跳过，不重复操作
        if self.training == mode:
            super().train(mode)
            return
        # 如果从train转换为eval
        if not mode:
            logger.info(
                "EMA: Switching from train to eval, backing up parameters and copying EMA params"
            )
            self.backup()
            self.copy_to_model()    # 目的是把EMA模型权重覆盖到训练所使用的模型，方便eval
        else:
            # 如果从eval转换到train
            logger.info("EMA: Switching from eval to train, restoring saved parameters")
            self.restore_to_model()

        super().train(mode)

    def update_ema(self) -> None:
        self.num_updates += 1
        num_updates = self.num_updates.item()
        decay = min(self.decay, (1 + num_updates) / (10 + num_updates)) # 获取最小的衰减值
        with torch.no_grad():
            params = [p for p in self.model.parameters() if p.requires_grad]
            for shadow, param in zip(self.shadow_params, params):
                shadow.sub_((1 - decay) * (shadow - param)) # 实际就是θema = decay * θema + (1 - decay) * θ_train

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.model(*args, **kwargs)

    def copy_to_model(self) -> None:
        # 目的是把EMA模型权重覆盖到训练所使用的模型，方便eval
        params = [p for p in self.model.parameters() if p.requires_grad]
        for shadow, param in zip(self.shadow_params, params):
            param.data.copy_(shadow.data)

    def backup(self) -> None:
        # 要保证当前为train
        assert (
            self.training
        ), "Backup can only be created in train mode to avoid backing-up ema weights."
        # 如果已经保存过了
        if len(self.backup_params) > 0:
            for p, b in zip(self.model.parameters(), self.backup_params):
                # 直接在内存原地址进行覆盖
                b.data.copy_(p.data)
        else:
            # 复制开启新的内存地址，保存到临时backup_params
            self.backup_params = [param.clone() for param in self.model.parameters()]

    def restore_to_model(self) -> None:
        # 目的是把临时backup_params权重覆盖到训练所使用的模型，方便train
        for param, backup in zip(self.model.parameters(), self.backup_params):
            param.data.copy_(backup.data)

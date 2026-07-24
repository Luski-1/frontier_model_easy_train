# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import gc
import logging
import math
from typing import Iterable

import torch
from flow_matching.path import CondOTProbPath, MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from models.ema import EMA
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.aggregation import MeanMetric
from training.grad_scaler import NativeScalerWithGradNormCount

logger = logging.getLogger(__name__)

MASK_TOKEN = 256
PRINT_FREQUENCY = 50


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    P_mean = -1.2
    P_std = 1.2
    # 1. 标准正态分布采样 N(0,1)
    rnd_normal = torch.randn((num_samples,), device=device)
    # 2. 变换为正态分布 N(P_mean, P_std²)
    # log_sigma ~ N(P_mean, P_std)
    log_sigma = rnd_normal * P_std + P_mean
    # 3. 指数得到 sigma
    sigma = log_sigma.exp()
    # 4. sigma -> 连续时间 t ∈ (0,1)
    time = 1 / (1 + sigma)
    # 5. 截断边界防止数值异常
    time = torch.clip(time, min=0.0001, max=1.0)
    # 得到t偏向靠近1的区域，可以理解为t越小则模型只需要还原粗略的信息，而t越大则模型需要还原更加精细的信息，会更难，因此让t更偏向于靠近1，提高模型学习
    return time


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    lr_schedule: torch.torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    loss_scaler: NativeScalerWithGradNormCount,
    args: argparse.Namespace,
):
    gc.collect()
    model.train(True)
    batch_loss = MeanMetric().to(device, non_blocking=True)
    epoch_loss = MeanMetric().to(device, non_blocking=True)

    accum_iter = args.accum_iter        # 梯度累积的次数
    if args.discrete_flow_matching:
        scheduler = PolynomialConvexScheduler(n=3.0)        # 多元幂的概率密度路径的相关参数
        path = MixtureDiscreteProbPath(scheduler=scheduler) # 多元幂的概率密度路径
    else:
        path = CondOTProbPath()     # 最优传输的概率密度路径

    for data_iter_step, (samples, labels) in enumerate(data_loader):
        if data_iter_step % accum_iter == 0:
            optimizer.zero_grad()                       # 提前清空梯度
            batch_loss.reset()
            if data_iter_step > 0 and args.test_run:    # 如果开启测试，直接退出
                break

        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if torch.rand(1) < args.class_drop_prob:        
            conditioning = {}
        else:
            conditioning = {"label": labels}        # 开启CFG

        if args.discrete_flow_matching:
            samples = (samples * 255.0).to(torch.long)          # 将数据转换为整数
            t = torch.torch.rand(samples.shape[0]).to(device)   # 随机抽样时间步t

            # sample probability path
            x_0 = (
                torch.zeros(samples.shape, dtype=torch.long, device=device) + MASK_TOKEN    # x0全为MASK TOKEN
            )
            path_sample = path.sample(t=t, x_0=x_0, x_1=samples)    # 获取xt

            # discrete flow matching loss
            logits = model(path_sample.x_t, t=t, extra=conditioning)    # extra=conditioning用于训练CFG，CFG可以等到后续CFG实现代码项目，先暂时理解为额外提供的条件参数即可
            loss = torch.nn.functional.cross_entropy(                   # 离散的损失函数是交叉熵损失，与MDLM/LLADA一致，建议留到离散文本模型相关课程时再学习
                logits.reshape([-1, 257]), samples.reshape([-1])
            ).mean()
        else:
            # Scaling to [-1, 1] from [0, 1]
            samples = samples * 2.0 - 1.0
            noise = torch.randn_like(samples).to(device)    # 获取噪声
            if args.skewed_timesteps:
                t = skewed_timestep_sample(samples.shape[0], device=device)
            else:
                t = torch.torch.rand(samples.shape[0]).to(device)
            path_sample = path.sample(t=t, x_0=noise, x_1=samples) 
            x_t = path_sample.x_t           # 获取xt
            u_t = path_sample.dx_t          # 获取速度v

            with torch.cuda.amp.autocast():
                # 如果开启EMA，那么会调用EMA.model进行forward
                # 损失就是L2损失，让模型预测的v与真实v靠近
                # extra=conditioning用于训练CFG，CFG可以等到后续CFG实现代码项目，先暂时理解为额外提供的条件参数即可
                loss = torch.pow(model(x_t, t, extra=conditioning) - u_t, 2).mean()     

        loss_value = loss.item()
        batch_loss.update(loss)
        epoch_loss.update(loss)

        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss /= accum_iter

        # Loss scaler applies the optimizer when update_grad is set to true.
        # Otherwise just updates the internal gradient scales
        apply_update = (data_iter_step + 1) % accum_iter == 0
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=apply_update,
        )
        if apply_update and isinstance(model, EMA): # 如果开启EMA
            model.update_ema()  # 需要更新EMA
        elif (
            apply_update
            and isinstance(model, DistributedDataParallel)  # 如果开启分布式
            and isinstance(model.module, EMA)   # 如果开启EMA
        ):
            model.module.update_ema()   # 需要更新EMA

        lr = optimizer.param_groups[0]["lr"]
        if data_iter_step % PRINT_FREQUENCY == 0:
            logger.info(
                f"Epoch {epoch} [{data_iter_step}/{len(data_loader)}]: loss = {batch_loss.compute()}, lr = {lr}"
            )

    lr_schedule.step()  # 更新学习率调度器
    return {"loss": float(epoch_loss.compute().detach().cpu())}

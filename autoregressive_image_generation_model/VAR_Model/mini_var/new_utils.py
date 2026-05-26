"""
精简版 VAR 项目工具模块
合并原文件：utils/amp_sc.py + utils/lr_control.py + utils/misc.py(精简版)
改动：去掉 MetricLogger/SyncPrint/DistLogger/TensorboardLogger/init_distributed_mode，
      去掉分布式函数，简化 AmpOptimizer（去掉梯度累积），保留 SmoothedValue，
      lr调度只保留 lin0 和 cos
"""

import math
from pprint import pformat
from typing import List, Dict, Union, Tuple, Optional

import torch
import torch.nn


# ===================== amp_sc.py 精简版 =====================

class NullCtx:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class AmpOptimizer:
    def __init__(
            self,
            mixed_precision: int,  # 混合精度模式：0=关闭,1=FP16,2=BF16
            optimizer: torch.optim.Optimizer,
            names: List[str],
            paras: List[torch.nn.Parameter],
            grad_clip: float,
    ):
        # ========== 1. 混合精度配置 ==========
        self.enable_amp = mixed_precision > 0
        self.using_fp16_rather_bf16 = mixed_precision == 1

        if self.enable_amp:
            # autocast：自动把计算切到FP16/BF16，省显存/加速
            self.amp_ctx = torch.autocast(
                'cuda', enabled=True,
                dtype=torch.float16 if self.using_fp16_rather_bf16 else torch.bfloat16,
                cache_enabled=True
            )
            # GradScaler：仅FP16需要，解决梯度下溢问题
            # init_scale=2**11=2048.0：初始缩放因子；
            # growth_interval=1000：每连续 1000 步无溢出，才尝试放大 scale。
            self.scaler = torch.cuda.amp.GradScaler(init_scale=2 ** 11, growth_interval=1000) if self.using_fp16_rather_bf16 else None
        else:
            self.amp_ctx = NullCtx()
            self.scaler = None

        # ========== 2. 保存核心成员 ==========
        self.optimizer, self.names, self.paras = optimizer, names, paras
        self.grad_clip = grad_clip

        # 梯度裁剪模式
        # early_clipping：普通优化器 → 用torch自带clip_grad_norm_，需要在optimizer.step()前手动执行裁剪
        # late_clipping：特殊优化器（带global_grad_norm）→ 用优化器自带裁剪，因此无需提前执行torch.clip，在optimizer.step()自行执行裁剪
        self.early_clipping = self.grad_clip > 0
        self.late_clipping = self.grad_clip > 0 and hasattr(optimizer, 'global_grad_norm')

    def backward_clip_step(self, loss: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[float]]:
        """反向传播 + 梯度裁剪 + 参数更新"""
        orig_norm = scaler_sc = None

        # 反向传播
        if self.scaler is not None:
            # FP16：缩放Loss后再反向传播（防止梯度下溢）
            self.scaler.scale(loss).backward()
        else:
            # BF16/FP32：直接反向传播
            loss.backward()

        # FP16：先取消梯度缩放
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        # 早期梯度裁剪
        if self.early_clipping:
            # 执行裁剪，并返回裁剪前的梯度
            orig_norm = torch.nn.utils.clip_grad_norm_(self.paras, self.grad_clip)

        # 优化器更新参数
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            scaler_sc: float = self.scaler.get_scale()
            # 安全限制：FP16最大缩放值不超过32768（防止溢出）
            if scaler_sc > 32768.:
                self.scaler.update(new_scale=32768.)
            else:
                self.scaler.update()    # 自动调整缩放比例
            scaler_sc = float(math.log2(scaler_sc))
        else:
            self.optimizer.step()

        # 晚期梯度裁剪
        if self.late_clipping:
            # 直接获取裁剪前的梯度即可
            orig_norm = self.optimizer.global_grad_norm

        # 清空梯度
        self.optimizer.zero_grad(set_to_none=True)

        return orig_norm, scaler_sc

    def state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict()
        } if self.scaler is None else {
            'scaler': self.scaler.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

    def load_state_dict(self, state, strict=True):
        if self.scaler is not None:
            try:
                self.scaler.load_state_dict(state['scaler'])
            except Exception as e:
                print(f'[fp16 load_state_dict err] {e}')
        self.optimizer.load_state_dict(state['optimizer'])


# ===================== lr_control.py 精简版 =====================

def lr_wd_annealing(sche_type: str, optimizer, peak_lr, wd, wd_end, cur_it, wp_it, max_it, wp0=0.005, wpe=0.001):
    """
    学习率和权重衰减调度（精简版，仅保留 lin0 和 cos）
    :param sche_type: 学习率衰减策略（lin0 或 cos）
    :param optimizer: 优化器
    :param peak_lr: 峰值学习率
    :param wd: 初始权重衰减值
    :param wd_end: 最终权重衰减值
    :param cur_it: 当前全局迭代步数
    :param wp_it: warmup总步数
    :param max_it: 训练总迭代步数
    :param wp0: 热身初始LR比例
    :param wpe: 学习率最终比例
    """
    wp_it = round(wp_it)
    # warmup阶段
    if cur_it < wp_it:
        # 线性热身：从 wp0 → 1 倍峰值LR
        cur_lr = wp0 + (1 - wp0) * cur_it / wp_it
    else:
        pasd = (cur_it - wp_it) / (max_it - 1 - wp_it)  # 完成比例[0, 1]
        rest = 1 - pasd  # 剩余比例[1, 0]
        if sche_type == 'cos':
            # 余弦衰减
            cur_lr = wpe + (1 - wpe) * (0.5 + 0.5 * math.cos(math.pi * pasd))
        elif sche_type == 'lin0':
            # lin0：前5%平台期，后95%线性下降
            T = 0.05
            max_rest = 1 - T
            if pasd < T:
                cur_lr = 1
            else:
                cur_lr = wpe + (1 - wpe) * rest / max_rest
        else:
            raise NotImplementedError(f'unknown sche_type {sche_type}, only support lin0 and cos')

    cur_lr *= peak_lr

    # 权重衰减余弦调度
    pasd = cur_it / (max_it - 1)
    cur_wd = wd_end + (wd - wd_end) * (0.5 + 0.5 * math.cos(math.pi * pasd))

    inf = 1e6
    min_lr, max_lr = inf, -1
    min_wd, max_wd = inf, -1
    for param_group in optimizer.param_groups:
        # 根据不同分组的学习率系数scale，×cur_lr，得到不同组的当前学习率
        # 随后优化器adamW会根据梯度动态再次调整学习率
        # 简单而言，一个参数的学习率，受到args.tlr（原计划中最高学习率）、训练步数（随着训练而变化）、不同参数分组（组内的学习率scale）、AdamW（根据梯度平方进行动态调整）
        param_group['lr'] = cur_lr * param_group.get('lr_sc', 1)
        max_lr = max(max_lr, param_group['lr'])
        min_lr = min(min_lr, param_group['lr'])
        param_group['weight_decay'] = cur_wd * param_group.get('wd_sc', 1)
        max_wd = max(max_wd, param_group['weight_decay'])
        if param_group['weight_decay'] > 0:
            min_wd = min(min_wd, param_group['weight_decay'])

    if min_lr == inf: min_lr = -1
    if min_wd == inf: min_wd = -1
    return min_lr, max_lr, min_wd, max_wd


def filter_params(model, nowd_keys=()) -> Tuple[
    List[str], List[torch.nn.Parameter], List[Dict[str, Union[torch.nn.Parameter, float]]]
]:
    """
    找出需要训练的参数，划分为需要权重衰减和不需要权重衰减的分组
    """
    para_groups = {}
    names, paras = [], []

    for name, para in model.named_parameters():
        if not para.requires_grad:
            continue

        names.append(name)
        paras.append(para)

        # 核心判断：是否需要权重衰减
        if (para.ndim == 1  # 一维参数（LayerNorm的gamma/beta、所有bias）
                or name.endswith('bias')    # 名字以bias结尾（偏置项）
                or any(k in name for k in nowd_keys)    # 包含黑名单关键词
        ):
            cur_wd_sc, group_name = 0., 'ND'    # wd_sc=0 → 权重衰减关闭
        else:
            cur_wd_sc, group_name = 1., 'D' # wd_sc=1 → 权重衰减开启
        cur_lr_sc = 1.

        if group_name not in para_groups:
            # 初始化分组：参数列表 + 权重衰减系数scale + 学习率系数scale（用于控制某些参数的学习率倍数）
            # 例如某些分组的参数就是要比其他分组的参数的学习率要低
            para_groups[group_name] = {'params': [], 'wd_sc': cur_wd_sc, 'lr_sc': cur_lr_sc}
        para_groups[group_name]['params'].append(para)

    count = len(names)
    numel = sum(p.numel() for p in paras)
    print(f'[filter_params] {type(model).__name__} trainable params: {count=}, {numel=}')

    return names, paras, list(para_groups.values())


# ===================== misc.py 精简版 =====================

import datetime
import glob
import os
import time
from collections import deque

import numpy as np


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a window or the global series average."""

    def __init__(self, window_size=30, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        return np.median(self.deque) if len(self.deque) else 0

    @property
    def avg(self):
        return sum(self.deque) / (len(self.deque) or 1)

    @property
    def global_avg(self):
        return self.total / (self.count or 1)

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1] if len(self.deque) else 0

    def time_preds(self, counts) -> Tuple[float, str, str]:
        remain_secs = counts * self.median
        return remain_secs, str(datetime.timedelta(seconds=round(remain_secs))), time.strftime("%Y-%m-%d %H:%M",
                                                                                               time.localtime(
                                                                                                   time.time() + remain_secs))

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def glob_with_latest_modified_first(pattern, recursive=False):
    return sorted(glob.glob(pattern, recursive=recursive), key=os.path.getmtime, reverse=True)


def auto_resume(output_dir, pattern='ckpt*.pth'):
    """搜索并加载最新的 checkpoint（简化版，去掉分布式）"""
    info = []
    file = os.path.join(output_dir, pattern)
    all_ckpt = glob_with_latest_modified_first(file)
    if len(all_ckpt) == 0:
        info.append(f'[auto_resume] no ckpt found @ {file}')
        return info, 0, 0, {}, {}
    else:
        info.append(f'[auto_resume] load ckpt from @ {all_ckpt[0]} ...')
        ckpt = torch.load(all_ckpt[0], map_location='cpu')
        ep, it = ckpt['epoch'], ckpt['iter']
        info.append(f'[auto_resume success] resume from ep{ep}, it{it}')
        return info, ep, it, ckpt.get('trainer', {}), ckpt.get('args', {})
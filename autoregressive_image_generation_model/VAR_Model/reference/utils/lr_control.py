import math
from pprint import pformat
from typing import Tuple, List, Dict, Union

import torch.nn

import dist


def lr_wd_annealing(sche_type: str, optimizer, peak_lr, wd, wd_end, cur_it, wp_it, max_it, wp0=0.005, wpe=0.001):
    """
    Decay the learning rate with half-cycle cosine after warmup
    :param sche_type:   学习率衰减策略类型（cos/lin/exp 等）
    :param optimizer:   PyTorch 优化器（支持多参数组）
    :param peak_lr:     峰值学习率（热身结束后的最大 LR）
    :param wd:          初始权重衰减值
    :param wd_end:      最终权重衰减值（训练结束时）
    :param cur_it:      当前全局迭代步数（总训练步数，非 epoch）
    :param wp_it:       学习率热身总步数
    :param max_it:      训练总迭代步数
    :param wp0:         热身初始 LR 比例（默认 0.005，即 0.5% 峰值 LR）
    :param wpe:         学习率最终比例（默认 0.01，即 1% 峰值 LR）
    :return:
    """
    wp_it = round(wp_it)
    # 1. 如果仍处于warmup阶段
    if cur_it < wp_it:
        # 线性热身：从 wp0 → 1 倍峰值LR
        cur_lr = wp0 + (1 - wp0) * cur_it / wp_it
    # 2. 如果已经超过warmup阶段
    else:
        pasd = (cur_it - wp_it) / (max_it - 1 - wp_it)  # 范围[0, 1] 超过wamrup后，当前步数与warmup的差值，在总步数与warmup的差值，的比例，即完成比例
        rest = 1 - pasd  # 范围[1, 0]  剩余比例
        if sche_type == 'cos':
            # 余弦衰减
            cur_lr = wpe + (1 - wpe) * (0.5 + 0.5 * math.cos(math.pi * pasd))
        elif sche_type == 'lin':
            # 前15%的区间
            T = 0.15
            max_rest = 1 - T # 0.85

            # 前15%的区间
            if pasd < T:
                # 平台期，就是维持学习率的倍率不变
                cur_lr = 1
            # 后85%的区间
            else:
                # 根据剩余比例，线性下降
                cur_lr = wpe + (1 - wpe) * rest / max_rest  # 1 to wpe
        elif sche_type == 'lin0':
            # 前5%的区间
            T = 0.05
            max_rest = 1 - T
            if pasd < T:
                cur_lr = 1
            else:
                cur_lr = wpe + (1 - wpe) * rest / max_rest
        elif sche_type == 'lin00':
            # 直接线性下降
            cur_lr = wpe + (1 - wpe) * rest
        elif sche_type.startswith('lin'):
            T = float(sche_type[3:])
            max_rest = 1 - T
            wpe_mid = wpe + (1 - wpe) * max_rest
            wpe_mid = (1 + wpe_mid) / 2
            if pasd < T:
                cur_lr = 1 + (wpe_mid - 1) * pasd / T
            else:
                cur_lr = wpe + (wpe_mid - wpe) * rest / max_rest
        elif sche_type == 'exp':
            T = 0.15
            max_rest = 1 - T
            if pasd < T:
                cur_lr = 1
            else:
                expo = (pasd - T) / max_rest * math.log(wpe)
                cur_lr = math.exp(expo)
        else:
            raise NotImplementedError(f'unknown sche_type {sche_type}')

    cur_lr *= peak_lr  # peak_lr就是args.tlr指定的学习率，×学习率倍率=当前学习率

    # # 范围[0, 1] 当前步数在总步数的完成比例
    pasd = cur_it / (max_it - 1)
    # 计算当前权重衰减值，余弦衰减
    cur_wd = wd_end + (wd - wd_end) * (0.5 + 0.5 * math.cos(math.pi * pasd))

    inf = 1e6
    min_lr, max_lr = inf, -1
    min_wd, max_wd = inf, -1
    for param_group in optimizer.param_groups:
        # 根据不同分组的学习率系数scale，×cur_lr，得到不同组的当前学习率
        # 随后优化器adamW会根据梯度动态再次调整学习率
        # 简单而言，一个参数的学习率，受到args.tlr（原计划中最高学习率）、训练步数（随着训练而变化）、不同参数分组（组内的学习率scale）、AdamW（根据梯度平方进行动态调整）
        param_group['lr'] = cur_lr * param_group.get('lr_sc', 1)  # 'lr_sc' could be assigned
        max_lr = max(max_lr, param_group['lr'])
        min_lr = min(min_lr, param_group['lr'])
        # 与学习率同理
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
    # ========== 1. 初始化变量 ==========
    para_groups = {}  # 【核心】优化器用的真实参数组（存参数张量）
    para_groups_dbg = {}  # 调试用参数组（存参数名，方便打印看）
    names, paras = [], []  # 所有可训练参数的 名称/参数本体
    names_no_grad = []  # 冻结参数（不需要梯度）的名称
    count, numel = 0, 0  # 统计：参数个数 / 总参数量

    # ========== 2. 遍历模型所有参数（核心循环）==========
    for name, para in model.named_parameters():
        # 兼容FSDP分布式训练：去掉FSDP包装的参数名前缀
        name = name.replace('_fsdp_wrapped_module.', '')

        # ========== 过滤冻结参数 ==========
        if not para.requires_grad:
            names_no_grad.append(name)
            continue  # 跳过：冻结的权重不参与训练/优化

        # 统计可训练参数
        count += 1
        numel += para.numel()
        names.append(name)
        paras.append(para)

        # ========== 核心判断：是否需要权重衰减！==========
        # 满足任一条件 → 不做权重衰减 (ND组)
        if (para.ndim == 1  # 一维参数（LayerNorm的gamma/beta、所有bias）
                or name.endswith('bias')  # 名字以bias结尾（偏置项）
                or any(k in name for k in nowd_keys)  # 包含黑名单关键词
        ):
            cur_wd_sc, group_name = 0., 'ND'  # wd_sc=0 → 权重衰减关闭
        else:
            cur_wd_sc, group_name = 1., 'D'  # wd_sc=1 → 权重衰减开启
        cur_lr_sc = 1.

        # ========== 分组存入字典 ==========
        if group_name not in para_groups:
            # 初始化分组：参数列表 + 权重衰减系数scale + 学习率系数scale（用于控制某些参数的学习率倍数）
            para_groups[group_name] = {'params': [], 'wd_sc': cur_wd_sc, 'lr_sc': cur_lr_sc}
            para_groups_dbg[group_name] = {'params': [], 'wd_sc': cur_wd_sc, 'lr_sc': cur_lr_sc}
        # 加入当前参数
        para_groups[group_name]['params'].append(para)
        para_groups_dbg[group_name]['params'].append(name)

    for g in para_groups_dbg.values():
        g['params'] = pformat(', '.join(g['params']), width=200)

    print(f'[get_param_groups] param_groups = \n{pformat(para_groups_dbg, indent=2, width=240)}\n')

    for rk in range(dist.get_world_size()):
        dist.barrier()
        if dist.get_rank() == rk:
            print(f'[get_param_groups][rank{dist.get_rank()}] {type(model).__name__=} {count=}, {numel=}', flush=True,
                  force=True)
    print('')

    assert len(
        names_no_grad) == 0, f'[get_param_groups] names_no_grad = \n{pformat(names_no_grad, indent=2, width=240)}\n'
    return names, paras, list(para_groups.values())

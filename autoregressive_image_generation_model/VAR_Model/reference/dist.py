import datetime
import functools
import os
import sys
from typing import List
from typing import Union

import torch
import torch.distributed as tdist
import torch.multiprocessing as mp

__rank, __local_rank, __world_size, __device = 0, 0, 1, 'cuda' if torch.cuda.is_available() else 'cpu'
__initialized = False

# 变量	含义
# __rank	全局进程编号（所有机器的所有进程统一编号，0~world_size-1）
# __local_rank	单机内进程编号（当前机器内的 GPU 编号，0~ 本机 GPU 数 - 1）
# __world_size	总进程数 = 总 GPU 数量
# __device	当前进程绑定的计算设备（cuda:0 /cuda:1 等）
# __initialized	分布式进程组是否初始化完成


def initialized():
    return __initialized


def initialize(fork=False, backend='nccl', gpu_id_if_not_distibuted=0, timeout=30):
    """
    设置分布式训练的环境变量
    :param fork:
    :param backend:
    :param gpu_id_if_not_distibuted:
    :param timeout:
    :return:
    """
    global __device
    if not torch.cuda.is_available():
        print(f'[dist initialize] cuda is not available, use cpu instead', file=sys.stderr)
        return
    # 只有用torchrun/torch.distributed.launch启动分布式训练时，系统才会自动注入RANK（全局进程号）
    elif 'RANK' not in os.environ:
        torch.cuda.set_device(gpu_id_if_not_distibuted)  # 绑定指定GPU
        __device = torch.empty(1).cuda().device  # 安全获取当前GPU设备对象
        print(f'[dist initialize] env variable "RANK" is not set, use {__device} as the device', file=sys.stderr)
        return
    # 从环境变量获取全局进程号，获取本机GPU总数
    global_rank, num_gpus = int(os.environ['RANK']), torch.cuda.device_count()
    # 核心公式：本地进程号 = 全局进程号 % 本机GPU数量
    local_rank = global_rank % num_gpus
    # 绑定当前进程到对应的GPU硬件
    torch.cuda.set_device(local_rank)

    # ref: https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/dist_utils.py#L29
    # spawn（默认，推荐）：安全、兼容；CUDA、跨平台（Windows / Linux），子进程独立初始化，是GPU分布式的标准选择；
    # fork：启动快，但CUDA多进程下极不稳定，容易崩溃，因此默认关闭。
    if mp.get_start_method(allow_none=True) is None:
        method = 'fork' if fork else 'spawn'
        print(f'[dist initialize] mp method={method}')
        mp.set_start_method(method)
    # 建立所有进程之间的通信网络
    tdist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=timeout * 60))

    global __rank, __local_rank, __world_size, __initialized
    __local_rank = local_rank  # 本地进程号
    __rank, __world_size = tdist.get_rank(), tdist.get_world_size()  # 全局号+总进程数
    __device = torch.empty(1).cuda().device  # 当前GPU设备
    __initialized = True  # 标记分布式初始化完成

    assert tdist.is_initialized(), 'torch.distributed is not initialized!'
    print(f'[lrk={get_local_rank()}, rk={get_rank()}]')


def get_rank():
    return __rank


def get_local_rank():
    return __local_rank


def get_world_size():
    return __world_size


def get_device():
    return __device


def set_gpu_id(gpu_id: int):
    if gpu_id is None: return
    global __device
    if isinstance(gpu_id, (str, int)):
        torch.cuda.set_device(int(gpu_id))
        __device = torch.empty(1).cuda().device
    else:
        raise NotImplementedError


def is_master():
    return __rank == 0


def is_local_master():
    return __local_rank == 0


def new_group(ranks: List[int]):
    if __initialized:
        return tdist.new_group(ranks=ranks)
    return None


def barrier():
    if __initialized:
        tdist.barrier()


def allreduce(t: torch.Tensor, async_op=False):
    if __initialized:
        if not t.is_cuda:
            cu = t.detach().cuda()
            ret = tdist.all_reduce(cu, async_op=async_op)
            t.copy_(cu.cpu())
        else:
            ret = tdist.all_reduce(t, async_op=async_op)
        return ret
    return None


def allgather(t: torch.Tensor, cat=True) -> Union[List[torch.Tensor], torch.Tensor]:
    if __initialized:
        if not t.is_cuda:
            t = t.cuda()
        ls = [torch.empty_like(t) for _ in range(__world_size)]
        tdist.all_gather(ls, t)
    else:
        ls = [t]
    if cat:
        ls = torch.cat(ls, dim=0)
    return ls


def allgather_diff_shape(t: torch.Tensor, cat=True) -> Union[List[torch.Tensor], torch.Tensor]:
    if __initialized:
        if not t.is_cuda:
            t = t.cuda()

        t_size = torch.tensor(t.size(), device=t.device)
        ls_size = [torch.empty_like(t_size) for _ in range(__world_size)]
        tdist.all_gather(ls_size, t_size)

        max_B = max(size[0].item() for size in ls_size)
        pad = max_B - t_size[0].item()
        if pad:
            pad_size = (pad, *t.size()[1:])
            t = torch.cat((t, t.new_empty(pad_size)), dim=0)

        ls_padded = [torch.empty_like(t) for _ in range(__world_size)]
        tdist.all_gather(ls_padded, t)
        ls = []
        for t, size in zip(ls_padded, ls_size):
            ls.append(t[:size[0].item()])
    else:
        ls = [t]
    if cat:
        ls = torch.cat(ls, dim=0)
    return ls


def broadcast(t: torch.Tensor, src_rank) -> None:
    if __initialized:
        if not t.is_cuda:
            cu = t.detach().cuda()
            tdist.broadcast(cu, src=src_rank)
            t.copy_(cu.cpu())
        else:
            tdist.broadcast(t, src=src_rank)


def dist_fmt_vals(val: float, fmt: Union[str, None] = '%.2f') -> Union[torch.Tensor, List]:
    if not initialized():
        return torch.tensor([val]) if fmt is None else [fmt % val]

    ts = torch.zeros(__world_size)
    ts[__rank] = val
    allreduce(ts)
    if fmt is None:
        return ts
    return [fmt % v for v in ts.cpu().numpy().tolist()]


def master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        force = kwargs.pop('force', False)
        if force or is_master():
            ret = func(*args, **kwargs)
        else:
            ret = None
        barrier()
        return ret

    return wrapper


def local_master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        force = kwargs.pop('force', False)
        if force or is_local_master():
            ret = func(*args, **kwargs)
        else:
            ret = None
        barrier()
        return ret

    return wrapper


def for_visualize(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_master():
            # with torch.no_grad():
            ret = func(*args, **kwargs)
        else:
            ret = None
        return ret

    return wrapper


def finalize():
    if __initialized:
        tdist.destroy_process_group()

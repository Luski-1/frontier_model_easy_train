import json
import os
import random
import re
import subprocess
import sys
import time
from collections import OrderedDict
from typing import Optional, Union

import numpy as np
import torch

try:
    from tap import Tap
except ImportError as e:
    print(f'`>>>>>>>> from tap import Tap` failed, please run:      pip3 install typed-argument-parser     <<<<<<<<',
          file=sys.stderr, flush=True)
    print(f'`>>>>>>>> from tap import Tap` failed, please run:      pip3 install typed-argument-parser     <<<<<<<<',
          file=sys.stderr, flush=True)
    time.sleep(5)
    raise e

import dist


class Args(Tap):
    data_path: str = '/path/to/imagenet'
    exp_name: str = 'text'

    # VAE
    vfast: int = 0  # torch.compile VAE; =0: not compile; 1: compile with 'reduce-overhead'; 2: compile with 'max-autotune' 【开启VAE编译加速】
    # VAR
    tfast: int = 0  # torch.compile VAR; =0: not compile; 1: compile with 'reduce-overhead'; 2: compile with 'max-autotune' 【开启VAR编译加速】
    depth: int = 16  # VAR depth 【模型层数，同时影响drop_path_rate[0.1 * depth/24]】
    # VAR initialization
    ini: float = -1  # -1: automated model parameter initialization 【模型参数初始化的方差，-1代表根据参数维度进行计算】
    hd: float = 0.02  # head.w *= hd 【输出头权重的缩放系数，目的是训练开始时预测相对均匀】
    aln: float = 0.5  # the multiplier of ada_lin.w's initialization 【AdaLN shift/scale的初始缩放，目的是训练开始时接近0】
    alng: float = 1e-5  # the multiplier of ada_lin.w[gamma channels]'s initialization 【AdaLN gamma 初始缩放接近0】
    # VAR optimization
    fp16: int = 0  # 1: using fp16, 2: bf16 【混合精度，1是fp16,2是bf16】
    tblr: float = 1e-4  # base lr 【train base learning rate，训练时基础学习率，默认1e-4，最终的基础学习率是tblr × (bs/256) × ac，遵循线性缩放规则】
    tlr: float = None  # lr = base lr * (bs / 256) 【train learning rate，训练时学习率，tblr * (bs/256)】
    twd: float = 0.05  # initial wd 【train weight decay，训练正则化的权重衰减，起始值为0.05】
    twde: float = 0  # final wd, =twde or twd 【train weight decay end，训练正则化的权重衰减的最终值，为0】
    tclip: float = 2.  # <=0 for not using grad clip 【梯度裁剪】
    ls: float = 0.0  # label smooth 【标签平滑，训练正则化的方法，让不是target label的类别也有微弱概率，让模型预测更加平滑，避免过拟合】

    bs: int = 768  # global batch size 【指定的全局batch size】
    batch_size: int = 0  # [automatically set; don't specify this] batch size per GPU = round(args.bs / args.ac / dist.get_world_size() / 8) * 8 【单卡batch size，round(bs/ac/world_size)】
    glb_batch_size: int = 0  # [automatically set; don't specify this] global batch size = args.batch_size * dist.get_world_size() 【最终的全局batch size，根据batch_size*world_size进行最终校正】
    ac: int = 1  # gradient accumulation 【梯度累计】

    ep: int = 250 # 【epoch】
    wp: float = 0 # 【warmup epoch自动跟踪ep / 50，即默认值是5】
    wp0: float = 0.005  # initial lr ratio at the begging of lr warm up 【warmup rate 0，即warmup期间起始lr比例】
    wpe: float = 0.01  # final lr ratio at the end of training 【warmup rate end，即warmup期间最终lr比例】
    sche: str = 'lin0'  # lr schedule 【lr调度策略】

    opt: str = 'adamw'  # lion: https://cloud.tencent.com/developer/article/2336657?areaId=106001 lr=5e-5 (0.25x) wd=0.8 (8x); Lion needs a large bs to work 【optimizer优化器】
    afuse: bool = True  # fused adamw   【融合优化器，CUDA优化优化器，加快训练速度】

    # other hps
    saln: bool = False  # whether to use shared adaln 【share AdaLN？仅当训练数据尺寸为512*512时开启，节省参数量】
    anorm: bool = True  # whether to use L2 normalized attention 【Attention模块中，在计算注意力得分前，Q/K是否开启归一化】
    fuse: bool = True  # whether to use fused op like flash attn, xformers, fused MLP, fused LayerNorm, etc. 【开启 FlashAttention、fused MLP、fused LayerNorm 等加速算子（省显存+加速）】

    # data
    pn: str = '1_2_3_4_5_6_8_10_13_16'  # 【关键参数，控制VAR不同的生成阶段的patch尺寸或者stride，最大16*16】
    patch_size: int = 16    # 【VAE的下采样倍率】
    patch_nums: tuple = None  # [automatically set; don't specify this] = tuple(map(int, args.pn.replace('-', '_').split('_'))) 【自动跟踪pn，(1,2,3,4,5,6,8,10,13,16)】
    resos: tuple = None  # [automatically set; don't specify this] = tuple(pn * args.patch_size for pn in args.patch_nums) 【不同的生成阶段的patch恢复上采样倍率的分辨率】

    data_load_reso: int = None  # [automatically set; don't specify this] would be max(patch_nums) * patch_size 【data_load_resolution 数据加载的最大分辨率，根据该参数对所有图像进行裁剪，自动跟踪训练图像的最大尺寸】
    mid_reso: float = 1.125  # aug: first resize to mid_reso = 1.125 * data_load_reso, then crop to data_load_reso 【mid_resolution 裁剪前先放大，避免图像中间信息丢失】
    hflip: bool = False  # augmentation: horizontal flip 【horizontal flip 图像水平翻转】
    workers: int = 0  # num workers; 0: auto, -1: don't use multiprocessing in DataLoader 【dataloader线程数，0代表自动跟踪，-1代表单线程加载】

    # progressive training
    pg: float = 0.0  # >0 for use progressive training during [0%, this] of training 【progressive traing rate，训练过程中属于渐进式训练的占比，如果处于渐进式训练期间，VAR的生成阶段的patch尺寸会受到控制，目的是让训练开始时难度不要过高】
    pg0: int = 4  # progressive initial stage, 0: from the 1st token map, 1: from the 2nd token map, etc 【progressive train patch 0，渐进式训练期间最大patch尺寸】
    pgwp: float = 0  # num of warmup epochs at each progressive stage 【progressive train warmup，渐进式训练中每次提高尺寸时需要开启warmup，自动跟踪ep/300作为每次提高尺寸的warmup轮次，目的是控制loss的权重让loss平稳】

    # would be automatically set in runtime
    cmd: str = ' '.join(sys.argv[1:])  # [automatically set; don't specify this]
    branch: str = subprocess.check_output(f'git symbolic-ref --short HEAD 2>/dev/null || git rev-parse HEAD',
                                          shell=True).decode(
        'utf-8').strip() or '[unknown]'  # [automatically set; don't specify this]
    commit_id: str = subprocess.check_output(f'git rev-parse HEAD', shell=True).decode(
        'utf-8').strip() or '[unknown]'  # [automatically set; don't specify this]
    commit_msg: str = \
        (subprocess.check_output(f'git log -1', shell=True).decode('utf-8').strip().splitlines() or ['[unknown]'])[
            -1].strip()  # [automatically set; don't specify this]
    acc_mean: float = None  # [automatically set; don't specify this]
    acc_tail: float = None  # [automatically set; don't specify this]
    L_mean: float = None  # [automatically set; don't specify this]
    L_tail: float = None  # [automatically set; don't specify this]
    vacc_mean: float = None  # [automatically set; don't specify this]
    vacc_tail: float = None  # [automatically set; don't specify this]
    vL_mean: float = None  # [automatically set; don't specify this]
    vL_tail: float = None  # [automatically set; don't specify this]
    grad_norm: float = None  # [automatically set; don't specify this]
    cur_lr: float = None  # [automatically set; don't specify this]
    cur_wd: float = None  # [automatically set; don't specify this]
    cur_it: str = ''  # [automatically set; don't specify this]
    cur_ep: str = ''  # [automatically set; don't specify this]
    remain_time: str = ''  # [automatically set; don't specify this]
    finish_time: str = ''  # [automatically set; don't specify this]

    # environment
    local_out_dir_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                           'local_output')  # [automatically set; don't specify this]
    tb_log_dir_path: str = '...tb-...'  # [automatically set; don't specify this]
    log_txt_path: str = '...'  # [automatically set; don't specify this]
    last_ckpt_path: str = '...'  # [automatically set; don't specify this]

    tf32: bool = True  # whether to use TensorFloat32 【是否使用TensorFloat32加速】
    device: str = 'cpu'  # [automatically set; don't specify this]
    seed: int = None  # seed 【基础随机种子】

    def seed_everything(self, benchmark: bool):
        torch.backends.cudnn.enabled = True
        # 不开启渐进式训练时，此时所有批次的训练数据长度时一致的，可以开启cudnn benchmark（加速卷积）
        torch.backends.cudnn.benchmark = benchmark
        # 设置种子
        if self.seed is None:
            torch.backends.cudnn.deterministic = False
        else:
            # 固定种子，保证可复现
            torch.backends.cudnn.deterministic = True
            # 分布式核心：每个进程种子不同 = 全局种子 * 总进程数 + 当前进程rank
            seed = self.seed * dist.get_world_size() + dist.get_rank()
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

    same_seed_for_all_ranks: int = 0  # this is only for distributed sampler

    def get_different_generator_for_each_rank(self) -> Optional[torch.Generator]:
        # 如果没有设置固定种子，返回None（使用默认随机数）
        if self.seed is None:
            return None
        # 创建PyTorch专用随机数生成器
        g = torch.Generator()
        # 为【每个GPU进程】设置独一无二的种子
        # 公式：全局种子 * 总进程数 + 当前进程rank
        g.manual_seed(self.seed * dist.get_world_size() + dist.get_rank())
        return g

    local_debug: bool = 'KEVIN_LOCAL' in os.environ # 【调试模式】
    dbg_nan: bool = False  # 'KEVIN_LOCAL' in os.environ

    def compile_model(self, m, fast):
        if fast == 0 or self.local_debug:
            return m
        return torch.compile(m, mode={
            1: 'reduce-overhead',
            2: 'max-autotune',
            3: 'default',
        }[fast]) if hasattr(torch, 'compile') else m

    def state_dict(self, key_ordered=True) -> Union[OrderedDict, dict]:
        d = (OrderedDict if key_ordered else dict)()
        # self.as_dict() would contain methods, but we only need variables
        for k in self.class_variables.keys():
            if k not in {'device'}:  # these are not serializable
                d[k] = getattr(self, k)
        return d

    def load_state_dict(self, d: Union[OrderedDict, dict, str]):
        if isinstance(d, str):  # for compatibility with old version
            d: dict = eval('\n'.join([l for l in d.splitlines() if '<bound' not in l and 'device(' not in l]))
        for k in d.keys():
            try:
                setattr(self, k, d[k])
            except Exception as e:
                print(f'k={k}, v={d[k]}')
                raise e

    @staticmethod
    def set_tf32(tf32: bool):
        if torch.cuda.is_available():
            # 开启卷积/矩阵乘法的TF32加速
            torch.backends.cudnn.allow_tf32 = bool(tf32)
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
            # 设置FP32矩阵乘法精度（TF32=high，速度更快）
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high' if tf32 else 'highest')
            # 打印配置日志（仅主进程）
            print(f'[tf32] [precis] torch.get_float32_matmul_precision(): {torch.get_float32_matmul_precision()}')
            print(f'[tf32] [ conv ] torch.backends.cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}')
            print(f'[tf32] [matmul] torch.backends.cuda.matmul.allow_tf32: {torch.backends.cudnn.allow_tf32}')

    def dump_log(self):
        if not dist.is_local_master():
            return
        if '1/' in self.cur_ep:  # first time to dump log
            with open(self.log_txt_path, 'w') as fp:
                json.dump(
                    {'is_master': dist.is_master(), 'name': self.exp_name, 'cmd': self.cmd, 'commit': self.commit_id,
                     'branch': self.branch, 'tb_log_dir_path': self.tb_log_dir_path}, fp, indent=0)
                fp.write('\n')

        log_dict = {}
        for k, v in {
            'it': self.cur_it, 'ep': self.cur_ep,
            'lr': self.cur_lr, 'wd': self.cur_wd, 'grad_norm': self.grad_norm,
            'L_mean': self.L_mean, 'L_tail': self.L_tail, 'acc_mean': self.acc_mean, 'acc_tail': self.acc_tail,
            'vL_mean': self.vL_mean, 'vL_tail': self.vL_tail, 'vacc_mean': self.vacc_mean, 'vacc_tail': self.vacc_tail,
            'remain_time': self.remain_time, 'finish_time': self.finish_time,
        }.items():
            if hasattr(v, 'item'): v = v.item()
            log_dict[k] = v
        with open(self.log_txt_path, 'a') as fp:
            fp.write(f'{log_dict}\n')

    def __str__(self):
        s = []
        for k in self.class_variables.keys():
            if k not in {'device', 'dbg_ks_fp'}:  # these are not serializable
                s.append(f'  {k:20s}: {getattr(self, k)}')
        s = '\n'.join(s)
        return f'{{\n{s}\n}}\n'


def init_dist_and_get_args():
    # 删除torchrun自动传入的--local-rank或--local_rank
    for i in range(len(sys.argv)):
        if sys.argv[i].startswith('--local-rank=') or sys.argv[i].startswith('--local_rank='):
            del sys.argv[i]
            break
    # explicit_bool=True，要求argparse的开关式参数要显式传入True或False
    # known_only，未定义参数不会报错，保存到.extra_args
    args = Args(explicit_bool=True).parse_args(known_only=True)
    # 如果是调试模式
    if args.local_debug:
        args.pn = '1_2_3'
        args.seed = 1
        args.aln = 1e-2
        args.alng = 1e-5
        args.saln = False
        args.afuse = False
        args.pg = 0.8
        args.pg0 = 1
    else:
        if args.data_path == '/path/to/imagenet':
            raise ValueError(f'{"*" * 40}  please specify --data_path=/path/to/imagenet  {"*" * 40}')

    # warn args.extra_args
    if len(args.extra_args) > 0:
        print(f'======================================================================================')
        print(
            f'=========================== WARNING: UNEXPECTED EXTRA ARGS ===========================\n{args.extra_args}')
        print(f'=========================== WARNING: UNEXPECTED EXTRA ARGS ===========================')
        print(f'======================================================================================\n\n')

    # init torch distributed
    from utils import misc
    os.makedirs(args.local_out_dir_path, exist_ok=True)
    misc.init_distributed_mode(local_out_path=args.local_out_dir_path, timeout=30) # 初始化分布式环境，替换打印方式，替换打印位置

    # set env
    args.set_tf32(args.tf32)
    args.seed_everything(benchmark=args.pg == 0)

    # update args: data loading
    args.device = dist.get_device()
    # 根据训练图片分辨率，调整训练数据的不同阶段=的尺寸
    if args.pn == '256':
        args.pn = '1_2_3_4_5_6_8_10_13_16'
    elif args.pn == '512':
        args.pn = '1_2_3_4_6_9_13_18_24_32'
    elif args.pn == '1024':
        args.pn = '1_2_3_4_5_7_9_12_16_21_27_36_48_64'
    args.patch_nums = tuple(map(int, args.pn.replace('-', '_').split('_')))  # 字符串转整数元组：'1_2_3' → (1,2,3)
    args.resos = tuple(pn * args.patch_size for pn in args.patch_nums) # 不同生成阶段的patch，恢复下采样倍率的长度/分辨率
    args.data_load_reso = max(args.resos) # 训练图片的最大分辨率

    # update args: bs and lr
    bs_per_gpu = round(
        args.bs / args.ac / dist.get_world_size())  # batchsize_per_gpu = batchsize / accumulation_gredient / worldsize
    args.batch_size = bs_per_gpu # 单卡batch size
    args.bs = args.glb_batch_size = args.batch_size * dist.get_world_size()  # 校正global batch size
    args.workers = min(max(0, args.workers), args.batch_size) # 计算dataloader的线程数

    # 计算学习率：lr = 基础lr × (全局batch / 256) × 梯度累积
    args.tlr = args.ac * args.tblr * args.glb_batch_size / 256
    # 权重衰减的最终值 = 传入值 或 权重衰减的衰减值
    args.twde = args.twde or args.twd
    # warmup epoch
    if args.wp == 0:
        args.wp = args.ep * 1 / 50

    # update args: progressive training
    # 训练过程中渐进式训练中，每次开启新尺寸阶段时，都需要开启warmup，控制该阶段的早期loss权重，让loss平稳。
    # wramup epoch：默认总epoch的1/300
    if args.pgwp == 0:
        args.pgwp = args.ep * 1 / 300
    # 如果开启渐进训练 → 根据渐进式训练在整体训练过程中的占比，调整对应的学习率调度策略
    if args.pg > 0:
        args.sche = f'lin{args.pg:g}'

    # update args: paths
    args.log_txt_path = os.path.join(args.local_out_dir_path, 'log.txt')
    args.last_ckpt_path = os.path.join(args.local_out_dir_path, f'ar-ckpt-last.pth')
    _reg_valid_name = re.compile(r'[^\w\-+,.]')
    tb_name = _reg_valid_name.sub(
        '_',
        f'tb-VARd{args.depth}'
        f'__pn{args.pn}'
        f'__b{args.bs}ep{args.ep}{args.opt[:4]}lr{args.tblr:g}wd{args.twd:g}'
    )
    args.tb_log_dir_path = os.path.join(args.local_out_dir_path, tb_name)

    return args

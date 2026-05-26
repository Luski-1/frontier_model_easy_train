import datetime
import functools
import glob
import os
import subprocess
import sys
import time
from collections import defaultdict, deque
from typing import Iterator, List, Tuple

import numpy as np
import pytz
import torch
import torch.distributed as tdist

import dist
from utils import arg_util

os_system = functools.partial(subprocess.call, shell=True)


def echo(info):
    os_system(
        f'echo "[$(date "+%m-%d-%H:%M:%S")] ({os.path.basename(sys._getframe().f_back.f_code.co_filename)}, line{sys._getframe().f_back.f_lineno})=> {info}"')


def os_system_get_stdout(cmd):
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')


def os_system_get_stdout_stderr(cmd):
    cnt = 0
    while True:
        try:
            sp = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        except subprocess.TimeoutExpired:
            cnt += 1
            print(f'[fetch free_port file] timeout cnt={cnt}')
        else:
            return sp.stdout.decode('utf-8'), sp.stderr.decode('utf-8')


def time_str(fmt='[%m-%d %H:%M:%S]'):
    return datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(fmt)


def init_distributed_mode(local_out_path, only_sync_master=False, timeout=30):
    """
    1. 设置分布式训练的环境变量
    2. 创建目录
    3. 调整分布式训练的输出方式
    :param local_out_path:
    :param only_sync_master:
    :param timeout:
    :return:
    """
    try:
        dist.initialize(fork=False, timeout=timeout) # 初始化分布式环境
        dist.barrier()  # 所有进程同步等待
    except RuntimeError:
        print(f'{">" * 75}  NCCL Error  {"<" * 75}', flush=True)
        time.sleep(10)

    if local_out_path is not None:
        os.makedirs(local_out_path, exist_ok=True)
    _change_builtin_print(dist.is_local_master()) # 替换打印方式，并且仅限node的master打印
    if (dist.is_master() if only_sync_master else dist.is_local_master()) and local_out_path is not None and len(
            local_out_path):
        # 替换打印位置，即重定向正常日志和错误日志
        sys.stdout, sys.stderr = SyncPrint(local_out_path, sync_stdout=True), SyncPrint(local_out_path,
                                                                                        sync_stdout=False)


def _change_builtin_print(is_master):
    # 1. 导入Python内置命名空间（所有内置函数：print/open/input 都存在这里）
    import builtins as __builtin__

    # 2. 保存【原生系统print函数】，避免直接覆盖后无法调用
    builtin_print = __builtin__.print

    # 3. 安全校验：如果原生print已经被重写过（不是函数类型），直接退出，防止重复修改
    if type(builtin_print) != type(open):
        return

    # 4. 定义【自定义新print函数】，替换原生print
    def prt(*args, **kwargs):
        # 提取3个自定义扩展参数（不会传给原生print）
        force = kwargs.pop('force', False)  # 强制打印（无视主进程权限）
        clean = kwargs.pop('clean', False)  # 纯净打印（不添加时间/文件名/行号）
        deeper = kwargs.pop('deeper', False)  # 追溯更深层的调用栈（精准定位代码）

        # 核心权限判断：是主进程 OR 强制打印 → 才执行打印
        if is_master or force:
            # clean=True：纯净打印，直接调用原生print
            if clean:
                builtin_print(*args, **kwargs)
            # clean=False：默认模式，添加日志前缀（时间+文件+行号）
            else:
                # 获取【调用print的上一层代码栈】（拿到在哪里调用的print）
                f_back = sys._getframe().f_back
                # deeper=True：再往上追溯一层（适配封装函数内的打印）
                if deeper and f_back.f_back is not None:
                    f_back = f_back.f_back

                # 截取文件名最后24个字符（避免路径太长）
                file_desc = f'{f_back.f_code.co_filename:24s}'[-24:]
                # 拼接前缀：时间 + (文件名, 行号) + 你的打印内容
                builtin_print(f'{time_str()} ({file_desc}, line{f_back.f_lineno:-4d})=>', *args, **kwargs)

    # 5. 全局替换：把Python系统自带的print，永久替换成我们自定义的prt函数
    __builtin__.print = prt


class SyncPrint(object):
    # 1. 构造函数：初始化流、打开日志文件、配置输出模式
    def __init__(self, local_output_dir, sync_stdout=True):
        # 标记当前是重定向 标准输出(stdout) 还是 标准错误(stderr)
        self.sync_stdout = sync_stdout
        # 保存【系统原始的终端流】（控制台打印），不覆盖原始输出
        self.terminal_stream = sys.stdout if sync_stdout else sys.stderr
        # 生成日志文件路径：输出目录下的 stdout.txt / stderr.txt
        fname = os.path.join(local_output_dir, 'stdout.txt' if sync_stdout else 'stderr.txt')

        # 判断文件是否已存在（用于区分重启训练）
        existing = os.path.exists(fname)
        # 以【追加模式】打开文件（a=append）：不覆盖历史日志，只在末尾新增
        self.file_stream = open(fname, 'a')

        # 如果是重启训练，写入醒目的分隔线 + 时间，区分新旧日志
        if existing:
            self.file_stream.write('\n' * 7 + '=' * 55 + f'   RESTART {time_str()}   ' + '=' * 55 + '\n')
        # 立即刷新文件缓存，确保内容写入磁盘，不丢失
        self.file_stream.flush()
        # 启用标记：防止重复关闭流
        self.enabled = True

    # 2. 核心写入方法：必须实现，流对象的核心接口
    def write(self, message):
        # 双输出核心：同时把日志写入【终端】和【文件】
        self.terminal_stream.write(message)
        self.file_stream.write(message)

    # 3. 刷新缓存：强制把缓存中的日志写入磁盘/终端
    def flush(self):
        # 同时刷新两个流，保证日志实时输出，不滞留内存
        self.terminal_stream.flush()
        self.file_stream.flush()

    # 4. 安全关闭流：释放文件句柄，恢复系统原始输出
    def close(self):
        # 仅执行一次，避免重复关闭报错
        if not self.enabled:
            return
        self.enabled = False

        # 刷新并关闭文件流
        self.file_stream.flush()
        self.file_stream.close()

        # 恢复系统原始的 stdout/stderr，防止输出流异常
        if self.sync_stdout:
            sys.stdout = self.terminal_stream
            sys.stdout.flush()
        else:
            sys.stderr = self.terminal_stream
            sys.stderr.flush()

    # 5. 析构函数：Python销毁对象时自动调用，兜底保障
    def __del__(self):
        # 自动关闭流，即使程序崩溃/退出，也能保存日志
        self.close()


class DistLogger(object):
    # 日志开关代理——根据 verbose 的值，决定底层 logger 的方法到底执行还是静默跳过
    def __init__(self, lg, verbose):
        self._lg, self._verbose = lg, verbose

    @staticmethod
    def do_nothing(*args, **kwargs):
        pass

    def __getattr__(self, attr: str):
        return getattr(self._lg, attr) if self._verbose else DistLogger.do_nothing


class TensorboardLogger(object):
    def __init__(self, log_dir, filename_suffix):
        try:
            import tensorflow_io as tfio
        except:
            pass
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=log_dir, filename_suffix=filename_suffix)
        self.step = 0

    def set_step(self, step=None):
        # 设置当前步数
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        # 写入标量指标
        for k, v in kwargs.items():
            if v is None:
                continue
            # assert isinstance(v, (float, int)), type(v)
            if step is None:  # iter wise
                it = self.step
                if it == 0 or (it + 1) % 500 == 0:
                    if hasattr(v, 'item'): v = v.item()
                    self.writer.add_scalar(f'{head}/{k}', v, it)
            else:  # epoch wise
                if hasattr(v, 'item'): v = v.item()
                self.writer.add_scalar(f'{head}/{k}', v, step)

    def log_tensor_as_distri(self, tag, tensor1d, step=None):
        # 写入分布直方图，用于观察某个 tensor 的统计分布（如权重、梯度的分布变化）
        # 每 500 步才写，epoch 模式每次都写
        if step is None:  # iter wise
            step = self.step
            loggable = step == 0 or (step + 1) % 500 == 0
        else:  # epoch wise
            loggable = True
        if loggable:
            try:
                self.writer.add_histogram(tag=tag, values=tensor1d, global_step=step)
            except Exception as e:
                print(f'[log_tensor_as_distri writer.add_histogram failed]: {e}')

    def log_image(self, tag, img_chw, step=None):
        #  写入单张图片，格式为 CHW（通道×高×宽），用于可视化生成效果
        if step is None:  # iter wise
            step = self.step
            loggable = step == 0 or (step + 1) % 500 == 0
        else:  # epoch wise
            loggable = True
        if loggable:
            self.writer.add_image(tag, img_chw, step, dataformats='CHW')

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

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

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        tdist.barrier()
        tdist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

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


class MetricLogger(object):
    def __init__(self, delimiter='  '):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.iter_end_t = time.time()
        self.log_iters = []

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if hasattr(v, 'item'): v = v.item()
            # assert isinstance(v, (float, int)), type(v)
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            if len(meter.deque):
                loss_str.append(
                    "{}: {}".format(name, str(meter))
                )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, start_it, max_iters, itrt, print_freq, header=None):
        self.log_iters = set(np.linspace(0, max_iters - 1, print_freq, dtype=int).tolist())
        self.log_iters.add(start_it)
        if not header:
            header = ''
        start_time = time.time()
        self.iter_end_t = time.time()
        self.iter_time = SmoothedValue(fmt='{avg:.4f}')
        self.data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(max_iters))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        log_msg = self.delimiter.join(log_msg)

        if isinstance(itrt, Iterator) and not hasattr(itrt, 'preload') and not hasattr(itrt, 'set_epoch'):
            for i in range(start_it, max_iters):
                obj = next(itrt)
                self.data_time.update(time.time() - self.iter_end_t)
                yield i, obj
                self.iter_time.update(time.time() - self.iter_end_t)
                if i in self.log_iters:
                    eta_seconds = self.iter_time.global_avg * (max_iters - i)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    print(log_msg.format(
                        i, max_iters, eta=eta_string,
                        meters=str(self),
                        time=str(self.iter_time), data=str(self.data_time)), flush=True)
                self.iter_end_t = time.time()
        else:
            if isinstance(itrt, int): itrt = range(itrt)
            for i, obj in enumerate(itrt):
                self.data_time.update(time.time() - self.iter_end_t)
                yield i, obj
                self.iter_time.update(time.time() - self.iter_end_t)
                if i in self.log_iters:
                    eta_seconds = self.iter_time.global_avg * (max_iters - i)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    print(log_msg.format(
                        i, max_iters, eta=eta_string,
                        meters=str(self),
                        time=str(self.iter_time), data=str(self.data_time)), flush=True)
                self.iter_end_t = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{}   Total time:      {}   ({:.3f} s / it)'.format(
            header, total_time_str, total_time / max_iters), flush=True)


def glob_with_latest_modified_first(pattern, recursive=False):
    return sorted(glob.glob(pattern, recursive=recursive), key=os.path.getmtime, reverse=True)


def auto_resume(args: arg_util.Args, pattern='ckpt*.pth') -> Tuple[List[str], int, int, dict, dict]:
    info = []
    file = os.path.join(args.local_out_dir_path, pattern)
    all_ckpt = glob_with_latest_modified_first(file)
    if len(all_ckpt) == 0:
        info.append(f'[auto_resume] no ckpt found @ {file}')
        info.append(f'[auto_resume quit]')
        return info, 0, 0, {}, {}
    else:
        info.append(f'[auto_resume] load ckpt from @ {all_ckpt[0]} ...')
        ckpt = torch.load(all_ckpt[0], map_location='cpu')
        ep, it = ckpt['epoch'], ckpt['iter']
        info.append(f'[auto_resume success] resume from ep{ep}, it{it}')
        return info, ep, it, ckpt['trainer'], ckpt['args']


def create_npz_from_sample_folder(sample_folder: str):
    """
    Builds a single .npz file from a folder of .png samples. Refer to DiT.
    """
    import os, glob
    import numpy as np
    from tqdm import tqdm
    from PIL import Image

    samples = []
    pngs = glob.glob(os.path.join(sample_folder, '*.png')) + glob.glob(os.path.join(sample_folder, '*.PNG'))
    assert len(pngs) == 50_000, f'{len(pngs)} png files found in {sample_folder}, but expected 50,000'
    for png in tqdm(pngs, desc='Building .npz file from samples (png only)'):
        with Image.open(png) as sample_pil:
            sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (50_000, samples.shape[1], samples.shape[2], 3)
    npz_path = f'{sample_folder}.npz'
    np.savez(npz_path, arr_0=samples)
    print(f'Saved .npz file to {npz_path} [shape={samples.shape}].')
    return npz_path

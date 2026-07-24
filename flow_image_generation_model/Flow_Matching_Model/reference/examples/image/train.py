# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. and affiliates.

import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from models.model_configs import instantiate_model
from train_arg_parser import get_args_parser

from training import distributed_mode
from training.data_transform import get_train_transform
from training.eval_loop import eval_model
from training.grad_scaler import NativeScalerWithGradNormCount as NativeScaler
from training.load_and_save import load_model, save_model
from training.train_loop import train_one_epoch

logger = logging.getLogger(__name__)


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    distributed_mode.init_distributed_mode(args)    # 设置分布式参数

    logger.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    logger.info("{}".format(args).replace(", ", ",\n"))
    if distributed_mode.is_main_process():                  # 如果是主进程，就把参数保存为本地文件
        args_filepath = Path(args.output_dir) / "args.json"
        logger.info(f"Saving args to {args_filepath}")
        with open(args_filepath, "w") as f:
            json.dump(vars(args), f)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + distributed_mode.get_rank()  # 设置当前进程的随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True  # 开启 测试卷积算法的实现，选用速度最快的

    logger.info(f"Initializing Dataset: {args.dataset}")
    transform_train = get_train_transform() # 数据转换方式
    # 加载数据集
    if args.dataset == "imagenet":
        dataset_train = datasets.ImageFolder(args.data_path, transform=transform_train)
    elif args.dataset == "cifar10":
        dataset_train = datasets.CIFAR10(
            root=args.data_path,
            train=True,
            download=True,
            transform=transform_train,
        )
    else:
        raise NotImplementedError(f"Unsupported dataset {args.dataset}")

    logger.info(dataset_train)

    logger.info("Intializing DataLoader")
    num_tasks = distributed_mode.get_world_size()   # 多少个进程
    global_rank = distributed_mode.get_rank()       # 当前全局进程
    sampler_train = torch.utils.data.DistributedSampler(    # 设置分布式抽样器
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    data_loader_train = torch.utils.data.DataLoader(        # 设置分布式dataloader，把抽样器传递给dataloader，控制数据的分配
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    logger.info(str(sampler_train))

    # 加载模型
    logger.info("Initializing Model")
    model = instantiate_model(                      
        architechture=args.dataset,
        is_discrete=args.discrete_flow_matching,    # 默认为连续，连续形式的模型为UNet，相信大家已经非常熟悉，因此不再对Unet或者离散形式所使用的模型进行人工注释
        use_ema=args.use_ema,                       # 默认不开启，EMA model就是指数平均移动的方式去更新备用模型
    )

    model.to(device)

    model_without_ddp = model               # 没有DDP封装的模型
    logger.info(str(model_without_ddp))

    eff_batch_size = (
        args.batch_size * args.accum_iter * distributed_mode.get_world_size()
    )

    logger.info(f"Learning rate: {args.lr:.2e}")

    logger.info(f"Accumulate grad iterations: {args.accum_iter}")
    logger.info(f"Effective batch size: {eff_batch_size}")

    if args.distributed:                    # 如果开启分布式
        model = torch.nn.parallel.DistributedDataParallel(      # DDP封装模型
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module                        # DDP拆包后的模型

    optimizer = torch.optim.AdamW(                              # 把没有DDP封装的模型参数交由AdamW管理
        model_without_ddp.parameters(), lr=args.lr, betas=args.optimizer_betas
    )
    if args.decay_lr:                                           # 是否开启学习率衰减
        lr_schedule = torch.optim.lr_scheduler.LinearLR(        # 如果开启，会在Warmup结束后进行学习率衰减，衰减起始倍率为1.0，衰减最终倍率为1e-8 / args.lr
            optimizer,
            total_iters=args.epochs,
            start_factor=1.0,
            end_factor=1e-8 / args.lr,
        )
    else:
        lr_schedule = torch.optim.lr_scheduler.ConstantLR(      # 如果不开启，则学习率恒定
            optimizer, total_iters=args.epochs, factor=1.0
        )

    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Learning-Rate Schedule: {lr_schedule}")

    loss_scaler = NativeScaler()        # 设置损失缩放器，避免损失过小导致梯度下溢

    load_model(                         # 就是加载过去的训练权重和相关参数
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        lr_schedule=lr_schedule,
    )

    logger.info(f"Start from {args.start_epoch} to {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)  # 设置分布式的开始epoch
        if not args.eval_only:
            train_stats = train_one_epoch(
                model=model,
                data_loader=data_loader_train,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                device=device,
                epoch=epoch,
                loss_scaler=loss_scaler,
                args=args,
            )
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
            }
        else:
            log_stats = {
                "epoch": epoch,
            }

        if args.output_dir and (
            (args.eval_frequency > 0 and (epoch + 1) % args.eval_frequency == 0)
            or args.eval_only
            or args.test_run
        ):
            if not args.eval_only:
                save_model(             # 保存模型
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    lr_schedule=lr_schedule,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                )
            if args.distributed:
                data_loader_train.sampler.set_epoch(0)  # 重新设置分布式抽样器
            if distributed_mode.is_main_process():
                # 主进程生成的图片数量，保证总数满足args.fid_samples
                fid_samples = args.fid_samples - (num_tasks - 1) * (
                    args.fid_samples // num_tasks
                )
            else:
                # 非主进程生成的图片数量
                fid_samples = args.fid_samples // num_tasks
            # 进行评估阶段
            eval_stats = eval_model(
                model,
                data_loader_train,
                device,
                epoch=epoch,
                fid_samples=fid_samples,
                args=args,
            )
            log_stats.update({f"eval_{k}": v for k, v in eval_stats.items()})

        if args.output_dir and distributed_mode.is_main_process():
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.test_run or args.eval_only:
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

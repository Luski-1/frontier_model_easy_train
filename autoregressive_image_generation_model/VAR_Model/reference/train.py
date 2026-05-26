import gc
import os
import shutil
import sys
import time
import warnings
from functools import partial

import torch
from torch.utils.data import DataLoader

import dist
from utils import arg_util, misc
from utils.data import build_dataset
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils.misc import auto_resume


def build_everything(args: arg_util.Args):
    # 1. 加载已有的训练信息

    # resume 开启继续训练，读取想干新
    # auto_resume_info：搜索信息，用于打印
    # start_ep 继续训练的epoch起始值，或者0
    # start_it 继续云联的iter起始值，或者0
    # trainer_state 训练器状态
    # args_state 训练器参数
    auto_resume_info, start_ep, start_it, trainer_state, args_state = auto_resume(args, 'ar-ckpt*.pth')

    # 2. 创建tensor board logger
    tb_lg: misc.TensorboardLogger
    with_tb_lg = dist.is_master()
    if with_tb_lg:
        os.makedirs(args.tb_log_dir_path, exist_ok=True)
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(
            misc.TensorboardLogger(log_dir=args.tb_log_dir_path, filename_suffix=f'__{misc.time_str("%m%d_%H%M")}'),
            verbose=True)
        tb_lg.flush()
    else:
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(None, verbose=False)
    dist.barrier()

    # log args
    print(f'global bs={args.glb_batch_size}, local bs={args.batch_size}')
    print(f'initial args:\n{str(args)}')

    # 3. 加载数据
    if not args.local_debug:
        print(f'[build PT data] ...\n')
        # 3.1. 创建dataset
        num_classes, dataset_train, dataset_val = build_dataset(
            args.data_path, final_reso=args.data_load_reso, hflip=args.hflip, mid_reso=args.mid_reso,
        )
        types = str((type(dataset_train).__name__, type(dataset_val).__name__))
        # 2. 创建dataloader
        ld_val = DataLoader(
            dataset_val, num_workers=0, pin_memory=True,
            batch_size=round(args.batch_size * 1.5),
            sampler=EvalDistributedSampler(dataset_val, num_replicas=dist.get_world_size(), rank=dist.get_rank()), # 创建数据分布器，用于分布式
            shuffle=False, drop_last=False,
        )
        del dataset_val

        ld_train = DataLoader(
            dataset=dataset_train, num_workers=args.workers, pin_memory=True,
            generator=args.get_different_generator_for_each_rank(),  # 随机数生成器
            batch_sampler=DistInfiniteBatchSampler(
                dataset_len=len(dataset_train),  # 训练集总样本数量
                glb_batch_size=args.glb_batch_size,  # 全局批次大小（所有GPU的总批次）
                same_seed_for_all_ranks=args.same_seed_for_all_ranks,  # 全局打乱共享种子
                shuffle=True,  # 开启全局随机打乱
                fill_last=True,  # 填充最后一个不完整批次
                rank=dist.get_rank(),  # 当前GPU进程编号（0/1/2/3...）
                world_size=dist.get_world_size(),  # 总GPU进程数
                start_ep=start_ep,  # 断点续训：起始轮次
                start_it=start_it,  # 断点续训：起始迭代数
            ),
        )
        del dataset_train

        [print(line) for line in auto_resume_info]
        print(f'[dataloader multi processing] ...', end='', flush=True)
        stt = time.time()
        iters_train = len(ld_train)
        ld_train = iter(ld_train)
        # noinspection PyArgumentList
        print(f'     [dataloader multi processing](*) finished! ({time.time() - stt:.2f}s)', flush=True, clean=True)
        print(
            f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size}, iters_train={iters_train}, types(tr, va)={types}')

    else:
        num_classes = 1000
        ld_val = ld_train = None
        iters_train = 10

    # build models
    from torch.nn.parallel import DistributedDataParallel as DDP
    from models import VAR, VQVAE, build_vae_var
    from trainer import VARTrainer
    from utils.amp_sc import AmpOptimizer
    from utils.lr_control import filter_params
    # 4. 创建模型
    vae_local, var_wo_ddp = build_vae_var(
        V=4096, # 词表维度
        Cvae=32, # vae的latent的维度
        ch=160, # 卷积通道数
        share_quant_resi=4,  # hard-coded VQVAE hyperparameters
        device=dist.get_device(),
        patch_nums=args.patch_nums,  # 256情况下 (1,2,3,4,5,6,8,10,13,16)
        num_classes=num_classes,  # imageNet种类数 1000
        depth=args.depth,  # 16
        shared_aln=args.saln,  # 是否共享adaln False
        attn_l2_norm=args.anorm,  # Attention计算前是否对Q/K使用L2归一化   True
        flash_if_available=args.fuse, # 是否开启算子加速
        fused_if_available=args.fuse, # 是否开启算子加速
        init_adaln=args.aln,  # 0.5 adaLN的shift/scale的初始缩放
        init_adaln_gamma=args.alng,  # 1e-5 adaLN的gamma的初始缩放
        init_head=args.hd,  # 0.02 输出头的缩放
        init_std=args.ini,  # -1 模型参数的初始化方差
    )
    # 5.加载VAE权重参数
    vae_ckpt = 'vae_ch160v4096z32.pth'
    if dist.is_local_master():
        if not os.path.exists(vae_ckpt):
            os.system(f'wget https://huggingface.co/FoundationVision/var/resolve/main/{vae_ckpt}')
    dist.barrier()
    vae_local.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    # 6.编译和封装模型
    # torch.compile PyTorch2.0新增的动态编译引擎，自动优化模型计算图，对Transformer加速效果极强。
    # fast值   编译模式    作用（通俗版）     适用场景
    # 1   reduce - overhead   轻量加速，开销小、速度快一点  快速调试、小模型
    # 2   max - autotune      极致加速，自动优化所有算子   正式训练 / 推理，首选
    # 3   default             平衡加速，速度和稳定性折中   兼容性优先
    vae_local: VQVAE = args.compile_model(vae_local, args.vfast)
    var_wo_ddp: VAR = args.compile_model(var_wo_ddp, args.tfast)
    var: DDP = (DDP if dist.initialized() else NullDDP)(var_wo_ddp, device_ids=[dist.get_local_rank()],
                                                        find_unused_parameters=False, broadcast_buffers=False)

    print(f'[INIT] VAR model = {var_wo_ddp}\n\n')
    count_p = lambda m: f'{sum(p.numel() for p in m.parameters()) / 1e6:.2f}'
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (
        ('VAE', vae_local), ('VAE.enc', vae_local.encoder), ('VAE.dec', vae_local.decoder),
        ('VAE.quant', vae_local.quantize))]))
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('VAR', var_wo_ddp),)]) + '\n\n')

    # 7. 创建优化器
    # 7.1 找出需要训练的参数，并且划分为需要权重衰减，和不需要权重衰减的分组
    # names: 所有可以训练的参数名称
    # paras: 所有可以训练的参数
    # para_groups: 训练的参数分组 [{'params': [...], 'wd_sc': 0, 'lr_sc': 1}, {'params': [...], 'wd_sc': 1, 'lr_sc': 1}]
    names, paras, para_groups = filter_params(var_wo_ddp, nowd_keys={
        'cls_token', 'start_token', 'task_token', 'cfg_uncond',  # 特殊标记token
        'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',  # 位置编码/层级编码（核心！）
        'gamma', 'beta',  # LayerNorm归一化参数
        'ada_gss', 'moe_bias',  # AdaLN自适应参数
        'scale_mul',
    })
    # 7.2获取优化器
    opt_clz = {
        'adam': partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),  # True
        'adamw': partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
    }[args.opt.lower().strip()]
    opt_kw = dict(lr=args.tlr, weight_decay=0) # tlr = tblr × (全局batch / 256) × 梯度累积 
    print(f'[INIT] optim={opt_clz}, opt_kw={opt_kw}\n')
    # 7.3 封装优化器
    var_optim = AmpOptimizer(
        mixed_precision=args.fp16,  # 0
        optimizer=opt_clz(params=para_groups, **opt_kw),
        names=names,
        paras=paras,
        grad_clip=args.tclip,  # 2
        n_gradient_accumulation=args.ac  # 1
    )
    del names, paras, para_groups

    # 8. 创建trainer
    trainer = VARTrainer(
        device=args.device,
        patch_nums=args.patch_nums,  # (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        resos=args.resos, # 不同生成阶段的patch，恢复下采样倍率后的长度/分辨率
        vae_local=vae_local,
        var_wo_ddp=var_wo_ddp, # without DDP
        var=var,    # with DDP
        var_opt=var_optim,  # AmpOptimizer AmpOptimizer.optimizer = AdamW
        label_smooth=args.ls,  # 标签平滑0
    )
    # 加载var参数
    if trainer_state is not None and len(trainer_state):
        trainer.load_state_dict(trainer_state, strict=False, skip_vae=True)  # don't load vae again
    del vae_local, var_wo_ddp, var, var_optim
    # 如果是调试模式
    if args.local_debug:
        rng = torch.Generator('cpu')
        rng.manual_seed(0)
        B = 4
        inp = torch.rand(B, 3, args.data_load_reso, args.data_load_reso)
        label = torch.ones(B, dtype=torch.long)

        me = misc.MetricLogger(delimiter='  ')
        trainer.train_step(
            it=0, g_it=0, stepping=True, metric_lg=me, tb_lg=tb_lg,
            inp_B3HW=inp, label_B=label, prog_si=args.pg0, prog_wp_it=20,
        )
        trainer.load_state_dict(trainer.state_dict())
        trainer.train_step(
            it=99, g_it=599, stepping=True, metric_lg=me, tb_lg=tb_lg,
            inp_B3HW=inp, label_B=label, prog_si=-1, prog_wp_it=20,
        )
        print({k: meter.global_avg for k, meter in me.meters.items()})

        args.dump_log()
        tb_lg.flush()
        tb_lg.close()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close(), sys.stderr.close()
        exit(0)

    dist.barrier()
    return (
        tb_lg, trainer, start_ep, start_it,
        iters_train, ld_train, ld_val
    )


def main_training():
    args: arg_util.Args = arg_util.init_dist_and_get_args() # 初始化分布式环境以及获取参数  
    if args.local_debug:
        torch.autograd.set_detect_anomaly(True)
    # tb_lg： tensorboard logger
    # start_ep: resume epoch or 0
    # start_it: resume iter or 0
    # iters_train: iter of one epoch 
    # ld_train: dataloader of train
    # ld_evl: dataloader of val
    (
        tb_lg, trainer,
        start_ep, start_it,
        iters_train, ld_train, ld_val
        ) = build_everything(args)

    # train
    # 3.1 记录训练开始时间
    start_time = time.time()

    # 3.2 初始化最佳训练指标（用于监控训练进度）
    best_L_mean, best_L_tail, best_acc_mean, best_acc_tail = 999., 999., -1., -1.

    # 3.3 初始化最佳验证指标（用于保存最佳 Checkpoint）
    best_val_loss_mean, best_val_loss_tail, best_val_acc_mean, best_val_acc_tail = 999, 999, -1, -1.

    # 3.4 初始化训练损失占位符
    L_mean, L_tail = -1, -1

    for ep in range(start_ep, args.ep):
        # False 为采样器设置epoch相关的随机种子
        if hasattr(ld_train, 'sampler') and hasattr(ld_train.sampler, 'set_epoch'):
            ld_train.sampler.set_epoch(ep)
            if ep < 3:
                # noinspection PyArgumentList
                print(f'[{type(ld_train).__name__}] [ld_train.sampler.set_epoch({ep})]', flush=True, force=True)
        tb_lg.set_step(ep * iters_train)

        stats, (sec, remain_time, finish_time) = train_one_ep(
            ep, # current epoch
            ep == start_ep, # 是否为起始epoch
            start_it if ep == start_ep else 0, # 如果是起始epoch就起始iter，否则iter从0开始
            args,
            tb_lg,
            ld_train, # dataloder of train
            iters_train, # 单epoch的总步数
            trainer
        )
        # 以下都是日志信息，eval测试，以及保存
        L_mean, L_tail, acc_mean, acc_tail, grad_norm = stats['Lm'], stats['Lt'], stats['Accm'], stats['Acct'], stats['tnm']
        best_L_mean, best_acc_mean = min(best_L_mean, L_mean), max(best_acc_mean, acc_mean)
        if L_tail != -1: best_L_tail, best_acc_tail = min(best_L_tail, L_tail), max(best_acc_tail, acc_tail)
        args.L_mean, args.L_tail, args.acc_mean, args.acc_tail, args.grad_norm = L_mean, L_tail, acc_mean, acc_tail, grad_norm
        args.cur_ep = f'{ep + 1}/{args.ep}'
        args.remain_time, args.finish_time = remain_time, finish_time

        AR_ep_loss = dict(L_mean=L_mean, L_tail=L_tail, acc_mean=acc_mean, acc_tail=acc_tail)
        # 验证和保存条件：每10个 Epoch 或 最后一个 Epoch
        is_val_and_also_saving = (ep + 1) % 10 == 0 or (ep + 1) == args.ep
        if is_val_and_also_saving:
            # 开启val测试
            val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail, tot, cost = trainer.eval_ep(ld_val)
            best_updated = best_val_loss_tail > val_loss_tail
            best_val_loss_mean, best_val_loss_tail = min(best_val_loss_mean, val_loss_mean), min(best_val_loss_tail,
                                                                                                 val_loss_tail)
            best_val_acc_mean, best_val_acc_tail = max(best_val_acc_mean, val_acc_mean), max(best_val_acc_tail,
                                                                                             val_acc_tail)
            AR_ep_loss.update(vL_mean=val_loss_mean, vL_tail=val_loss_tail, vacc_mean=val_acc_mean,
                              vacc_tail=val_acc_tail)
            args.vL_mean, args.vL_tail, args.vacc_mean, args.vacc_tail = val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail
            print(
                f' [*] [ep{ep}]  (val {tot})  Lm: {L_mean:.4f}, Lt: {L_tail:.4f}, Acc m&t: {acc_mean:.2f} {acc_tail:.2f},  Val cost: {cost:.2f}s')
            # 保存模型
            if dist.is_local_master():
                local_out_ckpt = os.path.join(args.local_out_dir_path, 'ar-ckpt-last.pth')
                local_out_ckpt_best = os.path.join(args.local_out_dir_path, 'ar-ckpt-best.pth')
                print(f'[saving ckpt] ...', end='', flush=True)
                torch.save({
                    'epoch': ep + 1,
                    'iter': 0,
                    'trainer': trainer.state_dict(),
                    'args': args.state_dict(),
                }, local_out_ckpt)
                if best_updated:
                    shutil.copy(local_out_ckpt, local_out_ckpt_best)
                print(f'     [saving ckpt](*) finished!  @ {local_out_ckpt}', flush=True, clean=True)
            dist.barrier()

        print(
            f'     [ep{ep}]  (training )  Lm: {best_L_mean:.3f} ({L_mean:.3f}), Lt: {best_L_tail:.3f} ({L_tail:.3f}),  Acc m&t: {best_acc_mean:.2f} {best_acc_tail:.2f},  Remain: {remain_time},  Finish: {finish_time}',
            flush=True)
        tb_lg.update(head='AR_ep_loss', step=ep + 1, **AR_ep_loss)
        tb_lg.update(head='AR_z_burnout', step=ep + 1, rest_hours=round(sec / 60 / 60, 2))
        args.dump_log()
        tb_lg.flush()

    total_time = f'{(time.time() - start_time) / 60 / 60:.1f}h'
    print('\n\n')
    print(
        f'  [*] [PT finished]  Total cost: {total_time},   Lm: {best_L_mean:.3f} ({L_mean}),   Lt: {best_L_tail:.3f} ({L_tail})')
    print('\n\n')

    del stats
    del iters_train, ld_train
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)

    args.remain_time, args.finish_time = '-', time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() - 60))
    print(f'final args:\n\n{str(args)}')
    args.dump_log()
    tb_lg.flush()
    tb_lg.close()
    dist.barrier()


def train_one_ep(
        ep: int, # 当前epoch
        is_first_ep: bool, # 是否为起始epoch
        start_it: int, # 如果是起始epoch就是起始iter，否则为0
        args: arg_util.Args, 
        tb_lg: misc.TensorboardLogger,
        ld_or_itrt, # dataloader
        iters_train: int, # 单epoch的总步数
        trainer):
    # import heavy packages after Dataloader object creation
    from trainer import VARTrainer
    from utils.lr_control import lr_wd_annealing
    trainer: VARTrainer

    step_cnt = 0
    me = misc.MetricLogger(delimiter='  ') # 指标logger
    me.add_meter('tlr', misc.SmoothedValue(window_size=1, fmt='{value:.2g}'))
    me.add_meter('tnm', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    [me.add_meter(x, misc.SmoothedValue(fmt='{median:.3f} ({global_avg:.3f})')) for x in ['Lm', 'Lt']]
    [me.add_meter(x, misc.SmoothedValue(fmt='{median:.2f} ({global_avg:.2f})')) for x in ['Accm', 'Acct']]
    header = f'[Ep]: [{ep:4d}/{args.ep}]'

    if is_first_ep:
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)

    # g_it：已完成训练的总步数
    # max_it：完整训练的总步数
    g_it, max_it = ep * iters_train, args.ep * iters_train

    # MetricLogger.log_every简单而言就是个封装器，在yield dataloader的数据前后计算指标
    # it是当前epoch内第几步；inp是dataloader的输入数据，label是dataloader的分类数据
    for it, (inp, label) in me.log_every(start_it, iters_train, ld_or_itrt, 30 if iters_train > 8000 else 5, header):
        # 累加已完成训练的总步数
        g_it = ep * iters_train + it
        if it < start_it: continue  # 断点续训：跳过已训练的 Iteration
        if is_first_ep and it == start_it: warnings.resetwarnings()  # 第一个 Epoch 的起始 Iteration：恢复警告

        inp = inp.to(args.device, non_blocking=True)
        label = label.to(args.device, non_blocking=True)

        args.cur_it = f'{it + 1}/{iters_train}'

        wp_it = args.wp * iters_train  # 计算warmup总步数
        min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(args.sche,  # lin0
                                                             trainer.var_opt.optimizer, # 用于调整优化器内的基础学习率以及权重衰减值
                                                             args.tlr,  # learning rate
                                                             args.twd,  # weight decay  0.05
                                                             args.twde,  # weight decay end 0.05
                                                             g_it,  # 当前已完成训练的总步数
                                                             wp_it,  # warmup的总步数
                                                             max_it,  # 完整训练的总步数
                                                             wp0=args.wp0,  # warmup初始，基础学习率的比例
                                                             wpe=args.wpe)  # warmup最终时，基础学习率的比例
        args.cur_lr, args.cur_wd = max_tlr, max_twd
        # 渐进式训练，默认关闭
        # 渐进式开启后：1) Warmup：只训1×1 2)中期：逐步增加到1×1 + 2×2 + ... + 13×13 3)后期：训所有尺度
        if args.pg:  # default: args.pg == 0.0, means no progressive training, won't get into this
            if g_it <= wp_it:
                # Warmup 阶段：只训练初始尺度（args.pg0）
                prog_si = args.pg0  # 4
            elif g_it >= max_it * args.pg:
                # 训练后期：训练所有尺度（prog_si = -1）
                prog_si = len(args.patch_nums) - 1
            else:
                # 中间阶段：线性增加训练的尺度
                delta = len(args.patch_nums) - 1 - args.pg0
                progress = min(max((g_it - wp_it) / (max_it * args.pg - wp_it), 0), 1)  # 进度从 0 到 1
                prog_si = args.pg0 + round(progress * delta)  # 从初始尺度线性增加到所有尺度
        else:
            prog_si = -1

        stepping = (g_it + 1) % args.ac == 0    # 计算是否满足参数更新
        step_cnt += int(stepping)   # 统计实际参数更新次数

        grad_norm, scale_log2 = trainer.train_step(
            it=it,  # 当前epoch内第几步
            g_it=g_it,  # 已完成训练的总步数
            stepping=stepping,  # 是否参数更新
            metric_lg=me,
            tb_lg=tb_lg,
            inp_B3HW=inp,   # 输入数据【B3HW】
            label_B=label,  # 输入标签
            prog_si=prog_si,    # 渐进式训练的最大尺度/stride，默认-1
            prog_wp_it=args.pgwp * iters_train, # 渐进式训练的各尺度/stride 的warmup总步数
        )

        me.update(tlr=max_tlr)
        tb_lg.set_step(step=g_it)
        tb_lg.update(head='AR_opt_lr/lr_min', sche_tlr=min_tlr)
        tb_lg.update(head='AR_opt_lr/lr_max', sche_tlr=max_tlr)
        tb_lg.update(head='AR_opt_wd/wd_max', sche_twd=max_twd)
        tb_lg.update(head='AR_opt_wd/wd_min', sche_twd=min_twd)
        tb_lg.update(head='AR_opt_grad/fp16', scale_log2=scale_log2)

        if args.tclip > 0:
            tb_lg.update(head='AR_opt_grad/grad', grad_norm=grad_norm)
            tb_lg.update(head='AR_opt_grad/grad', grad_clip=args.tclip)

    me.synchronize_between_processes()
    # 返回：
    # 1. 训练指标字典
    # 2. 时间统计（当前Epoch耗时、预计剩余时间、预计完成时间）
    return {k: meter.global_avg for k, meter in me.meters.items()}, me.iter_time.time_preds(
        max_it - (g_it + 1) + (args.ep - ep) * 15)  # +15: other cost


class NullDDP(torch.nn.Module):
    def __init__(self, module, *args, **kwargs):
        super(NullDDP, self).__init__()
        self.module = module
        self.require_backward_grad_sync = False

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


if __name__ == '__main__':
    try:
        main_training()
    finally:
        dist.finalize()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close(), sys.stderr.close()

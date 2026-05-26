
import gc
import os
import shutil
import time

import torch
import torch.nn as nn
from functools import partial
from torch.utils.data import DataLoader

from new_args import parse_args, seed_everything, set_tf32, PATCH_NUMS, SHARE_QUANT_RESI
from new_data import build_dataloaders
from new_var import VQVAE, VAR, build_vae_var
from new_utils import AmpOptimizer, lr_wd_annealing, filter_params, auto_resume, SmoothedValue


def main_training():
    # ===================== 1. 初始化 =====================
    args = parse_args()
    set_tf32(args.tf32)
    seed_everything(args.seed if args.seed is not None else 0, benchmark=True)

    print(f'batch_size={args.bs}')
    print(f'args:\n{vars(args)}')

    # ===================== 2. 加载数据 =====================
    num_classes, ld_train, ld_val = build_dataloaders(args)
    iters_train = len(ld_train)

    # ===================== 3. resume =====================
    auto_resume_info, start_ep, start_it, trainer_state, args_state = auto_resume(args.output_dir, 'ar-ckpt*.pth')
    [print(line) for line in auto_resume_info]

    # ===================== 4. 创建模型 =====================
    vae_local, var = build_vae_var(
        V=4096,
        Cvae=32,
        ch=160,
        share_quant_resi=SHARE_QUANT_RESI,
        device=args.device,
        patch_nums=args.patch_nums,
        num_classes=num_classes,
        depth=args.depth,
        attn_l2_norm=args.anorm,
        init_adaln=args.aln,
        init_adaln_gamma=args.alng,
        init_head=args.hd,
        init_std=args.ini,
    )

    # ===================== 5. 加载VAE权重 =====================
    if not os.path.exists(args.vae_ckpt):
        os.system(f'wget https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth')
    vae_local.load_state_dict(torch.load(args.vae_ckpt, map_location='cpu'), strict=True)

    print(f'[INIT] VAR model = {var}\n\n')
    count_p = lambda m: f'{sum(p.numel() for p in m.parameters()) / 1e6:.2f}'
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (
        ('VAE', vae_local), ('VAE.enc', vae_local.encoder), ('VAE.dec', vae_local.decoder),
        ('VAE.quant', vae_local.quantize))]))
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('VAR', var),)]) + '\n\n')

    # ===================== 6. 创建优化器 =====================
    names, paras, para_groups = filter_params(var, nowd_keys={
        'cls_token', 'start_token', 'task_token', 'cfg_uncond',
        'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
        'gamma', 'beta',
        'scale_mul',
    })

    optimizer = torch.optim.AdamW(params=para_groups, lr=args.tlr, weight_decay=0, betas=(0.9, 0.95))
    print(f'[INIT] optim=AdamW, lr={args.tlr}\n')

    var_optim = AmpOptimizer(
        mixed_precision=args.fp16,
        optimizer=optimizer,
        names=names,
        paras=paras,
        grad_clip=args.tclip,
    )
    del names, paras, para_groups

    # ===================== 7. 创建损失函数 =====================
    train_loss = nn.CrossEntropyLoss(label_smoothing=args.ls, reduction='none')
    val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')

    L = sum(pn * pn for pn in args.patch_nums)
    last_l = args.patch_nums[-1] * args.patch_nums[-1]
    loss_weight = torch.ones(1, L, device=args.device) / L

    # 加载 trainer state
    if trainer_state and len(trainer_state):
        var.load_state_dict(trainer_state.get('var', {}), strict=False)
        var_optim.load_state_dict(trainer_state.get('var_opt', {}), strict=False)

    # ===================== 8. 训练循环 =====================
    start_time = time.time()
    best_L_mean, best_acc_mean = 999., -1.
    best_val_loss_mean, best_val_loss_tail = 999, 999

    for ep in range(start_ep, args.ep):
        # 学习率调度
        wp_it = args.wp * iters_train
        max_it = args.ep * iters_train

        # 训练指标
        ep_loss_mean = SmoothedValue(fmt='{median:.4f} ({global_avg:.4f})')
        ep_loss_tail = SmoothedValue(fmt='{median:.4f} ({global_avg:.4f})')
        ep_acc_mean = SmoothedValue(fmt='{median:.2f} ({global_avg:.2f})')
        ep_acc_tail = SmoothedValue(fmt='{median:.2f} ({global_avg:.2f})')
        ep_grad_norm = SmoothedValue(fmt='{value:.2f}')
        ep_tlr = SmoothedValue(window_size=1, fmt='{value:.2g}')

        var.train()
        for it, (inp, label) in enumerate(ld_train):
            if ep == start_ep and it < start_it:
                continue  # 断点续训：跳过已训练的 iteration

            g_it = ep * iters_train + it

            # 学习率和权重衰减调度
            min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(
                args.sche, var_optim.optimizer,
                args.tlr, args.twd, args.twde,
                g_it, wp_it, max_it,
                wp0=args.wp0, wpe=args.wpe
            )

            inp = inp.to(args.device, non_blocking=True)
            label = label.to(args.device, non_blocking=True)

            # ========== 训练步 ==========
            B, V = label.shape[0], vae_local.vocab_size

            # 图像编码为真实 Token（Ground Truth）
            gt_idx_Bl = vae_local.img_to_idxBl(inp) # [[B,1], [B,4], [B,9]...]
            gt_BL = torch.cat(gt_idx_Bl, dim=1) # [B, L] L是总token数
            # 转换为 VAR 的训练输入数据（去掉第一个尺度的 Token，因为第一个尺度由 SOS 代替）
            # x_BLCv_wo_first_l: 维度[B,L,Cvae] without first level token
            x_BLCv_wo_first_l = vae_local.quantize.idxBl_to_var_input(gt_idx_Bl)

            with var_optim.amp_ctx:
                logits_BLV = var(label, x_BLCv_wo_first_l)  # 开启预训练
                loss = train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
                loss = loss.mul(loss_weight).sum(dim=-1).mean()

            grad_norm, scale_log2 = var_optim.backward_clip_step(loss)

            # 记录指标
            pred_BL = logits_BLV.data.argmax(dim=-1)
            Lmean = val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)).item()
            acc_mean = (pred_BL == gt_BL).float().mean().item() * 100
            Ltail = val_loss(logits_BLV.data[:, -last_l:].reshape(-1, V),
                             gt_BL[:, -last_l:].reshape(-1)).item()
            acc_tail = (pred_BL[:, -last_l:] == gt_BL[:, -last_l:]).float().mean().item() * 100

            ep_loss_mean.update(Lmean)
            ep_loss_tail.update(Ltail)
            ep_acc_mean.update(acc_mean)
            ep_acc_tail.update(acc_tail)
            if grad_norm is not None:
                ep_grad_norm.update(grad_norm.item())
            ep_tlr.update(max_tlr)

            # 打印训练进度
            if (it + 1) % 50 == 0 or it == 0:
                print(f'[Ep {ep+1}/{args.ep}] [It {it+1}/{iters_train}] '
                      f'Lm={Lmean:.4f}, Lt={Ltail:.4f}, Acc={acc_mean:.2f}/{acc_tail:.2f}, '
                      f'lr={max_tlr:.2g}, grad={grad_norm.item() if grad_norm else 0:.2f}')

        # ===================== epoch 统计 =====================
        L_mean = ep_loss_mean.global_avg
        L_tail = ep_loss_tail.global_avg
        acc_mean = ep_acc_mean.global_avg
        acc_tail = ep_acc_tail.global_avg
        best_L_mean = min(best_L_mean, L_mean)
        best_acc_mean = max(best_acc_mean, acc_mean)

        print(f'[ep{ep+1}] Lm={L_mean:.4f}, Lt={L_tail:.4f}, Acc={acc_mean:.2f}/{acc_tail:.2f}, '
              f'best_Lm={best_L_mean:.4f}, best_Acc={best_acc_mean:.2f}')

        # ===================== 验证 + 保存 =====================
        is_val = (ep + 1) % 10 == 0 or (ep + 1) == args.ep
        if is_val:
            # 验证
            val_L_mean, val_L_tail, val_acc_mean, val_acc_tail = eval_ep(
                var, vae_local, ld_val, val_loss, args.device, args.patch_nums
            )
            best_updated = best_val_loss_tail > val_L_tail
            best_val_loss_mean = min(best_val_loss_mean, val_L_mean)
            best_val_loss_tail = min(best_val_loss_tail, val_L_tail)

            print(f'[ep{ep+1} val] vLm={val_L_mean:.4f}, vLt={val_L_tail:.4f}, '
                  f'vAcc={val_acc_mean:.2f}/{val_acc_tail:.2f}')

            # 保存 checkpoint
            local_out_ckpt = os.path.join(args.output_dir, 'ar-ckpt-last.pth')
            print(f'[saving ckpt] ...', end='', flush=True)
            torch.save({
                'epoch': ep + 1,
                'iter': 0,
                'trainer': {
                    'var': var.state_dict(),
                    'var_opt': var_optim.state_dict(),
                },
                'args': vars(args),
            }, local_out_ckpt)
            if best_updated:
                local_out_ckpt_best = os.path.join(args.output_dir, 'ar-ckpt-best.pth')
                shutil.copy(local_out_ckpt, local_out_ckpt_best)
            print(f'     finished!  @ {local_out_ckpt}')

    # ===================== 训练完成 =====================
    total_time = f'{(time.time() - start_time) / 60 / 60:.1f}h'
    print(f'\n\n  [*] [Training finished] Total cost: {total_time}, best_Lm={best_L_mean:.4f}\n\n')

    time.sleep(3)
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def eval_ep(var, vae_local, ld_val, val_loss_fn, device, patch_nums):
    """验证一个 epoch（简化版，去掉分布式）"""
    tot = 0
    L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
    stt = time.time()

    last_l = patch_nums[-1] * patch_nums[-1]

    training = var.training
    var.eval()

    for inp_B3HW, label_B in ld_val:
        B, V = label_B.shape[0], vae_local.vocab_size
        inp_B3HW = inp_B3HW.to(device, non_blocking=True)
        label_B = label_B.to(device, non_blocking=True)

        gt_idx_Bl = vae_local.img_to_idxBl(inp_B3HW)
        gt_BL = torch.cat(gt_idx_Bl, dim=1)
        x_BLCv_wo_first_l = vae_local.quantize.idxBl_to_var_input(gt_idx_Bl)

        logits_BLV = var(label_B, x_BLCv_wo_first_l)
        L_mean += val_loss_fn(logits_BLV.data.view(-1, V), gt_BL.view(-1)) * B
        L_tail += val_loss_fn(logits_BLV.data[:, -last_l:].reshape(-1, V),
                              gt_BL[:, -last_l:].reshape(-1)) * B
        acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100 / gt_BL.shape[1])
        acc_tail += (logits_BLV.data[:, -last_l:].argmax(dim=-1) == gt_BL[:, -last_l:]).sum() * (100 / last_l)
        tot += B

    var.train(training)

    L_mean /= tot
    L_tail /= tot
    acc_mean /= tot
    acc_tail /= tot
    return L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), time.time() - stt


if __name__ == '__main__':
    main_training()
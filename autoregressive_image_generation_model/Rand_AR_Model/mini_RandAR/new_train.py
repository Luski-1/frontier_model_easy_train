import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from new_utils import instantiate_from_config, get_scheduler, make_grid, save_model_safetensors
from torch.utils.data import DataLoader
from new_dataset import build_dataset
from torchvision import transforms
from omegaconf import OmegaConf
import argparse
import shutil
import time
import os



def cycle(dl: torch.utils.data.DataLoader):
    # loop over the dataloader indefinitely
    while True:
        for data in dl:
            yield data


def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    device = torch.device("cuda")
    config = OmegaConf.load(args.config)

    #################### Setup ####################
    args.exp_name = args.exp_name + f'_bs_{config.global_batch_size}_lr_{config.optimizer.lr}'
    experiment_dir = os.path.join(args.results_dir, args.exp_name)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 设置随机种子
    torch.manual_seed(config.global_seed)

    # 梯度累积步数（从配置中读取，默认为 1）
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)

    # 混合精度设置
    # bf16 不需要 GradScaler；fp16 需要
    use_amp = args.mixed_precision != "none"
    amp_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda") if args.mixed_precision == "fp16" else None

    print(f"[INFO] Experiment directory: {experiment_dir}")
    print(f"[INFO] Checkpoint directory: {checkpoint_dir}")
    print(f"[INFO] Mixed precision: {args.mixed_precision} (amp_dtype={amp_dtype})")
    print(f"[INFO] Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"[INFO] Config: {dict(config)}")

    #################### Data ####################
    # 获取dataset
    dataset = build_dataset(is_train=True, args=args, transform=transforms.ToTensor())
    # 单 GPU，per_gpu_batch_size = global_batch_size / gradient_accumulation_steps
    per_gpu_batch_size = int(config.global_batch_size // gradient_accumulation_steps)
    # 获取dataloader
    data_loader = DataLoader(
        dataset,
        batch_size=per_gpu_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=8 if args.num_workers > 0 else None,
    )
    print(f"[INFO] Dataset contains {len(dataset)} samples, {len(dataset) // per_gpu_batch_size} batches per epoch")
    print(f"[INFO] Per-GPU batch size: {per_gpu_batch_size}, Effective global batch size: {per_gpu_batch_size * gradient_accumulation_steps}")

    #################### Model ####################
    # 参考configs/randar/xxx.yaml中ar_model.target
    model = instantiate_from_config(config.ar_model).to(device)
    print(f"[INFO] GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 参考configs/randar/xxx.yaml中tokenizer.target
    tokenizer = instantiate_from_config(config.tokenizer).to(device).eval()
    ckpt = torch.load(args.vq_ckpt, map_location="cpu")
    if 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    # 加载参数
    tokenizer.load_state_dict(state_dict)
    tokenizer.eval()
    for param in tokenizer.parameters():
        param.requires_grad = False
    del ckpt

    #################### Optimization ####################
    # 获取优化器
    optimizer = model.configure_optimizer(**config.optimizer)
    # 获取学习率调度器（单 GPU）

    lr_scheduler = get_scheduler(
        name=config.lr_scheduler.type,
        optimizer=optimizer,
        num_warmup_steps=config.lr_scheduler.warm_up_iters * gradient_accumulation_steps,
        num_training_steps=config.max_iters * gradient_accumulation_steps,
        min_lr_ratio=config.lr_scheduler.min_lr_ratio,
        num_cycles=config.lr_scheduler.num_cycles,
    )

    model.train()

    # 循环调用dataloader
    data_loader = cycle(data_loader)

    total_iters = config.max_iters

    ################## Resume Training ##################
    # 继续训练：从 checkpoint_dir 中找最新的 checkpoint
    if os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0:
        saved_ckpt_dirs = [_ for _ in os.listdir(checkpoint_dir) if _.startswith("iters")]
        saved_ckpt_dirs = sorted(saved_ckpt_dirs)
        ckpt_path = os.path.join(checkpoint_dir, saved_ckpt_dirs[-1])
        print(f"[INFO] Resuming from {ckpt_path}")

        # 加载模型权重
        ckpt_file = os.path.join(ckpt_path, "model.safetensors")
        if os.path.exists(ckpt_file):
            # safetensors 格式
            from new_utils import load_safetensors
            state_dict = load_safetensors(ckpt_file)
            model.load_state_dict(state_dict)
        else:
            # pytorch 格式
            ckpt_data = torch.load(os.path.join(ckpt_path, "checkpoint.pt"), map_location=device)
            model.load_state_dict(ckpt_data['model'])
            optimizer.load_state_dict(ckpt_data['optimizer'])
            lr_scheduler.load_state_dict(ckpt_data['lr_scheduler'])

        train_steps = int(saved_ckpt_dirs[-1].split("_")[-1])
    else:
        train_steps = 0

    #################### Training Loop ####################
    model.train()

    log_iters, running_loss, running_grad_norm, start_time = 0, 0.0, 0.0, time.time()
    micro_step = 0  # 梯度累积内的微步计数器

    print(f"[INFO] Starting training from iteration {train_steps} to {total_iters}")

    while train_steps < total_iters:
        model.train()
        # 图像x，分类y, 下标
        x, y, inat_index = next(data_loader)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        image_tokens = x.reshape(x.shape[0], -1) # 如果是原始图像，则维度是B, H*W*C；如果是VQVAE处理过的图像，则维度是B, block_size
        cond = y.reshape(-1) # B

        # 混合精度前向 + 反向
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            logits, loss, token_order = model(image_tokens, cond, targets=image_tokens) # logits : (bsz, seq_len, vocab)
            # 梯度累积：loss 除以累积步数，使得累积后的梯度等同于大 batch 的梯度
            loss = loss / gradient_accumulation_steps

        # 反向传播
        if scaler is not None:
            # fp16 模式：使用 GradScaler
            scaler.scale(loss).backward()
        else:
            # bf16 或 fp32 模式：直接 backward
            loss.backward()

        micro_step += 1

        # 只在累积完成后执行 optimizer step
        if micro_step % gradient_accumulation_steps == 0:
            # 梯度裁剪
            if config.optimizer.max_grad_norm != 0.0:
                if scaler is not None:
                    scaler.unscale_(optimizer)  # fp16 模式：先 unscale 才能正确裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)

            # 计算梯度范数
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item()

            # 当梯度大于skip_grad_norm并且训练步数超出skip_grad_iter时，不执行参数更新，保持训练稳定
            if grad_norm < config.optimizer.skip_grad_norm or train_steps < config.optimizer.skip_grad_iter:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
            else:
                print(f"[WARN] Skipping optimizer step at iter {train_steps}, grad_norm={grad_norm:.4f}")
                if scaler is not None:
                    scaler.update()  # 即使 skip，也需要更新 scaler

            optimizer.zero_grad()
            lr_scheduler.step()

            # 累积 loss 恢复原始值（loss 已经被 / gradient_accumulation_steps，所以 * 回来再 / log_every）
            running_loss += loss.item() * gradient_accumulation_steps
            running_grad_norm += grad_norm

            log_iters += 1
            train_steps += 1
        model.eval()  # 切到 eval 模式以做可视化/保存

        #################### Logging ####################
        if train_steps % args.log_every == 0:
            average_loss = running_loss / args.log_every
            average_grad_norm = running_grad_norm / args.log_every

            # speed
            end_time = time.time()
            average_time = (end_time - start_time) / args.log_every
            start_time = time.time()

            lr = lr_scheduler.get_last_lr()[0]

            print(
                f"[TRAIN] Step {train_steps:08d} | Loss {average_loss:.4f} | "
                f"Time {average_time:.4f}s | Grad Norm {average_grad_norm:.4f} | LR {lr:.5f}"
            )
            running_loss = 0.0
            running_grad_norm = 0.0

        #################### Visualization ####################
        if train_steps % args.visualize_every == 0:
            with torch.no_grad():
                visualize_logits = logits[:args.visualize_num]
                visualize_cond = cond[:args.visualize_num]
                visualize_token_order = token_order[:args.visualize_num]
                visualize_gt_indices = image_tokens[:args.visualize_num]
                orig_token_order = torch.argsort(visualize_token_order) # 如果按照从小到大排列下，返回token_order的下标调整顺序

                img_token_num = logits.shape[1] # seq_len

                # teacher forcing reconstruction
                pred_recon_indices = torch.zeros(args.visualize_num, img_token_num,
                                                 device=device).long()
                for i in range(img_token_num):
                    # 遍历每个token，获取argmax后的token id
                    pred_recon_indices[:, i: i + 1] = torch.argmax(visualize_logits[:, i: i + 1], dim=-1)
                pred_recon_indices = torch.gather(
                    pred_recon_indices.unsqueeze(-1),
                    dim=1,
                    index=orig_token_order.unsqueeze(-1)
                ).squeeze(-1)   # 调整为正确顺序 (bsz, seq_len)
                pred_recon_imgs = tokenizer.decode_codes_to_img(pred_recon_indices, args.image_size)

                # vq reconstruction
                gt_recon_indices = visualize_gt_indices
                gt_recon_imgs = tokenizer.decode_codes_to_img(gt_recon_indices, args.image_size)

                # | 对比维度 | **训练中的 Teacher-Forcing 可视化** | **`generate` 方法 (推理)** |
                # | :--- | :--- | :--- |
                # | **代码位置** | `train_c2i.py` 第 248-256 行 | `randar_gpt.py` 第 520-685 行 |
                # | **Token 来源** | **直接使用 Logits 的 Argmax** | **逐步采样 (Sample/Top-K/Top-P)** |
                # | **输入依赖** | 完全依赖训练时的 `forward` 返回的 `logits` | 使用 KV Cache 和循环，自己生产输入 |
                # | **并行性** | 无并行概念，只是把已经算好的一整串 logits 取最大值 | **核心特性**，通过 `num_inference_steps` 控制并行解码数量 |
                # | **CFG 支持** | 不支持 (仅展示模型拟合能力) | **支持** (Classifier-Free Guidance) |
                # | **用途** | 诊断模型是否 Overfit (Reconstruction) | 生成全新的、多样化的图像 (Generation) |

                # generation
                gen_indices = model.generate(
                    cond=visualize_cond,
                    token_order=None,
                    cfg_scales=(4.0, 4.0),  # CFG根本没用，class token永远为0
                    num_inference_steps=-1,
                    temperature=1.0,
                    top_k=0,
                    top_p=1.0,
                )
                model.remove_caches()
                gen_imgs = tokenizer.decode_codes_to_img(gen_indices, args.image_size)

                pred_recon_grid = make_grid(pred_recon_imgs)
                gt_recon_grid = make_grid(gt_recon_imgs)
                gen_grid = make_grid(gen_imgs)

                # 保存可视化图片到 experiment_dir
                from torchvision.utils import save_image as tv_save_image
                vis_dir = os.path.join(experiment_dir, "visualizations")
                os.makedirs(vis_dir, exist_ok=True)
                # make_grid 返回的是 numpy 数组 (H, W, C)，需要转为 CHW tensor
                tv_save_image(torch.from_numpy(pred_recon_grid).permute(2, 0, 1) / 255.0,
                              os.path.join(vis_dir, f"pred_recon_{train_steps:08d}.png"))
                tv_save_image(torch.from_numpy(gt_recon_grid).permute(2, 0, 1) / 255.0,
                              os.path.join(vis_dir, f"gt_recon_{train_steps:08d}.png"))
                tv_save_image(torch.from_numpy(gen_grid).permute(2, 0, 1) / 255.0,
                              os.path.join(vis_dir, f"gen_{train_steps:08d}.png"))
                print(f"[VIS] Saved visualization at step {train_steps}")

        #################### Checkpoint ####################
        if train_steps % args.ckpt_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"iters_{train_steps:08d}")
            os.makedirs(ckpt_path, exist_ok=True)

            # 使用 safetensors 保存模型权重
            save_model_safetensors(model, os.path.join(ckpt_path, "model.safetensors"))

            # 保存优化器和调度器状态（用 pytorch 格式）
            torch.save({
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'train_steps': train_steps,
            }, os.path.join(ckpt_path, "optimizer_scheduler.pt"))

            print(f"[CKPT] Saved Iter {train_steps} checkpoint to {ckpt_path}")

            # remove redundantly old checkpoints (keep last k + milestone checkpoints)
            milestones = [50000, 100000, 200000, 300000]
            for ckpt_dir in os.listdir(checkpoint_dir):
                if ckpt_dir.startswith("iters") and ckpt_dir != f"iters_{train_steps:08d}":
                    save_iter = int(ckpt_dir.split("_")[-1])
                    if save_iter < train_steps - args.keep_last_k * args.ckpt_every:
                        if save_iter not in milestones:
                            shutil.rmtree(os.path.join(checkpoint_dir, ckpt_dir))

        model.train()

    #################### Final Save ####################
    final_ckpt_dir = os.path.join(checkpoint_dir, f"iters_{train_steps:08d}_final")
    os.makedirs(final_ckpt_dir, exist_ok=True)
    save_model_safetensors(model, os.path.join(final_ckpt_dir, "model.safetensors"))
    torch.save({
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'train_steps': train_steps,
    }, os.path.join(final_ckpt_dir, "optimizer_scheduler.pt"))
    print(f"[CKPT] Saved Final Iter {train_steps} checkpoint to {final_ckpt_dir}")

    print("[INFO] Training Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="new_configs/randar_xl_0.7b.yaml")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--image-size", type=int, choices=[128, 256, 384, 448, 512], default=256)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--ckpt-every", type=int, default=5000)  # save every 5k iters
    # keep last k checkpoints; 1 means only keep the last checkpoint
    parser.add_argument("--keep-last-k", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    # vq checkpoint
    parser.add_argument("--vq-ckpt", type=str, default="./checkpoints/vq_ds16_c2i.pt")
    # data
    parser.add_argument("--dataset", type=str, default="latent")
    parser.add_argument("--data-path", type=str, required=True)  # /tmp/imagenet-llamagen-adm-256_codes
    parser.add_argument("--visualize-every", type=int, default=2000)
    parser.add_argument("--visualize-num", type=int, default=32)
    args = parser.parse_args()
    main(args)
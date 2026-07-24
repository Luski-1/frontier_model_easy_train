from utils import skewed_timestep_sample, save_samples
from torch.utils.data import DataLoader
from accelerate import Accelerator
from model import EnhancedUNet, FlowMatching
from torchvision import transforms
from dataset import CelebADataset
from config import get_args
from tqdm import tqdm

import torch.nn as nn
import torch
import os


def main():
    # 1. 初始化 Accelerator
    accelerator = Accelerator(gradient_accumulation_steps=1, log_with=None)

    # 2. 获取参数
    args = get_args()

    num_processes = accelerator.num_processes
    is_main = accelerator.is_main_process
    global_batch_size = args.batch_size * num_processes

    if is_main:
        print(f"[accelerate] num_processes={num_processes}, per_device_batch={args.batch_size}, global_batch={global_batch_size}")
        print(f"[accelerate] mixed_precision={accelerator.mixed_precision}")

    if is_main:
        os.makedirs(args.save_dir, exist_ok=True)

    # 3. 加载数据
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
    ])

    dataset = CelebADataset(root=args.data_root, image_size=args.image_size, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )

    # 4. 构建模型
    inner_model = EnhancedUNet(
        in_ch=3,
        base_ch=args.base_ch,
        time_emb_dim=args.time_ch,
        num_res_blocks=args.num_res_blocks
    )

    if args.ema:
        model = FlowMatching(inner_model, steps=args.flow_steps, decay=args.decay)
    else:
        model = FlowMatching(inner_model, steps=args.flow_steps)

    # 5. 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 6. 损失函数
    mse = nn.MSELoss()

    # 7. 让 accelerate 接管一切（自动 DDP / DeepSpeed / FSDP）
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    # ---------------- training loop ----------------
    if is_main:
        print(f"[starting training] device: {accelerator.device}")

    total_loss = 0.0
    step_count = 0
    global_step = 0

    epoch_pbar = tqdm(range(args.num_epochs), desc="Training Epoch", disable=not is_main)
    for epoch in epoch_pbar:
        batch_pbar = tqdm(loader, desc=f"Epoch {epoch:03d} Batch", leave=False, disable=not is_main)

        for z in batch_pbar:
            B = z.shape[0]

            # 抽样 x_0 ~ N(0,1)
            x_0 = torch.randn_like(z)

            # 抽样 t
            if not args.edm_train_time:
                t = torch.rand(B, device=accelerator.device, dtype=torch.float32)
            else:
                t = skewed_timestep_sample(B, accelerator.device)

            t_broadcast = t.view(B, 1, 1, 1)
            x_t = t_broadcast * z + (1.0 - t_broadcast) * x_0
            u_target = (z - x_0).detach()

            # accelerate 的上下文管理器（自动处理梯度缩放 / 混合精度）
            with accelerator.accumulate(model):
                pred = model(x_t, t)
                loss = mse(pred, u_target)
                accelerator.backward(loss)

                # 梯度裁剪（accelerate 会自动缩放 max_norm 以适配混合精度）
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            step_count += 1

            batch_pbar.set_postfix({"loss": f"{loss.item():.6f}"})

            if is_main and global_step % 500 == 0:
                avg_loss = total_loss / step_count
                print(f"\nEpoch {epoch:03d} Step {global_step:06d} Average Loss: {avg_loss:.6f}")
                total_loss = 0.0
                step_count = 0

            # 更新 EMA（仅主进程）
            if args.ema and is_main:
                first_phase = global_step < 2000
                model.module.update_ema(first_phase)  # 因为使用accelerator包装了model，因此需要通过model.module来拆包得到真正的FlowMatching对象

            global_step += 1


        # 采样（仅主进程）
        if is_main and epoch % args.sample_every == 0:
            samples = model.module.sample_flow(
                image_size=args.image_size,
                n_samples=args.num_sample_images,
                device=accelerator.device,
                edm_eval=args.edm_eval_time
            )
            save_samples(samples, epoch, args.save_dir)

        # 保存模型（仅主进程）
        if is_main:
            if args.ema:
                ckpt = {
                    "model_state_dict": model.module.model.state_dict(),
                    "ema_state_dict": model.module.ema_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch
                }
            else:
                ckpt = {
                    "model_state_dict": model.module.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch
                }

            ckpt_path = os.path.join(args.save_dir, "edm_flowmatch_ckpt_latest.pt")
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
            torch.save(ckpt, ckpt_path)
            print(f"\n[saved checkpoint] {ckpt_path}")

        # 同步所有进程
        accelerator.wait_for_everyone()

    if is_main:
        print("Training finished.")


if __name__ == "__main__":
    main()

from utils import skewed_timestep_sample, save_samples
from torch.utils.data import DataLoader
from model import EnhancedUNet, FlowMatching
from torchvision import transforms
from dataset import CelebADataset
from config import get_args
import torch.nn as nn
import torch
import os
from tqdm import tqdm  # 新增tqdm导入


def main():
    # 1. 获取参数
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 加载数据
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),  # [0,1]
    ])

    dataset = CelebADataset(root=args.data_root, image_size=args.image_size, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=args.pin_memory)

    # 3. 加载模型
    inner_model = EnhancedUNet(in_ch=3, base_ch=args.base_ch, time_emb_dim=args.time_ch, num_res_blocks=args.num_res_blocks).to(device)
    if args.ema:
        model = FlowMatching(inner_model, steps=args.flow_steps, decay=args.decay)
    else:
        model = FlowMatching(inner_model, steps=args.flow_steps)

    # 4. 加载优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 5. 设置损失函数
    mse = nn.MSELoss()

    # ---------------- training loop ----------------
    print("Starting training... device:", device)
    total_loss = 0.0
    step_count = 0
    global_step = 0

    # 外层epoch进度条
    epoch_pbar = tqdm(range(args.num_epochs), desc="Training Epoch")
    for epoch in epoch_pbar:
        # 内层dataloader进度条
        batch_pbar = tqdm(loader, desc=f"Epoch {epoch:03d} Batch", leave=False)
        for z in batch_pbar:
            z = z.to(device)  # (B,3,H,W)
            B = z.shape[0]

            # 抽样 x_0 ~ N(0,1)
            x_0 = torch.randn_like(z)

            # 抽样 t ~ Uniform(0,1)
            if not args.edm_train_time:
                t = torch.rand(B, device=device, dtype=torch.float32)
            else:
                # 如果采纳EDM，那么时间t会偏向于1附近
                t = skewed_timestep_sample(B, device)
            # 扩增维度
            t_broadcast = t.view(B, 1, 1, 1)
            # 构建 x_t = t * z + (1 - t) * x_0
            x_t = t_broadcast * z + (1.0 - t_broadcast) * x_0

            # 构建最优传输的概率密度路径的label：z - x_0 即(dx_t/dt)
            u_target = (z - x_0).detach()

            # 优化器清空
            optimizer.zero_grad()
            pred = model(x_t, t)
            # 求损失
            loss = mse(pred, u_target)
            # 反向传播
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # 参数更新
            optimizer.step()

            total_loss += loss.item()
            step_count += 1

            # 更新进度条显示loss
            batch_pbar.set_postfix({"loss": f"{loss.item():.6f}"})

            if global_step % 500 == 0:
                avg_loss = total_loss / step_count
                print(f"\nEpoch {epoch:03d} Step {global_step:06d} Average Loss: {avg_loss:.6f}")
                total_loss = 0.0
                step_count = 0

            # 更新EMA
            if args.ema:
                first_phase = global_step < 2000
                model.update_ema(first_phase)

            global_step += 1

        # epoch结束逻辑
        if epoch % args.sample_every == 0:
            # 进行ODE求解
            samples = model.sample_flow(image_size=args.image_size, n_samples=args.num_sample_images, device=device, edm_eval=args.edm_eval_time)
            save_samples(samples, epoch, args.save_dir)

        # 保存模型
        if args.ema:
            ckpt = {
                "model_state_dict": model.model.state_dict(),
                "ema_state_dict": model.ema_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "epoch": epoch
            }
        else:
            ckpt = {
                "model_state_dict": model.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "epoch": epoch
            }

        ckpt_path = os.path.join(args.save_dir, "flowmatch_ckpt_latest.pt")

        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

        torch.save(ckpt, ckpt_path)
        print(f"\n[saved checkpoint] {ckpt_path}")

    print("Training finished.")


if __name__ == "__main__":
    main()
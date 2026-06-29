import argparse
import datetime
import torch
import wandb

from torch.utils.data import DataLoader
from torchvision import datasets
from ddpm import script_utils


def main():
    args = create_argparser().parse_args()      # 获取参数
    device = args.device

    try:
        # 1. 获取模型
        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        # 2. 获取优化器
        optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.learning_rate)
        # 3. 加载参数
        if args.model_checkpoint is not None:
            diffusion.load_state_dict(torch.load(args.model_checkpoint))
        if args.optim_checkpoint is not None:
            optimizer.load_state_dict(torch.load(args.optim_checkpoint))

        if args.log_to_wandb:
            if args.project_name is None:
                raise ValueError("args.log_to_wandb set to True but args.project_name is None")

            run = wandb.init(
                project=args.project_name,
                entity='treaptofun',
                config=vars(args),
                name=args.run_name,
            )
            wandb.watch(diffusion)

        batch_size = args.batch_size
        # 4. 获取数据集
        train_dataset = datasets.CIFAR10(
            root='./cifar_train',
            train=True,
            download=True,
            transform=script_utils.get_transform(),
        )

        test_dataset = datasets.CIFAR10(
            root='./cifar_test',
            train=False,
            download=True,
            transform=script_utils.get_transform(),
        )
        # 永远循环
        train_loader = script_utils.cycle(DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=2,
        ))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, num_workers=2)
        
        acc_train_loss = 0

        for iteration in range(1, args.iterations + 1):
            diffusion.train()

            x, y = next(train_loader)
            x = x.to(device)
            y = y.to(device)

            if args.use_labels:         # 默认不用
                loss = diffusion(x, y)
            else:
                loss = diffusion(x)

            acc_train_loss += loss.item()

            optimizer.zero_grad()   # 清空优化器
            loss.backward()         # 梯度传播
            optimizer.step()        # 参数更新

            diffusion.update_ema()  # 更新EMA模型
            
            if iteration % args.log_rate == 0:
                test_loss = 0
                with torch.no_grad():
                    diffusion.eval()
                    for x, y in test_loader:
                        x = x.to(device)
                        y = y.to(device)

                        if args.use_labels:
                            loss = diffusion(x, y)
                        else:
                            loss = diffusion(x)

                        test_loss += loss.item()
                
                if args.use_labels:
                    samples = diffusion.sample(10, device, y=torch.arange(10, device=device))
                else:
                    samples = diffusion.sample(10, device)      # 先验抽样生成
                
                samples = ((samples + 1) / 2).clip(0, 1).permute(0, 2, 3, 1).numpy()    # 反归一化 [≈-1, ≈1] 裁剪 [-1, 1] | [B, C, H, W] 转换 [B, H, W, C]

                test_loss /= len(test_loader)
                acc_train_loss /= args.log_rate

                wandb.log({
                    "test_loss": test_loss,
                    "train_loss": acc_train_loss,
                    "samples": [wandb.Image(sample) for sample in samples],
                })

                acc_train_loss = 0
            
            if iteration % args.checkpoint_rate == 0:
                model_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-model.pth"
                optim_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-optim.pth"

                torch.save(diffusion.state_dict(), model_filename)      # 保存模型参数
                torch.save(optimizer.state_dict(), optim_filename)      # 保存优化器参数
        
        if args.log_to_wandb:
            run.finish()
    except KeyboardInterrupt:
        if args.log_to_wandb:
            run.finish()
        print("Keyboard interrupt, run finished early")


def create_argparser():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    run_name = datetime.datetime.now().strftime("ddpm-%Y-%m-%d-%H-%M")
    defaults = dict(
        learning_rate=2e-4,
        batch_size=128,
        iterations=800000,

        log_to_wandb=True,
        log_rate=1000,
        checkpoint_rate=1000,
        log_dir="~/ddpm_logs",
        project_name=None,
        run_name=run_name,

        model_checkpoint=None,
        optim_checkpoint=None,

        schedule_low=1e-4,
        schedule_high=0.02,

        device=device,
    )
    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
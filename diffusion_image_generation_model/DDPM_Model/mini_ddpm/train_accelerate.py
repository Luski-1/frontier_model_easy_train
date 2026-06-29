from transformers import Trainer, set_seed
from torch.utils.data import random_split
from config import DDPMTrainingArguments
from model import EnhancedUNet, Diffusion
from torchvision import transforms
from dataset import CelebADataset
from utils import save_image_grid
from ema import EMACallback
import torch
import os


class DDPMTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        inputs: 字典，包含 "pixel_values" -> (B, C, H, W) 在 [-1, 1] 范围
        """
        x_start = inputs["pixel_values"]
        batch_size = x_start.shape[0]
        t = torch.randint(0, self.args.T, (batch_size,), device=x_start.device)
        loss = model(x_start, t)
        return (loss, None) if return_outputs else loss

    # ========== evaluate 入口：重置标记 ==========
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Eval评估链路：
        trainer.evaluate()
            → evaluation_loop()
                → 对 eval_dataloader 的每个 batch 调用:
                prediction_step(model, inputs)
        每次开启evaluate时，重置eval_image_saved标记，确保每轮 evaluate 中 仅有第一次 prediction_step 保存图像

        可通过指定 Trainer 的 eval_strategy="steps" + eval_steps=500 或者 eval_strategy="epoch" 自动触发
        """
        self._eval_image_saved = False
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None, **kwargs):
        """
        评估步骤：
        1. 正常计算 loss（每个 batch 都做）
        2. 仅主进程的第一次 prediction_step 执行先验采样并保存图像

        多卡适配：
        - model 可能是 DDP/FSDP 封装的，需通过 accelerator.unwrap_model 解包后访问内部方法
        - 采样和文件保存仅在主进程执行，避免多进程重复写入
        """
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            x_start = inputs["pixel_values"]
            batch_size = x_start.shape[0]
            t = torch.randint(0, self.args.T, (batch_size,), device=x_start.device)
            loss = model(x_start, t)

        # 仅主进程 + 本次评估的第一个 batch 执行采样
        if not self._eval_image_saved and self.accelerator.is_main_process:
            self._eval_image_saved = True

            # 解包获取原始 Diffusion 模型，以调用 p_sample 方法
            diffusion = self.accelerator.unwrap_model(model)
            with torch.no_grad():
                x = torch.randn(self.args.num_sample_images, 3, self.args.image_size, self.args.image_size,
                                device=x_start.device)
                # 反向去噪
                for t_step in reversed(range(self.args.T)):
                    x = diffusion.p_sample(x, t_step)
                # 反归一化
                x = (x + 1.0) / 2.0
                x = torch.clamp(x, 0.0, 1.0)

            output_path = os.path.join(self.args.sample_dir, f"epoch{self.state.epoch}_ema.png")
            save_image_grid(x, output_path, nrow=8)

        if prediction_loss_only:
            return (loss, None, None)

        return loss, None, None


def train_hf():
    set_seed(42)

    # 1. 获取参数
    args = DDPMTrainingArguments(
        data_root="/workspace/data/img_align_celeba/img_align_celeba",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        num_train_epochs=300,
        bf16=True,
        report_to="tensorboard",
        resume_from_checkpoint=None,
    )

    # 仅全局主进程创建目录和打印
    if args.process_index == 0:
        os.makedirs(args.sample_dir, exist_ok=True)
        os.makedirs(args.output_dir, exist_ok=True)

    # 2. 构建模型
    inner_model = EnhancedUNet(in_ch=3,
                               base_ch=args.base_ch,
                               time_emb_dim=args.time_emb_dim,
                               num_res_blocks=args.num_res_blocks,
                               group_num=args.group_num)
    diffusion_model = Diffusion(model=inner_model, T=args.T)

    # 3. 加载数据
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),  # 转为RGB3通道
        transforms.Resize(args.image_size),                 # 调整大小
        transforms.CenterCrop(args.image_size),             # 裁剪
        transforms.ToTensor(),                              # 将离散[0,255]转换为连续[0,1]
    ])
    dataset = CelebADataset(root=args.data_root, transform=transform, image_size=args.image_size)
    train_dataset, eval_dataset = random_split(dataset, [0.99, 0.01], generator=torch.Generator().manual_seed(42))

    # 仅全局主进程打印
    if args.process_index == 0:
        print(f"train dataset loaded: {len(train_dataset)} images")
        print(f"eval dataset loaded: {len(eval_dataset)} images")

    # 4. 创建trainer
    trainer = DDPMTrainer(
        model=diffusion_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[]
    )

    ema_callback = EMACallback(
        decay=args.ema_decay,
        copy_step=args.copy_step,
        trainer=trainer
    )

    trainer.add_callback(ema_callback)

    # 5. 启动训练
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 保存最终模型
    final_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_path)

    if args.process_index == 0:
        print(f"\nTraining complete! Model saved to {final_path}")


if __name__ == "__main__":
    train_hf()

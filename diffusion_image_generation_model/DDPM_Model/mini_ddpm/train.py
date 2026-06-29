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
        # 从字典中提取图像（Trainer 会自动将 batch 数据堆叠）
        x_start = inputs["pixel_values"]

        # 生成随机时间步 t
        batch_size = x_start.shape[0]
        t = torch.randint(0, self.args.T, (batch_size,), device=x_start.device)

        # 使用全局的 p_losses 计算 MSE（预测噪声 vs 真实噪声）
        loss = model(x_start, t)

        # 返回格式需符合 Trainer 要求: (loss, outputs) 或 loss
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
        1. 正常计算 loss + reconstruction + labels（每个 batch 都做）
        2. 如果 save_eval_images=True 且是本次评估的第一个 batch：
           - 执行先验采样
        
        备注：此时model为单卡版本，无需拆包
        """
        # 正常执行eval步骤
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            # 从字典中提取图像（Trainer 会自动将 batch 数据堆叠）
            x_start = inputs["pixel_values"]

            # 生成随机时间步 t
            batch_size = x_start.shape[0]
            t = torch.randint(0, self.args.T, (batch_size,), device=x_start.device)

            # 使用全局的 p_losses 计算 MSE（预测噪声 vs 真实噪声）
            loss = model(x_start, t)

        # 可视化图像生成能力
        if not self._eval_image_saved:
            self._eval_image_saved = True

            with torch.no_grad():
                x = torch.randn(self.args.num_sample_images, 3, self.args.image_size, self.args.image_size, device=model.device)
                # 反向去噪
                for t in reversed(range(self.args.T)):
                    x = model.p_sample(x, t)
                # 反归一化
                x = (x + 1.0) / 2.0
                x = torch.clamp(x, 0.0, 1.0)

            output_path = os.path.join(self.args.sample_dir, f"epoch{self.state.epoch}_ema.png") # self.state由trainer自行维护
            save_image_grid(x, output_path, nrow=8)

        if prediction_loss_only:
            return (loss, None, None)

        # 返回loss, output, label
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
        bf16=torch.cuda.is_available(),
        report_to="tensorboard",
        resume_from_checkpoint=None,    # 同时也是EMA模型的参数加载的路径
    )

    os.makedirs(args.sample_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. 获取模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inner_model = EnhancedUNet(in_ch=3, 
                         base_ch=args.base_ch, 
                         time_emb_dim=args.time_emb_dim, 
                         num_res_blocks=args.num_res_blocks, 
                         group_num=args.group_num).to(device)
    diffusion_model = Diffusion(model=inner_model, T=args.T).to(device)

    # 3. 加载数据
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),  # 转为RGB3通道
        transforms.Resize(args.image_size),                 # 调整大小
        transforms.CenterCrop(args.image_size),             # 裁剪
        transforms.ToTensor(),                              # 将离散[0,255]转换为连续[0,1]
    ])
    dataset = CelebADataset(root=args.data_root, transform=transform, image_size=args.image_size)
    train_dataset, eval_dataset = random_split(dataset, [0.99, 0.01], generator=torch.Generator().manual_seed(42))
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
        trainer=trainer             # 这为了能够使用trainer.accelerator，可以在钩子方法中用accelerator解包model（事实上单卡训练传递给钩子方法的model都是没有任何封装的model）
    )

    trainer.add_callback(ema_callback)

    # 5. 启动训练
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 保存最终模型（包含EMA）
    final_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_path)
    print(f"\nTraining complete! Original model saved to {final_path}")


if __name__ == "__main__":
    train_hf()

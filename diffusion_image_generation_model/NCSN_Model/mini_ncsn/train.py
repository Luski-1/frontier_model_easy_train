from model import EnhancedUNet, anneal_dsm_score_estimation
from transformers import Trainer, set_seed
from dataset import CelebADataset, MNISTDataset
from config import NCSNTrainingArguments
from torchvision import transforms
from utils import SampleCallback
import numpy as np
import torch
import os


class NCSNTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        self.sigmas_list = kwargs.pop("sigmas_list", None)  # 弹出额外参数

        super().__init__(*args, **kwargs)

        if self.sigmas_list is None:
            raise ValueError("sigmas list cannot be None")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        inputs: 字典格式，包含 "pixel_values" -> (B, C, H, W) 在 [-1, 1] 范围
        """
        # 从字典中提取图像（Trainer 会自动将 batch 数据堆叠）
        x = inputs["pixel_values"]

        # 生成随机sigma的下标
        sigma_index = torch.randint(0, len(self.sigmas_list), (x.shape[0],), device=x.device)

        # 分数预测
        loss = anneal_dsm_score_estimation(model, x, sigma_index, self.sigmas_list)

        # 返回格式需符合 Trainer 要求: (loss, outputs) 或 loss
        return (loss, None) if return_outputs else loss


def main():

    # 1. 配置参数
    args = NCSNTrainingArguments(
        output_dir="./ncsn_mnist_checkpoints",
        save_dir="./ncsn_mnist_samples",
        data_root="./data",

        per_device_train_batch_size=256,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        num_train_epochs=50,

        image_size=32,
        channels=1,
        base_ch=128,

        bf16=torch.cuda.is_available(),
        max_grad_norm=1.0,

        report_to="tensorboard",
        resume_from_checkpoint=None
    )
    # 2. 创建目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(42)

    # 3. 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedUNet(in_ch=args.channels, base_ch=args.base_ch, time_emb_dim=512, num_res_blocks=2).to(device)

    # 4. 创建数据集
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),  # [0,1]
    ])
    # dataset = CelebADataset(root=args.data_root, transform=transform, image_size=args.image_size)
    dataset = MNISTDataset(root=args.data_root, transform=transform, image_size=args.image_size)

    print(f"Dataset loaded: {len(dataset)} images")

    # 5. 创建sigma等比序列
    sigmas_list = torch.tensor(
        np.exp(np.linspace(np.log(args.sigma_start), np.log(args.sigma_end), args.sigma_nums))).float().to(device)

    # 6. 创建Trainer
    trainer = NCSNTrainer(
        model=model,
        sigmas_list=sigmas_list,
        args=args,
        train_dataset=dataset,
    )

    sample_callback = SampleCallback(
        sample_every=args.sample_every,
        num_sample_images=args.num_sample_images,
        image_size=args.image_size,
        channels=args.channels,
        sigmas_list=sigmas_list,
        sigma_steps=args.sigma_steps,
        step_lr=args.step_lr,
        save_dir=args.save_dir
    )
    trainer.add_callback(sample_callback)

    # 7. 启动训练
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 8. 保存最终模型
    final_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_path)
    print(f"\nTraining complete! Original model saved to {final_path}")


if __name__ == "__main__":
    main()
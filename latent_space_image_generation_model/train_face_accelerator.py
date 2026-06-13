from torch.utils.data import random_split
from torchvision.utils import save_image
from dataset_face import CelebADataset
from config_face import VAEConfig
from model_face import ConvVAE
import torch
import os

from transformers import (
    TrainingArguments,
    Trainer,
    set_seed
)


# ========== 自定义 Trainer ==========
class VAETrainer(Trainer):
    """
    继承 HF Trainer 以覆盖特定行为：

    需要覆盖的方法：
    1. compute_loss: 执行计算损失+拆分损失用于后续打印
    2. prediction_step: 评估 + 可选的图像生成（重建对比 + 先验采样）
    3. log: 注入拆分 loss 到日志管线
    """

    def __init__(
        self,
        *args,
        save_eval_images: bool = False,                 # 是否在评估时保存生成图像
        sample_dir: str = "./vae_face_samples",         # 保存目录
        num_prior_samples: int = 8,                     # 先验采样数量
        num_recon_samples: int = 4,                     # 重建对比数量
        kl_anneal_steps: int = 0,                       # 退火步数，0=不退火
        kl_target_weight: float = 1.0,                  # 目标权重
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.save_eval_images = save_eval_images
        self.sample_dir = sample_dir
        self.num_prior_samples = num_prior_samples
        self.num_recon_samples = num_recon_samples
        self._eval_image_saved = False                   # 标记本次评估是否已保存图像
        self.kl_anneal_steps = kl_anneal_steps
        self.kl_target_weight = kl_target_weight
        self._loss_history = {}                            # 存储最近一次 compute_loss 的拆分 loss（Trainer 实例属性，各进程独立）
        os.makedirs(sample_dir, exist_ok=True)

    def _get_kld_weight(self, model, step: int) -> float:
        """
        根据 step 计算 KL 权重
        - kl_anneal_steps=0: 固定权重（用 config.kld_weight）
        - kl_anneal_steps>0: 从 0 线性增长到 kl_target_weight
        """
        if self.kl_anneal_steps == 0:
            return model.config.kld_weight  # 用模型配置中的固定值

        if step < self.kl_anneal_steps:
            return (step / self.kl_anneal_steps) * self.kl_target_weight
        else:
            return self.kl_target_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        执行模型的损失计算，并且拆分损失用于后续日志打印

        讲解 mini_step(batch)与step(batch)的关系：
            如果开启梯度积累，假设=2：
                会遍历两次dataloader > 每次是mini_batch > 2次是batch
                会执行两次compute_loss > 每次是mini_step > 2次是step
                global_step ≈ mini_step / gradient_accumulation_steps

        【多卡适配】model 在多卡时被 DDP/Accelerate 包装，需要解包后才能访问原始 ConvVAE 的属性
        """
        # 0. 解包分布式包装器（统一使用 accelerator.unwrap_model，兼容 DDP/FSDP/DeepSpeed）
        unwrapped_model = self.accelerator.unwrap_model(model)

        # 1. 计算KL散度的权重（操作原始模型实例的 config）
        kld_weight = self._get_kld_weight(unwrapped_model, self.state.global_step)
        unwrapped_model.config.kld_weight = kld_weight

        # 2. 前向+反向（model 仍用 DDP 包装器，DDP 会自动转发到 .module.forward）
        outputs = model(
            pixel_values=inputs.get("pixel_values"),
            labels=inputs.get("pixel_values"),
            return_loss=True
        )

        loss = outputs.loss

        # 3. 存储详细 loss 供 log 使用（存在 Trainer 实例上，各进程独立）
        if hasattr(outputs, 'loss_dict') and outputs.loss_dict is not None:
            self._loss_history = {
                'step': self.state.global_step if hasattr(self, 'state') else 0,
                'recon': outputs.loss_dict['recon_loss'].item(),
                'kld': outputs.loss_dict['kld_loss'].item(),
                'kld_weight': outputs.loss_dict['kld_weight'].item(),
            }

        return (loss, outputs) if return_outputs else loss
    
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
           - 保存重建对比图（前 N 张）
           - 执行先验采样
           - 插值生成

        【多卡适配】eval 时 model 也是分布式包装器，需要解包后访问 config 和 decode 方法
        """
        # 1. 解包分布式包装器
        unwrapped_model = self.accelerator.unwrap_model(model)

        # 2. 正常执行eval步骤（前向仍用封装后的 model）
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss

        # 3. 可视化图像生成能力（只在 global rank 0 执行，避免多卡重复保存，完成evaluate后会自动同步所有进程才开启train，只有显卡0生成图像不会导致进度不一致）
        if self.save_eval_images and not self._eval_image_saved and self.is_world_process_zero():
            self._eval_image_saved = True
            self._save_eval_images(unwrapped_model, inputs, outputs)

        if prediction_loss_only:
            return (loss, None, None)

        # 返回loss, output, label
        return loss, outputs.reconstruction, inputs.get("pixel_values")
    
    def _save_eval_images(self, unwrapped_model, inputs, outputs):
        """
        保存评估图像（只在每个 evaluate 的第一个 batch 调用一次）

        包含：
        1. 重建对比：取前 N 张 [真实 | 重建] 拼接
        2. 先验采样：从 N(0,I) 采样 z → decode
        3. 插值生成：空间进行线性插值，生成平滑过渡的图像
        """
        step = self.state.global_step
        device = inputs["pixel_values"].device

        # ===== 重建对比 =====
        n = min(self.num_recon_samples, inputs["pixel_values"].size(0))
        real = inputs["pixel_values"][:n]
        recon = outputs.reconstruction[:n]

        comparison = torch.cat([real, recon], dim=0)
        comparison = (comparison + 1) / 2           # 反归一化 [-1, 1] > [0, 2] > [0, 1]
        comparison = torch.clamp(comparison, 0, 1)  # 截断 

        recon_path = os.path.join(self.sample_dir, f"recon_compare_step{step:06d}.png")
        save_image(comparison, recon_path, nrow=n)
        print(f"  ✓ 重建对比已保存: {recon_path} (前{n}张)")

        # ===== 先验采样 =====
        config = unwrapped_model.config
        # 抽样
        z = torch.randn(self.num_prior_samples, config.latent_dim, device=device)
        # 重建
        samples = unwrapped_model.decode(z)
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        # 拼接对比 [真实 | 重建]
        prior_path = os.path.join(self.sample_dir, f"prior_sample_step{step:06d}.png")
        save_image(samples, prior_path, nrow=4)
        print(f"  ✓ 先验样本已保存: {prior_path}")

        # ==== 插值生成 ====
        # 随机选择两个 latent 向量
        z1 = torch.randn(1, config.latent_dim, device=device)
        z2 = torch.randn(1, config.latent_dim, device=device)

        # 插值系数 [0, 1]
        alphas = torch.linspace(0, 1, steps=8, device=device)

        # 线性插值: z = α*z1 + (1-α)*z2
        z_interp = torch.zeros(8, config.latent_dim, device=device)
        for i, alpha in enumerate(alphas):
            z_interp[i] = alpha * z1 + (1 - alpha) * z2

        images = unwrapped_model.decode(z_interp)

        images = (images + 1) / 2               # 反归一化 
        images = torch.clamp(images, 0, 1)      # 截断

        # 保存插值结果
        save_path = os.path.join(self.sample_dir, f"interp_step{step:06d}.png")
        save_image(images, save_path, nrow=8, padding=2)
        print(f"  ✓ 插值样本已保存: {save_path}")
    
    def log(self, logs: dict, start_time=None):
        """
        compute_loss → 返回 loss scalar → Trainer 内部累积 → 到 logging_steps 时 → 构建 logs dict（loss字段需要求均值） → 调用 self.log(logs)
        """
        if self._loss_history:
            # 1. 将本卡的 recon/kld 转为 tensor，跨卡 gather 后求平均
            device = self.accelerator.device
            local_recon = torch.tensor(self._loss_history['recon'], device=device)
            local_kld = torch.tensor(self._loss_history['kld'], device=device)

            # all_gather：收集所有卡的值，再求均值（weight 各卡一致，无需 gather）
            all_recon = self.accelerator.gather(local_recon).mean().item()
            all_kld = self.accelerator.gather(local_kld).mean().item()
            weight = self._loss_history['kld_weight']

            # 2. 注入 logs，后续自动写入 tensorboard
            logs['recon_loss'] = all_recon
            logs['kld_loss'] = all_kld
            logs['kld_weight'] = weight

            # 3. 只在 global rank 0 打印（is_world_process_zero 由 HF Trainer 设置）
            if self.is_world_process_zero():
                print(f"[Step {self.state.global_step}] "
                        f"recon={all_recon:.2f} "
                        f"kld={all_kld:.4f} "
                        f"weight={weight:.4f}")

        super().log(logs, start_time)


def main():
    set_seed(42)

    # 1. 获取配置对象
    config = VAEConfig(
        latent_dim=256,
        image_size=128,
        target_final_res=4,
        attention_resolutions=(16, 8, 4),
        channel_mult=(1, 2, 4, 8, 16),
        kld_weight=1,  # 标准VAE是1，但可能只适合简单且风格单一的数据集，主流的图像生成模型中的VAE插件训练过程中，kl散度损失的权重范围0.00001-0.1
    )

    training_args = TrainingArguments(
        output_dir="./vae_face_checkpoints",

        num_train_epochs=300,               # 训练epochs
        per_device_train_batch_size=256,    # 每张卡Batch size
        gradient_accumulation_steps=1,      # 梯度累积

        learning_rate=2e-4,                 # 学习率
        adam_beta1=0.9,                     # AdamW优化器参数，一阶矩（梯度均值）权重
        adam_beta2=0.999,                   # AdamW优化器参数，二阶矩（梯度方差）权重
        weight_decay=1e-5,                  # 权重衰减，起正则化
        max_grad_norm=1.0,                  # 梯度裁剪

        lr_scheduler_type="constant_with_warmup",       # 学习率调度方式："linear", "polynomial", "cosine", "constant_with_warmup", "constant"
        # warmup_steps=2000,                # warm up步数
        warmup_ratio=0.05,                  # warm up比例
        bf16=torch.cuda.is_available(),     # 混合精度训练

        logging_steps=50,                   # 打印步数
        save_strategy="epoch",
        save_total_limit=3,                 # 只保存3份checkpoint

        # 评估策略
        eval_strategy="epoch",
        # eval_strategy="steps",
        # eval_steps=500,
        # load_best_model_at_end=True,  # 训练结束加载最佳模型
        # metric_for_best_model="eval_loss",  # 根据eval_loss选择最佳
        # greater_is_better=False,  # eval_loss越小越好

        # 日志与监控
        report_to=["tensorboard"],          # 或 "wandb", "mlflow"
        logging_dir="./vae_face_logs",      # 日志输出位置

        # dataloader参数
        dataloader_num_workers=2,           # worker数量必须要大于tar文件数量
        dataloader_pin_memory=True,
        remove_unused_columns=False,        # 重要，必须设置False，否则model的forward方法的参数名称必须与dataset输出字典的key一致

        # 回复训练
        resume_from_checkpoint="/root/autodl-fs/vae_face_checkpoints/checkpoint-108504", # None 或 "..."
        seed = 42
    )

    # 2. 获取模型
    model = ConvVAE(config)

    # 3. 获取数据集
    data_root = "/root/autodl-tmp/img_align_celeba/img_align_celeba"
    dataset = CelebADataset(root=data_root, config=config)

    train_dataset, eval_dataset = random_split(dataset, [0.99, 0.01], generator=torch.Generator().manual_seed(42))


    # 4. 获取Trainer
    trainer = VAETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,            # 多卡训练中，会自动封装
        eval_dataset=eval_dataset,
        save_eval_images=True,                  # 开启评估图像生成
        sample_dir="./vae_face_samples",        # 图像生成保存位置
        num_prior_samples=8,                    # 先验抽样的图像生成数量
        num_recon_samples=4,                    # 重建的图像生成数量  
        # kl_anneal_steps=1000,                   # KL散度权重的warm up步数；仅当训练达到几十epoch时，图像重建仍然没能看到轮廓时尝试开启
        # kl_target_weight=1.0,                   # KL散度最终权重，默认=1.0
        # data_collator=...         # 如有需要自定义 batch 组装
    )

    # 5. 开启训练
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # 6. 保存最终模型
    trainer.save_model("./vae_final")

    if trainer.is_world_process_zero():
        print(f"Training finished!")
        print(f"Final loss: {train_result.training_loss:.4f}")
        print(f"Checkpoints saved to: {training_args.output_dir}")


if __name__ == "__main__":
    main()
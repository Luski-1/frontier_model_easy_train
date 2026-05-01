from transformers import AutoTokenizer, AutoProcessor, TrainingArguments, Trainer
from dataset import RealSynDataset, MyDataCollator
from model import SiglipModel, SiglipConfig


def train():
    # 1. 加载模型相关内容
    config = SiglipConfig(vision_model_name_or_path="../autodl-tmp/dinov3-vitl16-pretrain-lvd1689m",
                          text_model_name_or_path="../autodl-tmp/roberta-large")

    model = SiglipModel(config)
    tokenizer = AutoTokenizer.from_pretrained(config.text_model_name_or_path)
    processor = AutoProcessor.from_pretrained(config.vision_model_name_or_path)

    # 2. 设置训练参数
    args = TrainingArguments(
        output_dir='../autodl-fs/outputs',
        do_train=True,
        num_train_epochs=40,
        save_steps=2000,
        save_total_limit=3,
        logging_steps=100,
        report_to='none',
        dataloader_pin_memory=True,
        dataloader_num_workers=1,
        dataloader_drop_last=True,  # 抛弃不足够的一个批次
        # ========== 学习率调度 ==========
        optim="adamw_torch",  # 或 "adamw_hf"，推荐 torch 版本更快
        learning_rate=1e-4,  # 对比学习合适
        weight_decay=0.05,  # 🔥 重要：防止投影层过拟合，建议 0.01-0.1
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        # ========== 学习率调度 ==========
        lr_scheduler_type="cosine_with_restarts",  # 🔥 对比学习推荐：余弦退火
        warmup_ratio=0.05,  # 🔥 前5% steps 线性 warmup，防早期震荡
        # ========== 批次与梯度 ==========
        per_device_train_batch_size=32,
        gradient_accumulation_steps=16,  # 有效 batch=32*16=512，对比学习越大越好
        max_grad_norm=5.0,  # 梯度裁剪，防 fp16 爆炸
        # ========== 内存优化 ==========
        fp16=True,
        # gradient_checkpointing=True,    # 🔥 如果显存紧张，开启以时间换空间
        # ========== 其他 ==========
        seed=42,
    )

    # 3. 加载数据集
    dataset = RealSynDataset(data_dir="../autodl-tmp/realsyn15m_success_all",
                             tokenizer=tokenizer,
                             processor=processor,
                             max_seq_length=64)

    # 4. 组装Huggingface的Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=MyDataCollator()
    )

    # 5. 开启训练
    # trainer.train(resume_from_checkpoint=True)
    # trainer.train(resume_from_checkpoint='../autodl-fs/outputs/checkpoint-10000')
    trainer.train(resume_from_checkpoint=False)

    # 6. 保存结果
    trainer.save_model()
    trainer.save_state()


if __name__ == '__main__':
    train()

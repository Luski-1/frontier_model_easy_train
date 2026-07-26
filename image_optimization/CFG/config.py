from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class DDPMTrainingArguments(TrainingArguments):
    """
    DDPM训练参数配置类，继承自HuggingFace TrainingArguments

    在原有TrainingArguments基础上新增DDPM特定参数(扩散步数T、采样频率等)
    """

    # ---------- DDPM模型新增参数 ----------
    T: int = field(
        default=1000,
        metadata={"help": "Diffusion总时间步数，控制前向加噪和反向去噪的步数"}
    )
    base_ch: int = field(
        default=128,
        metadata={"help": "UNet基础通道数，决定模型容量"}
    )
    time_emb_dim: int = field(
        default=512,
        metadata={"help": "时间步向量的维度"}
    )
    num_res_blocks: int = field(
        default=2,
        metadata={"help": "encoder或decoder的所有层各自有多少个残差块module"}
    )
    group_num: int = field(
        default=16,
        metadata={"help": "GroupNorm中多少channel作为1组"}
    )
    # ⭐⭐⭐ 增加CFG参数
    # ---------- CFG新增参数 ----------
    w: float = field(
        default=1.8,
        metadata={"help": "CFG的类别信息强度"}
    )
    label_emb_dim: int = field(
        default=512,
        metadata={"help": "类别向量的维度"}
    )
    cfg_threshold: float = field(
        default=0.1,
        metadata={"help": "训练期间无类别信息的概率"}
    )
    class_num: int = field(
        default=1000,
        metadata={"help": "训练数据的类别数量"}
    )

    # ---------- EMA模型新增参数 ----------
    ema_decay: float = field(default=0.9999, metadata={"help": "EMA衰减率，通常0.999-0.9999"})
    copy_step: int = field(default=2000, metadata={"help": "在指定step之前，EMA模型直接复制于训练模型，避免初始更新太慢"})

    # ---------- 数据新增参数 ----------
    image_size: int = field(
        default=128,
        metadata={"help": "输入图像的宽高尺寸"}
    )
    train_root: str = field(
        default="/mnt/d/data/face/img/img_align_celeba",
        metadata={"help": "CelebA数据集根目录路径"}
    )
    eval_root: str = field(
        default="/mnt/d/data/face/img/img_align_celeba",
        metadata={"help": "CelebA数据集根目录路径"}
    )

    # ---------- 训练超参数 ----------
    per_device_train_batch_size: int = field(default=8, metadata={"help": "每个设备的训练批次大小"})
    learning_rate: float = field(default=1e-5, metadata={"help": "AdamW优化器的初始学习率"})
    num_train_epochs: int = field(default=100, metadata={"help": "总训练轮数"})
    max_grad_norm: float = field(default=1.0, metadata={"help": "梯度裁剪的最大范数"})
    lr_scheduler_type: str = field(default="constant_with_warmup", metadata={"help": "学习率调度策略"})
    # warmup_steps: int = field(default=0, metadata={"help": "学习率warmup步数"})
    warmup_ratio: float = field(default=0.1, metadata={"help": "学习率warmup比例"})

    # ---------- 保存与日志策略 ----------
    save_strategy: str = field(default="epoch", metadata={"help": "保存策略：epoch或steps"})
    save_total_limit: int = field(default=3, metadata={"help": "最多保留的检查点数量"})
    logging_strategy: str = field(default="steps", metadata={"help": "日志记录策略"})
    logging_steps: int = field(default=50, metadata={"help": "每N步记录一次训练日志"})
    output_dir: str = field(default="./checkpoints", metadata={"help": "模型保存路径"})
    logging_dir: str = field(default="./logs",  metadata={"help": "日志保存路径"})

    # ---------- eval参数 -------------
    eval_strategy: str = field(default="epoch", metadata={"help": "eval策略：epoch或steps"})
    # eval_steps: int = field(default=500, metadata={"help": "eval策略选择为steps时，需要指定间隔"})
    num_sample_images: int = field(default=16, metadata={"help": "新增参数——每次eval进行先验抽生成的图片数量，建议是4的倍数，每4个作为一种类别"})
    sample_dir: str = field(default="./samples", metadata={"help": "新增参数——保存策略路径"})

    # ---------- 性能优化参数 ----------
    bf16: bool = field(default=True, metadata={"help": "是否使用FP16混合精度训练"})
    dataloader_num_workers: int = field(default=4, metadata={"help": "数据加载的并行进程数"})
    dataloader_pin_memory: bool = field(default=True, metadata={"help": "是否使用pin_memory加速数据传输"})
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "必须设置False，否则model的forward方法的参数名称必须与dataset输出字典的key一致"}
    )
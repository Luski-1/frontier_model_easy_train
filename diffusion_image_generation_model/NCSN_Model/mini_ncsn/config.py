from transformers import TrainingArguments
from dataclasses import dataclass, field

@dataclass
class NCSNTrainingArguments(TrainingArguments):

    # ---------- NCSN 特定参数 ----------
    sigma_steps: int = field(default=100, metadata={"help": "ncsn中各阶段sigma的退火步数"})
    step_lr: float = field(default=0.00002, metadata={"help": "ncsn中退火基础步长/学习率"})
    sigma_nums: int = field(default=10, metadata={"help": "NCSN的sigma数量"})
    sigma_start: float = field(default=1.0, metadata={"help": "NCSN的最高sigma"})
    sigma_end: float = field(default=0.01, metadata={"help": "NCSN的最小sigma"})
    # ---------- 抽样参数 ----------
    sample_every: int = field(
        default=1,
        metadata={"help": "每N个epoch执行一次采样验证，设置为1表示每个epoch都采样"}
    )
    num_sample_images: int = field(
        default=8,
        metadata={"help": "每次采样验证时生成的图片数量"}
    )
    # ---------- 输入图像参数 ----------
    image_size: int = field(
        default=128,
        metadata={"help": "输入图像的宽高尺寸"}
    )
    channels: int = field(
        default=3,
        metadata={"help": "输入图像的通道"}
    )
    data_root: str = field(
        default="/mnt/d/data/face/img/img_align_celeba",
        metadata={"help": "CelebA数据集根目录路径"}
    )
    # ---------- 模型架构参数 ----------
    base_ch: int = field(
        default=128,
        metadata={"help": "UNet基础通道数，决定模型容量"}
    )
    # ---------- 训练超参数（覆盖父类默认值以匹配原有代码习惯） ----------
    per_device_train_batch_size: int = field(default=8, metadata={"help": "每个设备的训练批次大小"})
    learning_rate: float = field(default=1e-5, metadata={"help": "AdamW优化器的初始学习率"})
    num_train_epochs: int = field(default=100, metadata={"help": "总训练轮数"})
    max_grad_norm: float = field(default=1.0, metadata={"help": "梯度裁剪的最大范数"})
    lr_scheduler_type: str = field(default="constant", metadata={"help": "学习率调度策略"})
    # warmup_steps: int = field(default=0, metadata={"help": "学习率warmup步数"})
    warmup_ratio: float = field(default=0.1, metadata={"help": "学习率warmup比例"})

    # ---------- 保存与日志策略 ----------
    save_strategy: str = field(default="epoch", metadata={"help": "保存策略：epoch或steps"})
    save_total_limit: int = field(default=3, metadata={"help": "最多保留的检查点数量"})
    logging_strategy: str = field(default="steps", metadata={"help": "日志记录策略"})
    logging_steps: int = field(default=50, metadata={"help": "每N步记录一次训练日志"})
    save_dir: str = field(default="./ncsn_checkpoints", metadata={"help": "检查点和生成样本的保存目录"})

    # ---------- 性能优化参数 ----------
    bf16: bool = field(
        default=True,
        metadata={"help": "是否使用BF16混合精度训练"}
    )
    dataloader_num_workers: int = field(default=4, metadata={"help": "数据加载的并行进程数"})
    dataloader_pin_memory: bool = field(default=True, metadata={"help": "是否使用pin_memory加速数据传输"})
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "必须设为False，Huggingface的trainer会检测模型forward的签名来删除不存在的键"}
    )

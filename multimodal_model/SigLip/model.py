from transformers import PreTrainedModel, PretrainedConfig, AutoModel
from transformers.utils import ModelOutput
from dataclasses import dataclass
from typing import Optional
import torch.nn.functional as F
import torch.nn as nn
import torch
import math


@dataclass
class SiglipOutput(ModelOutput):
    loss: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    logits_per_image: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None


class SiglipConfig(PretrainedConfig):
    model_type = "siglip"

    def __init__(
            self,
            vision_model_name_or_path: str = "vit-base-patch16-224",
            text_model_name_or_path: str = "bert-base-chinese",
            projection_dim: Optional[int] = None,  # 可配置投影维度，None则保持1024
            **kwargs
    ):
        super().__init__(**kwargs)
        self.vision_model_name_or_path = vision_model_name_or_path
        self.text_model_name_or_path = text_model_name_or_path
        self.projection_dim = projection_dim


class SiglipModel(PreTrainedModel):
    config_class = SiglipConfig

    def __init__(self, config: SiglipConfig):
        super().__init__(config)
        self.config = config

        # 加载骨干网络
        self.vision_model = AutoModel.from_pretrained(config.vision_model_name_or_path)
        self.text_model = AutoModel.from_pretrained(config.text_model_name_or_path)

        # 自动获取 hidden_size（ViT和RoBERTa都是1024）
        self.vision_hidden_size = self.vision_model.config.hidden_size
        self.text_hidden_size = self.text_model.config.hidden_size

        # 确定投影维度（默认保持1024，也可配置为其他维度如512/768）
        proj_dim = config.projection_dim or self.vision_hidden_size

        # Vision MLP: 1024 -> 1024 (或 proj_dim)
        self.vision_projection = nn.Sequential(
            nn.Linear(self.vision_hidden_size, proj_dim),
            nn.LayerNorm(proj_dim),  # 添加LayerNorm稳定训练
            nn.GELU(),
            nn.Dropout(0.1),  # 可选：防止过拟合
            nn.Linear(proj_dim, proj_dim)
        )

        # Text MLP: 1024 -> 1024 (或 proj_dim)
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_hidden_size, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(proj_dim, proj_dim)
        )

        # 可学习的温度参数和偏置 (SigLIP风格)
        self.t = nn.Parameter(torch.ones(1) * math.log(10))  # ≈ 2.3026
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask, pixel_values, return_loss=True):
        # 1. 提取 [CLS] token 特征
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        vision_outputs = self.vision_model(pixel_values=pixel_values)

        # 取 [CLS] token 表示（维度 [batch_size, 1024]）
        # 使用 .last_hidden_state 而非 .pooler_output
        # roberta没有训练pool层
        text_features = text_outputs.last_hidden_state[:, 0, :]  # 第0个token是CLS
        vision_features = vision_outputs.last_hidden_state[:, 0, :]  # ViT/DINOv3 第0个token是CLS

        # 2. 通过 MLP 投影层
        text_features = self.text_projection(text_features)  # [B, 1024] -> [B, proj_dim]
        vision_features = self.vision_projection(vision_features)  # [B, 1024] -> [B, proj_dim]

        # 3. L2 归一化（必须在投影后进行）
        text_features = F.normalize(text_features, p=2, dim=-1)
        vision_features = F.normalize(vision_features, p=2, dim=-1)

        # 4. 计算相似度矩阵 (SigLIP 的 t-adjusted logistic loss)
        logits_per_text = torch.matmul(text_features, vision_features.t()) * self.t.exp() + self.b
        logits_per_image = logits_per_text.t()

        # 5. 计算损失（对抗性温度缩放 + 逐对 sigmoid）
        batch_size = logits_per_text.shape[0]
        # 创建标签：对角线为1（匹配对），非对角线为-1（非匹配对）
        labels = 2 * torch.eye(batch_size, device=logits_per_text.device) - 1

        # SigLIP 损失: -log(sigmoid(labels * logits)).mean()
        # 关键点1：sigmoid的范围为[0-1]，logsigmoid的范围[-∞, 0]，取负号为[0, +∞]
        # 关键点2：CLIP的损失函数为交叉熵，如文本侧输入去预测图像选项，图像侧输入与预测文本选项，因此按行（图像→文本）求loss和按列（文本→图像）求loss再合并
        # 关键点3：SigLip的损失函数为sigmod，是评判相似度的二元函数，本质是双向的，因此无需显式按行按列求和loss
        loss = None
        if return_loss:
            loglik = F.logsigmoid(labels * logits_per_text)
            # 对每个文本对所有图像求和，然后取平均
            nll = -loglik.sum(dim=-1)
            loss = nll.mean()

        return SiglipOutput(
            loss=loss,
            logits_per_text=logits_per_text,
            logits_per_image=logits_per_image,
            text_embeds=text_features,
            image_embeds=vision_features
        )

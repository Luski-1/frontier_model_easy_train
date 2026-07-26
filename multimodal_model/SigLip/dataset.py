from transformers import AutoTokenizer, AutoProcessor
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from pathlib import Path
from PIL import Image
import random
import torch


class RealSynDataset(Dataset):
    """
    适配纯数字ID格式的图像-文本数据集
    文件名格式: 000000000.jpg, 000000000.txt, 000000000.text1 等
    """

    def __init__(self,
                 data_dir: str,
                 tokenizer,
                 processor,
                 max_seq_length: int = 64,
                 text_field: str = 'caption',
                 shuffle: bool = True):
        """
        Args:
            data_dir: 包含 .jpg 和 .txt 文件的目录路径
            tokenizer: 文本tokenizer
            processor: 图像processor
            max_seq_length: 最大序列长度
            text_field: 使用哪个文本字段
            shuffle: 是否打乱数据
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.text_field = text_field

        # 验证参数
        assert text_field == 'caption' f"text_field必须是'caption'"

        # 扫描所有jpg文件并按数字排序
        image_files = sorted(self.data_dir.glob("*.jpg"),
                             key=lambda x: int(x.stem))  # 按数字ID排序

        print(f"扫描到 {len(image_files)} 个图像文件")

        # 构建样本列表
        self.datas = []
        missing_text = 0

        for img_path in image_files:
            sample_id = img_path.stem  # 如 "000000000"

            # 确定文本文件
            text_path = self.data_dir / f"{sample_id}.txt"

            if text_path.exists():
                self.datas.append({
                    'image_path': str(img_path),
                    'text_path': str(text_path),
                    'id': sample_id
                })
            else:
                missing_text += 1

        if missing_text > 0:
            print(f"警告: {missing_text} 个图像缺少对应的文本文件")
        print(f"成功加载 {len(self.datas)} 个有效样本")

        if shuffle:
            random.shuffle(self.datas)

    def __getitem__(self, index):
        sample = self.datas[index]

        # 1. 加载文本
        with open(sample['text_path'], 'r', encoding='utf-8') as f:
            text = f.read().strip()

        # Tokenize
        tok = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors=None  # 返回list，不是tensor
        )
        # tok = {"input_ids": [index, index, index], "attention_mask": [True, True, True]}

        # 2. 加载图像
        try:
            image = Image.open(sample['image_path']).convert("RGB")
        except Exception as e:
            print(f"警告: 加载图像失败 {sample['image_path']}, 使用黑图代替")
            image = Image.new('RGB', (224, 224), color='black')

        # Processor处理图像 [1, C, H, W]
        pixel_values = self.processor(
            images=image,
            return_tensors='pt'
        )['pixel_values']
        # pixel_values = torch.tensor([[[index, index, index], [index, index, index], [index, index, index]]])

        return {
            'input_ids': tok['input_ids'],
            'attention_mask': tok['attention_mask'],
            'pixel_values': pixel_values,
            # 'id': sample['id'],
            # 'text': text
        }

    def __len__(self):
        return len(self.datas)


class MyDataCollator:
    """聚合数据"""

    def __call__(self, features: List[Dict]):
        input_ids = [f['input_ids'] for f in features]
        attention_mask = [f['attention_mask'] for f in features]
        pixel_values = [f['pixel_values'] for f in features]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'pixel_values': torch.cat(pixel_values, dim=0),  # [B, C, H, W]
            # 'ids': [f['id'] for f in features],
            # 'texts': [f['text'] for f in features]
        }


# ==================== 使用示例 ====================
if __name__ == "__main__":

    # 1. 模型初始化（请替换为你的模型）
    text_model_path = "../autodl-tmp/roberta-large"  # 或本地路径
    tokenizer = AutoTokenizer.from_pretrained(text_model_path)

    picture_model_path = "../autodl-tmp/dinov3-vitl16-pretrain-lvd1689m"
    processor = AutoProcessor.from_pretrained(picture_model_path)

    # 2. 加载数据集
    dataset = RealSynDataset(
        data_dir="/autodl-tmp/realsyn15m_success_all",  # 数据目录
        tokenizer=tokenizer,
        processor=processor,
        max_seq_length=64,  # 截断长度
        text_field='caption',  # 使用 000000000.txt
        shuffle=True
    )

    # 3. 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,  # Dataset内部已shuffle，这里可设为False
        num_workers=0,
        collate_fn=MyDataCollator()
    )

    # 测试
    for batch in dataloader:
        print(f"Batch size: {len(batch['input_ids'])}")
        print(f"Input IDs shape: {batch['input_ids'].shape}")  # [32, 64]
        print(f"Pixel values shape: {batch['pixel_values'].shape}")  # [32, 3, 224, 224]
        print(f"Sample IDs: {batch['ids'][:3]}")  # ['000000000', '000000001', ...]
        break

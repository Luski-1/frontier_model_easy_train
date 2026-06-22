from torchvision.datasets import MNIST
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torch


# ---------- 数据集 ----------
class CelebADataset(Dataset):
    def __init__(self, root, transform=None, image_size=128, channels=3):
        root_path = Path(root)
        self.paths = []
        # 递归遍历指定后缀的文件，保存对应地址
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            self.paths.extend(root_path.rglob(ext))
        self.paths = [str(p) for p in self.paths]

        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {root}")

        self.transform = transform
        self.image_size = image_size  # 使用全局变量
        self.channels = channels  # 【新增】记录通道数

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        try:
            # 根据channels决定convert模式：1是L(灰度)，3是RGB
            mode = "L" if self.channels == 1 else "RGB"
            img = Image.open(img_path).convert(mode)
            # 归一化，转换为-1到1
            if self.transform:
                img = self.transform(img)
            img = img * 2.0 - 1.0

            return {"pixel_values": img}
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

            return {"pixel_values": torch.randn(self.channels, self.image_size, self.image_size) * 2.0 - 1.0}


class MNISTDataset(Dataset):
    """
    transformer库的trainer要求dataset返回的数据是字典形式，因此需要往外包裹一层
    """
    def __init__(self, root, transform, image_size=28, channels=1):
        
        self.dataset = MNIST(
            root=root,
            train=True,
            download=False,
            transform=transform
        )
        self.image_size = image_size
        self.channels = channels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            img, _ = self.dataset[idx]
            # 归一化，转换为-1到1
            img = img * 2.0 - 1.0
            return {"pixel_values": img}
        except Exception as e:
            print(f"Error loading: {e}")

            return {"pixel_values": torch.randn(self.channels, self.image_size, self.image_size) * 2.0 - 1.0}

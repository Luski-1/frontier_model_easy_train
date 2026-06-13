from torch.utils.data import Dataset
from torchvision import transforms
from config_face import VAEConfig
from pathlib import Path
from PIL import Image
import torch


class CelebADataset(Dataset):
    """
    返回 dict 格式：{"pixel_values": tensor}
    """

    def __init__(self, root, config: VAEConfig):

        root_path = Path(root)
        self.paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:  # 指定文件后缀
            self.paths.extend(root_path.rglob(ext))         # rglob递归搜索

        self.paths = [str(p) for p in self.paths]

        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {root}")

        self.transform = transforms.Compose([
            transforms.Resize(config.image_size),           # 缩放
            transforms.CenterCrop(config.image_size),       # 中心裁剪
            # transforms.RandomHorizontalFlip(p=0.5),       # 水平翻转
            transforms.ToTensor(),                          # [0, 1.0]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # 减均值除方差，[0, 1.0] > [-0.5, 0.5] > [-1, 1] 
        ])
        self.image_size = config.image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)

            return {"pixel_values": img}
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return {"pixel_values": torch.randn(3, self.image_size, self.image_size)}
        
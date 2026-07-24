from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torch

class CelebADataset(Dataset):
    def __init__(self, root, image_size, transform=None):
        root_path = Path(root)
        self.paths = []
        # 记录所有图像的路径
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            self.paths.extend(root_path.rglob(ext))
        self.paths = [str(p) for p in self.paths]

        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {root}")

        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            # 归一化，转换为-1到1
            if self.transform:
                img = self.transform(img)
            img = img * 2.0 - 1.0
            return img
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.randn(3, self.image_size, self.image_size) * 2.0 - 1.0
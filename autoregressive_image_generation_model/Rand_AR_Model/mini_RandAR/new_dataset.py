from torchvision import transforms
from torch.utils import data
from PIL import Image
import numpy as np
import tarfile
import torch
import random
import math
import os
#################################################################################
#                         Dataset Builder                                       #
#################################################################################

def build_dataset(is_train, args, transform):
    """构建数据集

    Args:
        is_train: 是否为训练集
        args: 包含 dataset 和 data_path 的参数对象
        transform: 图像变换函数
    """
    if args.dataset == "imagenet":
        root = os.path.join(args.data_path, "train.tar" if is_train else "val.tar")  # documentation.md说明建议使用tar格式的数据压缩包
        dataset = ImageTarDataset(root, return_labels=True, transform=transform)
        dataset.nb_classes = 1000   # 指定分类数量
    elif args.dataset == "latent":  # 默认走该分支
        dataset = INatLatentDataset(
            root_dir=args.data_path, transform=transform
        )
    else:
        raise NotImplementedError(f"Dataset type '{args.dataset}' not implemented")
    return dataset


#################################################################################
#                         ImageNet Tar Dataset                                  #
#################################################################################

class ImageTarDataset(data.Dataset):
    """从 tar 文件加载 ImageNet 数据集

    Args:
        tar_file: tar 文件路径
        return_labels: 是否返回标签
        transform: 图像变换函数
    """
    def __init__(self, tar_file, return_labels=False, transform=transforms.ToTensor()):
        self.tar_file = tar_file
        self.tar_handle = None
        categories_set = set()
        self.tar_members = []
        self.categories = {}
        self.categories_to_examples = {}
        # tar_file = /tmp/imagenet-llamagen-adm-256_codes/train.tar
        with tarfile.open(tar_file, 'r:') as tar:
            for index, tar_member in enumerate(tar.getmembers()):
                # 过滤不符合目录层级的文件
                if tar_member.name.count('/') != 2:
                    continue
                # 提取类别名
                category = self._get_category_from_filename(tar_member.name)
                categories_set.add(category)
                # 保存文件句柄对象而非文件名
                self.tar_members.append(tar_member)
                # 建立类别到样本索引的映射
                cte = self.categories_to_examples.get(category, [])
                cte.append(index)
                self.categories_to_examples[category] = cte
        categories_set = sorted(categories_set)
        # 将类别名排序并映射为整数 ID (0, 1, 2...)
        for index, category in enumerate(categories_set):
            self.categories[category] = index
        self.num_examples = len(self.tar_members)
        self.indices = np.arange(self.num_examples)
        self.num = self.__len__()
        print("Loaded the dataset from {}. It contains {} samples.".format(tar_file, self.num))
        self.return_labels = return_labels
        self.transform = transform  # ToTensor()
        self.nb_classes = 0

    def _get_category_from_filename(self, filename):
        # 示例：输入train/n01440764/n01440764_10029.JPEG
        # 第一个 / 在索引5，begin变为6
        # 从索引6开始找下一个/，在索引15，end变为15。截取filename[6:15]，返回n01440764
        begin = filename.find('/')
        begin += 1
        end = filename.find('/', begin)
        return filename[begin:end]

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        index = self.indices[index]
        # 延迟打开 tar 句柄：每个 Worker 进程只在第一次调用时打开
        if self.tar_handle is None:
            self.tar_handle = tarfile.open(self.tar_file, 'r:')
    
        sample = self.tar_handle.extractfile(self.tar_members[index])
        image = Image.open(sample).convert('RGB')
        image = self.transform(image)   # ToTensor()，转换为[0-1]
        # 默认开启
        if self.return_labels:
            # 获取类别名->类别id
            category = self.categories[self._get_category_from_filename(
                self.tar_members[index].name)]
            return image, category, index
        return image, index


#################################################################################
#                         Latent Dataset (for pre-tokenized data)               #
#################################################################################

class INatLatentDataset(data.Dataset):
    """从预编码的 latent token 文件加载数据集

    数据格式：root_dir/类别ID/样本.npy
    每个 .npy 文件的形状为 (1, aug_num, block_size)，其中 aug_num 是数据增强的数量

    Args:
        root_dir: 数据根目录路径
        transform: 数据变换函数
    """
    def __init__(self, root_dir, transform=transforms.ToTensor()):
        categories_set = set()
        # 扫描 root_dir 下的所有子文件夹名，转为 int 后排序
        # 例如 ['0', '1', '2', ..., '999'] → [0, 1, 2, ..., 999]
        self.categories = sorted([int(i) for i in list(os.listdir(root_dir))])
        self.samples = []

        for tgt_class in self.categories:
            # 类别目录
            tgt_dir = os.path.join(root_dir, str(tgt_class))
            for root, _, fnames in sorted(os.walk(tgt_dir, followlinks=True)):
                for fname in fnames:
                    path = os.path.join(root, fname)
                    item = (path, tgt_class)
                    self.samples.append(item)   # (具体路径, 类别)
        self.num_examples = len(self.samples)
        self.indices = np.arange(self.num_examples)
        self.num = self.__len__()
        print("Loaded the dataset from {}. It contains {} samples.".format(root_dir, self.num))
        self.transform = transform

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        index = self.indices[index]
        sample = self.samples[index]    # (具体路径, 类别)
        latents = np.load(sample[0])
        latents = self.transform(latents) # 1 * aug_num * block_size    # aug_num是数据增强的数量

        # select one of the augmented crops
        aug_idx = torch.randint(0, latents.shape[1], (1,)).item()   # 随机挑选1张
        latents = latents[:, aug_idx, :]
        label = sample[1]

        return latents, label, index
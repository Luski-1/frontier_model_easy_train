# Cloned from https://github.com/facebookresearch/deit/blob/main/datasets.py
# Modified from https://github.com/bfshi/AbSViT/blob/master/datasets.py

import os
import json
import numpy as np
import tarfile
from PIL import Image

import torch
from torch.utils import data
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


class ImageTarDataset(data.Dataset):
    def __init__(self, tar_file, return_labels=False, transform=transforms.ToTensor()):
        '''
        return_labels:
        Whether to return labels with the samples
        transform:
        A function/transform that takes in an PIL image and returns a transformed version. E.g, transforms.RandomCrop
        '''
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


class INatLatentDataset(data.Dataset):
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

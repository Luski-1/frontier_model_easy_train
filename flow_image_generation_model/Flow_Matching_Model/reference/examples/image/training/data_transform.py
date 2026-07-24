# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torchvision.transforms.v2 import Compose, RandomHorizontalFlip, ToDtype, ToImage


def get_train_transform():
    transform_list = [
        ToImage(),                          # 把离散[0,255] > 连续[0,1]
        RandomHorizontalFlip(),             # 随机水平翻转
        ToDtype(torch.float32, scale=True), # 转换为float32格式
    ]
    return Compose(transform_list)

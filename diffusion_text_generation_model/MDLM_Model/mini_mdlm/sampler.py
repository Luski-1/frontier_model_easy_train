import torch
import math

class FaultTolerantDistributedSampler(torch.utils.data.DistributedSampler):
    """
    该sampler用于给dataloader进行分布式抽样，但是accelerator会自动生成默认的sampler，因此仅做学习
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0
        self.restarting = False

    def state_dict(self):
        return {'epoch': self.epoch, 'counter': self.counter}

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.counter = state_dict['counter']
        self.restarting = True


    def __iter__(self):
    # 如果开启打乱
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)   # seed+epoch得到与epoch相关的种子
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # 随机生成dataset长度的下标列表【不会重复】
        else:
            indices = list(range(len(self.dataset))) # 生成dataset长度的下标列表[0,1,...,N-1]【不会重复】

        if not self.drop_last:    # 如果不开启删除最后一批不满足per_device_batch_size的功能

            # total_size是父类根据len(dataset)和进程数，计算出来能够保证平分给各进程的最佳数据总量
            # 最佳数据总量-当前数据总量=欠缺总量
            padding_size = self.total_size - len(indices)
            # 如果欠缺总量≤当前数据总量
            if padding_size <= len(indices):
                # 切片对应长度即可
                indices += indices[:padding_size]
            else:
                # 先计算欠缺总量是当前数据总量的多少倍，强制向上取整
                # 对当前数据总量扩增对应倍数，随后切片对应长度
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
        # 最简单的方式，直接切片最佳数据总量
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]  # 显卡进程下标~最佳数据总量的范围进行切片，补偿为显卡进程数，让每个显卡进程仅获取对应的部分
        assert len(indices) == self.num_samples

        # 用于断点续训
        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter:]
            self.restarting = False

        # 吐出index
        for index in indices:
            self.counter += 1
            yield index

        self.counter = 0
import numpy as np
import torch
from torch.utils.data.sampler import Sampler


# 定义分布式验证采样器，继承自PyTorch官方Sampler基类
class EvalDistributedSampler(Sampler):
    # 初始化函数：构造采样器，分配当前GPU进程要读取的验证集数据索引
    # 参数：
    # dataset: 验证集数据集对象（val_set）
    # num_replicas: 总进程数（总GPU数，dist.get_world_size()）
    # rank: 当前进程的编号（0,1,2...，dist.get_rank()）
    def __init__(self, dataset, num_replicas, rank):
        # 1. 生成数据分割点：把数据集总长度 等分为 num_replicas 份
        #    np.linspace(起点0, 终点len(dataset), 分割点数=总进程数+1)
        #    例：数据集100条，4个GPU → seps = [0,25,50,75,100]
        seps = np.linspace(0, len(dataset), num_replicas + 1, dtype=int)

        # 2. 拆分分割点为 起始索引列表 和 结束索引列表
        #    beg = [0,25,50,75]，end = [25,50,75,100]
        beg, end = seps[:-1], seps[1:]

        # 3. 根据当前进程rank，拿到自己专属的 起始、结束索引
        #    例：rank=0 → beg=0, end=25；rank=1 → beg=25, end=50
        beg, end = beg[rank], end[rank]

        # 4. 生成当前进程专属的索引列表（连续整数），并转为元组（ immutable，效率更高）
        #    例：rank0 → (0,1,2...24)，只负责这部分验证数据
        self.indices = tuple(range(beg, end))

    # 迭代器方法：PyTorch DataLoader 会调用这个方法获取数据索引
    # 作用：返回当前GPU要遍历的验证集索引列表
    def __iter__(self):
        return iter(self.indices)

    # 长度方法：返回当前进程负责的验证数据数量
    # 作用：告诉DataLoader当前GPU有多少条验证数据
    def __len__(self) -> int:
        return len(self.indices)


class InfiniteBatchSampler(Sampler):
    def __init__(self, dataset_len, batch_size, seed_for_all_rank=0, fill_last=False, shuffle=True, drop_last=False,
                 start_ep=0, start_it=0):
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.iters_per_ep = dataset_len // batch_size if drop_last else (dataset_len + batch_size - 1) // batch_size
        self.max_p = self.iters_per_ep * batch_size
        self.fill_last = fill_last
        self.shuffle = shuffle
        self.epoch = start_ep
        self.same_seed_for_all_ranks = seed_for_all_rank
        self.indices = self.gener_indices()
        self.start_ep, self.start_it = start_ep, start_it

    def gener_indices(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.same_seed_for_all_ranks)
            indices = torch.randperm(self.dataset_len, generator=g).numpy()
        else:
            indices = torch.arange(self.dataset_len).numpy()

        tails = self.batch_size - (self.dataset_len % self.batch_size)
        if tails != self.batch_size and self.fill_last:
            tails = indices[:tails]
            np.random.shuffle(indices)
            indices = np.concatenate((indices, tails))

        # built-in list/tuple is faster than np.ndarray (when collating the data via a for-loop)
        # noinspection PyTypeChecker
        return tuple(indices.tolist())

    def __iter__(self):
        self.epoch = self.start_ep
        while True:
            # BUG，self.epoch += 1导致self.epoch == self.start_ep永远为False，导致继续训练会有数据读取的问题
            # self.epoch += 1
            p = (self.start_it * self.batch_size) if self.epoch == self.start_ep else 0
            # p累计取数<单卡训练数据总量时
            while p < self.max_p:
                # 获得尾部下标
                q = p + self.batch_size
                yield self.indices[p:q]
                p = q
            # 把78行代码注释，并且迁移到87行
            self.epoch += 1
            # 完成epoch后打乱
            if self.shuffle:
                self.indices = self.gener_indices()

    def __len__(self):
        return self.iters_per_ep


class DistInfiniteBatchSampler(InfiniteBatchSampler):
    def __init__(self, world_size, rank, dataset_len, glb_batch_size, same_seed_for_all_ranks=0, repeated_aug=0,
                 fill_last=False, shuffle=True, start_ep=0, start_it=0):
        assert glb_batch_size % world_size == 0
        self.world_size, self.rank = world_size, rank
        self.dataset_len = dataset_len
        self.glb_batch_size = glb_batch_size
        self.batch_size = glb_batch_size // world_size # 单卡batch size
        # 保证即使数据量不能整除全局batch_size，也能够最接近且大于真实步数
        self.iters_per_ep = (dataset_len + glb_batch_size - 1) // glb_batch_size
        self.fill_last = fill_last
        self.shuffle = shuffle
        self.repeated_aug = repeated_aug # 图片重复使用
        self.epoch = start_ep
        self.same_seed_for_all_ranks = same_seed_for_all_ranks
        self.indices = self.gener_indices()
        self.start_ep, self.start_it = start_ep, start_it

    def gener_indices(self):
        # ===================== 1. 计算本轮总共需要多少个样本（全局） =====================
        # global_max_p = 一轮训练总共需要多少个样本（所有GPU加起来）
        # 关键保证：global_max_p 一定能被 world_size 整除，因为 glb_batch_size 是按GPU数对齐的
        global_max_p = self.iters_per_ep * self.glb_batch_size
        # ===================== 2. 生成全局索引（所有GPU共用同一份打乱顺序） =====================
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.same_seed_for_all_ranks) # 进一步细化随机种子，根据epoch来调整随机种子
            global_indices = torch.randperm(self.dataset_len, generator=g)
            # ===================== 【可选】重复增强：少部分数据重复多次 =====================
            # 比如数据集100万，repeated_aug=4 → 只用25万数据，每条重复4次，快速训练
            if self.repeated_aug > 1:
                # 取 1/repeated_aug 比例的数据
                global_indices = global_indices[:(self.dataset_len + self.repeated_aug - 1) // self.repeated_aug]
                # 每条数据重复 repeated_aug 次
                global_indices = global_indices.repeat_interleave(self.repeated_aug, dim=0)
                # 截断到 global_max_p 长度
                global_indices = global_indices[:global_max_p]
        else:
            # 不打乱：顺序索引
            global_indices = torch.arange(self.dataset_len)
        # ===================== 3. 填充尾部：不够的样本从头补齐 =====================
        # 计算还差多少个样本才能达到 global_max_p
        filling = global_max_p - global_indices.shape[0]

        # 如果需要填充、且确实不够 → 从开头复制数据补齐
        if filling > 0 and self.fill_last:
            global_indices = torch.cat((global_indices, global_indices[:filling]))

        # ===================== 4. 核心：把全局索引均分给所有GPU =====================
        # 把 global_indices 平均切分成 world_size 份
        seps = torch.linspace(0, global_indices.shape[0], self.world_size + 1, dtype=torch.int)

        # 当前GPU（rank）只取自己的那一段
        local_indices = global_indices[seps[self.rank].item(): seps[self.rank + 1].item()].tolist()

        # 记录当前GPU负责的样本总数
        self.max_p = len(local_indices)

        # 返回：当前GPU专属的、不与任何其他GPU重叠的索引列表
        return local_indices

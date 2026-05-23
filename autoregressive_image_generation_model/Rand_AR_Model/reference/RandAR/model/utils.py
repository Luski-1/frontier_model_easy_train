# drop_path cloned from https://github.com/FoundationVision/LlamaGen/blob/main/utils/drop_path.py
import torch
import math


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # (bs, 1, 1)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    # 随机抽样，决定哪条样本在当前层直接置0，等于当前层无效
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        # 维持均值，需要除以维持概率
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(torch.nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


def interleave_tokens(seq1, seq2):
    """ 交错拼接两个序列张量 """
    # 1. 创建一个和 seq1、seq2 拼接后形状、数据类型完全一致的全0张量
    result = torch.zeros_like(torch.cat((seq1, seq2), dim=1))
    # 2. 把 seq1 的所有元素赋值到结果张量的偶数索引位置（0,2,4...）
    result[:, ::2] = seq1
    # 3. 把 seq2 的所有元素赋值到结果张量的奇数索引位置（1,3,5...）
    result[:, 1::2] = seq2
    # 4. 返回交错拼接完成的张量
    return result


def calculate_num_query_tokens_for_parallel_decoding(
    cur_step,           # 当前是第几步（从 0 开始计数，如 0, 1, 2, ...）
    total_step,         # 总共计划用多少步（如 88）
    block_size,         # 总共有多少个 token 要生成（如 256）
    query_token_idx_cur_step,  # 当前步从第几个 token 开始预测
    num_query_token_cur_step   # 当前步预测了几个 token
):
    # 通过余弦公式，计算到cur_step为止，应该累计生成多少个token
    # 累计token数量的增加速度，先慢后快
    # cur_step是已经提前+1，因此其实是计算下一步累计的token数
    num_target_decoded_tokens = (
        1.0 - math.cos(math.pi / 2.0 * (cur_step + 1) / total_step)
    ) * block_size + 1
    # 不要超过block_size
    num_target_decoded_tokens = min(
        int(num_target_decoded_tokens), block_size
    )

    # 下一步生成数 = 目标累计 - 已经生成的 - 当前生成的
    num_query_tokens_next_step = (
        num_target_decoded_tokens - query_token_idx_cur_step - num_query_token_cur_step
    )
    # 不小于1
    num_query_tokens_next_step = max(num_query_tokens_next_step, 1)
    # 不大于剩余token
    num_query_tokens_next_step = min(
        num_query_tokens_next_step,
        block_size - query_token_idx_cur_step - num_query_token_cur_step,
    )

    return num_query_tokens_next_step
from torch.utils.checkpoint import checkpoint
from torch.nn import functional as F
from typing import Optional, Tuple
import torch.nn as nn
import torch
import math


#################################################################################
#                         Utility Functions                                      #
#################################################################################

def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)


def precompute_freqs_cis_2d(
    grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120
):
    # split the dimension into half, one for x and one for y
    # 除2，为了分割x轴和y轴
    half_dim = n_elem // 2
    # 假设n_elm = 64, half_dim = 32
    # arange步长为2，为了两两一组
    # 0,2,4,6,8 ... // 32 = 0/16,1/16,2/16,3/16....
    freqs = 1.0 / (
        base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim)
    )
    t = torch.arange(grid_size, device=freqs.device)
    # 位置索引 × 频率向量的外积：(grid_size, head_dim // 2)，得到t / 10000 ^ (i / 16)
    freqs = torch.outer(t, freqs)  # (grid_size, head_dim // 2)
    # 假设grid_size = 16
    # freqs[:, None, :].expand → 把 x 轴频率扩展为 (16,16,32)
    # freqs[None, :, :].expand → 把 y 轴频率扩展为 (16,16,32)
    freqs_grid = torch.concat(
        [
            freqs[:, None, :].expand(-1, grid_size, -1),
            freqs[None, :, :].expand(grid_size, -1, -1),
        ],
        dim=-1,
    )  # (grid_size, grid_size, head_dim // 2)
    # 对每个角度计算余弦、正弦值，并在最后维度stack
    cache_grid = torch.stack(
        [torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1
    )  # (grid_size, grid_size, head_dim // 2, 2)
    # 展平2D网格 → 1D序列
    cache = cache_grid.flatten(0, 1)
    # 给 CLS 分类 token 补 0 编码
    cond_cache = torch.cat(
        [torch.zeros(cls_token_num, n_elem // 2, 2), cache]
    )  # (cls_token_num+grid_size**2, head_dim // 2, 2)
    return cond_cache


def batch_apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (bs, seq_len, head_dim // 2, 2)
    bs, seq_len, n_head, head_dim = x.shape
    xshaped = x.float().reshape(
        *x.shape[:-1], head_dim // 2, 2
    )  # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(
        bs, xshaped.size(1), 1, xshaped.size(3), 2
    )  # (bs, seq_len, 1, head_dim//2, 2)
    # 原始的rope实现方式，dim=[d0, d1, d2, d3]，d0 = d0 * cos - d1 * sin, d1 = d1 * cos + d0 * sin
    # 能够兼容1D 或者 2D。如果纯1D，有更加快捷的计算方法.
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        dim=-1,
    )
    x_out2 = x_out2.flatten(3)  # (bs, seq_len, n_head, head_dim)
    return x_out2.type_as(x)


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


def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # 取出概率最大的 top_k 个值，取第 k 个作为门槛
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value    # 低于门槛的设为 -inf

    if top_p < 1.0:
        # 1. 排序：把 logits 从大到小排好
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        # 2. 累积概率：计算前缀和 [p1, p1+p2, p1+p2+p3, ...]
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)   # bsz, vocab

        # 3. 标记移除：只要累积概率超过了 0.9 (假设 top_p=0.9)，后面的都不要了
        sorted_indices_to_remove = cumulative_probs > top_p
        # 4. 保底机制：确保至少保留 min_tokens_to_keep 个
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        #    假设 Token 概率为 [0.6, 0.3, 0.1]，p=0.8。
        #    累积概率是 [0.6, 0.9, 1.0]。
        #    直接比较 > 0.8 会得到 [F, T, T]，也就是连 0.3 的那个 Token 都要扔掉！
        #    但 0.6+0.3=0.9 才是真正跨过 0.8 门槛的组合，所以必须保留 0.3 那个。变成[F, F, T]
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits


def sample(
    logits,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    sample_logits=True,
):
    """
    Note, torch.multinomial can only do probs of shape 1D or 2D.
    Args:
        logits: tensor of shape (batch_size, 1, vocab_size)

    Outs:
        idx: tensor of shape (batch_size, 1) of sampled indices
        probs: tensor of shape (batch_size, 1, vocab_size) of probabilities
    """
    logits = logits[:, -1, :] / max(temperature, 1e-5)  # bsz, vocab
    if top_k > 0 or top_p < 1.0:    # randar_gpt.py中调用sample时，默认top_k=0, top_p=1.0
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    if sample_logits:
        # torch.multinomial only accept 1D or 2D probs
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    return idx, probs


#################################################################################
#                      Embedding Layers for Class Labels                        #
#################################################################################
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0    # CFG训练期间的随机概率
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            # 随机值小于概率值，为True
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
            # True为1000，False保留原有label
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels).unsqueeze(1)
        return embeddings


#################################################################################
#                      Embedding Layers for Text Feature                        #
#################################################################################
class CaptionEmbedder(nn.Module):
    """
    Embeds text caption into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, in_channels, hidden_size, uncond_prob, token_num=120):
        super().__init__()
        self.cap_proj = MLP(
            in_features=in_channels,
            hidden_features=hidden_size,
            out_features=hidden_size,
        )
        self.register_buffer(
            "uncond_embedding",
            nn.Parameter(torch.randn(token_num, in_channels) / in_channels**0.5),
        )
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None], self.uncond_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        embeddings = self.cap_proj(caption)
        return embeddings


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


#################################################################################
#                                  GPT Model                                    #
#################################################################################
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(
        self, dim: int, ffn_dim_multiplier: int, multiple_of: int, ffn_dropout_p: float
    ):
        super().__init__()
        hidden_dim = 4 * dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        # 如果传入了 FFN 维度缩放系数，就对维度进行缩放
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        # 把计算后的维度，调整为 multiple_of 的整数倍
        hidden_dim = find_multiple(hidden_dim, multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.ffn_dropout = nn.Dropout(ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


""" Attention module modified for the parts updating KV cache
    Supporting slicing to accelerate inference
"""


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,  # 1280
            n_head: int,  # 20
            n_kv_head: int,  # None
            attn_dropout_p: float,  # 0.0
            resid_dropout_p: float,  # 0.1
    ):
        super().__init__()
        assert dim % n_head == 0
        self.dim = dim
        self.head_dim = dim // n_head
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head  # n_kv_head用于GQA，减低KV cache
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.kv_cache = None

        # regularization
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout_p)

    def forward(
            self,
            x: torch.Tensor,
            freqs_cis: torch.Tensor = None,
            input_pos: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
    ):
        """
        during inference:
        Args:
            x: [bsz, seqlen, dim], input tensor.
            freqs_cis: [bsz, seqlen, head_dim // 2, 2], used to apply rotary emb.
            input_pos: [seqlen], used to update KV cache.
            mask: [bsz, 1, seqlen, seqlen], used to mask out attention weights.
        """
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)

        # this part is modified from LLaMAGen
        xq = batch_apply_rotary_emb(xq, freqs_cis)
        xk = batch_apply_rotary_emb(xk, freqs_cis)  # (bs, seq_len, n_head, head_dim)   # 其中class token经过ROPE后直接置零

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))  # (bs, n_head, seq_len, head_dim)

        # this part is modified from LLaMAGen
        # 推理阶段才使用，存入和取出KV cache
        if self.kv_cache is not None:
            # [b, n_head, max_seq_len, head_dim]
            keys, values = self.kv_cache.update(input_pos, xk, xv)

            # assuming that all the samples in a batch have the same input_pos
            max_pos = torch.max(input_pos) + 1  # 强假设：同batch中输入长度一致
            keys = keys[:, :, :max_pos] # 获取之前存入的k以及当前传入的xk，包含(class token, query_0)
            values = values[:, :, :max_pos] # 获取之前存入的v以及当前传入的xv，包含(class token, query_0)
            if mask is not None:
                mask = mask[:, :, :, :max_pos]  # 结合463行，保证依然是因果自回归的下三角
        else:
            keys, values = xk, xv

        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        # TODO 极其非常奇怪，Class token的位置编码竟然设置为0，后续所有token如何通过attention获取class 信息？
        output = F.scaled_dot_product_attention(
            xq,
            keys,
            values,
            attn_mask=mask,
            is_causal=True if mask is None else False,  # 当mask为None时，代表训练期间，开启Causal掩码；否则推理期间会显式设置mask，无需再增加Causal掩码
            dropout_p=self.attn_dropout_p if self.training else 0,
        )
        # (bs, n_head, seq_len, head_dim) > (bs, seq_len, n_head, head_dim) > (bs, seq_len, dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))
        return output


""" Cloned from LLaMAGen: only the attention uses our customized version
"""


class TransformerBlock(nn.Module):
    def __init__(
            self,
            dim=4096,  # 1280
            n_layer=32,  # 36
            n_head=32,  # 20
            n_kv_head=None,  # None
            multiple_of=256,  # 256
            ffn_dim_multiplier=None,  # None
            rope_base=10000,  # 10000
            norm_eps=1e-5,  # 1e-5
            token_dropout_p=0.1,  # 0.1
            attn_dropout_p=0.0,  # 0.0
            resid_dropout_p=0.1,  # 0.1
            ffn_dropout_p=0.1,  # 0.1
            drop_path=0.0,  # 0.0
    ):
        super().__init__()
        self.attention = Attention(
            dim, n_head, n_kv_head, attn_dropout_p, resid_dropout_p
        )
        self.feed_forward = FeedForward(
            dim, ffn_dim_multiplier, multiple_of, ffn_dropout_p
        )
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
            self,
            x: torch.Tensor,
            freqs_cis: torch.Tensor,
            start_pos: int,
            mask: Optional[torch.Tensor] = None,
    ):
        h = x + self.drop_path(
            self.attention(self.attention_norm(x), freqs_cis, start_pos, mask)
        )
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out


class RandARTransformer(nn.Module):
    def __init__(
            self,
            dim=4096,  # dim: 1280
            n_layer=32,  # n_layer: 36
            n_head=32,  # n_head: 20
            n_kv_head=None,
            multiple_of=256,
            ffn_dim_multiplier=None,
            rope_base=10000,
            norm_eps=1e-5,
            initializer_range=0.02,
            token_dropout_p=0.1,  # token_dropout_p: 0.1
            attn_dropout_p=0.0,
            resid_dropout_p=0.1,  # resid_dropout_p: 0.1
            ffn_dropout_p=0.1,  # ffn_dropout_p: 0.1
            drop_path_rate=0.0,  # drop_path_rate: 0.0
            num_classes=1000,  # num_classes: 1000
            caption_dim=2048,
            class_dropout_prob=0.1,
            model_type="c2i",  # model_type: c2i
            vocab_size=16384,  # vocab_size: 16384
            cls_token_num=1,  # cls_token_num: 1
            block_size=256,  # block_size: 256
            max_batch_size=32,
            max_seq_len=2048,
            position_order="random",  # position_order: random
            num_inference_steps=88,  # num_inference_steps: 88
            zero_class_qk=True,  # zero_class_qk: True
            grad_checkpointing=True,  # grad_checkpointing: True
    ):
        super().__init__()
        self.dim = dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.rope_base = rope_base
        self.norm_eps = norm_eps
        self.token_dropout_p = token_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p
        self.ffn_dropout_p = ffn_dropout_p
        self.drop_path_rate = drop_path_rate
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.num_classes = num_classes
        self.model_type = model_type
        self.cls_token_num = cls_token_num
        if self.model_type == "c2i":
            self.cls_embedding = LabelEmbedder(num_classes, dim, class_dropout_prob)  # class_dropout_prob=0.1
        elif self.model_type == "t2i":
            self.cls_embedding = CaptionEmbedder(caption_dim, dim, class_dropout_prob)
        else:
            raise Exception("please check model type")
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.tok_dropout = nn.Dropout(token_dropout_p)

        # transformer blocks
        # dpr,Drop Path Rate,简单而言，就是对某一层的输出x进行随机dropout，等同于正则化，希望更加鲁棒
        # dpr一般是线性增加
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(n_layer):
            self.layers.append(
                TransformerBlock(
                    dim=dim,  # dim: 1280
                    n_layer=n_layer,  # n_layer: 36
                    n_head=n_head,  # n_head: 20
                    n_kv_head=n_kv_head,  # none
                    multiple_of=multiple_of,  # 256
                    ffn_dim_multiplier=ffn_dim_multiplier,  # none
                    rope_base=rope_base,  # 10000
                    norm_eps=norm_eps,  # 1e-5
                    token_dropout_p=token_dropout_p,  # 0.1
                    attn_dropout_p=attn_dropout_p,  # 0.0
                    resid_dropout_p=resid_dropout_p,  # 0.1
                    ffn_dropout_p=ffn_dropout_p,  # 0.1
                    drop_path=dpr[layer_id],
                )
            )

        # output layer
        self.norm = RMSNorm(dim, eps=norm_eps)
        self.output = nn.Linear(dim, vocab_size, bias=False)

        # 2d rotary pos embedding
        grid_size = int(self.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        # 计算2D ROPE矩阵，传入的单头下的head_dim
        # TODO 极其非常奇怪，Class token的位置编码竟然设置为0，后续所有token如何通过attention获取class 信息？
        self.freqs_cis = precompute_freqs_cis_2d(
            grid_size, self.dim // self.n_head, self.rope_base, self.cls_token_num
        )  # (cls_token_num+grid_size**2, head_dim // 2, 2)

        # KVCache
        self.max_batch_size = -1
        self.max_seq_length = -1

        # initialization
        self.initializer_range = initializer_range
        self.initialize_weights()

        # RandAR related parameters
        # 位置指令向量，仅存在1个，通过repeat+ROPE变成代表指向不同图片块的指令向量
        self.pos_instruct_embeddings = nn.Parameter(torch.randn(1, self.dim) * self.initializer_range)  # [1, dim]
        self.position_order = position_order  # random
        self.num_inference_steps = num_inference_steps  # 88
        self.zero_class_qk = zero_class_qk  # True
        self.grad_checkpointing = grad_checkpointing  # True

    def initialize_weights(self):
        # PyTorch 内置递归方法：遍历模型里所有的子模块
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        # Zero-out output layers:
        nn.init.constant_(self.output.weight, 0)

    def _init_weights(self, module):
        std = self.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        # if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
        #     return
        head_dim = self.dim // self.n_head  # 获取head dim
        max_seq_length = find_multiple(max_seq_length, 8)   # 确保是8的倍数
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:   # 为每一层attention开启KV cache
            b.attention.kv_cache = KVCache(
                max_batch_size, max_seq_length, self.n_head, head_dim, dtype
            )
        # 设置掩码
        causal_mask = torch.tril(
            torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
        )
        self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        grid_size = int(self.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        self.freqs_cis = precompute_freqs_cis_2d(
            grid_size, self.dim // self.n_head, self.rope_base, self.cls_token_num
        )

    def remove_caches(self):
        for l in self.layers:
            l.attention.kv_cache = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def forward(
            self,
            idx: torch.Tensor,
            cond_idx: torch.Tensor,  # cond_idx_or_embed
            token_order: Optional[torch.Tensor] = None,
            input_pos: Optional[torch.Tensor] = None,
            targets: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            valid: Optional[torch.Tensor] = None,
    ):
        if idx is not None and cond_idx is not None:
            return self.forward_train(idx, cond_idx, token_order, input_pos, targets, mask, valid)
        else:
            raise ValueError("idx and cond_idx cannot be both None")

    def forward_train(self,
                      idx: torch.Tensor,
                      cond_idx: torch.Tensor,
                      token_order: Optional[torch.Tensor] = None,
                      input_pos: Optional[torch.Tensor] = None,
                      targets: Optional[torch.Tensor] = None,
                      mask: Optional[torch.Tensor] = None,
                      valid: Optional[torch.Tensor] = None, ):
        """ Args:
            idx: [bsz, seq_len] GT image tokens for teacher forcing
            cond_idx: [bsz, cls_token_num] Cls tokens
            token_order: [bsz, seq_len] Position order for each token
            input_pos: [seq_len] Position index for each token (default None)
            targets: [bsz, seq_len] Target tokens for teacher forcing (default None)
            mask: [bsz, seq_len, seq_len] Causal mask for attention (default None)
            valid: [bsz, seq_len] Valid mask for loss calculation (default None)
        """
        # 1. Prepare orders
        bs = idx.shape[0]
        if token_order is None:
            if self.position_order == "random":
                token_order = torch.arange(self.block_size, device=self.tok_embeddings.weight.device, dtype=torch.long)
                token_order = token_order.unsqueeze(0).repeat(bs, 1)  # bs, block_size or seq_len
                for i in range(bs):
                    token_order[i] = token_order[i][torch.randperm(self.block_size)]
                token_order = token_order.contiguous()
            elif self.position_order == "raster":  # 光栅顺序
                token_order = torch.arange(self.block_size, device=idx.device)
                token_order = token_order.unsqueeze(0).repeat(bs, 1)
                token_order = token_order.contiguous()
            else:
                raise ValueError(f"Invalid position order: {self.position_order}")

        # permute the image tokens according to the random order
        # 按照token_order给出的顺序，重新排列idx里的元素
        idx = torch.gather(idx.unsqueeze(-1),  # b, s, 1
                           1,
                           token_order.unsqueeze(-1)).squeeze(-1).contiguous()  # [bsz, seq_len]
        targets = torch.gather(targets.unsqueeze(-1),
                               1,
                               token_order.unsqueeze(-1)).squeeze(-1).contiguous()  # [bsz, seq_len]

        # 2. Prepare embeddings and freqs_cis
        self.freqs_cis = self.freqs_cis.to(cond_idx.device)  # (cls_token_num+grid_size**2, head_dim // 2, 2)
        cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[  # 结合CFG，一定概率设置为None Class
                          :, : self.cls_token_num
                          ]  # [bsz, cls_token_num, dim]

        token_embeddings = self.tok_embeddings(idx)  # [bsz, seq_len, dim]
        token_embeddings = self.tok_dropout(token_embeddings)  # [bsz, seq_len, dim]
        position_instruction_tokens = self.get_position_instruction_tokens(token_order)  # 位置指令向量 [bsz, seq_len, dim]

        h = torch.cat(
            (cond_embeddings, interleave_tokens(position_instruction_tokens, token_embeddings)),
            dim=1
        )  # 把随机后的位置指令向量与图片块(x)向量交替插入> instruct[1], x[1], instruct[3], x[3] ...；随后拼接Class token

        token_freqs_cis = self.freqs_cis[self.cls_token_num:].clone().to(token_order.device)[
            token_order]  # (bsz, seq_len, head_dim // 2, 2)    并且顺序调整为token_order
        freqs_cis = torch.cat(
            (self.freqs_cis[:self.cls_token_num].unsqueeze(0).repeat(bs, 1, 1, 1),  # (bsz, 1, head_dim // 2, 2)
             interleave_tokens(token_freqs_cis, token_freqs_cis)),  # 可能是为了配合h已经翻倍seq_len?
            dim=1
        )

        # 3. Forward
        # 训练期间，并没有显式设置MASK
        for layer in self.layers:
            if self.grad_checkpointing:
                # 如果开启梯度激活点，使用checkpoint
                h = checkpoint(layer, h, freqs_cis, input_pos, mask, use_reentrant=False)
            else:
                # freqs_cis用于attention模块中旋转位置编码
                h = layer(h, freqs_cis, input_pos, mask)  # mask=None, input_pos=None

        h = self.norm(h)
        logits = self.output(h).float() # (bsz, seqlen, vocab)
        token_logits = logits[:, self.cls_token_num::2].contiguous()    # 取出非class token 非指令的token

        # 4. Loss computation
        loss = None
        if valid is not None:
            loss_all = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none"
            )
            valid_all = valid[:, None].repeat(1, targets.shape[1]).view(-1)
            loss = (loss_all * valid_all).sum() / max(valid_all.sum(), 1)
        elif targets is not None:
            loss = F.cross_entropy(token_logits.view(-1, token_logits.size(-1)), targets.view(-1))

        return token_logits, loss, token_order

    def forward_inference(self,
                          x: torch.Tensor,
                          freqs_cis: torch.Tensor,
                          input_pos: torch.Tensor):
        """ Args:
            x: [bs, query_num, dim] Input tokens
            freqs_cis: [bs, query_num, n_head, dim // n_head] Frequency embeddings
            input_pos: [query_num] Position index for each token
        """
        bs = x.shape[0]
        # mask = [bs, 1, input_pos, max_seq_length]
        mask = self.causal_mask[:bs, None, input_pos]   # MASK主要用于KV cache
        h = x
        for layer in self.layers:
            h = layer(h, freqs_cis, start_pos=input_pos, mask=mask)
        h = self.norm(h)
        logits = self.output(h).float() # [bsz, seq_len, vocab]
        return logits

    def get_position_instruction_tokens(self, token_order):
        # repeat和view，repeat目的是扩增指令向量，view是适配ROPE是针对单头的
        position_instruct_tokens = self.pos_instruct_embeddings.view(1, 1, self.n_head, self.dim // self.n_head)
        position_instruct_tokens = position_instruct_tokens.repeat(token_order.shape[0], self.block_size, 1,
                                                                   1)  # [batch_size, block_size, n_head, dim // n_head]

        # apply rotary embedding
        position_instruct_freqs_cis = self.freqs_cis[self.cls_token_num:].clone().to(token_order.device)[token_order]
        position_instruct_tokens = batch_apply_rotary_emb(position_instruct_tokens, position_instruct_freqs_cis)
        position_instruct_tokens = position_instruct_tokens.view(token_order.shape[0], self.block_size,
                                                                 self.dim).contiguous()
        return position_instruct_tokens

    def configure_optimizer(
            self, lr, weight_decay, beta1, beta2, max_grad_norm, **kwargs
    ):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]  # 类似矩阵的参数
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]  # 类似bias和norm的仿射参数
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # Create AdamW optimizer and use the fused version if it is available
        import inspect

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        extra_args = dict(fused=True) if fused_available else dict()

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
            **extra_args
        )
        return optimizer

    # large changes to LLaMAGen, directly supporting parallel decoding controlled by num_inference_steps
    def generate(
            self,
            cond: torch.Tensor,                                    # [B] 条件 token（如类别标签）
            token_order: torch.Tensor,                             # [B, 256] 预测顺序（None 则自动生成）
            cfg_scales: Tuple[float, float] = (1.0, 1.0),          # (CFG 起始系数, CFG 结束系数)
            num_inference_steps: int = 88,                         # 推理步数（控制并行度，-1 则纯自回归）
            temperature: float = 1.0,                              # 采样温度
            top_k: int = 0,                                        # Top-K 过滤
            top_p: float = 1.0,                                    # Top-P (Nucleus) 过滤
    ):
        """ Args:
            cond: [bsz, seq_len] Conditional tokens
            token_order: [bsz, seq_len] Position order for each token
            cfg_scales: Tuple (cfg_scale_start, cfg_scale_end) linear cfg scales, set start=end for constant cfg_scale
            num_inference_steps: int Number of inference steps, set to -1 or the number of image tokens to disable parallel decoding
            temperature: float Temperature for sampling
            top_k: int Top-k for sampling
            top_p: float Top-p for sampling
        """
        bs = cond.shape[0]

        # Step-1: Generate the token orders and result sequences
        if token_order is None:                                    # 如果没给顺序，就自己生成
            token_order = torch.arange(self.block_size, device=cond.device)  # [0,1,2,...,255]
            token_order = token_order.unsqueeze(0).repeat(bs, 1)             # 维度[B, 256]

            if self.position_order == "random":                              # 如果是随机顺序，逐行 shuffle
                for i in range(bs):
                    token_order[i] = token_order[i][torch.randperm(self.block_size)]
            token_order = token_order.contiguous()
        else:
            assert token_order.shape == (bs, self.block_size)
        # 创建结果容器 维度[B, 256]，初始全 0，用于存放生成的 Token IDs
        result_indices = torch.zeros((bs, self.block_size), dtype=torch.long, device=cond.device)

        # Step-2: Prepare the freqs_cis and position_instruction_tokens
        position_instruction_tokens = self.get_position_instruction_tokens(token_order)  # 维度[B, 256, dim] 获取 256 个位置指令向量，每个向量经过 ROPE 编码指向不同位置，再根据token_order调整位置
        img_token_freq_cis = self.freqs_cis[self.cls_token_num:].clone().to(token_order.device)[token_order]    # 维度[B, 256, head_dim//2, 2] 获取 256 个图像 token 位置的 ROPE 频率编码，再根据token_order调整位置

        # Step-3: Prepare CFG
        if cfg_scales[-1] > 1.0:                                   # 如果 CFG 系数 > 1，需要生成无条件样本
            cond_null = torch.ones_like(cond) * self.num_classes   # 维度[B] 生成 "Null/Empty" 类别 token
            cond_combined = torch.cat([cond, cond_null])           # 维度[2B] 拼接有条件 和 无条件
            img_token_freq_cis = torch.cat([img_token_freq_cis, img_token_freq_cis])  # 维度[2B, 256, ...]
            position_instruction_tokens = torch.cat([position_instruction_tokens, position_instruction_tokens])  # 维度[2B, 256, dim]
            bs *= 2                                                # batch size 翻倍为 2B
        else:
            cond_combined = cond                                   # 否则直接用 cond
        cond_combined_tokens = self.cls_embedding(cond_combined, train=False)   # 维度[2B, 1, dim] 将类别标签变为 Class Tok

        # Step-4: KV Cache setup
        max_seq_len = cond_combined_tokens.shape[1] + self.block_size * 2   # 最大序列长度 = 1 (Class) + 256 * 2 (指令和图像 token 交替)
        with torch.device(cond.device):
            self.setup_caches(max_batch_size=bs, max_seq_length=max_seq_len, dtype=self.tok_embeddings.weight.dtype)    # 开启KV Cache

        # Step-5: Autoregressive generation with parallel decoding
        if num_inference_steps == -1:
            num_inference_steps = self.block_size                  # 如果 -1，退化为传统逐 token 生成（256 步）

        cur_inference_step = 0                                     # 当前推理步数计数器
        num_query_token_cur_step = 1                               # 当前步要预测几个 token？初始为 1
        query_token_idx_cur_step = 0                               # 从 token_order 的第几个位置开始预测？初始为 0

        # Step 5-1: Prepare the first step
        # [cls_token, query_token_0, ..., query_token_n]
        x = torch.cat([cond_combined_tokens,                       # 维度[2B, 1, dim] Class Token
                    position_instruction_tokens[:, query_token_idx_cur_step: query_token_idx_cur_step + num_query_token_cur_step]],  # 维度[2B, 1, dim] 第一步的位置指令 ([:, 0:1] 指向第 0 个位置)
                    dim=1)                                       # 维度[2B, 2, dim] 作为拼接后作为输入
        # freqs_cis 维度[cls_token_num+grid_size**2, head_dim // 2, 2]
        cur_freqs_cis = torch.cat([self.freqs_cis[:self.cls_token_num].unsqueeze(0).repeat(bs, 1, 1, 1),
                                # 维度[2B, 1, ...] Class Token 的频率编码
                                img_token_freq_cis[:, query_token_idx_cur_step: query_token_idx_cur_step + num_query_token_cur_step]],
                                # 维度[2B, 1, ...] 第 0 个位置图像 Token 的频率编码
                                dim=1)                          # 维度[2B, 2, ...]

        input_pos = torch.arange(0, x.shape[1], device=cond.device)# [0, 1] 作为位置索引

        # Step 5-2: Start the loop
        # 1)如果不开启并行解码，按照训练数据的排列方式进行解码
        # 2.1)如果开启并行解码，那么逐渐增加每一步要解码的token数量，采取1-余弦公式的方式来缓慢增加解码token数量
        # 2.2)并且直接把多个要解码的位置指令向量query token放在末尾，推理获得最后多个img token，随后把query token和img token交替插入，最后添加新的位置指令向量，作为输入，重复循环直至解码完成
        while query_token_idx_cur_step <= self.block_size - num_query_token_cur_step and query_token_idx_cur_step <= self.block_size - 1:   # 直到256
            # Step 5-3: Decode the current step tokens
            # 通过input_pos列表，可以拿出mask的对应部分，以及重新设置KV Cache
            logits = self.forward_inference(x, cur_freqs_cis, input_pos)    # 维度[bsz, seq_len, vocab]  [class_token, query_0] | [img_0, query_1] | [img_1, query_2, query_3]

            # apply CFG
            if cfg_scales[-1] > 1.0:
                cur_cfg_scale = cfg_scales[0] + (cfg_scales[-1] - cfg_scales[0]) * query_token_idx_cur_step / self.block_size
                # 线性变化 CFG 系数
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                # Logits 融合：无条件 + CFG * (有条件 - 无条件) 即 (1 - λ) * uncond_logits + λ * cond_logits
                logits = uncond_logits + cur_cfg_scale * (cond_logits - uncond_logits)

            # query tokens' logits and indices
            logits = logits[:, -num_query_token_cur_step:]  # 维度[bs, query_num, vocab_size] 提取当前步骤需要生成的query_num个位置
            # 临时保存预测的token id的容器
            indices = torch.zeros(result_indices.shape[0], num_query_token_cur_step, dtype=torch.long,
                                  device=cond.device)
            for i in range(num_query_token_cur_step):
                indices[:, i: i + 1] = sample(logits[:, i: i + 1], temperature=temperature, top_k=top_k, top_p=top_p)[0]    # 对每个位置独立采样，得到 Token IDs [B, query_num]

            # 把临时容器的token id 转移到 结果容器 中
            result_indices[:,
            query_token_idx_cur_step: query_token_idx_cur_step + num_query_token_cur_step] = indices.clone()

            img_tokens = self.tok_embeddings(indices)
            # 如果开启CFG
            if cfg_scales[-1] > 1.0:
                # 翻倍Batch size
                img_tokens = torch.cat([img_tokens, img_tokens], dim=0)

            # Step 5-4: Prepare for the next step
            cur_inference_step += 1     # 0 | 1 | 2
            # 计算下一步要生成的token数     # 1 | 2(实际为1，后续x和cur_freqs_cis按照2进行推导) | 3(实际为1，后续x和cur_freqs_cis按照2进行推导)
            num_query_token_next_step = calculate_num_query_tokens_for_parallel_decoding(
                cur_inference_step, num_inference_steps, self.block_size,
                query_token_idx_cur_step, num_query_token_cur_step)

            ########## Important: Prepare the tokens ##########
            # [cur_img_0, cur_query_1, ..., cur_query_n, cur_img_n, next_query_0, ..., next_query_m]
            # [bs, 2, dim] | [bs, 3, dim] | [bs, 6, dim]
            # 计算公式就是step5.3 生成的image token数量 * 2（因为有query token）；-1是因为会继续使用step5.3 中x的第一个query(已保存在KV cache中)
            x = torch.zeros(bs, 2 * num_query_token_cur_step - 1 + num_query_token_next_step, self.dim, dtype=x.dtype,
                            device=cond.device)

            # cur_img_0 | cur_img_1 | cur_img_2
            # 获取step5.3 生成的第1个image token
            x[:, :1] = img_tokens[:, :1]

            # [cur_query_1, ..., cur_query_n]   获取已使用的位置向量 [:, 1:1] | [:, 2:2] | [:, 3:4](query_3)
            # 获取step5.3 中已使用的位置指令向量，通过query_token_idx_cur_step + 1避开step5.3 中x的第一个query
            cur_query_position_instruction_tokens = position_instruction_tokens[:,
                                                    query_token_idx_cur_step + 1: query_token_idx_cur_step + num_query_token_cur_step]
            # 交替插入位置指令向量 [:, 1:1][:, ::2] | [:, 1:1][:, ::2] | [:, 1:3][:, ::2]
            x[:, 1: 2 * num_query_token_cur_step - 1][:, ::2] = cur_query_position_instruction_tokens

            # 交替插入图片向量 [cur_img_1, ..., cur_img_n] [:, 1:1][:, 1::2] | [:, 1:1][:, 1::2] | [:, 1:3][:, 1::2](img_3)
            x[:, 1: 2 * num_query_token_cur_step - 1][:, 1::2] = img_tokens[:, 1: num_query_token_cur_step]

            # [next_query_0, ..., next_query_m]
            # 0 + 1 = 1 | 1 + 1 = 2 | 2 + 2 = 4
            query_token_idx_next_step = query_token_idx_cur_step + num_query_token_cur_step # 计算下一步要从 token_order 的第几个位置开始预测
            # 获取下一步需要的位置指令向量列表 [:, 1: 2] | [:, 2: 4] | [:, 4: 7]
            next_position_instruction_tokens = position_instruction_tokens[:,
                                               query_token_idx_next_step: query_token_idx_next_step + num_query_token_next_step]
            x[:, 2 * num_query_token_cur_step - 1:] = next_position_instruction_tokens  # 放在最后 x = [img_0, query_1] | [img_1, query_2, query_3] | [img_2, query_3, img_3, query_4, query_5, query_6]

            ########## Important: Prepare the freqs_cis ##########
            # [bs, 2, ...] | [bs, 3, ...] | [bs, 6, ...]
            cur_freqs_cis = torch.zeros(
                (bs, 2 * num_query_token_cur_step - 1 + num_query_token_next_step, *self.freqs_cis.shape[-2:]),
                dtype=cur_freqs_cis.dtype, device=cond.device)

            # cur_img_0 | cur_img_1 | cur_img_2
            # img_token_freq_cis [B, 256, head_dim//2, 2]
            # [:, 0:1] | [:, 1:2] | [:, 2:3]
            cur_freqs_cis[:, :1] = img_token_freq_cis[:, query_token_idx_cur_step: query_token_idx_cur_step + 1]

            # [cur_query_1, ..., cur_query_n] 获取已使用的旋转位置编码 [:, 1:1] | [:, 2:2] | [:, 3:4]
            cur_query_freq_cis = img_token_freq_cis[:,
                                 query_token_idx_cur_step + 1: query_token_idx_cur_step + num_query_token_cur_step]
            # 交替插入旋转位置编码 [:, 1:1][:, ::2] | [:, 1:1][:, ::2] | [:, 1:3][:, ::2]
            cur_freqs_cis[:, 1: 2 * num_query_token_cur_step - 1][:, ::2] = cur_query_freq_cis

            # [cur_img_1, ..., cur_img_n]
            # 交替插入旋转位置编码 [:, 1:1][:, 1::2] | [:, 1:1][:, 1::2] | [:, 1:3][:, 1::2]
            cur_freqs_cis[:, 1: 2 * num_query_token_cur_step - 1][:, 1::2] = cur_query_freq_cis

            # [next_query_0, ..., next_query_m]
            # 获取下一步需要的旋转位置编码 [:, 1:2] | [:, 2:4] | [:, 4:7]
            next_freq_cis = img_token_freq_cis[:,
                            query_token_idx_next_step: query_token_idx_next_step + num_query_token_next_step]
            cur_freqs_cis[:, 2 * num_query_token_cur_step - 1:] = next_freq_cis # 放在最后 [:, 1:] | [:, 1:]

            # Step 5-5: Move the query pointer idx
            query_token_idx_cur_step = query_token_idx_next_step    # 0 | 1 | 2 | 4
            if query_token_idx_cur_step > self.block_size:
                break
            # 获取step5.3 中使用到的input_pos中第1个query token所对应的位置(kv cache位置)
            last_input_pos = input_pos[input_pos.shape[0] - num_query_token_cur_step]  # position of cur_query_0  1 | 3 | 5
            # + last_input_pos + 1 代表沿用step5.3 中使用到的input_pos中第1个query token，新的x 在+ last_input_pos + 1这个位置存入kv cache
            # 因此，虽然是通过串联多个query-query-query来并行生成image token，但是每次存入和读取的kv cache中历史token顺序永远都是query - image - query - iamge
            input_pos = torch.arange(2 * num_query_token_cur_step - 1 + num_query_token_next_step, device=cond.device,
                                     dtype=torch.long) + last_input_pos + 1 # [0,1] + 2 = [2,3] | [0,1,2] + 4 = [4,5,6] | [0,1,2,3,4,5] + 6 = [6,7,8,9,10,11]
            num_query_token_cur_step = num_query_token_next_step    # 1 | 1 | 2 | 3

        # Step 6: Return to raster order for tokenizer decoding
        reverse_permutation = torch.argsort(token_order, dim=-1).long().unsqueeze(-1).expand(-1, -1, 1)
        result_indices = torch.gather(result_indices.unsqueeze(-1), 1, reverse_permutation).squeeze(-1)
        return result_indices
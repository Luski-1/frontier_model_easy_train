import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import flash_attn.layers.rotary
import flash_attn
import transformers
import omegaconf
import torch
import typing
import math
from noise import LogLinearNoise
from einops import rearrange

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

#################################################################################
#                                  dropout                                      #
#################################################################################
def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool) -> torch.Tensor:
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)    
    else: # 默认该分支
        out = scale * F.dropout(x, p=prob, training=training) # 设置training模式，即根据dropout放大最终结果 | 利用gate_msa缩放attention结果

    if residual is not None:
        out = residual + out  # 残差相加
    return out

# function overload
def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


@torch.jit.script # 算子融合
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, True)


@torch.jit.script # 算子融合
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, False)


@torch.jit.script # 算子融合
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
    return modulate(x, shift, scale)

#################################################################################
#                                旋转位置编码                                    #
#################################################################################
class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)) # 1 / 10000^[0, 1/32, 2/32, ..., 1]  就是正弦位置编码的频率
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]

        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len     # 缓存
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)  # 获得[0.0, seq_len)的浮点数列表
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone()) # 外积得到矩阵，向下代表不同token位置，向右代表不同频率
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)  # 拼接，向右的频率是[mθ₀, mθ₁, ..., mθ_{d/2‑1}, mθ₀, mθ₁, ..., mθ_{d/2‑1}]
            # dims are: batch, seq_len, qkv, head, dim
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)   # 在qkv维度复制3份
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
            # 不对V进行旋转
            self.cos_cached[:,:,2,:,:].fill_(1.)
            self.sin_cached[:,:,2,:,:].fill_(0.)

        # 外面一般会这样调用，通过把x(qkv)在hidden dim维度平分
        # def apply_rotary(qkv, cos, sin):
        #     # qkv: [B, L, 3, H, D]
        #     x1, x2 = qkv.chunk(2, dim=-1)
        #     x_rot = torch.cat([-x2, x1], dim=-1)
        #     return qkv * cos + x_rot * sin

        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(qkv, cos, sin):
    # flash_attention只需要传入[seq_len, head_dim//2]的cos旋转和sin旋转，内部会自行复制拼接和扩增维度
    cos = cos[0,:,0,0,:cos.shape[-1]//2]
    sin = sin[0,:,0,0,:sin.shape[-1]//2]
    return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)



#################################################################################
#                          关闭仿射变换的LayerNorm                               #
#################################################################################
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x.float(), [self.dim]) # 不开启放射变换
        return x * self.weight[None,None,:]


#################################################################################
#                               时间步的embedding                                #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
        nn.Linear(frequency_embedding_size, hidden_size, bias=True),
        nn.SiLU(),
        nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    def timestep_embedding(self, t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py

        # 就是最常见的正弦位置编码
        half = dim // 2       # 128

        freqs = torch.exp(    # exp( -log(10000) * [0, 1/half, 2/half, ...,1] ) = 1 / 10000^[0, 1/half, 2/half, ...,1]
        - math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half).to(device=t.device)
        
        args = t[:, None].float() * freqs[None] # 外积得到矩阵，向下代表不同时间，向右代表不同频率
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1) # [B, 128*2]
        if dim % 2:   # 非偶数补全
            embedding = torch.cat(
                [embedding,
                torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)  # 将单个数值的时间t转换为时间编码
        t_emb = self.mlp(t_freq)
        return t_emb

#################################################################################
#                               分类c的embedding                                 #
#################################################################################
class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations.
    
    Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, cond_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes

    # TODO think of initializing with 0.02 std deviation like in original DiT paper

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings
    

#################################################################################
#                                Dit Block架构                                  #
#################################################################################


class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads    # 12

        self.norm1 = LayerNorm(dim) # 768 不开启仿射变换
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False) # [768, 3*768]
        self.attn_out = nn.Linear(dim, dim, bias=False) # [768, 768]
        self.dropout1 = nn.Dropout(dropout) # 0.1

        self.norm2 = LayerNorm(dim) # 768
        self.mlp = nn.Sequential(
        nn.Linear(dim, mlp_ratio * dim, bias=True), # [768, 4*768]
        nn.GELU(approximate='tanh'),                # 使用tanh来模拟误差函数
        nn.Linear(mlp_ratio * dim, dim, bias=True)) # [4*768, 768]
        self.dropout2 = nn.Dropout(dropout) # 0.1
        self.dropout = dropout  # 0.1

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True) # [128, 6*128]
        self.adaLN_modulation.weight.data.zero_() # 置零
        self.adaLN_modulation.bias.data.zero_()   # 置零


    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference


    def forward(self, x, rotary_cos_sin, c, seqlens=None):
        """
        x: 输入的文本维度[B,S,D]
        rotary_cos_sin: 根据x得到的旋转cos和旋转sin，维度均为[1,L,3,1,head_dim]，对应为batch, seq_len, qkv, head, dim
        """
        batch_size, seq_len = x.shape[0], x.shape[1]

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        (shift_msa, scale_msa, gate_msa, shift_mlp,
        scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2) # 获取adaLN的仿射变换

        # attention operation
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa) # 执行x * (1 + scale) + shift

        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv,
                        'b s (three h d) -> b s three h d',
                        three=3,
                        h=self.n_heads) # 调整为[B, S, 3(qkv), head, head_dim]
        
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(                   # 把x和cos/sin传递，得到旋转后的x
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
            
            qkv = rearrange(qkv, 'b s ... -> (b s) ...')    # [B*S, head, head_dim]

        if seqlens is None:
            cu_seqlens = torch.arange(                    # 获得[0, seq_len, seq_len*2, ... seq_len*batch_size]的列表
                0, (batch_size + 1) * seq_len, step=seq_len,
                dtype=torch.int32, device=qkv.device)
        else:
            cu_seqlens = seqlens.cumsum(-1)

        x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
        qkv, cu_seqlens, seq_len, 0., causal=False)   # causal=False代表双向Attention，cu_seqlens=累积边界指明， 执行flash attention得到attention结果
        
        x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)  # [B, S, D]

        x = bias_dropout_scale_fn(self.attn_out(x), # W_O的结果
                                None,
                                gate_msa,         # adaLN的仿射变换之一
                                x_skip,           # 原始输入
                                self.dropout)     # 0.1

        # FFN操作，和Attention一致
        x = bias_dropout_scale_fn(
        self.mlp(modulate_fused(
            self.norm2(x), shift_mlp, scale_mlp)),
        None, gate_mlp, x, self.dropout)
        return x


#################################################################################
#                              token embedding架构                              #
#################################################################################
class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]

#################################################################################
#                              output head架构                                  #
#################################################################################
class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)  # 768
        self.linear = nn.Linear(hidden_size, out_channels)  # [768, vocab_size]
        self.linear.weight.data.zero_() # 置零
        self.linear.bias.data.zero_()   # 置零
        # adaLN的变体
        self.adaLN_modulation = nn.Linear(cond_dim,         # 128
                                        2 * hidden_size,  # 768*2
                                        bias=True)
        self.adaLN_modulation.weight.data.zero_() # 置零
        self.adaLN_modulation.bias.data.zero_()   # 置零


    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)   # 执行layernorm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)  # 输出logits
        return x

#################################################################################
#                                    Dit架构                                    #
#################################################################################
class DIT(nn.Module):
    def __init__(self, config, vocab_size: int):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)

        self.config = config
        self.vocab_size = vocab_size

        self.vocab_embed = EmbeddingLayer(config.model.hidden_size,     # [vocab_size, 768] 获得token的embedding
                                        vocab_size)
        self.sigma_map = TimestepEmbedder(config.model.cond_dim)        # cond_dim = 128  获得时间的embedding
        self.rotary_emb = Rotary(
        config.model.hidden_size // config.model.n_heads)               # 768/12=64  对token embeding进行旋转位置编码

        blocks = []
        for _ in range(config.model.n_blocks):
        
            blocks.append(DDiTBlock(config.model.hidden_size, # 768
                                    config.model.n_heads,     # 12
                                    config.model.cond_dim,    # 128
                                    dropout=config.model.dropout))  # 0.1
        self.blocks = nn.ModuleList(blocks)

        self.output_layer = DDitFinalLayer( # 最终映射
        config.model.hidden_size,     # 768
        vocab_size,               
        config.model.cond_dim)        # 128
        

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return  bias_dropout_add_scale_fused_inference

    def forward(self, indices, sigma):
        x = self.vocab_embed(indices)       # 获取token embedding
        c = F.silu(self.sigma_map(sigma))   # 获取时间t embedding

        rotary_cos_sin = self.rotary_emb(x) # 获取旋转的cos和sin

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)  # 遍历dit架构
        x = self.output_layer(x, c)       # 输出最终logits

        return x
  
#################################################################################
#                                文本扩散模型架构                                 #
#################################################################################
class Diffusion(nn.Module):
    def __init__(self, config, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.sampler = self.config["sample"]["predictor"]                       # ddpm_cache
        self.antithetic_sampling = self.config["train"]["antithetic_sampling"]     # True
        self.importance_sampling = self.config["train"]["importance_sampling"]     # False

        if (not hasattr(self.tokenizer, 'mask_token')
            or self.tokenizer.mask_token is None):
            self.mask_index = self.vocab_size                   # 设置mask token
            self.vocab_size += 1                                # vocab siez+1
        else:
            self.mask_index = self.tokenizer.mask_token_id      # 获得mask token的id

        self.backbone = DIT(config, self.vocab_size)

        self.T = self.config["T"]                                 # 0

        self.sampling_eps = self.config["train"]["sampling_eps"]
        self.noise = LogLinearNoise(self.sampling_eps)            # 噪声调度器

        self.lr = self.config["optim"]["lr"]      # 3e-4
        self.time_conditioning = self.config["time_conditioning"]  # False
        self.neg_infinity = -1000000.0

    def inner_forward(self, xt, unet_conditioning):
        """
        提取公共逻辑，方便推理阶段使用
        """
        # 4、sigma与t存在对应关系，准备模型前向计算的前处理
        if unet_conditioning.ndim > 1:
            unet_conditioning = unet_conditioning.squeeze(-1)
        if not self.time_conditioning:  # MDLM默认，返回零向量，因为文献中说明模型无需显式接受时间t作为条件参数，因为时间t已经天然隐含在x_t中，即MASK越多t越大
            unet_conditioning = torch.zeros_like(unet_conditioning)

        # 5、模型前向计算
        logits = self.backbone(xt, unet_conditioning)

        # 6、执行MDLM特有的替换处理，RB1和RB2替换logits（logP）
        logits = self.subs_parameterization(logits, xt)

        return logits

    def forward(self, batch):
        x0 = batch['input_ids']
        attention_mask = batch['attention_mask'] 

        # 1、抽样时间t
        t = self.sample_t(x0.shape[0], x0.device)  # 抽样时间t

        # 2、获取前向加噪概率的相关概率
        sigma, dsigma = self.noise(t) # 作者定义a_t = exp(-σ(t))，sigma就是σ(t)，dsigma就是σ(t)导数
        unet_conditioning = sigma[:, None] # [B, 1]
        move_chance = 1 - torch.exp(-sigma[:, None])  # 获得1-a_t，就是加噪为mask的概率，因为a_t实际就是1-t，那么1-a_t就是t

        # 3、加噪得到xt
        xt = self.q_xt(x0, move_chance) # 加噪得到xt

        # 4-6、最终得到MDLM特有的替换处理后的logits（logP）
        logits = self.inner_forward(xt, unet_conditioning)

        # 7、取出正确token的logP
        log_p_theta = torch.gather(         # 获取真实label对应的logP
            input=logits,
            dim=-1,
            index=x0[:, :, None]).squeeze(-1)
        
        # 8、计算loss
        if self.importance_sampling:
            # 正如importance_sampling_transformation方法中提到，如果开启重要性采样，会把损失的权重从1/t变成-ln(eps)
            # log_p_theta是负数，需要再乘负号，而ln(eps)就是负数，两者相乘即可
            loss = log_p_theta * torch.log(self.sampling_eps)
        else:
            # dsigma = (1 - self.eps) / (1 - (1 - self.eps) * t)
            # 原本sigma = -log(1 - (1 - eps) * t)
            # torch.expm1(sigma) = exp(-log(1 - (1 - eps) * t)) - 1 = 1 / [1 - (1 - eps) * t] - 1 = (1 - eps) * t/ [1 - (1 - eps) * t]
            # dsigma / torch.expm1(sigma) = 1/t
            loss = - log_p_theta * (dsigma / torch.expm1(sigma))[:, None]

        # 9、调整有效范围的loss
        nlls = loss * attention_mask        # 该attention mask用于挑选非padding
        count = attention_mask.sum()

        batch_nll = nlls.sum()
        token_nll = batch_nll / count

        return token_nll
    
    def sample_t(self, n, device):
        """
        n: batch_size
        """
        _eps_t = torch.rand(n, device=device) # 均匀分布抽样时间t

        if self.antithetic_sampling:
            offset = torch.arange(n, device=device) / n # [1, 2/n, ..., (n-1)/n] 作为偏移值
            _eps_t = (_eps_t / n + offset) % 1  # 均匀分布抽样的时间t / n 那么方差缩小N^2，在[0, 0.25)，随后增加偏移值得到每个1/N的小区间都有一个时间t | %1避免超出1

        t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps  # 平移到 [eps, 1]，分布下限很重要

        if self.importance_sampling:
            return self.noise.importance_sampling_transformation(t) # 逻辑就是通过重要性采样把均匀分布的t转换为其他分布的t
        
        return t
    
    def q_xt(self, x, move_chance):

        # 就是从均匀分布抽样z，如果z比MASK概率数值小，就换成MASK
        move_indices = torch.rand(
        * x.shape, device=x.device) < move_chance
        xt = torch.where(move_indices, self.mask_index, x)
        return xt
    
    def subs_parameterization(self, logits, xt):
        """
        logits: 模型预测的结果[B,S,V]
        xt: 加噪结果[B, S]
        """
        logits[:, :, self.mask_index] += self.neg_infinity  # RB2，xt预测为MASK的logit永远为负无穷
        
        logits = logits - torch.logsumexp(logits, dim=-1, # log( exp(x_i) / ∑exp(x...) ) = x_i - log(∑exp(x...)) 直接获取logP
                                        keepdim=True)

        unmasked_indices = (xt != self.mask_index)    # RB1，xt中非MASK位置永远不变
        logits[unmasked_indices] = self.neg_infinity  # 在xt的[B,S]维度中mask位置直接负无穷
        logits[unmasked_indices, xt[unmasked_indices]] = 0  # 在xt的V维度，指定非mask token的logP=0
        return logits

    def sample_categorical(self, categorical_probs):
        """
        常见的gumble_max的使用方式
        u = torch.rand_like(p)
        g = -torch.log(-torch.log(u + eps) + eps)
        return (torch.log(p) + g).argmax(dim=-1)
        """

        gumbel_norm = (
            1e-10
            - (torch.rand_like(categorical_probs) + 1e-10).log())
        return (categorical_probs / gumbel_norm).argmax(dim=-1)

    def sample_prior(self, *batch_dims):
        """
        获取先验分布
        """
        return self.mask_index * torch.ones(*batch_dims, dtype=torch.int64)

    def ddpm_update(self, x, t, dt):
        """
        模型输入显式依赖时间输入
        x: 输入的x_t
        t: 当前时间t，[B,]，浮点数
        dt: 步长
        """

        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)

        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1:
            sigma_s = sigma_s.squeeze(-1)

        assert sigma_t.ndim == 1, sigma_t.shape
        assert sigma_s.ndim == 1, sigma_s.shape

        move_chance_t = 1 - torch.exp(-sigma_t)   # 获取时间t的加噪MASK概率
        move_chance_s = 1 - torch.exp(-sigma_s)   # 获取时间s的加噪MASK概率
        move_chance_t = move_chance_t[:, None, None]
        move_chance_s = move_chance_s[:, None, None]

        unet_conditioning = sigma_t
        log_p_x0 = self.inner_forward(x, unet_conditioning) # 传递时间
        assert move_chance_t.ndim == log_p_x0.ndim

        # 以下部分参考ddpm_caching_update即可
        q_xs = log_p_x0.exp() * (move_chance_t- move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        _x = self.sample_categorical(q_xs)

        copy_flag = (x != self.mask_index).to(x.dtype)
        return copy_flag * x + (1 - copy_flag) * _x 

    def ddpm_caching_update(self, x, t, dt, p_x0=None):
        """
        模型输入不显式依赖时间输入
        x: 输入的x_t
        t: 当前时间t，[B,]，浮点数
        dt: 步长
        p_x0: None or ?

        参考文献A.2.2，模型的后验公式p(z_s | z_t = m) = Cat(z_s | Q_(t|s) @ M(one-hot) * (Q_s)^T @ x_θ / (M(one-hot)^T) @ (Q_t)^T @ @ x_θ)
        参考文献A.2.2，最后化简得到：
        p_θ(z_s = x | z_t = m) = (a_s - a_t) * <x_θ, x> / (a_t * <x_θ, m> + 1 - a_t)
        p_θ(z_s = m | z_t = m) = (a_s * <x_θ, m> + 1 - a_s) / (a_t * <x_θ, m> + 1 - a_t)

        利用RB2和RB1，那么<x_θ, m> = 0，最终化简得到
        p_θ(z_s = x | z_t = m) = (a_s - a_t) * <x_θ, x> / (1 - a_t)
        p_θ(z_s = m | z_t = m) = (1 - a_s) / (1 - a_t)
        """
        sigma_t, _ = self.noise(t)  # 获取sigma

        if t.ndim > 1:
            t = t.squeeze(-1)
        assert t.ndim == 1

        move_chance_t = t[:, None, None]        # 当前时间t，此时也是1 - a_t
        move_chance_s = (t - dt)[:, None, None] # 上一步时间s，此时也是1 - a_s
        assert move_chance_t.ndim == 3, move_chance_t.shape

        if p_x0 is None:
            p_x0 = self.inner_forward(x, sigma_t).exp() # 预测x0的概率情况
        
        assert move_chance_t.ndim == p_x0.ndim

        # move_chance_t - move_chance_s = (1 - a_t) - (1 - a_s) = a_s - a_t
        # q_xs = (a_s - a_t) * <x_θ, x>
        q_xs = p_x0 * (move_chance_t - move_chance_s)
        # 设置mask位置为1 - a_s
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        _x = self.sample_categorical(q_xs)  # gumble_max采样
        
        # p_x0作为缓存，只要_x没有新解码任何一个token，就可以重复使用
        # copy_flag * x + (1 - copy_flag) * _x，就是维持原有非mask的token
        copy_flag = (x != self.mask_index).to(x.dtype)
        return p_x0, copy_flag * x + (1 - copy_flag) * _x

    def semi_ar_sample(self, stride_length, num_strides, dt, n_samples, device):
        """
        这种半自回归的方式并不好，因为首次解码时就确定model.length数量的全部字符
        可以考虑首次的model.length也采取自回归方式，每次确定stride_length数量的字符，剩余部分转为MASK，使得首次解码时需要执行model.length/stride_length次，增加修改机会
        """


        # stride_length=128 | num_strides=1 | dt=1/128 | n_samples=2
        ones = torch.ones(n_samples, dtype=torch.float32,
                        device=device)

        num_steps = int(1 / dt) # 128 + 1 就是总的时间点
        sampling_steps = 0
        intermediate_tokens = []
        target = None
        for _ in range(num_strides + 1):  # 4 + 1
            p_x0_cache = None
            x = self.sample_prior(      # 得到[1, 1024]维度的全mask token id
                n_samples,              # 1
                self.config["model"]["length"]).to(device) # 1024
        
            # 把上一次预测结果的后半段，替换当前MASK序列的前半段
            if target is not None:
                x[:, : -stride_length] = target

            for i in range(num_steps + 1):                # 遍历129次
                # p_x0_cache就是模型预测的x0概率，可以用于缓存
                # x_next是抽样后的状态
                p_x0_cache, x_next = self.ddpm_caching_update(
                x=x, t=(1 - i * dt) * ones, dt=dt, p_x0=p_x0_cache)
                
                # 如果x_next与x非近似相等，或者模型的输入依赖时间条件
                if (not torch.allclose(x_next, x)
                    or self.time_conditioning):
                    # 清空缓存
                    p_x0_cache = None
                    sampling_steps += 1

                x = x_next
            # 再执行一次时间t=0的预测，随后直接取max避免还有mask
            x = self.inner_forward(x, 0 * ones).argmax(dim=-1)

            intermediate_tokens.append(
                x[:, :stride_length].cpu().numpy()) # 第1轮挤出的128个token  → shape [1, 128] | 第2轮挤出的1个token  → shape [1, 128] ...
            target = x[:, stride_length:]
        
        intermediate_tokens.append(target.cpu().numpy())    # 最后的prefix → shape [1, 896]
        intermediate_text_samples = []

        sequence_lengths = ((
            np.concatenate(intermediate_tokens, axis=1)[:, 1:]          # np.concatenate(intermediate_tokens, axis=1)得到[1, 1536]，随后去掉首个token
            == self.tokenizer.eos_token_id).cumsum(-1) == 0).sum(-1)    # == self.tokenizer.eos_token_id 得到属于eos_token_id的布尔数组，cumsum累计就能知道连续为False的片段在哪里
        
        for i in range(2, len(intermediate_tokens) + 1):      # 遍历执行tokenizer解码
            intermediate_text_samples.append(
                self.tokenizer.batch_decode(
                    np.concatenate(intermediate_tokens[:i], axis=1)))
        
        return (sampling_steps, intermediate_text_samples,
                sequence_lengths)

    def default_sample(self, num_steps, n_samples, device, eps=1e-5):
        """MDLM的默认采样方式"""
        # num_steps = 128
        # eps = 1e-5
        
        if num_steps is None:
            num_steps = self.config["sample"]["steps"]

        x = self.sample_prior(   # 获得全为MASK token id 的xt
            n_samples,
            self.config["model"]["length"]).to(device)
        
        timesteps = torch.linspace(1, eps, num_steps + 1, device=device)  # 获得时间点的列表
        dt = (1 - eps) / num_steps  # 获得时间点的间隔
        p_x0_cache = None
        intermediate_tokens = [x]

        for i in range(num_steps):

            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)  # 获取时间t
        
            if self.sampler == 'ddpm':
                x = self.ddpm_update(x, t, dt)
            else:
                p_x0_cache, x_next = self.ddpm_caching_update(x, t, dt, p_x0=p_x0_cache)
                # 如果x_next和x有差异（浮点数），或者模型是时间参数依赖的
                if (not torch.allclose(x_next, x) or self.time_conditioning):
                    # 清空缓存
                    p_x0_cache = None
                x = x_next

            intermediate_tokens.append(x) 
            
        if self.config["sample"]["noise_removal"]:
            # 设置时间0
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)

            unet_conditioning = self.noise(t)[0]  # unet_conditioning = 0
            x = self.inner_forward(x, unet_conditioning).argmax(dim=-1)

        intermediate_tokens.append(x) 

        intermediate_text_samples = []
        for step_x in intermediate_tokens:
            # step_x: [n_samples, seq_len]
            # 把每个 sample 的 1D token id list 取出，传给 batch_decode
            seqs = [step_x[i].tolist() for i in range(step_x.shape[0])]
            decoded = self.tokenizer.batch_decode(seqs, skip_special_tokens=False)
            intermediate_text_samples.append(decoded)

        return num_steps, intermediate_text_samples, self.config["model"]["length"]

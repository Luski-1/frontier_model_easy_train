import math
import typing

import flash_attn
import flash_attn.layers.rotary
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


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


def get_bias_dropout_add_scale(training):
  def _bias_dropout_add(x, bias, scale, residual, prob):
    return bias_dropout_add_scale(
      x, bias, scale, residual, prob, training)

  return _bias_dropout_add


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
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, True)


@torch.jit.script # 算子融合
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, False)


@torch.jit.script # 算子融合
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
  return modulate(x, shift, scale)


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


# function overload
def modulate(x, shift, scale):
  return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                                  Layers                                       #
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


def residual_linear(x, W, x_skip, residual_scale):
  """x_skip + residual_scale * W @ x"""
  dim_out, dim_in = W.shape[0], W.shape[1]
  return torch.addmm(
    x_skip.view(-1, dim_out),
    x.view(-1, dim_in),
    W.T,
    alpha=residual_scale).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
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

  @staticmethod
  def timestep_embedding(t, dim, max_period=10000):
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
#                                 Core Model                                    #
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



class EmbeddingLayer(nn.Module):
  def __init__(self, dim, vocab_dim):
    super().__init__()
    self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
    torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

  def forward(self, x):
    return self.embedding[x]


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


class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
  def __init__(self, config, vocab_size: int):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)

    self.config = config
    self.vocab_size = vocab_size

    self.vocab_embed = EmbeddingLayer(config.model.hidden_size,   # [vocab_size, 768] 获得token的embedding
                                      vocab_size)
    self.sigma_map = TimestepEmbedder(config.model.cond_dim)      # cond_dim = 128  获得时间的embedding
    self.rotary_emb = Rotary(
      config.model.hidden_size // config.model.n_heads)           # 768/12=64  对token embeding进行旋转位置编码

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
    
    self.scale_by_sigma = config.model.scale_by_sigma # True

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
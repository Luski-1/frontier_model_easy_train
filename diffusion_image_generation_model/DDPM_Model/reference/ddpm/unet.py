import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.normalization import GroupNorm


def get_norm(norm, num_channels, num_groups):
    if norm == "in":
        return nn.InstanceNorm2d(num_channels, affine=True)
    elif norm == "bn":
        return nn.BatchNorm2d(num_channels)
    elif norm == "gn":
        return nn.GroupNorm(num_groups, num_channels)
    elif norm is None:
        return nn.Identity()
    else:
        raise ValueError("unknown normalization type")


class PositionalEmbedding(nn.Module):
    __doc__ = r"""Computes a positional embedding of timesteps.

    Input:
        x: tensor of shape (N)
    Output:
        tensor of shape (N, dim)
    Args:
        dim (int): embedding dimension
        scale (float): linear scale to be applied to timesteps. Default: 1.0
    """

    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim      # 128
        self.scale = scale  # 1.0

    def forward(self, x):
        device = x.device                       # x，时间步
        half_dim = self.dim // 2                # 64
        emb = math.log(10000) / half_dim        # 1/64 * log(10000)
        # [0, 63] 区间 / 64 = [0, 63/64] 区间 * -log(10000) = log(10000^[-63/64, 0]) > exp > 10000^[-63/64, 0] = 1 / 10000^[0, 63/64]  | 即向量各维度的位置频率，维度[64]
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)  # 外积，即时间步*位置频率的矩阵，维度[B,64]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1) # transformer的位置编码，维度[B,128]
        return emb


class Downsample(nn.Module):
    __doc__ = r"""Downsamples a given tensor by a factor of 2. Uses strided convolution. Assumes even height and width.

    Input:
        x: tensor of shape (N, in_channels, H, W)
        time_emb: ignored
        y: ignored
    Output:
        tensor of shape (N, in_channels, H // 2, W // 2)
    Args:
        in_channels (int): number of input channels
    """

    def __init__(self, in_channels):
        super().__init__()

        self.downsample = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)
    
    def forward(self, x, time_emb, y):
        if x.shape[2] % 2 == 1:
            raise ValueError("downsampling tensor height should be even")
        if x.shape[3] % 2 == 1:
            raise ValueError("downsampling tensor width should be even")

        return self.downsample(x)


class Upsample(nn.Module):
    __doc__ = r"""Upsamples a given tensor by a factor of 2. Uses resize convolution to avoid checkerboard artifacts.

    Input:
        x: tensor of shape (N, in_channels, H, W)
        time_emb: ignored
        y: ignored
    Output:
        tensor of shape (N, in_channels, H * 2, W * 2)
    Args:
        in_channels (int): number of input channels
    """

    def __init__(self, in_channels):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )
    
    def forward(self, x, time_emb, y):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    __doc__ = r"""Applies QKV self-attention with a residual connection.
    
    Input:
        x: tensor of shape (N, in_channels, H, W)
        norm (string or None): which normalization to use (instance, group, batch, or none). Default: "gn"
        num_groups (int): number of groups used in group normalization. Default: 32
    Output:
        tensor of shape (N, in_channels, H, W)
    Args:
        in_channels (int): number of input channels
    """
    def __init__(self, in_channels, norm="gn", num_groups=32):
        super().__init__()
        
        self.in_channels = in_channels
        self.norm = get_norm(norm, in_channels, num_groups)
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, 1)    # Q K V合并计算
        self.to_out = nn.Conv2d(in_channels, in_channels, 1)        # W_O

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = torch.split(self.to_qkv(self.norm(x)), self.in_channels, dim=1)   # 计算QKV并拆分

        q = q.permute(0, 2, 3, 1).view(b, h * w, c) 
        k = k.view(b, c, h * w)
        v = v.permute(0, 2, 3, 1).view(b, h * w, c)

        dot_products = torch.bmm(q, k) * (c ** (-0.5))      # 缩放点积注意力
        assert dot_products.shape == (b, h * w, h * w)

        attention = torch.softmax(dot_products, dim=-1)     # softmax
        out = torch.bmm(attention, v)                       # QK^T @ V
        assert out.shape == (b, h * w, c)
        out = out.view(b, h, w, c).permute(0, 3, 1, 2)      # [B, C, H, W]

        return self.to_out(out) + x         # W_O + X


class ResidualBlock(nn.Module):
    __doc__ = r"""Applies two conv blocks with resudual connection. Adds time and class conditioning by adding bias after first convolution.

    Input:
        x: tensor of shape (N, in_channels, H, W)
        time_emb: time embedding tensor of shape (N, time_emb_dim) or None if the block doesn't use time conditioning
        y: classes tensor of shape (N) or None if the block doesn't use class conditioning
    Output:
        tensor of shape (N, out_channels, H, W)
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        time_emb_dim (int or None): time embedding dimension or None if the block doesn't use time conditioning. Default: None
        num_classes (int or None): number of classes or None if the block doesn't use class conditioning. Default: None
        activation (function): activation function. Default: torch.nn.functional.relu
        norm (string or None): which normalization to use (instance, group, batch, or none). Default: "gn"
        num_groups (int): number of groups used in group normalization. Default: 32
        use_attention (bool): if True applies AttentionBlock to the output. Default: False
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout,
        time_emb_dim=None,
        num_classes=None,
        activation=F.relu,
        norm="gn",
        num_groups=32,
        use_attention=False,
    ):
        super().__init__()

        self.activation = activation

        self.norm_1 = get_norm(norm, in_channels, num_groups)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm_2 = get_norm(norm, out_channels, num_groups)
        self.conv_2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.time_bias = nn.Linear(time_emb_dim, out_channels) if time_emb_dim is not None else None        # 时间步向量的bias
        self.class_bias = nn.Embedding(num_classes, out_channels) if num_classes is not None else None      # None

        self.residual_connection = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()    # 维度转换
        self.attention = nn.Identity() if not use_attention else AttentionBlock(out_channels, norm, num_groups)
    
    def forward(self, x, time_emb=None, y=None):
        out = self.activation(self.norm_1(x))   # 归一化 > 激活
        oua = self.conv_1(out)                  # 卷积

        if self.time_bias is not None:
            if time_emb is None:
                raise ValueError("time conditioning was specified but time_emb is not passed")
            out += self.time_bias(self.activation(time_emb))[:, :, None, None]      # 时间步归一化 > Linear > 时间步bias

        if self.class_bias is not None:
            if y is None:
                raise ValueError("class conditioning was specified but y is not passed")

            out += self.class_bias(y)[:, :, None, None]

        out = self.activation(self.norm_2(out))                 # 归一化 > 激活
        out = self.conv_2(out) + self.residual_connection(x)    # 分支1：卷积，即残差 | 分支2：x的恒等映射
        out = self.attention(out)                               # Attention

        return out


class UNet(nn.Module):
    __doc__ = """UNet model used to estimate noise.

    Input:
        x: tensor of shape (N, in_channels, H, W)
        time_emb: time embedding tensor of shape (N, time_emb_dim) or None if the block doesn't use time conditioning
        y: classes tensor of shape (N) or None if the block doesn't use class conditioning
    Output:
        tensor of shape (N, out_channels, H, W)
    Args:
        img_channels (int): number of image channels
        base_channels (int): number of base channels (after first convolution)
        channel_mults (tuple): tuple of channel multiplers. Default: (1, 2, 4, 8)
        time_emb_dim (int or None): time embedding dimension or None if the block doesn't use time conditioning. Default: None
        time_emb_scale (float): linear scale to be applied to timesteps. Default: 1.0
        num_classes (int or None): number of classes or None if the block doesn't use class conditioning. Default: None
        activation (function): activation function. Default: torch.nn.functional.relu
        dropout (float): dropout rate at the end of each residual block
        attention_resolutions (tuple): list of relative resolutions at which to apply attention. Default: ()
        norm (string or None): which normalization to use (instance, group, batch, or none). Default: "gn"
        num_groups (int): number of groups used in group normalization. Default: 32
        initial_pad (int): initial padding applied to image. Should be used if height or width is not a power of 2. Default: 0
    """

    def __init__(
        self,
        img_channels,
        base_channels,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=None,
        time_emb_scale=1.0,
        num_classes=None,
        activation=F.relu,
        dropout=0.1,
        attention_resolutions=(),
        norm="gn",
        num_groups=32,
        initial_pad=0,
    ):
        super().__init__()

        self.activation = activation            # silu
        self.initial_pad = initial_pad          # 0

        self.num_classes = num_classes          # None
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(base_channels, time_emb_scale), # 128, 1.0
            nn.Linear(base_channels, time_emb_dim),             # 128 > 512
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),              # 512 > 512
        ) if time_emb_dim is not None else None
    
        self.init_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)   # 初始卷积：图像channel转换为基础channel

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        channels = [base_channels]
        now_channels = base_channels
        # ======== encoder层 ========
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult         # 输出channel

            for _ in range(num_res_blocks):
                self.downs.append(ResidualBlock(
                    now_channels,                       # 输入channel
                    out_channels,                       # 输出channel
                    dropout,                            # dropout率0.1
                    time_emb_dim=time_emb_dim,          # 时间步向量维度512
                    num_classes=num_classes,            # None
                    activation=activation,              # Silu激活函数
                    norm=norm,                          # GroupNorm归一化
                    num_groups=num_groups,              # 32组
                    use_attention=i in attention_resolutions,   # attention_resolutions = (1)，即第一层使用Attention
                ))
                now_channels = out_channels             # 更新channel
                channels.append(now_channels)           # 记录每个module的输出channel
            
            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(now_channels)) # 最后一层不下采样 | 卷积下采样
                channels.append(now_channels)           # 记录每个module的输出channel
        
        # ======== 中间层 ========
        self.mid = nn.ModuleList([
            ResidualBlock(
                now_channels,
                now_channels,
                dropout,
                time_emb_dim=time_emb_dim,
                num_classes=num_classes,
                activation=activation,
                norm=norm,
                num_groups=num_groups,
                use_attention=True,     # 指定开启Attention
            ),
            ResidualBlock(
                now_channels,
                now_channels,
                dropout,
                time_emb_dim=time_emb_dim,
                num_classes=num_classes,
                activation=activation,
                norm=norm,
                num_groups=num_groups,
                use_attention=False,    # 指定开启Attention
            ),
        ])
        # ======== 上采样层 ========
        for i, mult in reversed(list(enumerate(channel_mults))):        # 翻转基础channel的倍数
            out_channels = base_channels * mult

            for _ in range(num_res_blocks + 1):                 # 多一个ResidualBlock是因为相同的下采样层虽然只有num_res_blocks个ResidualBlock，但是会记录num_res_blocks个输出（downsample输出结果）
                self.ups.append(ResidualBlock(
                    channels.pop() + now_channels,              # 既包含上一个模块的输出channel，又包含对应层对应module的输出channel（即encoder直连decoder）
                    out_channels,
                    dropout,
                    time_emb_dim=time_emb_dim,
                    num_classes=num_classes,
                    activation=activation,
                    norm=norm,
                    num_groups=num_groups,
                    use_attention=i in attention_resolutions,
                ))
                now_channels = out_channels
            
            if i != 0:                                      # 最上层不上采样
                self.ups.append(Upsample(now_channels))     # nearest最近邻上采样 
        
        assert len(channels) == 0
        
        self.out_norm = get_norm(norm, base_channels, num_groups)       # 最终归一化
        self.out_conv = nn.Conv2d(base_channels, img_channels, 3, padding=1)    # 最终卷积：基础channel转换为图像channel
    
    def forward(self, x, time=None, y=None):
        ip = self.initial_pad           # padding填充
        if ip != 0:
            x = F.pad(x, (ip,) * 4)

        if self.time_mlp is not None:
            if time is None:
                raise ValueError("time conditioning was specified but tim is not passed")
            
            time_emb = self.time_mlp(time)  # 时间步向量
        else:
            time_emb = None
        
        if self.num_classes is not None and y is None:
            raise ValueError("class conditioning was specified but y is not passed")
        
        x = self.init_conv(x)

        skips = [x]

        for layer in self.downs:
            x = layer(x, time_emb, y)
            skips.append(x)             # 记录每个module输出
        
        for layer in self.mid:
            x = layer(x, time_emb, y)
        
        for layer in self.ups:
            if isinstance(layer, ResidualBlock):
                x = torch.cat([x, skips.pop()], dim=1)  # 上一个module的输出 + 对应层对应module的输出（即encoder直连decoder）
            x = layer(x, time_emb, y)

        x = self.activation(self.out_norm(x))   # 归一化 > 激活
        x = self.out_conv(x)    # 最终卷积 > 图像channel
        
        if self.initial_pad != 0:               # 减去填充长度
            return x[:, :, ip:-ip, ip:-ip]
        else:
            return x
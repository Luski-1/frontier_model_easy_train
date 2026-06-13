from transformers.utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple
from config_face import VAEConfig
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch


# ========== 模型输出定义 ==========
@dataclass
class VAEOutput(ModelOutput):
    """
    模型输出
    """
    loss: Optional[torch.FloatTensor] = None
    reconstruction: torch.FloatTensor = None
    latent_dist: Tuple[torch.FloatTensor, torch.FloatTensor] = None
    z: torch.FloatTensor = None
    loss_dict: Optional[dict] = None


# ========== ResNet残差块 ==========
class ResnetBlock(nn.Module):
    """
    优势：GroupNorm对小batch更稳定（BatchNorm在batch_size<16时统计波动大）
    """

    def __init__(self, in_channels, out_channels, dropout=0.0, num_groups=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 第一层：GroupNorm + SiLU + Conv
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # 第二层：GroupNorm + SiLU + Conv
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # 跳跃连接：当输入与输出通道不匹配时需要投影
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        # 使用SiLU（Swish）
        self.act = nn.SiLU()

    def forward(self, x):
        h = x
        # Pre-Norm操作
        # 第一层：Norm -> Act -> Conv
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv1(h)

        # 第二层：Norm -> Act -> Dropout -> Conv
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # 残差连接：允许梯度直接回传，训练更深网络
        return h + self.shortcut(x)

        # 如果想进一步降低激活值的方差，可以开启下方代码
        # return (h + self.shortcut(x)) * 0.7071  # 1/sqrt(2) ≈ 0.707


# ========== Attention模块（在低分辨率层使用）==========
class AttentionBlock(nn.Module):
    """
    Attention模块，在较低分辨率（如16x16或8x8）添加
    """

    def __init__(self, channels, num_groups=32):
        super().__init__()
        self.channels = channels

        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)  # Q, K, V合并计算  Conv或者MLP都可以
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        residual = x

        # GroupNorm
        h = self.norm(x)

        # 计算QKV并拆分
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        # 关于reshape、permute的转换维度，详情可以查看维度转换.md
        q = q.reshape(B, C, H * W).permute(0, 2, 1)
        k = k.reshape(B, C, H * W).permute(0, 2, 1)
        v = v.reshape(B, C, H * W).permute(0, 2, 1)

        # 缩放点积注意力
        scale = 1.0 / np.sqrt(C)
        attn = torch.bmm(q, k.permute(0, 2, 1)) * scale  # (B, H*W, C) @ (B, C, H*W) > (B, H*W, H*W)
        attn = F.softmax(attn, dim=-1)

        # 应用注意力到V
        h = torch.bmm(attn, v)  # (B, H*W, H*W) @ (B, H*W, C) -> (B, H*W, C)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)

        # 输出投影
        h = self.proj_out(h)

        return residual + h  # 残差连接


# ========== 下采样模块 ==========
class Downsample(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


# ========== 上采样模块 ==========
class Upsample(nn.Module):
    """
    使用Nearest Neighbor插值+卷积进行上采样
    """

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class ConvVAE(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        self.scaling_factor = self.config.scaling_factor  # 如果是SD模型，建议是0.18215；如果是VAE模型，默认是1

        
        ch = config.ch                                              # 基础channel
        target_final_res = getattr(config, 'target_final_res', 8)   # 目标最低分辨率，可选4/8/16

        # 自动计算stage阶段数量
        self.num_downsamples = int(np.log2(config.image_size / target_final_res))
        self.num_downsamples = max(3, self.num_downsamples) + 1  # 至少3层，并且增加1层，因为最后一层不下采样

        # 自动计算channel_mult
        # 1. 如果下采样的次数>channel_mult数量，如果显存充足，可以直接把超出次数的channel设置为channel_mult[-1]或者增加channel_mult
        # 2. 如果下采样的次数>channel_mult数量，如果显存不充足，可以参考以下自动计算方法
        # 3. 如果下采样的次数<channel_mult数量，直接获取channel_mult对应片段即可
        if self.num_downsamples > len(config.channel_mult):
            loop = (self.num_downsamples - len(config.channel_mult)) // (len(config.channel_mult) - 1) + 1          # 得到是多少倍数
            residue = (self.num_downsamples - len(config.channel_mult)) % (len(config.channel_mult) - 1)            # 得到是多少余数
            mults = ([config.channel_mult[0]] +
                     [config.channel_mult[i] for i in range(1, len(config.channel_mult)) for _ in range(loop)] +    # 根据倍数重复当前channel
                     [config.channel_mult[-1] for _ in range(residue)])                                             # 根据余数重复最后channel
        else:
            mults = config.channel_mult[:self.num_downsamples]
        self.channel_mult = tuple(mults)

        self.final_res = config.image_size // (2 ** (self.num_downsamples - 1)) # 最终分辨率
        self.encoder_out_ch = ch * self.channel_mult[-1]                        # 编码器最终输出的channel

        print(f"[VAE] {config.image_size} -> {self.final_res}, "
              f"stages={self.num_downsamples}, mults={self.channel_mult}, "
              f"decode_dim={self.encoder_out_ch * self.final_res ** 2}")

        # ========== 编码器构建（使用ResNet块堆叠）==========
        self.encoder_blocks = nn.ModuleList()
        self.conv_in = nn.Conv2d(config.in_channels, ch, kernel_size=3, padding=1)      # 原始图像 > ConV

        # 构建下采样阶段
        in_ch = ch                      # 起始输入channel
        now_res = config.image_size     # 起始分辨率

        for i_level, mult in enumerate(self.channel_mult):
            out_ch = ch * mult          # 输出channel
            blocks = []                 # 残差块列表

            # 每个阶段堆叠ResNet块
            for i_block in range(config.num_res_blocks):
                blocks.append(ResnetBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    dropout=config.dropout,
                    num_groups=config.num_groups
                ))
                in_ch = out_ch

                # 如果处于开启Attention所指定的分辨率
                if now_res in config.attention_resolutions:
                    blocks.append(AttentionBlock(out_ch, num_groups=config.num_groups))

            down = nn.Module()
            down.blocks = nn.ModuleList(blocks)

            # 添加下采样（最后一个阶段不下采样）
            if i_level != len(self.channel_mult) - 1:
                down.downsample = Downsample(out_ch)
                now_res = now_res // 2
            else:
                down.downsample = nn.Identity()

            self.encoder_blocks.append(down)    # [down(block[ResnetBlock, AttentionBlock...], downsample)]

        # 中间层
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(self.encoder_out_ch, self.encoder_out_ch, config.dropout, config.num_groups)
        self.mid.attn_1 = AttentionBlock(self.encoder_out_ch, config.num_groups)  # 中间层注意力
        self.mid.block_2 = ResnetBlock(self.encoder_out_ch, self.encoder_out_ch, config.dropout, config.num_groups)

        # ========== 潜在空间 ==========
        self.use_gap = getattr(config, 'use_gap', False)
        if self.use_gap:    # 平均池化
            self.fc_mu = nn.Linear(self.encoder_out_ch, config.latent_dim)  # 均值：编码器最终输出的channel > 隐变量的channel
            # self.fc_logvar = nn.Linear(self.encoder_out_ch, config.latent_dim)
            self.fc_scale = nn.Linear(self.encoder_out_ch, config.latent_dim)   # 方差：模型输出log σ^2（配合模型可以输出正值负值），后续exp(log σ^2)容易方差爆炸，后续必须限制
        else:               # 拉平
            self.flatten_dim = self.encoder_out_ch * self.final_res * self.final_res
            self.fc_mu = nn.Linear(self.flatten_dim, config.latent_dim)
            # self.fc_logvar = nn.Linear(self.flatten_dim, config.latent_dim)
            self.fc_scale = nn.Linear(self.flatten_dim, config.latent_dim)

        # ========== 解码器构建（对称结构）==========
        self.fc_decode = nn.Linear(config.latent_dim, self.encoder_out_ch * self.final_res * self.final_res)

        # ========== 中间层（对称结构）==========
        self.mid_dec = nn.Module()
        self.mid_dec.block_1 = ResnetBlock(self.encoder_out_ch, self.encoder_out_ch,
                                           config.dropout, config.num_groups)
        self.mid_dec.attn_1 = AttentionBlock(self.encoder_out_ch, config.num_groups)
        self.mid_dec.block_2 = ResnetBlock(self.encoder_out_ch, self.encoder_out_ch,
                                           config.dropout, config.num_groups)

        # ========== 解码器（对称结构）==========
        self.decoder_blocks = nn.ModuleList()
        in_ch = self.encoder_out_ch
        now_res = self.final_res

        for i_level in reversed(range(len(self.channel_mult))):     # 翻转channel倍数
            out_ch = ch * self.channel_mult[i_level]
            blocks = []

            # 每层的解码器比编码器多一次循环
            for _ in range(config.num_res_blocks + 1):
                blocks.append(ResnetBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    dropout=config.dropout,
                    num_groups=config.num_groups
                ))
                in_ch = out_ch

                if now_res in config.attention_resolutions:
                    blocks.append(AttentionBlock(out_ch, num_groups=config.num_groups))

            up = nn.Module()
            up.blocks = nn.ModuleList(blocks)

            # 添加上采样（最后一个阶段不上采样）
            if i_level != 0:
                up.upsample = Upsample(out_ch)
                now_res *= 2
            else:
                up.upsample = nn.Identity()

            self.decoder_blocks.append(up)

        # 最终输出层
        self.norm_out = nn.GroupNorm(num_groups=config.num_groups, num_channels=ch)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(ch, 3, 3, padding=1)

        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # fan_in 当前层的输入维度 或 输入神经元总数 来决定参数随机化时的标准差
            # fan_out 当前层的输出维度 或 输出神经元总数 来决定参数随机化时的标准差
            # Kaiming初始化适合ReLU/SiLU激活函数
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.GroupNorm):
            # GroupNorm的标准初始化
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def encode(self, x):
        # 初始卷积
        h = self.conv_in(x)

        # 编码器各阶段：残差块 -> [注意力] -> 残差块 -> [注意力] -> 下采样
        for down in self.encoder_blocks:
            for block in down.blocks:
                h = block(h)
            h = down.downsample(h)

        # 中间层
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # 全局平均池化
        if self.use_gap:
            h = h.mean(dim=[2, 3])  # (B, C, H, W) -> (B, C)
        else:
        # 拉平
            h = h.reshape(h.size(0), -1)  # (B, C*H*W)

        # 在潜在空间计算时，强制使用 FP32（防止 FP16 指数溢出）
        with torch.amp.autocast("cuda", dtype=torch.float32):

            # 在训练VAE期间，如果采取VAE默认代码（logvar = self.fc_scale(h)），那么最难处理的就是隐变量方差💥
            # 假设如果raw_scale直接=log σ^2，模型训练初期输出值范围偏大，很容易就导致σ^2 = exp(log σ^2) = ∞
            raw_scale = self.fc_scale(h)  # 无界，log σ^2可以是任意实数

            # 经过多次尝试，调整为以下方差限制
            # Softplus：log(1+exp(x))，输出 > 0，且增长缓慢（线性而非指数）[0, ~10]
            # 1e-4：防止 std=0
            std = F.softplus(raw_scale) + 1e-4
            logvar = torch.log(std ** 2)


            # 如果还是方差💥，请选择以下任一备选方案
            # 【备选方案1】Sigmoid 约束：严格限制 std ∈ (0, 5)，适合对稳定性要求极高时
            # std = 5 * torch.sigmoid(raw_scale) + 1e-4  # 最大 std=5，对应 logvar≈3.2

            # 【备选方案2】方差永远固定

            # 【备选方案3】在 forward 中立即截断，不让大数值流向下游
            # 方式1：Hard Clamp（梯度会消失，但数值安全）
            # logvar = torch.clamp(logvar, -20, 20)

            # 【备选方案4】在 forward 中限制过大
            # 方式2：Soft Clamp（tanh，梯度平滑）
            # logvar = 20 * torch.tanh(logvar / 20)

            mu = self.fc_mu(h) * self.scaling_factor  # VAE模型，默认不需要缩放 mu；SD模型，可以缩放 mu

            return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # 潜在空间投影并恢复空间维度
        h = self.fc_decode(z)
        # 恢复为4x4空间维度（假设经过4次下采样：128->64->32->16->8，你可以根据实际调整）
        h = h.reshape(h.size(0), self.encoder_out_ch, self.final_res, self.final_res)

        # 解码器中间层
        h = self.mid_dec.block_1(h)
        h = self.mid_dec.attn_1(h)
        h = self.mid_dec.block_2(h)

        # 解码器各阶段：残差块 -> [注意力] -> 残差块 -> [注意力] -> 上采样
        for up in self.decoder_blocks:
            for block in up.blocks:
                h = block(h)
            h = up.upsample(h)

        # 最终归一化和输出
        h = self.norm_out(h)
        h = self.act_out(h)
        h = self.conv_out(h)
        return torch.tanh(h)  # 限制到 [-1, 1]

    def forward(
            self,
            pixel_values: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            return_loss: bool = True,
            **kwargs
    ) -> VAEOutput:
        if labels is None:
            labels = pixel_values

        mu, logvar = self.encode(pixel_values)
        z = self.reparameterize(mu, logvar)

        z = z / self.scaling_factor # 反缩放encoder输出的mu
        recon = self.decode(z)

        loss_dict = None
        if return_loss:
            loss_dict = self.compute_loss(recon, labels, mu, logvar)

        return VAEOutput(
            loss=loss_dict['loss'] if loss_dict else None,
            reconstruction=recon,
            latent_dist=(mu, logvar),
            z=z,
            loss_dict=loss_dict
        )

    def compute_loss(self, recon, x, mu, logvar):
        batch_size = x.size(0)
        # 图像重建损失，默认MSE
        if self.config.recon_loss_type == "mse":
            recon_loss = F.mse_loss(recon, x, reduction='sum') / batch_size
        else:
            recon_loss = F.binary_cross_entropy(recon, x, reduction='sum') / batch_size

        # KL 散度计算建议在 FP32 中进行，防止 exp(logvar) 溢出
        with torch.amp.autocast("cuda", dtype=torch.float32):
            # 将参与指数运算的变量转为 FP32
            mu_f32 = mu.float()
            logvar_f32 = logvar.float()

            # 如果还是训练期间方差💥，可以尝试打开以下注释
            # logvar_f32_clamped = torch.clamp(logvar_f32, -20, 20)

            # 即使 logvar=20，exp(20) 在 FP32 中也能表示（约 4.8e8）
            # 而在 FP16 中 exp(20) 会直接溢出为 inf
            # 当然 也可以选择 BF16
            kld = -0.5 * torch.sum(
                1 + logvar_f32 - mu_f32.pow(2) - logvar_f32.exp()
            ) / batch_size

            # 转回 FP16
            kld = kld.to(recon.dtype)

        total_loss = recon_loss + self.config.kld_weight * kld

        # ✅ 返回字典
        return {
            'loss': total_loss,
            'recon_loss': recon_loss.detach(),
            'kld_loss': kld.detach(),
            'kld_weight': torch.tensor(self.config.kld_weight, device=recon.device)
        }
    


import torch.nn.functional as F
import torch.nn as nn
import torch
import math



# ---------- sigma embedding ----------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: sigma，维度[B]
        """
        device = t.device
        half = self.dim // 2
        # 获取transformer位置编码的频率矩阵[1, dim/2] exp(log(10000) * - [0, 255] / 255) = 10000 ^(- [0, 255] / 255) = 1 / 10000 ^ (- [0, 255] / 255)
        emb = torch.exp(torch.arange(half, device=device) * -(math.log(10000) / (half - 1)))
        # 构建时间-频率矩阵[B, dim/2]
        emb = t[:, None].float() * emb[None, :]
        # 构建时间-频率（sin+cos）矩阵[B, dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# ---------- Enhanced UNet with Residual Blocks + Attention ----------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),                                 # 归一化
            nn.SiLU(),                                              # 激活
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)      # 卷积
        )

        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),                              # 激活
            nn.Linear(time_emb_dim, out_ch)         # mlp
        )

        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),                                # 归一化
            nn.SiLU(),                                              # 激活
            nn.Dropout(dropout),                                    # 正则化
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)     # 卷积
        )
        # 残差分支，当入参和出参的channel不一致时使用1*1卷积改变通道，当入参和出参的channel一致时直接穿透
        self.residual_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        residual = self.residual_conv(x)        # 恒等映射
        h = self.conv1(x)
        t_emb = self.time_emb_proj(t_emb)
        h = h + t_emb[:, :, None, None]         # 相加
        h = self.conv2(h)                       # 残差
        return h + residual


# ---------- Self-Attention Layer ----------
class SelfAttention2D(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, in_channels)
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(B, self.num_heads, C // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W)

        attn = torch.softmax(torch.matmul(q.transpose(-2, -1), k) / math.sqrt(C // self.num_heads), dim=-1)
        out = torch.matmul(attn, v.transpose(-2, -1)).transpose(-2, -1)
        out = out.contiguous().view(B, C, H, W)
        out = self.proj_out(out)
        return x + out


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, num_blocks=2, downsample=True, use_attention=False):
        """
        in_ch: 输入通道
        out_ch: 输出入通道
        time_emb_dim: sigma向量
        num_blocks: 残差块
        downsample: 是否开启下采样
        use_attention: 是否开启Attention
        """
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualBlock(in_ch if i == 0 else out_ch, out_ch, time_emb_dim)
            for i in range(num_blocks)
        ])
        self.attn = SelfAttention2D(out_ch) if use_attention else nn.Identity()
        self.downsample = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1) if downsample else nn.Identity()

    def forward(self, x, t_emb):
        skips = []
        for block in self.blocks:       # 收集每个残差块的输出
            x = block(x, t_emb)
            skips.append(x)
        x = self.attn(x)
        x = self.downsample(x)
        return x, skips


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, num_blocks=2, upsample=True, use_attention=False):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1) if upsample else nn.Identity()
        self.blocks = nn.ModuleList([
            ResidualBlock(in_ch + out_ch, out_ch, time_emb_dim)     # in_ch是当前层的x输入，out_ch是对应的下采样层的输出
            for _ in range(num_blocks)
        ])
        self.attn = SelfAttention2D(out_ch) if use_attention else nn.Identity()

    def forward(self, x, skips, t_emb):
        x = self.upsample(x)                                # 先上采样
        for block in self.blocks:
            if skips:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, t_emb)
        x = self.attn(x)
        return x


class MidBlock(nn.Module):
    def __init__(self, channels, time_emb_dim, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualBlock(channels, channels, time_emb_dim)
            for _ in range(num_blocks)
        ])
        self.attn = SelfAttention2D(channels)

    def forward(self, x, t_emb):
        for block in self.blocks:
            x = block(x, t_emb)
        x = self.attn(x)
        return x


# ---------- Full Enhanced UNet with Attention ----------
class EnhancedUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=128, time_emb_dim=512, num_res_blocks=2):
        """
        in_ch: 输入图像的通道数
        base_ch: 转换后的基础通道数
        time_emb_dim: sigma参数的dim
        num_res_blocks: 每一层残差模块的数量
        """
        super().__init__()
        # sigma模块
        self.sigma_mlp = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        # 图像初始转换模块
        self.init_conv = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)

        # 下采样模块
        self.down1 = DownBlock(base_ch, base_ch, time_emb_dim, num_res_blocks, downsample=False, use_attention=False)
        self.down2 = DownBlock(base_ch, base_ch * 2, time_emb_dim, num_res_blocks, downsample=True, use_attention=False)
        self.down3 = DownBlock(base_ch * 2, base_ch * 4, time_emb_dim, num_res_blocks, downsample=True, use_attention=True)
        self.down4 = DownBlock(base_ch * 4, base_ch * 8, time_emb_dim, num_res_blocks, downsample=True, use_attention=True)

        # 中间层模块
        self.mid = MidBlock(base_ch * 8, time_emb_dim, num_res_blocks * 2)

        # 上采样模块
        self.up4 = UpBlock(base_ch * 8, base_ch * 4, time_emb_dim, num_res_blocks, upsample=True, use_attention=True)
        self.up3 = UpBlock(base_ch * 4, base_ch * 2, time_emb_dim, num_res_blocks, upsample=True, use_attention=True)
        self.up2 = UpBlock(base_ch * 2, base_ch, time_emb_dim, num_res_blocks, upsample=True, use_attention=False)
        self.up1 = UpBlock(base_ch, base_ch, time_emb_dim, num_res_blocks, upsample=False, use_attention=False)

        # 最终转换模块
        self.final = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, in_ch, kernel_size=3, padding=1)
        )

    def forward(self, x, t):
        t_emb = self.sigma_mlp(t)
        x = self.init_conv(x)

        skips = []
        x, s1 = self.down1(x, t_emb)
        skips.extend(s1)
        x, s2 = self.down2(x, t_emb)
        skips.extend(s2)
        x, s3 = self.down3(x, t_emb)
        skips.extend(s3)
        x, s4 = self.down4(x, t_emb)
        skips.extend(s4)

        x = self.mid(x, t_emb)

        x = self.up4(x, skips, t_emb)
        x = self.up3(x, skips, t_emb)
        x = self.up2(x, skips, t_emb)
        x = self.up1(x, skips, t_emb)

        return self.final(x)


# ---------- 损失函数 ----------
def anneal_dsm_score_estimation(model, x, sigma_index, sigmas_list, anneal_power=2.):
    """
    :param model: 模型
    :param x: 输入数据   [B, C, H, W]
    :param sigma_index: sigma下标   [B]
    :param sigmas_list: sigma等比序列   [10]
    :param anneal_power: 2.0
    :return:
    """
    sigmas_list = sigmas_list.to(x.device)
    # 得到图像维度的具体sigma
    used_sigmas = sigmas_list[sigma_index].view(x.shape[0], *([1] * len(x.shape[1:])))  # [B, C, H, W]
    # 加噪
    perturbed_samples = x + torch.randn_like(x) * used_sigmas   # x_t = x_0 + ε * σ  ; ε ~ N(0, I)
    # 真实目标
    target = (- 1 / (used_sigmas ** 2) * (perturbed_samples - x)).detach()  # score: - (x_t - x_0) / σ^2
    # 预测目标
    scores = model(perturbed_samples, sigma_index)  # label是sigma级别，在模型中就是分类信息

    # 1) 1 / 2. * ((scores - target) ** 2) 即模型预测与目标score的MSE损失
    # 2.1) used_sigmas.squeeze() ** 2 即score的分子x_t - x_0是σ级别，score的分母是σ^2级别，分子/分母=1/σ级别
    # 2.2) MSE损失是平方，|scores - target|^2是1/σ^2级别，那么σ越小导致损失越大，因此需要×上used_sigmas.squeeze() ** anneal_power
    loss = F.mse_loss(scores, target, reduction='none')  # [B, C, H, W]
    loss = (loss.sum(dim=[1, 2, 3]) * 0.5) * (used_sigmas.squeeze() ** anneal_power)
    return loss.mean()
import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import copy


# ---------- β的线性获取 ----------
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


# ---------- 时间步 embedding ----------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim          # 128

    def forward(self, t):
        device = t.device       # t 时间步
        half = self.dim // 2    # 64
        # [0, 63] 区间 / 63 = [0, 63/63] 区间 * -log(10000) = log(10000^[-63/63, 0]) > exp > 10000^[-63/63, 0] = 1 / 10000^[0, 63/63]  | 即向量各维度的位置频率，维度[64]
        emb = torch.exp(torch.arange(half, device=device) * -(math.log(10000) / (half - 1)))
        emb = t[:, None].float() * emb[None, :]                     # 外积，即时间步*位置频率的矩阵，维度[B,64]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)   # transformer的位置编码，维度[B,128]
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# ---------- 残差Module ----------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, label_emb_dim, group_num=16, dropout=0.1):
        super().__init__()
        # 卷积+channel调整
        self.conv1 = nn.Sequential(
            nn.GroupNorm(group_num, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        )
        # 时间分支
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        # ⭐⭐⭐ 增加类别分支
        self.label_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(label_emb_dim, out_ch)
        )
        # channel维持
        self.conv2 = nn.Sequential(
            nn.GroupNorm(group_num, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        )
        # 残差分支，当入参出参的channel不一致时使用1*1卷积改变通道，当入参出参的channel一致时直接恒等映射
        self.residual_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    # ⭐⭐⭐ 增加类别信息的传入
    def forward(self, x, t_emb, l_emb):
        residual = self.residual_conv(x)    # 恒等映射
        h = self.conv1(x)                   # 卷积1
        t_emb = self.time_emb_proj(t_emb)   # 时间向量
        l_emb = self.label_emb_proj(l_emb)  # 类别向量
        h = h + t_emb[:, :, None, None]     # 相加
        h = h + l_emb[:, :, None, None]     # 相加
        h = self.conv2(h)                   # 卷积2
        return h + residual


# ---------- Self-Attention Layer ----------
class SelfAttention2D(nn.Module):
    def __init__(self, in_channels, group_num=16, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(group_num, in_channels)
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)   # Q K V 合并计算
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        # 以H*W作为tokens，channel作为hidden_dim，num_heads=4，C//num_heads = head_dim
        q = q.view(B, self.num_heads, C // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W)
        # 常见的缩放点积注意力计算
        attn = torch.softmax(torch.matmul(q.transpose(-2, -1), k) / math.sqrt(C // self.num_heads), dim=-1)
        out = torch.matmul(attn, v.transpose(-2, -1)).transpose(-2, -1)
        out = out.contiguous().view(B, C, H, W)
        out = self.proj_out(out)
        return x + out


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, label_emb_dim, num_blocks=2, downsample=True, use_attention=False):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ResidualBlock(in_ch if i == 0 else out_ch, out_ch, time_emb_dim, label_emb_dim) for i in range(num_blocks)])  # 首个ResidualBlock的输入通道是in_ch，后续的输入通道都是out_ch
        self.attn = SelfAttention2D(out_ch) if use_attention else nn.Identity()
        self.downsample = nn.Sequential(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)) if downsample else nn.Identity() # 不改变channel

    # ⭐⭐⭐ 增加类别信息的传入
    def forward(self, x, t_emb, l_emb):
        skips = []
        for block in self.blocks:
            x = block(x, t_emb, l_emb)
            skips.append(x)             # 每个ResBlock输出都收集
        x = self.attn(x)                # 简化，不收集
        x = self.downsample(x)          # 简化，不收集
        return x, skips


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_chs, time_emb_dim, label_emb_dim, num_blocks=3, upsample=True, use_attention=False):
        super().__init__()

        self.blocks = nn.ModuleList()
        prev_out_ch = in_ch
        for i in range(num_blocks):
            block_in = prev_out_ch + skip_chs[i]
            # 第0个Residual Block接收上一层的输出以及对应DownBlock的对应Residual Block
            # 其余Residual Block接收上一个Residual Block(out_ch通道数的输出)以及对应DownBlock的对应Residual Block
            self.blocks.append(ResidualBlock(block_in, out_ch, time_emb_dim, label_emb_dim))   
            prev_out_ch = out_ch
        self.attn = SelfAttention2D(out_ch) if use_attention else nn.Identity()
        self.upsample = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1) if upsample else nn.Identity()   # 不改变channel

    # ⭐⭐⭐ 增加类别信息的传入
    def forward(self, x, skips, t_emb, l_emb):
        for block in self.blocks:
            x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, t_emb, l_emb)
        x = self.attn(x)
        x = self.upsample(x)
        return x


class MidBlock(nn.Module):
    def __init__(self, channels, time_emb_dim, label_emb_dim, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualBlock(channels, channels, time_emb_dim, label_emb_dim)
            for _ in range(num_blocks)
        ])
        self.attn = SelfAttention2D(channels)

    # ⭐⭐⭐ 增加类别信息的传入
    def forward(self, x, t_emb, l_emb):
        for block in self.blocks:
            x = block(x, t_emb, l_emb)
        x = self.attn(x)
        return x


# ---------- Unet模型 ----------
class EnhancedUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=128, time_emb_dim=512, num_res_blocks=2, group_num=16, label_nums=1000, label_emb_dim=512):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        # ⭐⭐⭐ 增加类别信息的embedding
        self.class_mlp = nn.Sequential(
            nn.Embedding(num_embeddings=label_nums + 1, embedding_dim=base_ch),
            nn.Linear(base_ch, label_emb_dim),
            nn.SiLU(),
            nn.Linear(label_emb_dim, label_emb_dim)
        )

        self.init_conv = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)            # 初始卷积，将图像channel转换为base channel

        # encoder层
        # ⭐⭐⭐ 增加类别信息的dim
        self.down1 = DownBlock(base_ch, base_ch, time_emb_dim, label_emb_dim, num_res_blocks, downsample=True, use_attention=False)
        self.down2 = DownBlock(base_ch, base_ch * 2, time_emb_dim, label_emb_dim, num_res_blocks, downsample=True, use_attention=False)
        self.down3 = DownBlock(base_ch * 2, base_ch * 4, time_emb_dim, label_emb_dim, num_res_blocks, downsample=True, use_attention=True)
        self.down4 = DownBlock(base_ch * 4, base_ch * 8, time_emb_dim, label_emb_dim, num_res_blocks, downsample=False, use_attention=True)

        # Middle block
        # ⭐⭐⭐ 增加类别信息的dim
        self.mid = MidBlock(base_ch * 8, time_emb_dim, label_emb_dim, num_res_blocks * 2)

        # decoder层
        # ⭐⭐⭐ 增加类别信息的dim
        self.up4 = UpBlock(base_ch * 8, base_ch * 4, skip_chs=[base_ch * 8] * 2, time_emb_dim=time_emb_dim, label_emb_dim=label_emb_dim, num_blocks=2, upsample=True, use_attention=True)
        self.up3 = UpBlock(base_ch * 4, base_ch * 2, skip_chs=[base_ch * 4] * 2, time_emb_dim=time_emb_dim, label_emb_dim=label_emb_dim, num_blocks=2, upsample=True, use_attention=True)
        self.up2 = UpBlock(base_ch * 2, base_ch,     skip_chs=[base_ch * 2] * 2, time_emb_dim=time_emb_dim, label_emb_dim=label_emb_dim, num_blocks=2, upsample=True, use_attention=False)
        self.up1 = UpBlock(base_ch,     base_ch,     skip_chs=[base_ch] * 3,     time_emb_dim=time_emb_dim, label_emb_dim=label_emb_dim, num_blocks=3, upsample=False, use_attention=False)

        self.final = nn.Sequential(                                 # 最终卷积，将base channel转换为图像channel
            nn.GroupNorm(group_num, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, in_ch, kernel_size=3, padding=1)
        )

    # ⭐⭐⭐ 增加类别信息的传入
    def forward(self, x, t, label):
        t_emb = self.time_mlp(t)
        l_emb = self.class_mlp(label)
        x = self.init_conv(x)

        skips = [x]                     # 收集init_conv输出
        x, s1 = self.down1(x, t_emb, l_emb)
        skips.extend(s1)                # 2 skips: base_ch
        x, s2 = self.down2(x, t_emb, l_emb)
        skips.extend(s2)                # 2 skips: 2 * base_ch
        x, s3 = self.down3(x, t_emb, l_emb)
        skips.extend(s3)                # 2 skips: 4 * base_ch
        x, s4 = self.down4(x, t_emb, l_emb)
        skips.extend(s4)                # 2 skips: 8 * base_ch

        x = self.mid(x, t_emb, l_emb)

        x = self.up4(x, skips, t_emb, l_emb)
        x = self.up3(x, skips, t_emb, l_emb)
        x = self.up2(x, skips, t_emb, l_emb)
        x = self.up1(x, skips, t_emb, l_emb)

        return self.final(x)
    
# ---------- 整体模型 ----------
class Diffusion(nn.Module):

    def __init__(self, model: nn.Module, T=1000, w=1.8):
        super().__init__()
        # 训练模型
        self.model = model
        # EMA模型
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
        # ⭐⭐⭐ 增加类别信息的权重
        self.w = w

        betas = linear_beta_schedule(T)             # 线性递增的β
        self.register_buffer('betas', betas)        # β
        self.register_buffer('alphas', 1.0 - betas) # α = 1 - β 随时间越来越小
        self.register_buffer('alpha_cumprod', torch.cumprod(self.alphas, dim=0))    # alpha_bar_t 随时间越来越小
        self.register_buffer('alpha_cumprod_prev',
                             torch.cat([torch.tensor([1.0], dtype=betas.dtype), self.alpha_cumprod[:-1]]))  # alpha_bar_t-1
        self.register_buffer('sqrt_alpha_cumprod', torch.sqrt(self.alpha_cumprod))                  # √(alpha_bar_t)    前向加噪中控制图像比例
        self.register_buffer('sqrt_one_minus_alpha_cumprod', torch.sqrt(1.0 - self.alpha_cumprod))  # √(1 - alpha_bar_t) 前向加噪中控制噪声比例，随时间越来越大
        self.register_buffer('posterior_variance',
                             self.betas * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)) # (1 - αt)(1 - alpha_bar_t-1) / (1 - alpha_bar_t) 去噪生成中理论方差

    # ---------- 前向加噪过程 ----------
    def q_sample(self, x_start, t, noise=None):
        """
        前向加噪过程: x_t = √(alpha_bar_t) * x_0 +  √(1 - alpha_bar_t) * ε | 随时间越来越模糊
        x_start: 原始图像(B,C,H,W) in [-1,1]
        t: 时间步(B,)in [0,T-1]
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise, noise

    # ---------- MSE损失 ----------
    # ⭐⭐⭐ 增加类别信息的传入
    def p_losses(self, x_start, t, label):
        x_noisy, noise = self.q_sample(x_start, t)
        predicted_noise = self.model(x_noisy, t, label)
        loss = F.mse_loss(predicted_noise, noise, reduction='mean')
        return loss
    
    # 直接返回损失即可
    # ⭐⭐⭐ 增加类别信息的传入
    def forward(self, x, t, label):
        return self.p_losses(x, t, label)


    # ---------- DDPM先验抽样 ----------
    @torch.no_grad()
    # ⭐⭐⭐ 增加类别信息的传入
    def p_sample(self, x_t, t, label):
        """
        去噪过程 x_t at timestep t -> x_{t-1}
        去噪公式: x_t-1 =  [1 / √(αt)] * [x_t - (1 - αt) / √(1 - alpha_bar_t) * ε(模型预测噪声) ] + σ * z (z ~ N(0, I))
        均值: [1 / √(αt)] * [x_t - (1 - αt) / √(1 - alpha_bar_t) * ε(模型预测噪声) ]
        方差: σ^2 = (1 - αt)(1 - alpha_bar_t-1) / (1 - alpha_bar_t)
        """
        # 确保 t 是标量
        if isinstance(t, torch.Tensor):
            t = t.item()

        # 当前步的参数
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alpha_cumprod[t]

        # 预测带条件的噪声 ε    | 建议使用EMA模型
        pred_noise_cond = self.ema_model(x_t, torch.full((x_t.size(0),), t, dtype=torch.long, device=x_t.device), label)

        # 生成无条件
        label = torch.full((x_t.size(0),), self.model.class_mlp[0].num_embeddings - 1,
                         dtype=torch.long, device=x_t.device)

        # 预测无条件的噪声 ε    | 建议使用EMA模型
        pred_noise_uncond = self.ema_model(x_t, torch.full((x_t.size(0),), t, dtype=torch.long, device=x_t.device), label)

        # 使用CFG公式(1+w)*P(x|y) - w*P(x)，即带条件的预测 - 无条件的预测 = 增强条件信息
        pred_noise = (1 + self.w) * pred_noise_cond - self.w * pred_noise_uncond

        # ---- 计算均值 μ ----
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)

        mu = (1.0 / sqrt_alpha_t) * (
                x_t - ((1.0 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * pred_noise
        )

        # ---- 计算方差 σ² ----
        sigma2 = self.posterior_variance[t]
        sigma = torch.sqrt(sigma2)

        # ---- 采样 x_{t-1} ----
        # 如果最后一步，均值就是概率密度最高的
        # 如果非最后一步，需要从分布中抽样
        if t == 0:
            return mu
        else:
            # 重参数化
            noise = torch.randn_like(x_t, dtype=x_t.dtype)
            return mu + sigma.view(1, 1, 1, 1) * noise

from utils import get_time_discretization
from typing import Optional
import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import copy


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
    def __init__(self, in_ch, out_ch, time_emb_dim, group_num=16, dropout=0.1):
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
        # channel维持
        self.conv2 = nn.Sequential(
            nn.GroupNorm(group_num, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        )
        # 残差分支，当入参出参的channel不一致时使用1*1卷积改变通道，当入参出参的channel一致时直接恒等映射
        self.residual_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        residual = self.residual_conv(x)    # 恒等映射
        h = self.conv1(x)                   # 卷积1
        t_emb = self.time_emb_proj(t_emb)   # 时间向量
        h = h + t_emb[:, :, None, None]     # 相加
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
    def __init__(self, in_ch, out_ch, time_emb_dim, num_blocks=2, downsample=True, use_attention=False):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ResidualBlock(in_ch if i == 0 else out_ch, out_ch, time_emb_dim) for i in range(num_blocks)])  # 首个ResidualBlock的输入通道是in_ch，后续的输入通道都是out_ch
        self.attn = SelfAttention2D(out_ch) if use_attention else nn.Identity()
        self.downsample = nn.Sequential(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)) if downsample else nn.Identity() # 不改变channel

    def forward(self, x, t_emb):
        skips = []
        for block in self.blocks:
            x = block(x, t_emb)
            skips.append(x)             # 每个ResBlock输出都收集
        x = self.attn(x)                # 简化，不收集
        x = self.downsample(x)          # 简化，不收集
        return x, skips

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_chs, time_emb_dim, num_blocks=3, upsample=True, use_attention=False):
        super().__init__()

        self.blocks = nn.ModuleList()
        prev_out_ch = in_ch
        for i in range(num_blocks):
            block_in = prev_out_ch + skip_chs[i]
            # 第0个Residual Block接收上一层的输出以及对应DownBlock的对应Residual Block
            # 其余Residual Block接收上一个Residual Block(out_ch通道数的输出)以及对应DownBlock的对应Residual Block
            self.blocks.append(ResidualBlock(block_in, out_ch, time_emb_dim))   
            prev_out_ch = out_ch
        self.attn = SelfAttention2D(out_ch) if use_attention else nn.Identity()
        self.upsample = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1) if upsample else nn.Identity()   # 不改变channel

    def forward(self, x, skips, t_emb):
        for block in self.blocks:
            x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, t_emb)
        x = self.attn(x)
        x = self.upsample(x)
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

# ---------- Unet模型 ----------
class EnhancedUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=128, time_emb_dim=512, num_res_blocks=2, group_num=16):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.init_conv = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)            # 初始卷积，将图像channel转换为base channel

        # encoder层
        self.down1 = DownBlock(base_ch, base_ch, time_emb_dim, num_res_blocks, downsample=True, use_attention=False)
        self.down2 = DownBlock(base_ch, base_ch * 2, time_emb_dim, num_res_blocks, downsample=True, use_attention=False)
        self.down3 = DownBlock(base_ch * 2, base_ch * 4, time_emb_dim, num_res_blocks, downsample=True, use_attention=True)
        self.down4 = DownBlock(base_ch * 4, base_ch * 8, time_emb_dim, num_res_blocks, downsample=False, use_attention=True)

        # Middle block
        self.mid = MidBlock(base_ch * 8, time_emb_dim, num_res_blocks * 2)

        # decoder层
        self.up4 = UpBlock(base_ch * 8, base_ch * 4, skip_chs=[base_ch * 8] * 2, time_emb_dim=time_emb_dim, num_blocks=2, upsample=True, use_attention=True)
        self.up3 = UpBlock(base_ch * 4, base_ch * 2, skip_chs=[base_ch * 4] * 2, time_emb_dim=time_emb_dim, num_blocks=2, upsample=True, use_attention=True)
        self.up2 = UpBlock(base_ch * 2, base_ch,     skip_chs=[base_ch * 2] * 2, time_emb_dim=time_emb_dim, num_blocks=2, upsample=True, use_attention=False)
        self.up1 = UpBlock(base_ch,     base_ch,     skip_chs=[base_ch] * 3,     time_emb_dim=time_emb_dim, num_blocks=3, upsample=False, use_attention=False)

        self.final = nn.Sequential(                                 # 最终卷积，将base channel转换为图像channel
            nn.GroupNorm(group_num, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, in_ch, kernel_size=3, padding=1)
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = self.init_conv(x)

        skips = [x]                     # 收集init_conv输出
        x, s1 = self.down1(x, t_emb)
        skips.extend(s1)                # 2 skips: base_ch
        x, s2 = self.down2(x, t_emb)
        skips.extend(s2)                # 2 skips: 2 * base_ch
        x, s3 = self.down3(x, t_emb)
        skips.extend(s3)                # 2 skips: 4 * base_ch
        x, s4 = self.down4(x, t_emb)
        skips.extend(s4)                # 2 skips: 8 * base_ch

        x = self.mid(x, t_emb)

        x = self.up4(x, skips, t_emb)
        x = self.up3(x, skips, t_emb)
        x = self.up2(x, skips, t_emb)
        x = self.up1(x, skips, t_emb)

        return self.final(x)

class FlowMatching(nn.Module):

    def __init__(self, model: nn.Module, steps=200, decay: Optional[float] = None):
        super().__init__()
        # 训练模型
        self.model: nn.Module = model
        # 如果开启EMA模型
        if decay:
            self.ema_model = copy.deepcopy(model)
            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad_(False)

        self.steps = steps
        self.decay = decay

    def forward(self, x, t):
        result = self.model(x, t)
        return result

    
    def update_ema(self, first_phase=True):

        # 阶段1：前2000步硬拷贝（预热期）
        if first_phase:
            # 前2000步直接复制参数
            with torch.no_grad():
                for ema_p, model_p in zip(self.ema_model.parameters(), self.model.parameters()):
                    ema_p.copy_(model_p)
        else:
            with torch.no_grad():
                for current_params, ema_params in zip(self.model.parameters(), self.ema_model.parameters()):
                    ema_params.data = ema_params.data * self.decay + (1 - self.decay) * current_params.data

    @torch.no_grad()
    def sample_flow(self, image_size: int, device: torch.device, n_samples=8, edm_eval=True):
        self.model.eval()
        # 抽样 x ~ N(0,1)
        x = torch.randn(n_samples, 3, image_size, image_size, device=device)

        # 生成完整时间序列 (N+1 个点, 定义 N 个区间, 递增: 0→1)
        if edm_eval:
            # get_time_discretization 已返回递增的 t: [~0.012, ..., 0.998, 1.0]
            t_seq = get_time_discretization(self.steps, device)
        else:
            t_seq = torch.linspace(0.0, 1.0, self.steps + 1, device=device)

        for i in range(self.steps):
            # 取出当前步时间并扩展到batch维度
            t = t_seq[i].expand(n_samples)
            # 实际步长: t_seq 有 N+1 个点, 第 i 步从 t_seq[i] 走到 t_seq[i+1]
            dt = t_seq[i + 1] - t_seq[i]
            # 如果开启EMA模型
            if self.decay:
                u = self.ema_model(x, t)
            else:
                u = self(x, t)
            x = x + u * dt
        self.model.train()
        return x.clamp(-1,1)
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from tqdm import tqdm
import os  # 用于在保存采样结果前确保输出目录存在

import torch
import torch.nn as nn

# 定义基础下采样模块（无分辨率变化，仅特征提取）
# 输入: ic (输入通道数), oc (输出通道数)
# 结构: 3个5x5卷积层 + GroupNorm + LeakyReLU
blk = lambda ic, oc: nn.Sequential(
    nn.Conv2d(ic, oc, 5, padding=2),    # 5x5卷积，保持分辨率 (H,W) 不变
    nn.GroupNorm(oc // 8, oc),          # GroupNorm，分组数为 oc//8
    nn.LeakyReLU(),                     # 激活函数
    nn.Conv2d(oc, oc, 5, padding=2),    # 第二个5x5卷积，保持分辨率
    nn.GroupNorm(oc // 8, oc),
    nn.LeakyReLU(),
    nn.Conv2d(oc, oc, 5, padding=2),    # 第三个5x5卷积，保持分辨率
    nn.GroupNorm(oc // 8, oc),
    nn.LeakyReLU(),
)

# 定义上采样模块（末尾含转置卷积，分辨率x2）
# 输入: ic (输入通道数), oc (输出通道数)
# 结构: 与blk类似，最后增加 ConvTranspose2d 进行上采样
blku = lambda ic, oc: nn.Sequential(
    nn.Conv2d(ic, oc, 5, padding=2),   # 5x5卷积，保持分辨率
    nn.GroupNorm(oc // 8, oc),
    nn.LeakyReLU(),
    nn.Conv2d(oc, oc, 5, padding=2),   # 第二个5x5卷积
    nn.GroupNorm(oc // 8, oc),
    nn.LeakyReLU(),
    nn.Conv2d(oc, oc, 5, padding=2),   # 第三个5x5卷积
    nn.GroupNorm(oc // 8, oc),
    nn.LeakyReLU(),
    nn.ConvTranspose2d(oc, oc, 2, stride=2),  # 转置卷积，分辨率x2 (H,W)->(2H,2W)
    nn.GroupNorm(oc // 8, oc),
    nn.LeakyReLU(),
)

# 想理解上面的代码，需要先理解F.conv_transpose2d（本人感觉目前主流都是使用torch.nn.functional.interpolate进行插值上采样）
# F.conv_transpose2d有两种理解方式
# a)
# 对x（2*2）的每个元素，单独逐元素相乘kernal（假设3*3），那么每个元素变成3*3大小
# 随后根据stride进行排布，例如处于[0,0]的元素要霸占[0-2, 0-2]位置，处于[0,1]的的元素要霸占[0-2,1-3]位置，隔了stride步。
# 同理[1,0]的元素要霸占[1-3,0-2]位置，[1,1]的元素要霸占[1-3,1-3]位置
# 重叠位置的数值进行相加即可
# b)
# 在每个元素之间插入stride-1个0
# 旋转卷积核180°
# 两边各做kernal_size - 1 - padding的填充0向量，随后进行stride固定=1的普通卷积
# PS：padding在F.conv_transpose2d的参数中是代表要裁剪两边各多少元素，output_padding是仅在最后结果的右边/下边填充多少0元素
# 还要理解卷积和转置卷积的分辨率计算公式
# 卷积 = (input_size + 2 * padding - kernal_size) / 2
# 转置卷积 = (input_size - 1) * stride - 2 * padding + kernal_size  + output_padding


class DummyX0Model(nn.Module):
    """
    一个用于预测 x0 的 U-Net 风格模型，结合了 Transformer 层、条件嵌入和时间嵌入
    输入:
        - x: 输入图像 (B, C, H, W)
        - t: 时间步 (B,)
        - cond: 条件标签 (B,)，用于类别条件生成
    输出:
        - y: 预测的 logits (B, C, H, W, N)，N 是离散化的类别数
    """
    def __init__(self, n_channel: int, N: int = 16) -> None:
        super(DummyX0Model, self).__init__()
        # 下采样路径（编码器）
        self.down1 = blk(n_channel, 16)     # 第1个下采样块，通道 16
        self.down2 = blk(16, 32)            # 第2个下采样块，通道 32
        self.down3 = blk(32, 64)            # 第3个下采样块，通道 64
        self.down4 = blk(64, 512)           # 第4个下采样块，通道 512
        self.down5 = blk(512, 512)          # 第5个下采样块，通道 512

        # 上采样路径（解码器）
        self.up1 = blku(512, 512)           # 第1个上采样块，通道 512，分辨率x2
        self.up2 = blku(512 + 512, 64)      # 第2个上采样块，输入是 concat(x4, y)，通道 64，分辨率x2
        self.up3 = blku(64, 32)             # 第3个上采样块，通道 32，分辨率x2
        self.up4 = blku(32, 16)             # 第4个上采样块，通道 16，分辨率x2

        # 最终卷积层
        self.convlast = blk(16, 16)        # 最终特征提取块
        self.final = nn.Conv2d(16, N * n_channel, 1, bias=False)  # 1x1卷积，输出通道 N*C

        # Transformer 编码器层（用于全局特征建模）
        self.tr1 = nn.TransformerEncoderLayer(d_model=512, nhead=8)  # 处理 512 维特征
        self.tr2 = nn.TransformerEncoderLayer(d_model=512, nhead=8)  # 处理 512 维特征
        self.tr3 = nn.TransformerEncoderLayer(d_model=64, nhead=8)   # 处理 64 维特征

        # 条件嵌入（类别嵌入，用于条件生成）
        self.cond_embedding_1 = nn.Embedding(10, 16)   # 对应 down1 输出通道 16
        self.cond_embedding_2 = nn.Embedding(10, 32)   # 对应 down2 输出通道 32
        self.cond_embedding_3 = nn.Embedding(10, 64)   # 对应 down3 输出通道 64
        self.cond_embedding_4 = nn.Embedding(10, 512)  # 对应 down4 输出通道 512
        self.cond_embedding_5 = nn.Embedding(10, 512)  # 对应 up1 输出通道 512
        self.cond_embedding_6 = nn.Embedding(10, 64)   # 对应 up2 输出通道 64

        # 时间嵌入（将时间步映射到特征空间）
        self.temb_1 = nn.Linear(32, 16)   # 对应 down1 输出通道 16
        self.temb_2 = nn.Linear(32, 32)   # 对应 down2 输出通道 32
        self.temb_3 = nn.Linear(32, 64)   # 对应 down3 输出通道 64
        self.temb_4 = nn.Linear(32, 512)  # 对应 down4 输出通道 512

        self.N = N  # 离散化的类别数

    def forward(self, x, t, cond) -> torch.Tensor:
        """
        x和t还是需要通过除最大值方式转换为连续型数据，cond通过embedding的方式转换为连续型数据
        """

        # 输入归一化：将 x 从 [0, N-1] 尽量映射到 [-1, 1)，当N越大，上限越接近1
        x = (2 * x.float() / self.N) - 1.0

        # 时间步归一化：t 从 [0, 999] 映射到 [0, 1)
        t = t.float().reshape(-1, 1) / 1000

        # 计算时间特征：使用 sin/cos 位置编码（类似 Transformer 的位置编码）
        # 生成 16 个 sin 和 16 个 cos 特征，拼接成 32 维
        # 该方法为傅里叶特征映射（Fourier Features）：i小（低频）：对t缓慢变化敏感，捕捉长期、宏观时间趋势；i大（高频）：t微小扰动就会剧烈改变输出，捕捉瞬时、短时突变。
        t_features = [torch.sin(t * 3.1415 * 2**i) for i in range(16)] + [
            torch.cos(t * 3.1415 * 2**i) for i in range(16)
        ]
        tx = torch.cat(t_features, dim=1).to(x.device)  # (B, 32)

        # 计算时间嵌入：将 32 维时间特征映射到各层对应的通道数，并扩展为 4D 张量
        t_emb_1 = self.temb_1(tx).unsqueeze(-1).unsqueeze(-1)  # (B, 16, 1, 1)
        t_emb_2 = self.temb_2(tx).unsqueeze(-1).unsqueeze(-1)  # (B, 32, 1, 1)
        t_emb_3 = self.temb_3(tx).unsqueeze(-1).unsqueeze(-1)  # (B, 64, 1, 1)
        t_emb_4 = self.temb_4(tx).unsqueeze(-1).unsqueeze(-1)  # (B, 512, 1, 1)

        # 计算条件嵌入：将类别标签映射到各层对应的通道数，并扩展为 4D 张量
        cond_emb_1 = self.cond_embedding_1(cond).unsqueeze(-1).unsqueeze(-1)  # (B, 16, 1, 1)
        cond_emb_2 = self.cond_embedding_2(cond).unsqueeze(-1).unsqueeze(-1)  # (B, 32, 1, 1)
        cond_emb_3 = self.cond_embedding_3(cond).unsqueeze(-1).unsqueeze(-1)  # (B, 64, 1, 1)
        cond_emb_4 = self.cond_embedding_4(cond).unsqueeze(-1).unsqueeze(-1)  # (B, 512, 1, 1)
        cond_emb_5 = self.cond_embedding_5(cond).unsqueeze(-1).unsqueeze(-1)  # (B, 512, 1, 1)
        cond_emb_6 = self.cond_embedding_6(cond).unsqueeze(-1).unsqueeze(-1)  # (B, 64, 1, 1)

        # 下采样路径（编码器）
        x1 = self.down1(x) + t_emb_1 + cond_emb_1  # (B, 16, H, W)，加入时间和条件嵌入
        x2 = self.down2(nn.functional.avg_pool2d(x1, 2)) + t_emb_2 + cond_emb_2  # (B, 32, H/2, W/2)，先下采样再卷积
        x3 = self.down3(nn.functional.avg_pool2d(x2, 2)) + t_emb_3 + cond_emb_3  # (B, 64, H/4, W/4)，先下采样再卷积
        x4 = self.down4(nn.functional.avg_pool2d(x3, 2)) + t_emb_4 + cond_emb_4  # (B, 512, H/8, W/8)，先下采样再卷积
        x5 = self.down5(nn.functional.avg_pool2d(x4, 2))  # (B, 512, H/16, W/16)，先下采样再卷积，无时间/条件嵌入

        # 第1个 Transformer 层：处理 x5 的全局特征
        # 将 4D 张量 reshape 为 3D (B, SeqLen, Dim)，其中 SeqLen = H/16 * W/16，Dim = 512
        # 维度转换的目的是，模型把输入当做离散文本[B,L,D]，因此需要把channel维度转换到最后维
        x5 = (
            self.tr1(x5.reshape(x5.shape[0], x5.shape[1], -1).transpose(1, 2))
            .transpose(1, 2)
            .reshape(x5.shape)
        )  # 输出形状与 x5 相同 (B, 512, H/16, W/16)

        # 上采样路径（解码器）
        y = self.up1(x5) + cond_emb_5  # (B, 512, H/8, W/8)，上采样后加入条件嵌入

        # 第2个 Transformer 层：处理 y 的全局特征
        y = (
            self.tr2(y.reshape(y.shape[0], y.shape[1], -1).transpose(1, 2))
            .transpose(1, 2)
            .reshape(y.shape)
        )  # 输出形状与 y 相同 (B, 512, H/8, W/8)

        # 拼接 x4 和 y（跳跃连接），然后上采样
        y = self.up2(torch.cat([x4, y], dim=1)) + cond_emb_6  # (B, 64, H/4, W/4)，输入通道 512+512=1024

        # 第3个 Transformer 层：处理 y 的全局特征
        y = (
            self.tr3(y.reshape(y.shape[0], y.shape[1], -1).transpose(1, 2))
            .transpose(1, 2)
            .reshape(y.shape)
        )  # 输出形状与 y 相同 (B, 64, H/4, W/4)

        # 继续上采样
        y = self.up3(y)  # (B, 32, H/2, W/2)
        y = self.up4(y)  # (B, 16, H, W)

        # 最终卷积层
        y = self.convlast(y)  # (B, 16, H, W)
        y = self.final(y)     # (B, N*C, H, W)，输出 logits

        # Reshape 为 (B, C, H, W, N)，便于后续计算 loss（如交叉熵）
        y = (
            y.reshape(y.shape[0], -1, self.N, *x.shape[2:])  # (B, C, N, H, W)
            .transpose(2, -1)  # (B, C, H, W, N)
            .contiguous()
        )

        return y


class D3PM(nn.Module):
    def __init__(
        self,
        x0_model: nn.Module,
        n_T: int,
        num_classes: int = 10,
        forward_type="uniform",
        hybrid_loss_coeff=0.001,
    ) -> None:
        super(D3PM, self).__init__()
        self.x0_model = x0_model

        self.n_T = n_T  # 1000
        self.hybrid_loss_coeff = hybrid_loss_coeff  # 用于控制KL散度损失的占比

        steps = torch.arange(n_T + 1, dtype=torch.float64) / n_T    # [0, 1]
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2)   # 余弦噪声调度，越来越小，因为域在[0, π/2]范围，因此值从1>0
        self.beta_t = torch.minimum(            # βt是用来构建每个时刻的转移矩阵，越来越大
            1 - alpha_bar[1:] / alpha_bar[:-1], torch.ones_like(alpha_bar[1:]) * 0.999
        ) # 1 - a_t / a_t-1 越来越趋近1，通过minimum限制最高只能0.999

        # self.beta_t = [1 / (self.n_T - t + 1) for t in range(1, self.n_T + 1)]
        self.eps = 1e-6
        self.num_classses = num_classes     # 2
        q_onestep_mats = []
        q_mats = []  # these are cumulative
        # 设置不同时刻的转移矩阵
        for beta in self.beta_t:

            if forward_type == "uniform":
                mat = torch.ones(num_classes, num_classes) * beta / num_classes     # 构建N*N的矩阵，以1 / K * β_t填充
                mat.diagonal().fill_(1 - (num_classes - 1) * beta / num_classes)    # 设置对角线是 1 - (K - 1) / K * β_t
                q_onestep_mats.append(mat)  # 累加，共1000个
            else:
                raise NotImplementedError
        q_one_step_mats = torch.stack(q_onestep_mats, dim=0)    # 堆叠，[1000, 2, 2]

        q_one_step_transposed = q_one_step_mats.transpose(      # 转置，用于后续计算
            1, 2
        )  # this will be used for q_posterior_logits
        # 计算连乘的转移矩阵
        q_mat_t = q_onestep_mats[0]
        q_mats = [q_mat_t]
        for idx in range(1, self.n_T):
            q_mat_t = q_mat_t @ q_onestep_mats[idx]
            q_mats.append(q_mat_t)
        q_mats = torch.stack(q_mats, dim=0) # Q_t_bar 即Q_t转移矩阵连乘
        self.logit_type = "logit"

        # register
        self.register_buffer("q_one_step_transposed", q_one_step_transposed)
        self.register_buffer("q_mats", q_mats)

        assert self.q_mats.shape == (
            self.n_T,
            num_classes,
            num_classes,
        ), self.q_mats.shape

    def _at(self, a, t, x):
        """
        a: 连乘的转移矩阵Q_t_bar[1000,2,2]
        t: 时间t[B]
        x: 数据[B,C,H,W]

        a[t - 1, x, :]，代表先对t减1【因为要从0下标开始】，按顺序遍历t矩阵[B,C,H,W]【因为会跟随x自动扩增】的每一个元素，作为Q_t_bar的0维【第几个Q_t_bar】
        再按顺序遍历x矩阵[B,C,H,W]的每一个像素，作为作为Q_t_bar的1维【第几行】

        即这个时间t所对应的Q_t_bar矩阵，然后对x的每一个像素，都取出随后取出这个像素对应的转移概率行，维度是[B,C,H,W,N]
        
        也等价于，对x进行one-hot化，随后与Q_t_bar相乘

        return: [B,C,H,W,N]，连续型数据
        """
        # t is 1-d, x is integer value of 0 to num_classes - 1

        a = a.to(x.device)
        t = t.to(x.device)
        bs = t.shape[0]             # 获取batch_size维度
        t = t.reshape((bs, *[1] * (x.dim() - 1)))   # 维度[B,1,1,1]
        # out[i, j, k, l, m] = a[t[i, j, k, l], x[i, j, k, l], m]
        return a[t - 1, x, :]

    def q_posterior_logits(self, x_0, x_t, t):
        """
        q(x_t-1 | x_t, x0) = q(x_t | x_t-1, x0) * q(x_t-1 | x0) / q(xt | x0)

        0.
        转移矩阵的每一行代表该类型【行号】进行转移到各种类型的概率【横和=1】
        转移矩阵的每一列代表所有类型转移到指定类型【列号】的数值，不是概率【因为列和!=1】

        1.
        因为马尔科夫性质，q(x_t | x_t-1, x0) = q(x_t | x_t-1)，
        因为未知是x_t-1，已知的是x_t，所以要利用x_t与转移矩阵的列【转置后是行】相乘得到所有x_t-1转移到x_t的score【此时不是概率！！！】，即x_t@(Q_t)^T

        2.
        q(x_t-1 | x0)就是x0@Q_t-1_bar

        3. 
        q(x_t | x_t-1, x0) * q(x_t-1 | x0) = x_t@(Q_t)^T ⊙ x0@Q_t-1_bar

        4.
        q(xt | x0)是x0 @ Q_t_bar @ x_t^T，其中x0 @ Q_t_bar代表获取从x0转移到x_t的各种情况的概率，但是实际上x_t已知，所以再@ x_t^T
        但是实际上q(xt | x0)=∑x_t-1 q(x_t | x_t-1, x0) * q(x_t-1 | x0)，即是分子项的配分函数，此时才是真正的概率

        return: log( q(x_t | x_t-1, x0) * q(x_t-1 | x0) ) = log ( x_t@(Q_t)^T ⊙ x0@Q_t-1_bar )
        """

        # if t == 1, this means we return the L_0 loss, so directly try to x_0 logits.
        # otherwise, we return the L_{t-1} loss.
        # Also, we never have t == 0.

        if x_0.dtype == torch.int64 or x_0.dtype == torch.int32:
            x_0_logits = torch.log(
                torch.nn.functional.one_hot(x_0, self.num_classses) + self.eps  # 转换为one-hot[B,C,H,W,N]，并且+eps避免log0
            )
        else:
            x_0_logits = x_0.clone()

        assert x_0_logits.shape == x_t.shape + (self.num_classses,), print(
            f"x_0_logits.shape: {x_0_logits.shape}, x_t.shape: {x_t.shape}"
        )

        # Here, we caclulate equation (3) of the paper. Note that the x_0 Q_t x_t^T is a normalizing constant, so we don't deal with that.

        # fact1 is "guess of x_{t-1}" from x_t
        # fact2 is "guess of x_{t-1}" from x_0

        fact1 = self._at(self.q_one_step_transposed, t, x_t)    # 获得x_t@(Q_t)^T [B,C,H,W,N]

        softmaxed = torch.softmax(x_0_logits, dim=-1)  # softmax包含exp，与之前的log抵消【+ self.eps，影响不大】，实际上还是one-hot，维度[B,C,H,W,N]
        qmats2 = self.q_mats[t - 2].to(dtype=softmaxed.dtype, device=softmaxed.device)   # Q_t-1_bar，维度[N,N]

        # x_0 @ Q_t-1_bar
        # softmaxed [b,c,h,w,n] qmats2 [b,n,n] -> fact2 [b,c,h,w,n]
        fact2 = torch.einsum("b...c,bcd->b...d", softmaxed, qmats2)

        # log(x*y) = log(x) + log(y) 
        # x_0@Q_t_bar^T * x_0 @ Q_t-1_bar
        # +eps避免log0
        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)

        t_broadcast = t.reshape((t.shape[0], *[1] * (x_t.dim())))
        # if t == 1, this means we return the L_0 loss, so directly try to x_0 logits.otherwise, we return the L_{t-1} loss.
        # 如果t=1【即最开始的时刻】，直接返回x_0_logits，即x_0的log真实概率
        # 否则直接返回log(x*y)
        bc = torch.where(t_broadcast == 1, x_0_logits, out)

        return bc

    def vb(self, dist1, dist2):
        """
        dist1: 真实的log(x_t@(Q_t)^T ⊙ x0@Q_t-1_bar)
        dist2: 预测的log(x_t@(Q_t)^T ⊙ x0@Q_t-1_bar)

        softmax含有exp，会对消log，随后进行求和归一化，即对x_t@(Q_t)^T ⊙ x0@Q_t-1_bar求和【即∑x_t-1 q(x_t | x_t-1, x0) * q(x_t-1 | x0)】
        随后x_t@(Q_t)^T ⊙ x0@Q_t-1_bar中每一个值都除以总和，得到真正的概率

        KL散度公式=P(x) * log(P(x) / Q(x))
        """

        # flatten dist1 and dist2
        dist1 = dist1.flatten(start_dim=0, end_dim=-2)  # 拉平，等价于单独计算每个像素
        dist2 = dist2.flatten(start_dim=0, end_dim=-2)
        # softmax中exp()抵消之前的log，q * log(q) / log(p)
        out = torch.softmax(dist1 + self.eps, dim=-1) * (   # 得到真正的概率
            torch.log_softmax(dist1 + self.eps, dim=-1)     # 得到真正的概率再log
            - torch.log_softmax(dist2 + self.eps, dim=-1)   # 得到真正的概率再log
        )
        return out.sum(dim=-1).mean()

    def q_sample(self, x_0, t, noise):
        """
        x_0: 图像数据，离散的整数
        t: 时间t，来自[1, 1000]
        noise: 噪声，来自均匀分布抽样

        执行gumble-max抽样法，返回[B,C,H,W]的整数型数据
        """
        # forward process, x_0 is the clean input.
        logits = torch.log(self._at(self.q_mats, t, x_0) + self.eps)    # logits：未归一化的logist概率  | + self.eps避免log0
        noise = torch.clip(noise, self.eps, 1.0)        # 扰动 | 截断避免Log0
        gumbel_noise = -torch.log(-torch.log(noise))    # 标准 Gumbel 分布采样公式。
        return torch.argmax(logits + gumbel_noise, dim=-1)  # 扰动后的分数取最大值，gumble-max抽样法

    def model_predict(self, x_0, t, cond):
        # this part exists because in general, manipulation of logits from model's logit
        # so they are in form of x_0's logit might be independent to model choice.
        # for example, you can convert 2 * N channel output of model output to logit via get_logits_from_logistic_pars
        # they introduce at appendix A.8.

        predicted_x0_logits = self.x0_model(x_0, t, cond)

        return predicted_x0_logits

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        """
        x: 图像的离散数据，[B,C,H,W]每个元素都属于{0,1,...,N-1}整数集合里面
        cond: 图像的类别标签，也是整数
        """
        t = torch.randint(1, self.n_T, (x.shape[0],), device=x.device)          # 随机获取时间t[1, 1000)
        x_t = self.q_sample(                                                    # 采样获取xt
            x, t, torch.rand((*x.shape, self.num_classses), device=x.device)    # 噪声来自均匀分布
        )   # batch_size, channel, h, w
        # x_t is same shape as x
        assert x_t.shape == x.shape, print(
            f"x_t.shape: {x_t.shape}, x.shape: {x.shape}"
        )
        # we use hybrid loss.
        # 预测的x0的logits，还没进行softmax，并不是概率；维度[B,C,H,W,N]
        predicted_x0_logits = self.model_predict(x_t, t, cond)

        # based on this, we first do vb loss.
        true_q_posterior_logits = self.q_posterior_logits(x, x_t, t)    # [b,c,h,w,n]
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x_t, t) # [b,c,h,w,n]

        vb_loss = self.vb(true_q_posterior_logits, pred_q_posterior_logits) # KL散度损失

        predicted_x0_logits = predicted_x0_logits.flatten(start_dim=0, end_dim=-2)

        x = x.flatten(start_dim=0, end_dim=-1)  # x为整数
        # x已经是0和1，增加近似L_0项的辅助损失（即交叉熵损失），即E~q(x_t|x_0)(log(P_θ(x_0|x_t))，完美契合用x_0得到的x_t给模型预测x_0
        ce_loss = torch.nn.CrossEntropyLoss()(predicted_x0_logits, x)

        return self.hybrid_loss_coeff * vb_loss + ce_loss, {
            "vb_loss": vb_loss.detach().item(),
            "ce_loss": ce_loss.detach().item(),
        }

    def p_sample(self, x, t, cond, noise):
        """
        x: x_t图像  [B,C,H,W]
        t: 时间t    [B]
        cond: 类别标签 [B]
        noise: 来自均匀分布的随机抽样的噪声 [B,C,H,W,N]
        """

        predicted_x0_logits = self.model_predict(x, t, cond)    # 预测x0

        # 返回log( q(x_t | x_t-1, x0) * q(x_t-1 | x0) ) = log( x_t@(Q_t)^T ⊙ x0@Q_t-1_bar ) 
        # 维度[b,c,h,w,n]
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x, t)    

        noise = torch.clip(noise, self.eps, 1.0)    # 截断

        not_first_step = (t != 1).float().reshape((x.shape[0], *[1] * (x.dim())))   # MASK，维度[B,C,H,W,N]，代表最后一步放弃随机性

        gumbel_noise = -torch.log(-torch.log(noise))    # gumbel_max的抽样方法
        sample = torch.argmax(
            pred_q_posterior_logits + gumbel_noise * not_first_step, dim=-1
        )
        return sample # [B,C,H,W]

    def sample(self, x, cond=None):
        for t in reversed(range(1, self.n_T)):
            t = torch.tensor([t] * x.shape[0], device=x.device)
            x = self.p_sample(
                x, t, cond, torch.rand((*x.shape, self.num_classses), device=x.device)
            )

        return x

    def sample_with_image_sequence(self, x, cond=None, stride=10):
        """
        x: 整数噪声 [4,1,32,32]
        cond: 类别标签 [4]
        """
        steps = 0
        images = []
        for t in reversed(range(1, self.n_T)):      # 1000
            t = torch.tensor([t] * x.shape[0], device=x.device)
            x = self.p_sample(
                x, t, cond, torch.rand((*x.shape, self.num_classses), device=x.device)
            )
            steps += 1
            if steps % stride == 0:
                images.append(x)

        # if last step is not divisible by stride, we add the last image.
        if steps % stride != 0:
            images.append(x)

        return images


if __name__ == "__main__":
    # 1、加载模型
    N = 2  # MNIST是黑白图，每个像素非0即1，因此类别数为2
    d3pm = D3PM(DummyX0Model(1, N), 1000, num_classes=N, hybrid_loss_coeff=1.0) # 不理解为什么作者选择设置hybrid_loss_coeff=0
    print(f"Total Param Count: {sum([p.numel() for p in d3pm.x0_model.parameters()])}")
    # 2、加载数据
    dataset = MNIST(
        r"/workspace/data",
        train=True,
        download=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Pad(2),
            ]
        ),
    )
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=2)
    # 3、加载优化器
    device = "cuda"  # 训练设备，提前定义供后续使用
    optim = torch.optim.AdamW(d3pm.x0_model.parameters(), lr=1e-3)
    d3pm.to(device)
    d3pm.train()

    # 4、遍历训练
    n_epoch = 100

    for epoch_i in range(n_epoch):

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, cond in pbar:
            optim.zero_grad()
            x = x.to(device)        # 图像数据
            cond = cond.to(device)  # 数字类别

            # 离散化：将图像x的连续空间[0,1] * (N-1) 得到[0, N-1]，进行四舍五入后取整，得到{0,1,2,..,N-1}的整数
            x = (x * (N - 1)).round().long().clamp(0, N - 1)
            loss, info = d3pm(x, cond)

            loss.backward() # 反向传播
            norm = torch.nn.utils.clip_grad_norm_(d3pm.x0_model.parameters(), 0.1)  # 梯度裁剪

            with torch.no_grad():
                param_norm = sum([torch.norm(p) for p in d3pm.x0_model.parameters()])

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.99 * loss_ema + 0.01 * loss.item()
            pbar.set_description(
                f"loss: {loss_ema:.4f}, norm: {norm:.4f}, param_norm: {param_norm:.4f}, vb_loss: {info['vb_loss']:.4f}, ce_loss: {info['ce_loss']:.4f}"
            )
            optim.step()

        if (epoch_i + 1) % 10 == 0:
            d3pm.eval()

            with torch.no_grad():
                cond = (torch.arange(0, 4) % 10).to(device)               # 获得类别标签
                init_noise = torch.randint(0, N, (4, 1, 32, 32), device=device) # 均匀分布抽样获得整数噪声

                images = d3pm.sample_with_image_sequence(       # 先验抽样
                    init_noise, cond, stride=40
                )
                # image sequences to gif
                gif = []
                for image in images:
                    x_as_image = make_grid(image.float() / (N - 1), nrow=2)
                    img = x_as_image.permute(1, 2, 0).cpu().numpy()
                    img = (img * 255).astype(np.uint8)
                    gif.append(Image.fromarray(img))

                gif[0].save(
                    f"./sample_{epoch_i + 1}.gif",
                    save_all=True,
                    append_images=gif[1:],
                    duration=100,
                    loop=0,
                )

                last_img = gif[-1]
                last_img.save(f"./sample_{epoch_i + 1}_last.png")

            d3pm.train()

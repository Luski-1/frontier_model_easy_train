import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from copy import deepcopy

from .ema import EMA
from .utils import extract

class GaussianDiffusion(nn.Module):
    __doc__ = r"""Gaussian Diffusion model. Forwarding through the module returns diffusion reversal scalar loss tensor.

    Input:
        x: tensor of shape (N, img_channels, *img_size)
        y: tensor of shape (N)
    Output:
        scalar loss tensor
    Args:
        model (nn.Module): model which estimates diffusion noise
        img_size (tuple): image size tuple (H, W)
        img_channels (int): number of image channels
        betas (np.ndarray): numpy array of diffusion betas
        loss_type (string): loss type, "l1" or "l2"
        ema_decay (float): model weights exponential moving average decay
        ema_start (int): number of steps before EMA
        ema_update_rate (int): number of steps before each EMA update
    """
    def __init__(
        self,
        model,
        img_size,           # (32, 32)
        img_channels,       # 3
        num_classes,        # 10
        betas,              # [β...] 1000
        loss_type="l2",     # L2损失函数
        ema_decay=0.9999,   # 0.9999
        ema_start=5000,     # 2000
        ema_update_rate=1,  # 1
    ):
        super().__init__()

        self.model = model
        self.ema_model = deepcopy(model)    # 复制原始模型作为EMA模型，指数平均移动的方式进行更新

        self.ema = EMA(ema_decay)
        self.ema_decay = ema_decay              # ema 更新比例
        self.ema_start = ema_start              # ema 更新起始点
        self.ema_update_rate = ema_update_rate  # ema 更新频率
        self.step = 0

        self.img_size = img_size
        self.img_channels = img_channels
        self.num_classes = num_classes

        if loss_type not in ["l1", "l2"]:
            raise ValueError("__init__() got unknown loss type")

        self.loss_type = loss_type
        self.num_timesteps = len(betas)     # 1000

        alphas = 1.0 - betas                # α = 1 - β
        alphas_cumprod = np.cumprod(alphas) # alpha_bar_t = α1 * α2 ... * αt = 控制图像比例

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))                      # 随时间越来越大
        self.register_buffer("alphas", to_torch(alphas))                    # 随时间越来越小
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))    # alpha_bar_t 随时间越来越小

        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))    # √(1 - alpha_bar_t) 控制噪声比例，随时间越来越大
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))                   # 1 / √(αt) 用于推理生成公式中，当前图像的比例

        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))       # 1 - αt / √(1 - alpha_bar_t) 用于推理生成公式中，当前模型预测噪声的比例
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))                                         # σ 用于推理生成公式中，重参数化的新增噪声的标准差 | 理论值是(1 - αt)(1 - alpha_bar_t-1) / (1 - alpha_bar_t)，但与βt接近

    def update_ema(self):
        self.step += 1
        if self.step % self.ema_update_rate == 0:
            if self.step < self.ema_start:
                self.ema_model.load_state_dict(self.model.state_dict())     # 直接=训练模型
            else:
                self.ema.update_model_average(self.ema_model, self.model)   # 0.9999 ema模型参数 + 0.0001 训练模型参数 = ema模型参数

    @torch.no_grad()
    def remove_noise(self, x, t, y, use_ema=True):
        """
        去噪公式: x_t-1 =  [1 / √(αt)] * [x_t - (1 - αt) / √(1 - alpha_bar_t) * ε(模型预测噪声) ] + σ * z (z ~ N(0, I))
        均值: [1 / √(αt)] * [x_t - (1 - αt) / √(1 - alpha_bar_t) * ε(模型预测噪声) ]
        方差: σ^2 = (1 - αt)(1 - alpha_bar_t-1) / (1 - alpha_bar_t)
        """
        if use_ema:
            return (
                (x - extract(self.remove_noise_coeff, t, x.shape) * self.ema_model(x, t, y)) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )
        else:
            return (
                (x - extract(self.remove_noise_coeff, t, x.shape) * self.model(x, t, y)) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )

    @torch.no_grad()
    def sample(self, batch_size, device, y=None, use_ema=True):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        
        for t in range(self.num_timesteps - 1, -1, -1):                     # 反向
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y, use_ema)       # 得到均值
            # 1. 不是最后一步，那么需要抽样噪声进行重参数化，即代表着满足最大似然的图像分布中进行抽样
            # 2. 最后一步，均值就是最高概率密度的数值，无需增加噪声
            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
        
        return x.cpu().detach()

    @torch.no_grad()
    def sample_diffusion_sequence(self, batch_size, device, y=None, use_ema=True):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        diffusion_sequence = [x.cpu().detach()]
        
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y, use_ema)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
            
            diffusion_sequence.append(x.cpu().detach())
        
        return diffusion_sequence

    def perturb_x(self, x, t, noise):
        """
        前向加噪: x_t = √(alpha_bar_t) * x_0 +  √(1 - alpha_bar_t) * ε | 随时间越来越模糊
        """
        return (
            extract(self.sqrt_alphas_cumprod, t, x.shape) * x +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )   

    def get_losses(self, x, t, y):
        noise = torch.randn_like(x)                             # 随机获取噪声 ε ~ N(0, I)

        perturbed_x = self.perturb_x(x, t, noise)               # 前向加噪
        estimated_noise = self.model(perturbed_x, t, y)         # 预测噪声，其中时间步通过transfomer的默认位置编码方式获得时间向量

        if self.loss_type == "l1":
            loss = F.l1_loss(estimated_noise, noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(estimated_noise, noise)           # MSE损失

        return loss

    def forward(self, x, y=None):
        b, c, h, w = x.shape
        device = x.device

        if h != self.img_size[0]:
            raise ValueError("image height does not match diffusion parameters")
        if w != self.img_size[0]:
            raise ValueError("image width does not match diffusion parameters")
        
        t = torch.randint(0, self.num_timesteps, (b,), device=device)       # 随机获取离散时间步 [0, 1000]
        return self.get_losses(x, t, y)


def generate_cosine_schedule(T, s=0.008):
    # (t / 1000 + 0.008) / (1 + 0.008) = [0, 1]区间
    # cos([0, 1] * π/2) = [0, 1]区间
    # cos^2 = [0, 1]区间
    def f(t, T):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2
    
    alphas = []
    f0 = f(0, T)

    for t in range(T + 1):
        alphas.append(f(t, T) / f0) # alphas = alpha_bar = a1 * a2 * a3 ... = (1-β1) * (1-β2) * (1-β3) ... = 控制图像比例，随时间t越来越小
    
    betas = []

    for t in range(1, T + 1):
        betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999)) # alpha_bar_t / alpha_bar_t-1 = 1-β_t  ==> β_t = 1 - alpha_bar_t / alpha_bar_t-1 = 随时间t越来越大
    
    return np.array(betas)


def generate_linear_schedule(T, low, high):
    return np.linspace(low, high, T)
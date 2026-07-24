from torchvision import utils
import torch
import math
import os


def save_samples(x: torch.tensor, name: str, save_dir: str):
    # x: tensor in [-1,1], shape (N,3,H,W)
    out = (x.clamp(-1,1) + 1.0) / 2.0  # to [0,1]
    grid = utils.make_grid(out, nrow=int(math.sqrt(out.shape[0]) + 0.999), padding=2)
    filename = os.path.join(save_dir, f"edm_sample_epoch_{name}.png")
    utils.save_image(grid, filename)
    print(f"[saved] {filename}")


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    P_mean = -1.2
    P_std = 1.2
    # 1. 标准正态分布采样 N(0,1)
    rnd_normal = torch.randn((num_samples,), device=device, dtype=torch.float32)
    # 2. 变换为正态分布 N(P_mean, P_std²)
    # log_sigma ~ N(P_mean, P_std)
    log_sigma = rnd_normal * P_std + P_mean
    # 3. 指数得到 sigma
    sigma = log_sigma.exp()
    # 4. sigma -> 连续时间 t ∈ (0,1)
    time = 1 / (1 + sigma)
    # 5. 截断边界防止数值异常
    time = torch.clip(time, min=0.0001, max=1.0)
    # 得到t偏向靠近1的区域，可以理解为t越小则模型只需要还原粗略的信息，而t越大则模型需要还原更加精细的信息，会更难，因此让t更偏向于靠近1，提高模型学习
    return time


def get_time_discretization(nfes: int, device: torch.device, rho=7):
    step_indices = torch.arange(nfes, dtype=torch.float64)  # [0, 1, 2, ..., nfes-1]
    sigma_min = 0.002
    sigma_max = 80.0
    # 对应的公式：( σ_max^(1/ρ) + i / (N-1) * (σ_min^(1/ρ) - σ_max^(1/ρ)) )^ρ，其中i是列表step_indices，递减性质
    sigma_vec = (
        sigma_max ** (1 / rho)
        + step_indices / (nfes - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho    # {80^(1/7) + [0, 1/nfes-1, 2/nfes-1, ..., 1] * (0.002^(1/7) - 80^(1/7))} ^ 7，首位是80
    sigma_vec = torch.cat([sigma_vec, torch.zeros_like(sigma_vec[:1])]) # 末尾+0
    time_vec = (sigma_vec / (1 + sigma_vec)).squeeze()  # 递减列表，σ从80到0 → time从80/81到0
    t_samples = 1.0 - torch.clip(time_vec, min=0.0, max=1.0)    # 递增列表，从1/81到1.0
    return t_samples.to(device=device)
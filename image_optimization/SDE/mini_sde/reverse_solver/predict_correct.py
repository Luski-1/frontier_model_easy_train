from utils import get_data_inverse_scaler
from sde.rsde import RSDE
from sde.vpsde import VPSDE
from sde.vesde import VESDE
from typing import Union
from tqdm import tqdm
import torch.nn as nn
import torch


class ReverseDiffusionPredictor:
    def __init__(self, rsde: Union[RSDE]):
       self.rsde = rsde

    # NCSN/DDPM的denoise_update_fn: probability_flow=False  | NCSN的PC: probability_flow=False
    def update_x(self, x, t, score, probability_flow):
        """
        调用NCSN或DDPM特有的反向SDE离散方法，获得f和g的部分结果，再通过rsde得到F和G，F已经包含f*dt，G已经包含g*dt
        x <= x - F + G * z
        """
        f, G = self.rsde.get_special_reverse_discretize_f_and_g(score, x, t, probability_flow)
        z = torch.randn_like(x)
        x_mean = x - f  # 添加漂移项
        x = x_mean + G[:, None, None, None] * z # 添加扩散项
        return x, x_mean
    

class EulerMaruyamaPredictor:
    def __init__(self, rsde: Union[RSDE]):
        self.rsde = rsde

    # DDPM的PC: probability_flow=False
    def update_x(self, x, t, score, probability_flow=False):
        """
        反向离散的方法是常规的RSDE，而不是DDPM特有的反向SDE离散方法，求解算法使用欧拉法

        反向求解的ODE：dx = [f(前向) - 1/2 * g(前向)^2 * score] * dt
        因此在反向ODE中，f(反向) = [f(前向) - 1/2 * g(前向)^2 * score]，g(反向) = 0
        """

        # 反向SDE，时间步为负
        dt = -1. / self.rsde.N
        # 抽样噪声
        z = torch.randn_like(x)
        # 获取反向SDE的f和g
        drift, diffusion = self.rsde.get_reverse_f_and_g(score, x, t, probability_flow)
        # x <= x - f * dt + g * dw
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * ((-dt) ** 0.5) * z
        return x, x_mean
    

class NoneCorrector:

    def __init__(self, sde, snr, n_steps):
        pass

    def update_x(self, x, t=None, score=None, probability_flow=False):
        return x, x
    
class LangevinCorrector:
  def __init__(self, sde, snr, n_steps):

    self.sde = sde
    # self.score_fn = score_fn
    self.snr = snr              # 信噪比
    self.n_steps = n_steps      # 纠正次数


  def update_x(self, x, t, score: torch.tensor, probability_flow=False):

    target_snr = self.snr
    if isinstance(self.sde, VPSDE):
        timestep = (t * (self.sde.N - 1) / self.sde.T).long()
        alpha = self.sde.alphas.to(t.device)[timestep]
    else:
        # NCSN分支
        alpha = torch.ones_like(t)

    for _ in range(self.n_steps):
        noise = torch.randn_like(score) # 获得高斯分布的噪声
        grad_norm = torch.norm(score.reshape(score.shape[0], -1), dim=-1).mean()  # 求出score的信号强度
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean() # 求出噪声的信号强度
        step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha    # 求出步长
        # 调用朗之万动力学退火：xi <= xi + step * score + sqrt(2 * step) * ε | ε ~ N(0, I)
        # 其中step在一开始的NCSN项目中，是当前sigma/最小sigma作为自动步长
        x_mean = x + step_size[:, None, None, None] * score
        x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

    return x, x_mean
    
PREDICTOR_MAP = {"euler_maruyama": EulerMaruyamaPredictor, "reverse_diffusion": ReverseDiffusionPredictor}
CORRECTOR_MAP = {"none": NoneCorrector, "langevin": LangevinCorrector}
    

def get_pc_solve_result(model: nn.Module,
                        sde: Union[VPSDE, VESDE], shape,
                        predictor_name: str,
                        corrector_name: str,
                        probability_flow=False,
                        snr=0.16,
                        n_step=1,
                        denoise=True, eps=1e-3, device='cuda'):
    
    def get_score(x, t):
        """
        用于调用模型并且适配为score
        """
        # 时间放大999倍，是为了放大位置编码的频率，但实际上还是连续的浮点数
        labels = t * 999
        score = model(x, labels)
        std = sde.marginal_prob(torch.zeros_like(x), t)[1]

        # 但是这个项目是SDE，是通过score来去噪获得图像的，所以根据原始的DDPM的公式，除std=score
        if isinstance(sde, VPSDE):
            score = -score / std[:, None, None, None]
            return score
        else:
            return score / std[:, None, None, None]
    
    with torch.no_grad():
        # 获得时刻1的噪声图像
        x = sde.prior_sampling(shape).to(device)

        rsde = RSDE(sde)

        predictor = PREDICTOR_MAP[predictor_name](rsde)
        corrector = CORRECTOR_MAP[corrector_name](sde, snr, n_step)

        # 获得NCSN区间[1, 1e-5]或DDPM区间[1, 1e-3]，间隔1000的等差时间
        timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

        for i in tqdm(range(sde.N), desc="PC Sampling", leave=False):
            t = timesteps[i]
            vec_t = torch.ones(shape[0], device=t.device) * t # 调整时间维度
            score = get_score(x, vec_t)
            # 先P后C或者先C后P，问题不大
            x, x_mean = corrector.update_x(x, vec_t, score, probability_flow)
            x, x_mean = predictor.update_x(x, vec_t, score, probability_flow)

            # denoise=True，因为已经使用朗之万动力学矫正，就没必要再执行额外去噪了
        return get_data_inverse_scaler(x_mean if denoise else x), sde.N * (n_step + 1)


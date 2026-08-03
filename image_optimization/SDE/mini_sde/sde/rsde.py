from sde.vpsde import VPSDE
from typing import Optional
import torch

class RSDE:

    def __init__(self, sde: Optional[VPSDE]):
        self.N = sde.N
        self.T = sde.T
        self.sde = sde

    def get_reverse_f_and_g(self, score, x, t, probability_flow=False):
        # 反向求解的ODE：dx = [f(前向) - 1/2 * g(前向)^2 * score] * dt
        # 因此在反向ODE中，f(反向) = [f(前向) - 1/2 * g(前向)^2 * score]，g(反向) = 0

        # 获取NCSN的前向SDE：drift=0 | diffusion = σmin * (σmax/σmin)^t * sqrt( 2 * log(σmax/σmin) )
        # 获取DDPM的前向SDE：drift=-1/2 * ( β_bar_min + t * (β_bar_max - β_bar_min)) * x | diffusion=sqrt(β_bar_min + t * (β_bar_max - β_bar_min))

        # 反向求解的SDE: dx = [f(前向) - g(前向)^2 * score] * dt + g(前向) * dw_bar
        # 因此在反向SDE中，f(反向) = [f(前向) - g(前向)^2 * score], g(反向) = g(前向)
        drift, diffusion = self.sde.get_forward_f_and_g(x, t)
        # 参考上述公式
        drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if probability_flow else 1.)
        # 参考上述公式
        diffusion = 0. if probability_flow else diffusion
        return drift, diffusion
    

    def get_special_reverse_discretize_f_and_g(self, score, x, t, probability_flow=False):
        # VESDE的get_special_reverse_discretize_parameters方法返回结果：f=0 | G = sqrt( σ(i)^2 - σ(i - 1)^2 )
        # NCSN分支denoise_update_fn方法中probability_flow=False，即反向SDE的NCSN专有离散方法：x_i-1 = x_i + ( σ(i)^2 - σ(i - 1)^2 ) * score + sqrt( σ(i)^2 - σ(i - 1)^2 ) * ε | 其中 ε ~ N(0, I)
        # ref_f = 0 - (σ(i)^2 - σ(i - 1)^2) * score
        # ref_G = sqrt( σ(i)^2 - σ(i - 1)^2 )

        # VPSDE的get_special_reverse_discretize_parameters方法返货结果：f=sqrt(1 - β_i_t) * x_i - x_i  | G=sqrt(β_i_t)
        # DDPM分支denoise_update_fn方法中probability_flow=False，即反向SDE的NCSN专有离散方法：x_i-1 = (2 - sqrt(1 - β_i_t)) * x_i + β_i_t * score + sqrt(β_i_t) * ε | 其中 ε ~ N(0, I)
        # ref_f = sqrt(1 - β_i_t) * x_i - x_i - β_i_t * score
        # ref_G = sqrt(β_i_t)
        partial_drift, partial_diffusion = self.sde.get_special_reverse_discretize_parameters(x, t)
        # 参考上述公式
        drift = partial_drift - partial_diffusion[:, None, None, None] ** 2 * score * (0.5 if probability_flow else 1.)
        # 参考上述公式
        diffusion = torch.zeros_like(partial_diffusion) if probability_flow else partial_diffusion
        return drift, diffusion
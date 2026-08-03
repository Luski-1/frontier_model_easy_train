from utils import from_flattened_numpy, to_flattened_numpy, get_data_inverse_scaler
from reverse_solver.predict_correct import ReverseDiffusionPredictor
from typing import Union
from sde.vpsde import VPSDE
from sde.vesde import VESDE
from scipy import integrate
from sde.rsde import RSDE
from tqdm import tqdm
import torch.nn as nn
import torch


def get_ode_solve_result(model: nn.Module, 
                         sde: Union[VPSDE, VESDE], shape,
                        probability_flow=True,
                        denoise=True, rtol=1e-5, atol=1e-5,
                        method='RK45', eps=1e-3, device='cuda'):
  
    # 创建反向SDE过程的管理对象
    rsde = RSDE(sde)
  
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


    def denoise_update_x(rsde: RSDE, x):
        """
        用于最后一步去噪
        """
        vec_eps = torch.ones(x.shape[0], device=x.device) * eps # eps此时为时间，调整时间的维度
        score = get_score(x, vec_eps)  # 获得score

        # 获取执行反向SDE的对象
        predictor_obj = ReverseDiffusionPredictor(rsde)
        _, x = predictor_obj.update_x(x, vec_eps, score, probability_flow=False)  # 执行NCSN特有反向SDE的离散方法，_是包含扩散项的最终结果，x是剔除扩散项的最终结果
        return x

    def get_reverse_f(rsde: RSDE, x, t):
        """
        用于获取反向ODE的f
        """
        score = get_score(x, t) # 获得score
        f, _ = rsde.get_reverse_f_and_g(score, x, t, probability_flow=True) # probability_flow用于控制是否为ODE
        return f
    

    with torch.no_grad():

        # 获取先验分布
        x = sde.prior_sampling(shape).to(device)


        def ode_func(t, x):
            # 重新调整x维度
            x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
            # 重新调整t维度
            vec_t = torch.ones(shape[0], device=x.device) * t
            # 获得反向求解ODE的漂移项f
            drift = get_reverse_f(rsde, x, vec_t)
            # 又拍平为一维
            return to_flattened_numpy(drift)

        # Black-box ODE solver for the probability flow ODE
        # ode_func参数是ODE求解的函数f，即dx/dt = f(x, t)
        # (sde.T, eps)是求解时间区间，[1, 1E-5] 或 [1, 1E-3]
        # to_flattened_numpy(x)是初始状态的x
        # RK45是具体数值求解方法
        # rtol=1e-5，是相对误差忍耐，即|error| > rtol * |x|则缩小求解步长
        # atol=1e-5，是绝对误差忍耐，即|error| > atol则缩小求解步长
        # |error|具体数值求解方法中使用高阶求解与低阶求解的结果差异，例如欧拉法是一阶，heun法是二阶
        solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                        rtol=rtol, atol=atol, method=method)
        nfe = solution.nfev # 求解的步数
        x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32) # 最终时刻的x

        # 是否执行最后一步去噪
        if denoise:
            x = denoise_update_x(rsde, x)

        # 反归一化
        x = get_data_inverse_scaler(x)
        return x, nfe
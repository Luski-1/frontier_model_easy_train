def extract(a, t, x_shape):
    b, *_ = t.shape             # 获取时间步的batch size
    out = a.gather(-1, t)       # -1 指 从a的最后1个维度取数，取数的位置由t最后1个维度的数字来决定 [B]
    return out.reshape(b, *((1,) * (len(x_shape) - 1))) # DDPM是像素空间模型，x的维度是[B,C,H,W]
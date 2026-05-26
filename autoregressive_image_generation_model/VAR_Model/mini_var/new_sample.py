"""
精简版 VAR 项目采样脚本
原文件：demo_sample.ipynb
改动：转为独立 .py 脚本，去掉 512 分支，保留 256 采样功能
"""

import os
import os.path as osp
import random

import numpy as np
import PIL.Image as PImage
import torch
import torchvision

from new_args import PATCH_NUMS, NUM_CLASSES, set_tf32, parse_args
from new_var import build_vae_var


def main_sample():
    # ===================== 1. 配置 =====================
    MODEL_DEPTH = 16       # 模型深度
    seed = 0               # 随机种子
    cfg = 4                # Classifier-Free Guidance 强度
    top_k = 900            # top-k 采样
    top_p = 0.95           # top-p 采样
    more_smooth = False    # 是否使用 Gumbel Softmax 平滑（仅用于可视化）
    class_labels = (980, 980, 437, 437, 22, 22, 562, 562)  # ImageNet 类别标签

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = parse_args()

    # ===================== 2. TF32 加速 =====================
    tf32 = True
    set_tf32(tf32)

    # ===================== 3. 下载 checkpoint =====================
    hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
    var_ckpt = f'{args.output_dir}/var_d{MODEL_DEPTH}.pth'
    if not osp.exists(args.vae_ckpt):
        os.system(f'wget {hf_home}/vae_ch160v4096z32.pth')
    if not osp.exists(var_ckpt):
        os.system(f'wget {hf_home}/{var_ckpt}')

    # ===================== 4. 构建模型 =====================
    # 禁用PyTorch默认参数初始化
    setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
    setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=PATCH_NUMS,
        num_classes=NUM_CLASSES, depth=MODEL_DEPTH,
    )

    # ===================== 5. 加载权重 =====================
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
    vae.eval()
    var.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)
    print(f'prepare finished.')

    # ===================== 6. 设置种子 =====================
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ===================== 7. 采样 =====================
    B = len(class_labels)
    label_B: torch.LongTensor = torch.tensor(class_labels, device=device)

    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
            recon_B3HW = var.autoregressive_infer_cfg(
                B=B, label_B=label_B, cfg=cfg,
                top_k=top_k, top_p=top_p, g_seed=seed, more_smooth=more_smooth
            )

    # ===================== 8. 可视化 =====================
    chw = torchvision.utils.make_grid(recon_B3HW, nrow=8, padding=0, pad_value=1.0)
    chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
    img = PImage.fromarray(chw.astype(np.uint8))

    # 保存
    output_path = 'sample_output.png'
    img.save(output_path)
    print(f'Sampled image saved to {output_path}')

    # 显示
    img.show()


if __name__ == '__main__':
    main_sample()
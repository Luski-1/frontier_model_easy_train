from utils import save_samples
from model import EnhancedUNet, FlowMatching
from config import get_args
import torch
import os


@torch.no_grad()
def evaluate(ckpt_path: str, args, device: torch.device):
    """加载检查点并使用修正后的t序列进行先验采样"""
    
    # 1. 加载检查点
    print(f"[loading checkpoint] {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # 2. 重建模型
    inner_model = EnhancedUNet(
        in_ch=3,
        base_ch=args.base_ch,
        time_emb_dim=args.time_ch,
        num_res_blocks=args.num_res_blocks
    ).to(device)
    
    if args.ema:
        model = FlowMatching(inner_model, steps=args.flow_steps, decay=args.decay)
        model.model.load_state_dict(ckpt["model_state_dict"])
        model.ema_model.load_state_dict(ckpt["ema_state_dict"])
        print("[loaded] model + ema_model")
    else:
        model = FlowMatching(inner_model, steps=args.flow_steps)
        model.model.load_state_dict(ckpt["model_state_dict"])
        print("[loaded] model (no ema)")
    
    model.eval()
    
    # 3. 先验采样（修正后的t方向：从0→1）
    print(f"[sampling] {args.num_sample_images} images, {args.flow_steps} steps, edm_eval={args.edm_eval_time}")
    samples = model.sample_flow(
        image_size=args.image_size,
        device=device,
        n_samples=args.num_sample_images,
        edm_eval=args.edm_eval_time
    )
    
    # 4. 保存结果
    save_dir = os.path.dirname(ckpt_path) if os.path.dirname(ckpt_path) else "./eval_output"
    os.makedirs(save_dir, exist_ok=True)
    save_samples(samples, "eval", save_dir=save_dir)
    print("[done]")


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 默认加载最新检查点
    ckpt_path = os.path.join(args.save_dir, "flowmatch_ckpt_epoch_075.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    evaluate(ckpt_path, args, device)


if __name__ == "__main__":
    main()

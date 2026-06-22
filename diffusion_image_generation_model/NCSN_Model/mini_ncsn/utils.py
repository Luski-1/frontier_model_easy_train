from transformers import TrainerCallback
from torchvision import utils
import torch
import tqdm
import os


def save_image_grid(tensor, filename, nrow=8):
    # tensor expected in [0,1]
    utils.save_image(tensor, filename, nrow=nrow, padding=2)


# ==================== 回调函数 ====================
class SampleCallback(TrainerCallback):
    def __init__(self, sample_every, num_sample_images, image_size, channels,
                sigmas_list, sigma_steps, step_lr, save_dir):
        """
        sample_every: 先验抽样的周期
        num_sample_images: 生成图像的数量
        image_size: 先验抽样的图像大小
        channel: 先验抽样的图像channel
        sigmas_list: sigma的等比序列
        sigma_steps: 每个sigma的退火步数
        step_lr: 退火的基础步长
        save_dir: 图像的保存位置
        """
        self.sample_every = sample_every
        self.num_sample_images = num_sample_images
        self.image_size = image_size
        self.channels = channels
        self.sigmas_list = sigmas_list
        self.sigma_steps = sigma_steps
        self.step_lr = step_lr
        self.save_dir = save_dir

    def on_epoch_end(self, args, state, control, **kwargs):
        """
        args: TrainingArguments
        state: TrainerState，例如epoch、max_steps
        control: TrainerControl，例如should_training_stop、should_evaluate
        kwargs: 不同阶段传入的参数不一样
        """
        current_epoch = int(state.epoch)
        if current_epoch % self.sample_every != 0 and current_epoch != 1:
            return control

        model = kwargs.get("model")
        if model is None:
            print("[SampleCallback] Warning: model not found in kwargs, skip sampling")
            return control

        was_training = model.training
        model.eval()

        print(f"\n[SampleCallback] Epoch {current_epoch}: 生成 {self.num_sample_images} 张样本...")

        try:
            with torch.no_grad():
                # 先验抽样的朗之万动力学退火
                samples = self.anneal_Langevin_dynamics(model)
            # 保存图像
            os.makedirs(self.save_dir, exist_ok=True)
            output_path = os.path.join(self.save_dir, f"sample_epoch{current_epoch}.png")
            save_image_grid(samples, output_path, nrow=2)
            print(f"[SampleCallback] 样本已保存到 {output_path}")
        finally:
            # 恢复之前的训练状态，而非强制 train()
            if was_training:
                model.train()

        return control

    @torch.no_grad()
    def anneal_Langevin_dynamics(self, model):
        
        device = next(model.parameters()).device
        # 获取噪声
        x_mod = torch.randn(self.num_sample_images, self.channels, self.image_size,
                            self.image_size, device=device)

        sigmas_list = self.sigmas_list.to(device)

        for c, sigma in enumerate(tqdm.tqdm(sigmas_list, desc='annealed Langevin dynamics')):
            labels = torch.full((x_mod.shape[0],), c, device=x_mod.device, dtype=torch.long)    # c作为sigma类别的下标
            step_size = self.step_lr * (sigma / sigmas_list[-1]) ** 2                           # 当前步伐长度 = (当前sigma/最小sigma)^2 * step_lr 

            for s in range(self.sigma_steps):
                grad = model(x_mod, labels)                                                     # score
                noise = torch.randn_like(x_mod) * torch.sqrt(step_size * 2)                     # sqrt(2 * step_size) * ε
                x_mod = x_mod + step_size * grad + noise                                        # x_t+1 = x_t + step_size * score + sqrt(2 * step_size) * ε; ε ~ N(0, I)

        x_mod = (x_mod + 1.0) / 2.0                 # 反归一化
        x_mod = torch.clamp(x_mod, 0.0, 1.0)        # 裁剪

        # 单通道复制为3通道（GRAY->RGB），否则save_image报错
        if x_mod.shape[1] == 1:
            x_mod = x_mod.repeat(1, 3, 1, 1)

        return x_mod

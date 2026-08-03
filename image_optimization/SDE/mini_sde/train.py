from utils import get_data_scaler, save_image_grid, save_checkpoint
from reverse_solver.predict_correct import get_pc_solve_result
from reverse_solver.ode import get_ode_solve_result
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import get_scheduler
from loss import calculate_loss
from model import EnhancedUNet
from sde.vpsde import VPSDE
from sde.vesde import VESDE
from tqdm import tqdm
from ema import EMA
import torch.optim as optim
import torchvision
import torch
import yaml
import os


def main(yaml_path: str = "./config.yaml", sample_step: int=5000):

    # 1、加载配置文件以及创建文件夹
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    os.makedirs("./samples", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)

    # pytorch cpu
    torch.manual_seed(config["seed"])
    # pytorch all gpu
    torch.cuda.manual_seed_all(config["seed"])

    # 2、加载模型
    model = EnhancedUNet(in_ch=config["data"]["num_channels"], base_ch=config["model"]["nf"], num_res_blocks=config["model"]["num_res_blocks"])

    # 3、获取优化器
    optimizer = optim.AdamW(model.parameters(), lr=config["optim"]["lr"], betas=(config["optim"]["beta1"], 0.999), eps=config["optim"]["eps"],
                        weight_decay=config["optim"]["weight_decay"])

    
    # 4、获取数据集
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),          # 转为RGB3通道
        transforms.RandomHorizontalFlip(),                          # 随机水平翻转
        transforms.Resize(config["data"]["image_size"]),                    # 调整大小
        transforms.CenterCrop(config["data"]["image_size"]),                # 裁剪
        transforms.ToTensor(),                                      # 将离散[0,255]转换为连续[0,1]
        # transforms.Lambda(lambda img: (torch.rand(img) + img * 255.) / 256.)    # 将图像进一步连续化
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=config["data"]["dataset_path"], train=True, download=False, transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=config["training"]["per_device_batch_size"], shuffle=True, drop_last=True)


    # 5、初始化Accelerator 和 学习率调度器
    accelerator = Accelerator(
        gradient_accumulation_steps=1,  # 梯度累积=1
        mixed_precision="bf16",         # 开启混合精度
    )

    scheduler = get_scheduler(
        "constant_with_warmup",         # warmup后维持学习率不变
        optimizer=optimizer,
        num_warmup_steps=config["optim"]["warmup"],
    )

    # 6、封装
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    ema = EMA(model.parameters(), config["model"]["ema_rate"])

    # 7、创建SDE过程的管理对象
    if config["training"]["sde"] == "vpsde":
        sde = VPSDE(beta_min=config["model"]["beta_min"], beta_max=config["model"]["beta_max"], N=config["model"]["num_scales"])
        model_name = "vpsde_ddpm"
        accelerator.print("本次训练的模型是VPSDE类型的DDPM")
    else:
        sde = VESDE(sigma_min=config["model"]["sigma_min"], sigma_max=config["model"]["sigma_max"], N=config["model"]["num_scales"])
        model_name = "vesde_ncsn"
        accelerator.print("本次训练的模型是VESDE类型的NCSN")
    

    # 8、 开启训练
    dataloader_iter = iter(dataloader)
    total_iters = config["training"]["n_iters"] // accelerator.num_processes # 根据显卡总数调整训练总步数


    model.train()
    epoch_counter = 0
    progress_bar = tqdm(range(total_iters), disable=not accelerator.is_main_process, desc="Training")
    for global_step in range(total_iters):
            
        try:
            x, _ = next(dataloader_iter)
        except StopIteration:
            # 数据取完，重新来一轮       
            epoch_counter += 1
            dataloader.set_epoch(epoch_counter)  # 重新设置epoch数，确保各进程数据shuffle不同
            dataloader_iter = iter(dataloader)
            x, _ = next(dataloader_iter)

        # 让accelerator管理梯度累积
        with accelerator.accumulate(model):
            x = get_data_scaler(x)  # 数据归一化
            
            loss = calculate_loss(model, x, sde) # DDPM的前向SDE训练
            real_loss = loss.detach()
            loss = loss / accelerator.gradient_accumulation_steps

            accelerator.backward(loss)
            if accelerator.sync_gradients:  # 只有最后一个累积步才执行
                accelerator.clip_grad_norm_(model.parameters(), max_norm=config["optim"]["grad_clip"])

            optimizer.step()        # 由accelerator控制是否参数更新，
            optimizer.zero_grad()   # 由accelerator控制是否梯度置零


        if accelerator.sync_gradients:  # 只有最后一个累积步才执行
            scheduler.step()
            ema.update(model.parameters())


        if global_step != 0 and global_step % sample_step == 0:
            # 1. 同步所有进程，确保梯度更新完毕
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:  # 只在主进程推理
                # 2. 切换到 eval
                model.eval()

                # 3. 切换EMA权重
                ema.store(model.parameters())     # 保存训练模型的权重
                ema.copy_to(model.parameters())   # 把EMA模型的权重复制到训练模型

                # unwrapped = accelerator.unwrap_model(model) 如果模型有其他方法需要调用，可以通过unwarp_model来拆包获得真正的模型
                with torch.no_grad():
                    # 获得要生成多少张图像
                    shape = x.shape[:16]
                    sample_name = f'global_step_{global_step}_{model_name}_ema_{config["sampling"]["method"]}.png'

                    if config["sampling"]["method"] == "ode":
                        samples, nfe = get_ode_solve_result(model, sde, shape=shape,
                                                            denoise=config["sampling"]["noise_removal"], eps=float(config["sampling"]["eps"]))
                    elif config["sampling"]["method"] == "pc":
                        samples, nfe = get_pc_solve_result(model, sde, shape=shape, 
                                                           predictor_name=config["sampling"]["predictor"], corrector_name=config["sampling"]["corrector"],
                                                           snr=config["sampling"]["snr"], n_step=config["sampling"]["n_steps_each"],
                                                           denoise=config["sampling"]["noise_removal"], eps=float(config["sampling"]["eps"]))
                    else:
                        raise ValueError("暂不支持其他求解方法")

                    samples = torch.clamp(samples, 0.0, 1.0)
                    samples_path = os.path.join("./samples", sample_name)
                    save_image_grid(samples, samples_path, 8)
                
                ema.restore(model.parameters()) # 回复训练模型的权重

                checkpoint_path = os.path.join("./checkpoints", f"{model_name}.pt")
                save_checkpoint(checkpoint_path, accelerator.unwrap_model(model), ema)

                # 4. 切回训练模式
                model.train()

            # 5. 再次同步，确保所有进程一起继续训练
            accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            progress_bar.set_postfix(lr=f"{scheduler.get_last_lr()[0]:.8f}", loss=f"{real_loss.item():.4f}")
            progress_bar.update(1)

    progress_bar.close()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        checkpoint_path = os.path.join("./checkpoints", f"{model_name}.pt")
        save_checkpoint(checkpoint_path, accelerator.unwrap_model(model), ema)

if __name__ == "__main__":
    main(yaml_path="./config/ddpm_config.yaml", sample_step=10000)

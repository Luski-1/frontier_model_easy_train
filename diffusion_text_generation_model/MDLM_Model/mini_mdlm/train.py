from utils import save_checkpoint, save_intermediate_gif
from transformers import AutoTokenizer
from accelerate import Accelerator
from transformers import get_scheduler
from model import Diffusion
from tqdm import tqdm
import torch.optim as optim
import datasets
import torch
import yaml
import os


def main(yaml_path: str = "./config.yaml"):

    # 1、加载配置文件以及创建文件夹
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    os.makedirs(config["train"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["sample"]["sample_dir"], exist_ok=True)

    # pytorch cpu
    torch.manual_seed(config["seed"])
    # pytorch all gpu
    torch.cuda.manual_seed_all(config["seed"])

    # 2、获取tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["data"]["tokenizer_name_or_path"])
    if tokenizer.pad_token is None:
        raise ValueError("tokenizer不存在PAD TOKEN 请更换tokenizer或者增加PAD TOKEN替换代码")
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.sep_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.cls_token

    # 3、获取数据集
    dataset = datasets.load_from_disk(config["data"]["packed_dataset_path"])
    dataset =  dataset.with_format('torch', columns=['input_ids', 'attention_mask'])

    dataloader = torch.utils.data.DataLoader(       # 获取dataloader
        dataset,
        batch_size=config["train"]["per_device_batch_size"],
        num_workers=config["loader"]["num_workers"],
        pin_memory=config["loader"]["pin_memory"],
        shuffle=True,
        persistent_workers=True)
    dataloader.tokenizer = tokenizer                # 设置tokenizer


    # 4、加载模型
    model = Diffusion(config=config, tokenizer=tokenizer)


    # 5、获取优化器
    optimizer = optim.AdamW(model.parameters(), lr=config["optim"]["lr"], betas=(config["optim"]["beta1"], config["optim"]["beta2"]), 
                            eps=config["optim"]["eps"], weight_decay=config["optim"]["weight_decay"])

    # 6、初始化Accelerator 和 学习率调度器
    accelerator = Accelerator(
        gradient_accumulation_steps=1,                          # 梯度累积=1
        mixed_precision=config["train"]["precision"],           # 开启混合精度
    )

    scheduler = get_scheduler(
        config["train"]["lr_scheduler"],                    # warmup后维持学习率不变
        optimizer=optimizer,
        num_warmup_steps=config["train"]["wramup_step"],
    )

    # 7、封装
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # 8、 开启训练
    dataloader_iter = iter(dataloader)
    total_iters = config["train"]["train_steps"] // accelerator.num_processes # 根据显卡总数调整训练总步数

    model.train()
    epoch_counter = 0
    log_steps = config["train"]["log_steps"]
    progress_bar = tqdm(range(total_iters), disable=not accelerator.is_main_process, desc="Training")
    for global_step in range(total_iters):
            
        try:
            batch_data = next(dataloader_iter)
        except StopIteration:
            # 数据取完，重新来一轮       
            epoch_counter += 1
            dataloader.set_epoch(epoch_counter)  # 重新设置epoch数，确保各进程数据shuffle不同
            dataloader_iter = iter(dataloader)
            batch_data = next(dataloader_iter)

        # 让accelerator管理梯度累积
        with accelerator.accumulate(model):
            
            loss = model(batch_data)
            real_loss = loss.detach()
            loss = loss / accelerator.gradient_accumulation_steps

            accelerator.backward(loss)
            if accelerator.sync_gradients:  # 只有最后一个累积步才执行
                accelerator.clip_grad_norm_(model.parameters(), max_norm=config["train"]["gradient_clip"])

            optimizer.step()        # 由accelerator控制是否参数更新
            optimizer.zero_grad()   # 由accelerator控制是否梯度置零


        if accelerator.sync_gradients:  # 只有最后一个累积步才执行
            scheduler.step()

        if global_step != 0 and global_step % log_steps == 0:
            # 1. 同步所有进程，确保梯度更新完毕
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:  # 只在主进程推理

                model.eval()
                ar_intermediate_samples = None

                with torch.no_grad():
                        unwrap_model: Diffusion = accelerator.unwrap_model(model)   # 需要拆包

                        if config["sample"]["semi_ar"]:
                            stride_length = config["sample"]["stride_length"] # 1
                            num_strides = config["sample"]["num_strides"] # 1

                            # intermediate_samples：长度递进的文本片段
                            _, ar_intermediate_samples, _ = unwrap_model.semi_ar_sample(
                                stride_length=stride_length,
                                num_strides=num_strides,
                                dt=1 / config["sample"]["steps"],
                                n_samples=config["sample"]["per_device_batch_size"],
                                device=batch_data["input_ids"].device)
                            
                        _, default_intermediate_samples, _ = unwrap_model.default_sample(
                            num_steps=config["sample"]["steps"], 
                            n_samples=config["sample"]["per_device_batch_size"],
                            device=batch_data["input_ids"].device)
                        
                        # 保存为gif图片
                        gif_dir = config["sample"]["sample_dir"]
                        step_tag = f"step{global_step}"
                        if ar_intermediate_samples is not None:
                            save_intermediate_gif(
                                ar_intermediate_samples,
                                gif_path=os.path.join(gif_dir, f"{step_tag}_semi_ar.gif"),
                                tokenizer=tokenizer,
                                title=f"semi-ar  step={global_step}",
                                width_px=1000, font_size=18, duration=450, max_frames=64,
                            )
                        if default_intermediate_samples is not None:
                            save_intermediate_gif(
                                default_intermediate_samples,
                                gif_path=os.path.join(gif_dir, f"{step_tag}_default.gif"),
                                tokenizer=tokenizer,
                                title=f"default  step={global_step}",
                                width_px=1000, font_size=18, duration=450, max_frames=64,
                            )

                checkpoint_path = os.path.join(config["train"]["checkpoint_dir"], "mdlm.pt")
                save_checkpoint(checkpoint_path, accelerator.unwrap_model(model))

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
        checkpoint_path = os.path.join(config["train"]["checkpoint_dir"], "mdlm.pt")
        save_checkpoint(checkpoint_path, accelerator.unwrap_model(model))

if __name__ == "__main__":
    main(yaml_path="./config.yaml")

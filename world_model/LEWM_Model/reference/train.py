import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size  # 3
    n_preds = cfg.num_preds     # 1
    lambd = cfg.loss.sigreg.weight  # 0.09

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"] # [B, T, D]

    ctx_emb = emb[:, :ctx_len]  # x of pixel
    ctx_act = act_emb[:, : ctx_len] # x of action

    tgt_emb = emb[:, n_preds:] # y of pixel
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred of pixel

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean() # MSE LOSS
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1)) # SIGReg LOSS
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    # 0. 加载参数
    # cfg由hydra构建，cfg.data指代lewm.yaml中defaults.data

    # +------------+------------------------------------------------------+
    # | dataset参数名 | 含义 |
    # +------------+------------------------------------------------------+
    # | num_steps | 最终想要多少个连续的数据点（模型输入的序列长度） |
    # +------------+------------------------------------------------------+
    # | frameskip | 从原始数据里，每隔几帧取1个数据点（跳帧降采样，降低数据量） |
    # +------------+------------------------------------------------------+
    # | span | 为了取到这些点，原始数据必须占多少连续帧（总跨度 / 最小长度） |
    # +------------+------------------------------------------------------+

    # 1. 加载dataset
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)   # 默认None
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    # 2. 设置transforms
    # transforms = tv_tensors.Image类型 > 自动缩放uint8(0-255) → float32(0-1) > ImageNet归一化 > resize 224
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            # 之前get_img_preprocessor中pixel字段已经imagenet均值/标准差归一化
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col) # 根据该类型的数据计算mean、std进行归一化
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action") # 5 * 2 = 10

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform
    #  3. 划分数据集
    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )
    # 4. 加载dataloader
    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################
    
    # 5. 加载模型
    # hydra.utils.instantiate方法分析传入的参数信息，递归参考_target_导入对应的类/方法，剩余参数作为类实例化/调用参数执行，返回对象/调用结果
    # from stable_pretraining.backbone.utils import vit_hf      encoder  == vit
    # from module import ARPredictor                            predictor   == dit
    # from module import embedder                               action embedder == MLP
    # from module import MLP                                    encoder projecter   == MLP
    # from module import MLP                                    prodictor projecter == MLP
    # from jepa import JEPA
    world_model = hydra.utils.instantiate(cfg.model)

    # 6. 设置优化器
    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }
    # 通过调用data_module.train_dataloader返回train dataloader；同理eval dataloader
    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    # 获取Path(cache_dir/checkpoints)对象，如果cache_dir为None则获取STABLEWM_HOME环境变量的Path对象，STABLEWM_HOME环境变量默认为os.path.expanduser('~/.stable_worldmodel')
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    # 设置保存CallBack
    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()

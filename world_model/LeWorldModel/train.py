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

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


def lejepa_forward(self, batch, stage, cfg):
    """
    1. encode observations,
    2. predict next states,
    3. compute losses
    """

    ctx_len = cfg.wm.history_size  # 3
    n_preds = cfg.wm.num_preds  # 1
    lambd = cfg.loss.sigreg.weight  # 0.09

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)  # world_model(spt.Module类的对象).model.encode

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]  # (B, T, D)

    ctx_emb = emb[:, :ctx_len]  # [:,:3]
    ctx_act = act_emb[:, : ctx_len]  # [:,:3]

    tgt_emb = emb[:, n_preds:]  # label [:,1:]
    # transformer模型，输入current token，预测next token，那么传进去3帧，就会预测下三帧
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # world_model(spt.Module类的对象).model.predict

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))  # world_model(spt.Module类的对象).sigreg
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    # 0. 加载参数
    # cfg有hydra构建，cfg.data指代lewm.yaml中defaults.data

    # +------------+------------------------------------------------------+
    # | dataset参数名 | 含义 |
    # +------------+------------------------------------------------------+
    # | num_steps | 最终想要多少个连续的数据点（模型输入的序列长度） |
    # +------------+------------------------------------------------------+
    # | frameskip | 从原始数据里，每隔几帧取1个点（跳帧降采样，降低数据量） |
    # +------------+------------------------------------------------------+
    # | span | 为了取到这些点，原始数据必须占多少连续帧（总跨度 / 最小长度） |
    # +------------+------------------------------------------------------+

    # 1. 加载dataset
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)

    # 2. 数据处理
    # 转换为Image类型 > 根据ImageNet的均值和标准差进行归一化 > resize
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            # 之前get_img_preprocessor中pixel字段已经imagenet均值/标准差归一化
            if col.startswith("pixels"):
                continue

            # 遍历字段，获取对应字段的数据，计算mean和std，用于归一化
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            # 通过np.prod(shape[1:])计算各字段的维度
            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    # 设置随机种子
    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    # 划分训练集/测试集
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    # 3. 获取dataloader
    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    # 4. 加载模型
    # 获取vit作为encoder，如果pretrained为True那么从huggingface下载模型，否则通过默认参数初始化vit
    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size  # tiny : 192
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)  # 192
    # 有效动作维度：表示模型每预测一次，环境实际执行的动作步数(frameskip)。即：5 * 2
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim  # action_dim来自81行

    # predictor是DiT模型
    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,  # 3
        input_dim=embed_dim,  # 192
        hidden_dim=hidden_dim,  # 192
        output_dim=hidden_dim,  # 192
        **cfg.predictor,
    )
    # action_encoder是Conv1d
    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    # encoder后的转换器
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )
    # predictor后的转换器
    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )
    # 组装模型
    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }
    # 封装数据
    data_module = spt.data.DataModule(train=train, val=val)
    # 封装为完整功能的模型
    world_model = spt.Module(
        model=world_model,  # 在spt.Module中设置self.model
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),  # 在spt.Module中设置self.sigreg
        forward=partial(lejepa_forward, cfg=cfg),  # 在spt.Module中指定训练的forward
        optim=optimizers,  # 在spt.Module中设置self.optim
    )

    # 5. 开始训练
    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    # 保存模型的Callback
    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )
    # 创建lightning的trainer
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )
    # 组装最终的trainer
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    return


if __name__ == "__main__":
    run()

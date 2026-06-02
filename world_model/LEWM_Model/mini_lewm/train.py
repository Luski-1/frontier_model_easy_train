import os
import time

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from model import (
    ARPredictor, Embedder, JEPA, MLP, SIGReg, create_encoder,
)
from dataset import make_dataloaders
from loss import compute_loss
from utils import (
    AverageMeter, create_optimizer, create_scheduler,
    load_config, save_checkpoint,
)


def build_model(cfg, action_dim):
    """根据配置构建 JEPA 模型。

    所有组件显式创建，不依赖 hydra.utils.instantiate。

    注意：这里的模型结构与原始 jepa.py + module.py 完全一致，
    因此 state_dict 的 key 完全相同，可以加载到原始 eval.py 的模型中。

    Parameters
    ----------
    cfg : dict
        训练配置
    action_dim : int
        原始 action 维度（pusht 为 2），frameskip * action_dim = action_encoder.input_dim

    Returns
    -------
    model : JEPA
    """
    encoder = create_encoder(
        size=cfg["encoder_size"],
        patch_size=cfg["encoder_patch_size"],
        image_size=cfg["img_size"],
        pretrained=cfg["encoder_pretrained"],
    )
    # Dit架构
    predictor = ARPredictor(
        num_frames=cfg["history_size"],
        input_dim=cfg["embed_dim"],
        hidden_dim=cfg["embed_dim"],
        output_dim=cfg["embed_dim"],
        depth=cfg["predictor_depth"],
        heads=cfg["predictor_heads"],
        mlp_dim=cfg["predictor_mlp_dim"],
        dim_head=cfg["predictor_dim_head"],
        dropout=cfg["predictor_dropout"],
        emb_dropout=cfg["predictor_emb_dropout"]
    )
    # cfg.data.dataset.frameskip * dataset.get_dim("action") # 5 * 2 = 10
    action_encoder = Embedder(
        input_dim=cfg["frameskip"] * action_dim,   # 5 * 2 = 10
        emb_dim=cfg["embed_dim"],
    )
    projector = MLP(
        input_dim=cfg["embed_dim"],
        hidden_dim=2048,
        output_dim=cfg["embed_dim"],
        norm_fn=nn.BatchNorm1d,
    )
    pred_proj = MLP(
        input_dim=cfg["embed_dim"],
        hidden_dim=2048,
        output_dim=cfg["embed_dim"],
        norm_fn=nn.BatchNorm1d,
    )

    return JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    )


def train():
    """纯 PyTorch 训练循环入口。"""

    # ---- 加载配置 ----
    config_path = os.path.join(os.path.dirname(__file__), "configs", "train.yaml")
    cfg = load_config(config_path)

    # ---- seed ----
    torch.manual_seed(cfg["seed"])

    # ---- 数据 ----
    train_loader, val_loader, action_dim = make_dataloaders(
        h5_path=cfg["h5_path"],
        frameskip=cfg["frameskip"],
        history_size=cfg["history_size"],
        num_preds=cfg["num_preds"],
        img_size=cfg["img_size"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        train_split=cfg["train_split"],
        seed=cfg["seed"],
        keys_to_load=cfg["keys_to_load"],
        keys_to_cache=cfg["keys_to_cache"]
    )

    # ---- 模型 ----
    model = build_model(cfg, action_dim).to(cfg["device"])
    sigreg = SIGReg(knots=cfg["sigreg_knots"], num_proj=cfg["sigreg_num_proj"])
    sigreg = sigreg.to(cfg["device"])

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"  encoder:  {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"  predictor: {sum(p.numel() for p in model.predictor.parameters()):,}")
    print(f"  action_encoder: {sum(p.numel() for p in model.action_encoder.parameters()):,}")
    print(f"  projector: {sum(p.numel() for p in model.projector.parameters()):,}")
    print(f"  pred_proj: {sum(p.numel() for p in model.pred_proj.parameters()):,}")

    # ---- 优化器 + 调度器 ----
    optimizer = create_optimizer(model, cfg)
    scheduler = create_scheduler(optimizer, cfg, cfg["max_epochs"], len(train_loader))

    # ---- 混合精度 ----
    use_bf16 = cfg["precision"] == "bf16"
    scaler = GradScaler() if use_bf16 else None

    # ---- 训练循环----
    print(f"\nTraining start: {cfg['max_epochs']} epochs, {len(train_loader)} steps/epoch")
    print(f"  history_size={cfg['history_size']}, num_preds={cfg['num_preds']}, "
          f"frameskip={cfg['frameskip']}, span={(cfg['history_size'] + cfg['num_preds']) * cfg['frameskip']}")
    print(f"  sigreg_weight={cfg['sigreg_weight']}, lr={cfg['lr']}, "
          f"warmup={cfg.get('warmup', 0.01)*100:.0f}%")
    print()

    steps_per_epoch = len(train_loader)
    log_interval = max(steps_per_epoch // 100, 1)

    for epoch in range(cfg["max_epochs"]):
        epoch_start = time.time()

        # === train ===
        model.train()
        loss_meter = AverageMeter()
        pred_loss_meter = AverageMeter()
        sigreg_loss_meter = AverageMeter()

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(cfg["device"]) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)

            if use_bf16:  # bf16 混合精度
                with autocast("cuda", dtype=torch.bfloat16):
                    output = compute_loss(
                        model, sigreg, batch,
                        history_size=cfg["history_size"],
                        num_preds=cfg["num_preds"],
                        sigreg_weight=cfg["sigreg_weight"],
                    )
                scaler.scale(output["loss"]).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_val"])
                scaler.step(optimizer)
                scaler.update()
            else:        # fp32
                output = compute_loss(
                    model, sigreg, batch,
                    history_size=cfg["history_size"],
                    num_preds=cfg["num_preds"],
                    sigreg_weight=cfg["sigreg_weight"],
                )
                output["loss"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_val"])
                optimizer.step()

            scheduler.step()

            loss_meter.update(output["loss"].item())
            pred_loss_meter.update(output["pred_loss"].item())
            sigreg_loss_meter.update(output["sigreg_loss"].item())

            if (batch_idx + 1) % log_interval == 0 or batch_idx + 1 == steps_per_epoch:
                elapsed = time.time() - epoch_start
                print(
                    f"Epoch {epoch+1:>3d}/{cfg['max_epochs']} "
                    f"[{batch_idx+1:>4d}/{steps_per_epoch}] "
                    f"train_loss={loss_meter.avg:.4f} "
                    f"pred_loss={pred_loss_meter.avg:.4f} "
                    f"sigreg_loss={sigreg_loss_meter.avg:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} "
                    f"time={elapsed:.1f}s"
                )

        # === val ===
        model.eval()
        val_loss_meter = AverageMeter()
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(cfg["device"]) for k, v in batch.items()}
                if use_bf16:
                    with autocast("cuda", dtype=torch.bfloat16):
                        output = compute_loss(
                            model, sigreg, batch,
                            history_size=cfg["history_size"],
                            num_preds=cfg["num_preds"],
                            sigreg_weight=cfg["sigreg_weight"],
                        )
                else:
                    output = compute_loss(
                        model, sigreg, batch,
                        history_size=cfg["history_size"],
                        num_preds=cfg["num_preds"],
                        sigreg_weight=cfg["sigreg_weight"],
                    )
                val_loss_meter.update(output["loss"].item())
        print(f"  val_loss={val_loss_meter.avg:.4f}")

        # === save ===
        if (epoch + 1) % cfg["save_every"] == 0:
            save_dir = os.path.join(
                cfg["output_dir"], cfg["output_model_name"], f"epoch_{epoch+1}"
            )
            save_checkpoint(model, optimizer, scheduler, epoch + 1, save_dir, cfg, action_dim)

    print("\nTraining complete.")


if __name__ == "__main__":
    train()

import torch


def compute_loss(model, sigreg, batch, history_size, num_preds, sigreg_weight):
    """encode observations, predict next states, compute losses.

    Parameters
    ----------
    model : JEPA
        JEPA 世界模型
    sigreg : SIGReg
        Sketch Isotropic Gaussian Regularizer
    batch : dict
        包含 'pixels' (B, T, C, H, W) 和 'action' (B, T, D) 等
    history_size : int
        上下文步数（如 3）
    num_preds : int
        预测步数（如 1）
    sigreg_weight : float
        SIGReg 正则权重 lambda（如 0.09）

    Returns
    -------
    dict : 包含 'loss', 'pred_loss', 'sigreg_loss'
    """

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"] # [B, T, D]

    ctx_emb = emb[:, :history_size]  # x of pixel
    ctx_act = act_emb[:, : history_size] # x of action

    tgt_emb = emb[:, num_preds:] # y of pixel
    pred_emb = model.predict(ctx_emb, ctx_act) # pred of pixel

    # LeWM loss
    pred_loss = (pred_emb - tgt_emb).pow(2).mean() # MSE LOSS
    sigreg_loss = sigreg(emb.transpose(0, 1)) # SIGReg LOSS
    loss = pred_loss + sigreg_weight * sigreg_loss

    return {
        "loss": loss,
        "pred_loss": pred_loss.detach(),
        "sigreg_loss": sigreg_loss.detach(),
    }

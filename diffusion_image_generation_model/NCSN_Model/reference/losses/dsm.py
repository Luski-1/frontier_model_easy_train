import torch
import torch.autograd as autograd


def dsm(energy_net, samples, sigma=1):
    samples.requires_grad_(True)
    vector = torch.randn_like(samples) * sigma
    perturbed_inputs = samples + vector
    logp = -energy_net(perturbed_inputs)
    dlogp = sigma ** 2 * autograd.grad(logp.sum(), perturbed_inputs, create_graph=True)[0]
    kernel = vector
    loss = torch.norm(dlogp + kernel, dim=-1) ** 2
    loss = loss.mean() / 2.

    return loss


def dsm_score_estimation(scorenet, samples, sigma=0.01):
    perturbed_samples = samples + torch.randn_like(samples) * sigma
    target = - 1 / (sigma ** 2) * (perturbed_samples - samples)
    scores = scorenet(perturbed_samples)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1).mean(dim=0)

    return loss


def anneal_dsm_score_estimation(scorenet, samples, labels, sigmas, anneal_power=2.):
    """
    scorenet: 模型
    samples: 输入数据   [B, C, H, W]
    labels: sigma下标   [B]
    sigmas: sigma等比序列   [10]
    anneal_power: 2.0
    """
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))    # [B, C, H, W]
    perturbed_samples = samples + torch.randn_like(samples) * used_sigmas   # x_t = x_0 + ε * σ  ; ε ~ N(0, I)
    target = - 1 / (used_sigmas ** 2) * (perturbed_samples - samples)       # score: - (x_t - x_0) / σ^2
    scores = scorenet(perturbed_samples, labels)                            # label是sigma级别，在模型中就是分类信息
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    # 1) 1 / 2. * ((scores - target) ** 2) 即模型预测与目标score的MSE损失
    # 2.1) used_sigmas.squeeze() ** 2 即score的分子x_t - x_0是σ级别，score的分母是σ^2级别，分子/分母=1/σ级别
    # 2.2) MSE损失是平方，|scores - target|^2是1/σ^2级别，那么σ越小导致损失越大，因此需要×上used_sigmas.squeeze() ** anneal_power
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

    return loss.mean(dim=0)

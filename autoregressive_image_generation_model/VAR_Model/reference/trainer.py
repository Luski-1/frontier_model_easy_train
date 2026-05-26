import time
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import dist
from models import VAR, VQVAE, VectorQuantizer2
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger, TensorboardLogger

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor


class VARTrainer(object):
    def __init__(
            self,
            device,  # 当前 GPU 设备
            patch_nums: Tuple[int, ...],  # VAR生成图像的不同阶段的stride/尺寸，如 (1,2,3,4,5,6,8,10,13,16)
            resos: Tuple[int, ...],  # 不同生成阶段的stride/尺寸，恢复下采样倍率后的长度/分辨率，如 (16,32,48,64,80,96,128,160,208,256)
            vae_local: VQVAE,  # 本地 VQVAE（无 DDP 包装，用于推理/编码）
            var_wo_ddp: VAR,  # 本地 VAR（无 DDP 包装，可能经过 torch.compile）
            var: DDP,  # DDP 包装后的 VAR（用于分布式训练）
            var_opt: AmpOptimizer,  # 混合精度优化器
            label_smooth: float,  # 标签平滑系数（防止过拟合）
    ):
        super(VARTrainer, self).__init__()

        self.var, self.vae_local, self.quantize_local = var, vae_local, vae_local.quantize
        self.quantize_local: VectorQuantizer2
        self.var_wo_ddp: VAR = var_wo_ddp  # after torch.compile
        self.var_opt = var_opt

        del self.var_wo_ddp.rng
        self.var_wo_ddp.rng = torch.Generator(device=device)

        self.label_smooth = label_smooth  # 0 不使用标签平滑
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')

        # self.L (length) 1²+2²+3²+...+16² = 1+4+9+...+256 = 666
        self.L = sum(pn * pn for pn in patch_nums)  
        # self.last_l (last level length) = 16*16 = 256
        self.last_l = patch_nums[-1] * patch_nums[-1]

        # 损失权重：初始化为全 1，除以总 Token 数（平均每个 Token 的损失）
        self.loss_weight = torch.ones(1, self.L, device=device) / self.L

        # self.patch_nums = (1,2,3,4,5,6,8,10,13,16)
        # self.resos = (16,32,48,64,80,96,128,160,208,256)
        self.patch_nums, self.resos = patch_nums, resos

        # 计算每个尺度的 Token 起止索引（和 VAR 里的 begin_ends 完全一致）
        # 例如：[(0,1), (1,5), (1,14), ..., (410,666)]
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn * pn

        self.prog_it = 0  # 渐进式训练中，记录当前阶段/尺寸/stride的训练步数
        self.last_prog_si = -1  # 渐进式训练中，记录上一个阶段/尺寸/stride
        self.first_prog = True  # 渐进式训练中，记录是否为第一个阶段/尺寸/stride（第一个阶段不需要 warmup）

    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        training = self.var_wo_ddp.training
        self.var_wo_ddp.eval()
        for inp_B3HW, label_B in ld_val:
            B, V = label_B.shape[0], self.vae_local.vocab_size
            inp_B3HW = inp_B3HW.to(dist.get_device(), non_blocking=True)
            label_B = label_B.to(dist.get_device(), non_blocking=True)

            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)

            logits_BLV = self.var_wo_ddp(label_B, x_BLCv_wo_first_l)
            L_mean += self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)) * B
            L_tail += self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V),
                                    gt_BL[:, -self.last_l:].reshape(-1)) * B
            acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100 / gt_BL.shape[1])
            acc_tail += (logits_BLV.data[:, -self.last_l:].argmax(dim=-1) == gt_BL[:, -self.last_l:]).sum() * (
                    100 / self.last_l)
            tot += B
        self.var_wo_ddp.train(training)

        stats = L_mean.new_tensor([L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()
        return L_mean, L_tail, acc_mean, acc_tail, tot, time.time() - stt

    def train_step(
            self,
            it: int,  # 当前epoch内第几步
            g_it: int,  # 已完成训练的总步数
            stepping: bool,  # 是否参数更新
            metric_lg: MetricLogger,
            tb_lg: TensorboardLogger,
            inp_B3HW: FTen,  # 图像数据 [B3HW]
            label_B: Union[ITen, FTen],  # 图像标签[B]
            prog_si: int,  # 渐进式训练的最大尺度/stride，默认-1，即全阶段/尺寸/stride训练
            prog_wp_it: float,  # 渐进式训练的各尺度/stride 的warmup总步数
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        # if progressive training
        # ========== 1.1 设置当前渐进式训练尺度 ==========
        # 同时设置 VAR 和 VQVAE Quantizer 的 prog_si，保证两者一致
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = prog_si

        # ========== 1.2 判断是否进入新的阶段/尺寸/stride ==========
        if self.last_prog_si != prog_si:
            # 如果不是第一个阶段，标记 first_prog=False
            if self.last_prog_si != -1:
                self.first_prog = False
            # 更新当前阶段/尺寸/stride
            self.last_prog_si = prog_si
            # 重置当前阶段的迭代次数
            self.prog_it = 0

        # ========== 1.3 更新当前阶段/尺寸/stride的迭代次数 ==========
        self.prog_it += 1
        # ========== 1.4 计算渐进式训练的 Warmup 系数 ==========
        # prog_wp：当前阶段/尺寸/stride的 Warmup 系数（从 0.01 到 1），实际上是不同尺寸/stride的损失加权，与self.loss_weight相乘
        prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
        # ========== 1.5 第一个阶段或全阶段训练不需要 阶段式Warmup ==========
        if self.first_prog:
            prog_wp = 1

        # ========== 1.6 判断是否全阶段训练 prog_si 设为 -1 ==========
        # 表示训练所有尺度
        if prog_si == len(self.patch_nums) - 1:
            prog_si = -1

        # forward
        # ========== 2.1 获取 Batch 大小和码本大小 ==========
        B, V = label_B.shape[0], self.vae_local.vocab_size
        # ========== 2.2 控制 DDP 分布式梯度同步（关键！） ==========
        self.var.require_backward_grad_sync = stepping
        # ========== 2.3 图像编码为真实 Token（Ground Truth） ==========
        # 获得prog_si控制下的阶段/尺寸/stirde的token id [[B,1], [B,4], [B,9]...]
        gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
        # [B, L] L是总token数
        gt_BL = torch.cat(gt_idx_Bl, dim=1)
        # 转换为 VAR 的训练输入数据（去掉第一个尺度的 Token，因为第一个尺度由 SOS 代替）
        # x_BLCv_wo_first_l: 维度[B,L,Cvae] without first level token
        x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)

        with self.var_opt.amp_ctx:
            logits_BLV = self.var(label_B, x_BLCv_wo_first_l)   # [B,L,V]
            loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1) # [B,L]
            # 如果开展阶段式训练
            if prog_si >= 0:  # in progressive training
                bg, ed = self.begin_ends[prog_si]
                assert logits_BLV.shape[1] == gt_BL.shape[1] == ed
                lw = self.loss_weight[:, :ed].clone() # 不同阶段/尺寸/stride的损失比例
                # 不同阶段/stride有对应的warmup的区间，但控制的是损失而不是学习率
                lw[:, bg:ed] *= min(max(prog_wp, 0), 1)
            else:  # not in progressive training
                lw = self.loss_weight # 不同阶段/尺寸/stride的损失比例
            loss = loss.mul(lw).sum(dim=-1).mean()

        # backward
        # 返回的是，裁剪前的梯度范数，log缩放系数
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=loss, stepping=stepping)

        # log
        pred_BL = logits_BLV.data.argmax(dim=-1)
        if it == 0 or it in metric_lg.log_iters:
            Lmean = self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)).item()
            acc_mean = (pred_BL == gt_BL).float().mean().item() * 100
            if prog_si >= 0:  # in progressive training
                Ltail = acc_tail = -1
            else:  # not in progressive training
                Ltail = self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V),
                                      gt_BL[:, -self.last_l:].reshape(-1)).item()
                acc_tail = (pred_BL[:, -self.last_l:] == gt_BL[:, -self.last_l:]).float().mean().item() * 100
            grad_norm = grad_norm.item()
            metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm)

        # log to tensorboard
        if g_it == 0 or (g_it + 1) % 500 == 0:
            prob_per_class_is_chosen = pred_BL.view(-1).bincount(minlength=V).float()
            dist.allreduce(prob_per_class_is_chosen)
            prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()
            cluster_usage = (prob_per_class_is_chosen > 0.001 / V).float().mean().item() * 100
            if dist.is_master():
                if g_it == 0:
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-10000)
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-1000)
                kw = dict(z_voc_usage=cluster_usage)
                for si, (bg, ed) in enumerate(self.begin_ends):
                    if 0 <= prog_si < si: break
                    pred, tar = logits_BLV.data[:, bg:ed].reshape(-1, V), gt_BL[:, bg:ed].reshape(-1)
                    acc = (pred.argmax(dim=-1) == tar).float().mean().item() * 100
                    ce = self.val_loss(pred, tar).item()
                    kw[f'acc_{self.resos[si]}'] = acc
                    kw[f'L_{self.resos[si]}'] = ce
                tb_lg.update(head='AR_iter_loss', **kw, step=g_it)
                tb_lg.update(head='AR_iter_schedule', prog_a_reso=self.resos[prog_si], prog_si=prog_si, prog_wp=prog_wp,
                             step=g_it)
        # 重新设置prog_si，因为外部epoch会实时计算并传入prog_si
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = -1
        return grad_norm, scale_log2

    def get_config(self):
        return {
            'patch_nums': self.patch_nums, 'resos': self.resos,
            'label_smooth': self.label_smooth,
            'prog_it': self.prog_it, 'last_prog_si': self.last_prog_si, 'first_prog': self.first_prog,
        }

    def state_dict(self):
        state = {'config': self.get_config()}
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state

    def load_state_dict(self, state, strict=True, skip_vae=False):
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            if skip_vae and 'vae' in k: continue
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[VARTrainer.load_state_dict] {k} missing:  {missing}')
                    print(f'[VARTrainer.load_state_dict] {k} unexpected:  {unexpected}')

        config: dict = state.pop('config', None)
        self.prog_it = config.get('prog_it', 0)
        self.last_prog_si = config.get('last_prog_si', -1)
        self.first_prog = config.get('first_prog', True)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VAR.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err)

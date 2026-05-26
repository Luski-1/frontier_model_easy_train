import math
from typing import List, Optional, Tuple, Union

import torch


class NullCtx:
    def __enter__(self):
        pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class AmpOptimizer:
    def __init__(
            self,
            mixed_precision: int,  # 混合精度模式：0=关闭,1=FP16,2=BF16
            optimizer: torch.optim.Optimizer,  # 底层真实优化器（AdamW）
            names: List[str], paras: List[torch.nn.Parameter],  # 可训练的参数名字，可训练的参数
            grad_clip: float,  # 梯度裁剪阈值
            n_gradient_accumulation: int = 1,  # 梯度累积步数
    ):
        # ========== 1. 混合精度配置 ==========
        self.enable_amp = mixed_precision > 0  # 是否开启混合精度
        self.using_fp16_rather_bf16 = mixed_precision == 1  # 区分FP16/BF16

        # 初始化AMP上下文（autocast）+ 梯度缩放器（仅FP16需要）
        if self.enable_amp:
            # autocast：自动把计算切到FP16/BF16，省显存/加速
            self.amp_ctx = torch.autocast(
                'cuda', enabled=True,
                dtype=torch.float16 if self.using_fp16_rather_bf16 else torch.bfloat16,
                cache_enabled=True
            )
            # GradScaler：【仅FP16需要】解决梯度下溢问题（FP16数值范围小，梯度容易变成0）
            # init_scale=2**11=2048.0：初始缩放因子；
            # growth_interval=1000：每连续 1000 步无溢出，才尝试放大 scale。
            self.scaler = torch.cuda.amp.GradScaler(init_scale=2 ** 11,
                                                    growth_interval=1000) if self.using_fp16_rather_bf16 else None
        else:
            self.amp_ctx = NullCtx()  # 关闭精度，用空上下文
            self.scaler = None # 不需要梯度缩放

        # ========== 2. 保存核心成员 ==========
        self.optimizer, self.names, self.paras = optimizer, names, paras  # 可训练参数
        self.grad_clip = grad_clip  # 梯度裁剪阈值 2

        # 梯度裁剪模式：
        # early_clipping：普通优化器 → 用torch自带clip_grad_norm_，需要在optimizer.step()前手动执行裁剪
        # late_clipping：特殊优化器（带global_grad_norm）→ 用优化器自带裁剪，因此无需提前执行torch.clip，在optimizer.step()自行执行裁剪
        self.early_clipping = self.grad_clip > 0 and not hasattr(optimizer, 'global_grad_norm')
        self.late_clipping = self.grad_clip > 0 and hasattr(optimizer, 'global_grad_norm')

        # ========== 3. 梯度累积系数 ==========
        # 梯度累积：把N步的梯度累加，等效于大Batch，公式：loss = loss * (1/N)
        self.r_accu = 1 / n_gradient_accumulation# r_accu == 1.0 / n_gradient_accumulation

    def backward_clip_step(
            self,
            stepping: bool,  # 是否更新参数（True=累积满N步，更新；False=只累加梯度）
            loss: torch.Tensor,  # 模型损失
    ) -> Tuple[Optional[torch.Tensor, float], Optional[float]]:
        # ========== 1. 梯度累积：Loss 乘以系数 ==========
        loss = loss.mul(self.r_accu)  # 等效：loss = loss / 累积步数
        orig_norm = scaler_sc = None

        # ========== 2. 反向传播（自动处理混合精度）==========
        if self.scaler is not None:
            # FP16：缩放Loss后再反向传播（防止梯度下溢）
            self.scaler.scale(loss).backward()
        else:
            # BF16/FP32：直接反向传播
            loss.backward()

        # ========== 3. 仅当 stepping=True（累积满步），才更新参数 ==========
        if stepping:
            # FP16：先取消梯度缩放（才能正常裁剪/更新）
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)

            # ========== 4. 早期梯度裁剪（普通优化器）==========
            if self.early_clipping:
                # 执行裁剪，并返回裁剪前的梯度
                orig_norm = torch.nn.utils.clip_grad_norm_(self.paras, self.grad_clip)

            # ========== 5. 优化器更新参数（自动处理FP16缩放）==========
            if self.scaler is not None:
                self.scaler.step(self.optimizer)  # FP16：取消缩放后更新参数
                # 安全限制：FP16最大缩放值不超过32768（防止溢出）
                scaler_sc: float = self.scaler.get_scale()
                if scaler_sc > 32768.:
                    self.scaler.update(new_scale=32768.)
                else:
                    self.scaler.update()  # 自动调整缩放比例
                scaler_sc = float(math.log2(scaler_sc))  # 记录log2值，方便打印
            else:
                self.optimizer.step()  # BF16/FP32：直接更新

            # ========== 6. 晚期梯度裁剪（特殊优化器）==========
            if self.late_clipping:
                # 直接获取裁剪前的梯度即可
                orig_norm = self.optimizer.global_grad_norm

            # ========== 7. 清空梯度 ==========
            self.optimizer.zero_grad(set_to_none=True)

        # 返回：梯度范数 + FP16缩放系数（用于日志打印）
        return orig_norm, scaler_sc
    
    def state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict()
        } if self.scaler is None else {
            'scaler': self.scaler.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
    
    def load_state_dict(self, state, strict=True):
        if self.scaler is not None:
            try: 
                self.scaler.load_state_dict(state['scaler'])
            except Exception as e: 
                print(f'[fp16 load_state_dict err] {e}')
        self.optimizer.load_state_dict(state['optimizer'])

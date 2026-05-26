import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

import dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)  # B16C


class VAR(nn.Module):
    def __init__(
            self,
            vae_local: VQVAE,
            num_classes=1000,
            depth=16,
            embed_dim=1024,
            num_heads=16,
            mlp_ratio=4.,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_eps=1e-6,
            shared_aln=False,
            cond_drop_rate=0.1,
            attn_l2_norm=False,
            patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # 10 steps by default
            flash_if_available=True, fused_if_available=True,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0
        # self.Cvae = 32, VAE的latent维度
        # self.V = 4096 VAE词表
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        # self.depth = 16
        # self.C = 1024,  token embedding dim
        # self.D = 1024, condition embedding dim
        # self.num_heads = 16
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        # self.cond_drop_rate = 0.1 CFG概率
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1  # 是否开启渐进式训练 progressive train
        # self.patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        self.patch_nums: Tuple[int] = patch_nums
        # self.L(length) = sum(1, 4, 9, 16, 25, 36, 64, 100, 169, 256) 计算预测的总token数
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        # self_first_l = 1 第1个尺寸/阶段/stride需要预测的token数
        self.first_l = self.patch_nums[0] ** 2
        # self.begin_ends = [(0,1), (1, 5)...] 每stride的区间下标
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur + pn ** 2))  # [(0,1), (1, 5)...]，每stride的区间下标
            cur += pn ** 2
        # self.num_stages_minus_1 = 9 用于推理阶段计算当前阶段/尺寸/stride在全阶段的占比
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        # 随机生成器
        self.rng = torch.Generator(device=dist.get_device())

        # input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        # [32, 1024]
        self.word_embed = nn.Linear(self.Cvae, self.C)  

        # 计算初始化方差
        # 目的是让总方差=1，每一个token的输入包含3部分：token embedding、stride/level embedding、 absolute position embedding，因此需要/3
        init_std = math.sqrt(1 / self.C / 3)
        # 类别数量 1000
        self.num_classes = num_classes
        # 维度[1, 1000]的0.001 用于推理阶段没有显式传入分类标签时进行抽样
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32,
                                       device=dist.get_device()) 
        # [1001, 1024] 类别Embedding，+1代表新增'无分类'
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        # [1, 1, 1024] 起始token
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)

        # absolute position embedding，即把所有尺寸/stride对应的token数总和作为总长度
        # self.pos_1LC的维度 [1, L(length), C]
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn * pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)  # 维度[1, L(length), C]
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)

        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        # self.lvl_embed 维度[10(level,即stride阶段), 1024]
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)

        # 大多数情况下shared_aln=False，当训练512*512时即开启共享adaLN
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False),
                                            SharedAdaLin(self.D, 6 * self.C) # 维度[self.D(1024), 6*self.C(6144)] 后续拆分为6个参数控制attention和FFN
                                            ) if shared_aln else nn.Identity() # 维度[B,1,6,C]

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate  # 层(深度)dropput 0.1 * depth / 24 ≈ 0.066
        # dpr：层(深度)dropout的线性递增，层数越小，dropout概率越小
        dpr = [x.item() for x in
               torch.linspace(0, drop_path_rate, depth)]
        # DIT风格的adaLN
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D,  # 1024
                shared_aln=shared_aln,  # False
                block_idx=block_idx,
                embed_dim=self.C,  # 1024
                norm_layer=norm_layer, # layerNorm
                num_heads=num_heads,  # 16
                mlp_ratio=mlp_ratio,  # 4 控制FFN的中间维度的倍数
                drop=drop_rate,  # 0 linear 的drouput概率，例如W_O, FFN
                attn_drop=attn_drop_rate,  # 0 Attention 的dropout概率
                drop_path=dpr[block_idx], # dpr[x]
                last_drop_p=0 if block_idx == 0 else dpr[block_idx - 1], # 整个流程没使用到
                attn_l2_norm=attn_l2_norm,  # True
                flash_if_available=flash_if_available,
                fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])

        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]  # [False, ...]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )

        # 获得Attention mask，使得相同尺寸/stride内token互相看见，大尺寸/stride可以看见小尺寸/stride内的token
        # Attention mask不会应用到推理阶段，因为开启了KV cache
        d: torch.Tensor = torch.cat(
            [torch.full((pn * pn,), i) for i, pn in enumerate(self.patch_nums)] # [0, | 1,1,1,1 | 2,2,2,2,2,2,2,2,2 | 3,... | ... | 9,...]
        ).view(1, self.L, 1)  # [[[0],[1],[1],[1],[1],[2],[2],[2],[2],[2],[2],[2],[2],[2].....]]
        dT = d.transpose(1, 2)  # dT: [1,1,L]
        lvl_1L = dT[:, 0].contiguous()  # 1L, 1是batch, [[0,1,1,1,1,2,2,2,2,2,2,2,2....]]，记录不同token属于什么尺寸/stride
        self.register_buffer('lvl_1L', lvl_1L)
        # d [1, L, 1] 和 dT [1, 1, L] 会自动广播为 [1, L, L]，参考以下mask矩阵
        # 0 - -
        # - 0 0
        # - 0 0
        # 维度[1, 1, L, L]
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())

        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)  # layerNorm，偏移与缩放由adaLN控制
        self.head = nn.Linear(self.C, self.V)  # 预测头，预测稀疏token，4096

    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                   cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual  # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:  # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()

    @torch.no_grad()
    def autoregressive_infer_cfg(
            self,
            B: int,  # Batch 大小
            label_B: Optional[Union[int, torch.LongTensor]],  # 生成图像的类别标签（None=随机，int=指定类别）
            g_seed: Optional[int] = None,  # 随机种子（保证可复现性）
            cfg=1.5,  # Classifier-Free Guidance 强度（1.5=默认，越大越贴合类别但多样性降低）
            top_k=0,  # top-k 采样（0=禁用）
            top_p=0.0,  # top-p 采样（0.0=禁用）
            more_smooth=False,  # 是否用 Gumbel Softmax 平滑（仅用于可视化，不用于 FID/IS 评估）
    ) -> torch.Tensor:  # 返回生成的图像 [B, 3, H, W]，范围 [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        # ========== 1.1 初始化随机数生成器（保证可复现性） ==========
        if g_seed is None:
            rng = None  # 不指定种子，完全随机
        else:
            self.rng.manual_seed(g_seed)  # 指定种子，保证可复现
            rng = self.rng

        # ========== 1.2 处理生成标签 ==========
        # [B]
        if label_B is None:
            # 标签为 None：随机采样类别（均匀分布）
            label_B = torch.multinomial(
                self.uniform_prob,  # [1, 1000]，均匀分布
                num_samples=B,  # 采样 B 个
                replacement=True,  # 可重复
                generator=rng
            ).reshape(B)
        elif isinstance(label_B, int):
            # 标签为 int：指定类别（所有样本都生成这个类别）
            label_B = torch.full(
                (B,),
                fill_value=self.num_classes if label_B < 0 else label_B,  # label_B < 0 时用无条件标签
                device=self.lvl_1L.device
            )

        # ========== 2.1 构建 CFG 双 Batch 条件向量 ==========
        # 解释：
        # - 前 B 个：条件生成（用真实标签）
        # - 后 B 个：无条件生成（用 self.num_classes 标签）
        # 拼接起来：[label_B, label_B_uncond]
        sos = cond_BD = self.class_emb(
            torch.cat(
                (label_B, torch.full_like(label_B, fill_value=self.num_classes)),
                dim=0
            )
        )
        # sos = cond_BD 形状：[2*B, C]（C=1024）

        # ========== 3.1 预计算层级编码 + 绝对位置编码 ==========
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        # lvl_pos 形状：[1, L, C]（L=length，C=1024）
        # ========== 3.2 初始化起始输入（分类标签信息 + 生成开始信号+ 层级编码 + 绝对编码） ==========
        # next_token_map 形状：[2*B, first_l, C]（first_l=1，对应阶段/尺寸/stride 0的要生成的token数量）
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) \
                         + self.pos_start.expand(2 * B, self.first_l, -1) \
                         + lvl_pos[:, :self.first_l]

        # ========== 3.3 初始化当前 Token 长度和重建特征图 ==========
        cur_L = 0  # 当前已生成的 Token 长度
        # f_hat 形状：[B, Cvae, H, W]（H=W=16，对应encoder输出的latent的维度，也是decoder输入的维度）
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        
        # ========== 4.1 开启所有 Transformer block 的 KV Cache ==========
        for b in self.blocks:
            b.attn.kv_caching(True)
        # ========== 5.1 遍历每个尺度（从粗到细） ==========
        for si, pn in enumerate(self.patch_nums):
            ratio = si / self.num_stages_minus_1  # 当前阶段/尺寸/stride在全阶段的占比，用于计算抽样的温度
            cur_L += pn * pn  # 更新当前已生成的 Token 长度

            # ========== 5.2 处理共享 AdaLN（仅 512x512 模型开启） ==========
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)

            # ========== 5.3 Transformer 前向传播（用 KV Cache） ==========
            x = next_token_map
            for b in self.blocks:
                # 注意：attn_bias=None，因为推理时用 KV Cache，不需要因果掩码
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
                
            # ========== 5.4 计算 logits ==========
            logits_BlV = self.get_logits(x, cond_BD)
            # logits_BlV 形状：[2*B, cur_L, V]（V=4096，码本大小）

            # ========== 5.5 应用 Classifier-Free Guidance (CFG) ==========
            t = cfg * ratio  # CFG 强度随尺度增加（细尺度用更强的 CFG）
            logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]
            # 解释：
            # - logits_BlV[:B]：条件生成的 logits
            # - logits_BlV[B:]：无条件生成的 logits
            # - 公式：logits = (1+t)*logits_cond - t*logits_uncond
            # - 效果：放大条件信号，抑制无条件噪声，生成更贴合类别的图像

            # idx_Bl 维度[B,l(当前阶段/尺寸/stride的token数)]，通过top_p和top_k在V维度剔除不符合要求的候选，随后softmax采样
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]

            # h_BChw 维度[B, pn*pn, Cvae]（Cvae=32）
            if not more_smooth:  # this is the default case
                # [B, l, Cvae]
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)  
            else:  # not used when evaluating FID/IS/Precision/Recall
                # gum_t 从0.27 > 0.05
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
                # hard控制返回的是硬标签还是软标签
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ \
                         self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            # h_BChw 维度[B, Cvae, pn, pn]
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)  

            # f_hat是重建信息的累计，next_token_map是下一个尺寸的输入
            # f_hat [B,Cvae(32),H(16),W(16)]
            # next_token_map [B, Cvae(32), pn+1, pn+1)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums),
                                                                                              f_hat, h_BChw)
            # 非最后阶段/尺寸/stride才需要整理输入信息
            if si != self.num_stages_minus_1:  # prepare for next stage
                # [B, Cvae(32), pn+1, pn+1] > [B, Cvae(32), pn+1*pn+1], [B,pn+1*pn+1,C)]
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                # [B, pn+1*pn+1, Cvae] > [B, pn+1*pn+1, C(1024)] + 对应长度的层级编码 + 绝对位置编码
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:,cur_L:cur_L + self.patch_nums[si + 1] ** 2]
                # [2B, pn+1*pn+1, C]
                next_token_map = next_token_map.repeat(2, 1, 1)  # 双 Batch，用于 CFG

        for b in self.blocks: b.attn.kv_caching(False)
        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)  # de-normalize, from [-1, 1] to [0, 1]

    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:  # returns logits_BLV
        """
        :param label_B: 分类标签，维度[B]
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae) # 输入数据
        :return: logits BLV, V is vocab_size
        """
        # bg没什么用
        # 如果开启渐进式训练，ed就是截止可允许的阶段/尺寸/stride的输入x的有效末尾
        # 如果不开启渐进式训练，ed就是完整x
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)  # prog_si=-1代表训练所有尺度
        B = x_BLCv_wo_first_l.shape[0]
        with torch.cuda.amp.autocast(enabled=False):
            # CFG训练的概率，低于cond_drop_rate就设置num_classes，即1000，否则保持原始label
            label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
            # 构建起始sos embedding
            # 先获取分类向量
            sos = cond_BD = self.class_emb(label_B)  # 获取分类信息condition embedding的向量 [B, 1024]
            # 再扩增到first level/stride的token数量的维度，再加上生成开始信号 pos_start embedding
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)

            if self.prog_si == 0:
                x_BLC = sos
            else:
                # x_BLCv_wo_first_l [B, L - 1, 32] > [B, L - 1, 1024]
                # x_BLC [B, L, 1024]
                x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
                # 加上level/stride embedding和absolute position embedding
                # self.lvl_1L 维度[1, L] self.lvl_embed 维度[10, C(1024)]
                # self.pos_1LC 维度[1, L, C]
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed]  # lvl: BLC;  pos: 1LC
        # 获取对应的mask
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        # 如果是训练512*512就会开启share_ada_ln，就是提前把条件向量升维到6*1024
        # 这样就能使得每个block的adaln只需要设置nn.parameter而不是nn.linear
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)

        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype

        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        for i, b in enumerate(self.blocks):
            # cond_BD_or_gss用于给每个block获取adaLN的6个参数
            # 如果shared_aln=True，那么cond_BD_or_gss维度是[B,1,6,C]可直接与block中self.ada_gss相加
            # 如果shared_aln=True，那么cond_BD_or_gss维度是[B,C]可直接与block中self.ada_lin[C,6C]进行矩阵相乘
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        #  归一化+输出分类，传入的是原始分类信息cond_BD
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)

        if self.prog_si == 0:  # prog_si = -1
            if isinstance(self.word_embed, nn.Linear):
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                x_BLC[0, 0, 0] += s
        return x_BLC  # logits BLV, V is vocab_size, L is all token

    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        # 基于He初始化的变体，保证输入输出方差一致。
        # 输入的embedding，除了token embdding，还有level embedding和absulute embedding，所以/3
        # sqrt(1/(3*1024)) ≈ 0.018，接近手动设置的 0.02
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5  # init_std < 0: automated

        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)  # 只取 [-2*std, 2*std] 范围内的值，避免极端值导致训练初期不稳定
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()  # padding 位置强制为 0，避免干扰
            elif isinstance(m, (
                    nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm,
                    nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                # 归一化层初始化为恒等变换：y = 1*x + 0，保证训练初期网络行为稳定
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (
                    nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0:
                    nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else:
                    nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()

        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)  # 将输出头的初始权重* 0.02，保证输出头的初始预测接近均匀分布
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()

        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)  # 将输出头前layerNorm的shift/scale的权重 *乘以 0.5，保证输出头的初始预测接近均匀分布
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()

        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            # 方差选择2 * depth的原因：每一层都会增加方差，并且每一层有attn和ffn
            # W_O
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            # FFN
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2 * self.C:].mul_(init_adaln)  # 后 2C 部分（shift/scale）乘以 0.5
                sab.ada_lin[-1].weight.data[:2 * self.C].mul_(init_adaln_gamma)  # 前 2C 部分（gamma）乘以 1e-5（极小）约等于恒等映射
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):   # 与ada_lin同理
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)

    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'


class VARHF(VAR, PyTorchModelHubMixin):
    # repo_url="https://github.com/FoundationVision/VAR",
    # tags=["image-generation"]):
    def __init__(
            self,
            vae_kwargs,
            num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
            attn_l2_norm=False,
            patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # 10 steps by default
            flash_if_available=True, fused_if_available=True,
    ):
        vae_local = VQVAE(**vae_kwargs)
        super().__init__(
            vae_local=vae_local,
            num_classes=num_classes, depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_eps=norm_eps, shared_aln=shared_aln, cond_drop_rate=cond_drop_rate,
            attn_l2_norm=attn_l2_norm,
            patch_nums=patch_nums,
            flash_if_available=flash_if_available, fused_if_available=fused_if_available,
        )

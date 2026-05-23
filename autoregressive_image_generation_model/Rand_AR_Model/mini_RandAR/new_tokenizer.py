import torch
import torch.nn as nn
import torch.nn.functional as F


#################################################################################
#                        Shared Utility Functions                                #
#################################################################################

def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def normalize_group(in_channels, norm_type='group'):
    """LLaMAGen tokenizer 使用的归一化函数"""
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return nn.SyncBatchNorm(in_channels)


def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    # 1. 展平 (b, n) → (b, n)
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    # 2. 温度系数：让分布更锐化/平滑
    flat_affinity /= temperature
    # 3. 转成概率分布
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    # 按照dim=0维度求平均，即按照每个码本（列）求平均 (b,n) > (n)，得到是全局码本的预测使用情况
    avg_probs = torch.mean(target_probs, dim=0)
    # 全局码本熵：熵计算，并求和
    avg_entropy = - torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    # 按照dim=1进行熵计算，即按照当前token（行）去获得每个码本的预测使用情况，并求平均，得到当前token的码本熵
    sample_entropy = - torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    # 希望当前token的码本熵越小越好（更偏向某一个码本），全局码本熵越大越好（更多码本被使用）
    loss = sample_entropy - avg_entropy
    return loss


#################################################################################
#                 LLaMAGen Tokenizer: Building Blocks                           #
#################################################################################

class LlamaResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = normalize_group(in_channels, norm_type)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalize_group(out_channels, norm_type)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)  # swish激活函数
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class LlamaAttnBlock(nn.Module):
    def __init__(self, in_channels, norm_type='group'):
        super().__init__()
        self.norm = normalize_group(in_channels, norm_type)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class LlamaUpsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class LlamaDownsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


#################################################################################
#            LLaMAGen Tokenizer: VectorQuantizer                                #
#################################################################################

class LlamaVectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, entropy_loss_ratio, l2_norm, show_usage):
        super().__init__()
        self.n_e = n_e  # 16384
        self.e_dim = e_dim  # 8
        self.beta = beta    # 0.25
        self.entropy_loss_ratio = entropy_loss_ratio    # 0.0
        self.l2_norm = l2_norm  # True
        self.show_usage = show_usage    # True

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if self.l2_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)
        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(65536)))

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = torch.einsum('b c h w -> b h w c', z).contiguous()
        z_flattened = z.view(-1, self.e_dim)    # b*h*w, c
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, torch.einsum('n d -> d n', embedding))   # (b,1) + (n) + (b,n) = (b,n)
        # 找出最接近的离散token 向量
        min_encoding_indices = torch.argmin(d, dim=1)   # (b*h*w, 1)
        z_q = embedding[min_encoding_indices].view(z.shape) # (b*h*w, d) > (b, h, w, d)
        perplexity = None
        min_encodings = None
        vq_loss = None
        commit_loss = None
        entropy_loss = None
        codebook_usage = 0

        if self.show_usage and self.training:
            # 滑动窗口：丢掉最旧的，加入最新的
            cur_len = min_encoding_indices.shape[0] # b*h*w
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = min_encoding_indices
            # 统计最近用到了多少个不同的码本条目
            codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e

        # compute loss for embedding
        if self.training:
            # 1. VQ 损失：让码本向量靠近特征
            vq_loss = torch.mean((z_q - z.detach()) ** 2)
            # 2. 承诺损失：让特征靠近码本向量
            commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2)
            # 3. 熵损失：让码本被均匀使用（防崩溃）
            entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)

        # preserve gradients
        # STE，梯度直通
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = torch.einsum('b h w c -> b c h w', z_q)

        return z_q, (vq_loss, commit_loss, entropy_loss, codebook_usage), (
            perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        # shape = (batch, channel, height, width) if channel_first else (batch, height, width, channel)
        if self.l2_norm:
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight
        z_q = embedding[indices]  # (b, h*w, e_dim)
        # 1D转为2D
        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                # reshape back to match original input shape
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            else:
                z_q = z_q.view(shape)
        return z_q


#################################################################################
#            LLaMAGen Tokenizer: Encoder                                        #
#################################################################################

class LlamaEncoder(nn.Module):
    def __init__(self, in_channels=3, ch=128, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,
                 norm_type='group', dropout=0.0, resamp_with_conv=True, z_channels=256):
        super().__init__()
        self.num_resolutions = len(ch_mult)  # 5
        self.num_res_blocks = num_res_blocks
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        # downsampling
        in_ch_mult = (1,) + tuple(ch_mult)  # in_ch_mult是输入通道的倍数，ch_mult是输出通道的倍数
        self.conv_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            conv_block = nn.Module()
            # res & attn
            res_block = nn.ModuleList()  # 残差模块
            attn_block = nn.ModuleList()  # Attention模块
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                res_block.append(LlamaResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if i_level == self.num_resolutions - 1:  # 仅当最后一层开展Attention
                    attn_block.append(LlamaAttnBlock(block_in, norm_type))
            conv_block.res = res_block
            conv_block.attn = attn_block
            # downsample
            if i_level != self.num_resolutions - 1:
                conv_block.downsample = LlamaDownsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)

        # middle
        self.mid = nn.ModuleList()
        self.mid.append(LlamaResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.mid.append(LlamaAttnBlock(block_in, norm_type))
        self.mid.append(LlamaResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))

        # end
        self.norm_out = normalize_group(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        # downsampling
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.downsample(h)

        # middle
        for mid_block in self.mid:
            h = mid_block(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


#################################################################################
#            LLaMAGen Tokenizer: Decoder                                        #
#################################################################################

class LlamaDecoder(nn.Module):
    def __init__(self, z_channels=256, ch=128, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2, norm_type="group",
                 dropout=0.0, resamp_with_conv=True, out_channels=3):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        block_in = ch * ch_mult[self.num_resolutions - 1]
        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.ModuleList()
        self.mid.append(LlamaResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.mid.append(LlamaAttnBlock(block_in, norm_type))
        self.mid.append(LlamaResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))

        # upsampling
        self.conv_blocks = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            conv_block = nn.Module()
            # res & attn
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                res_block.append(LlamaResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(LlamaAttnBlock(block_in, norm_type))
            conv_block.res = res_block
            conv_block.attn = attn_block
            # downsample
            if i_level != 0:
                conv_block.upsample = LlamaUpsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)

        # end
        self.norm_out = normalize_group(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    @property
    def last_layer(self):
        return self.conv_out.weight

    def forward(self, z):
        # z to block_in
        h = self.conv_in(z)

        # middle
        for mid_block in self.mid:
            h = mid_block(h)

        # upsampling
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks + 1):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


#################################################################################
#            LLaMAGen Tokenizer: VQModel (Main Entry)                           #
#################################################################################

class VQModel(nn.Module):
    """LLaMAGen 的 VQ-VAE tokenizer，将图像编码为离散 token 序列"""

    def __init__(self,
                 codebook_size=16384,  # 16384
                 codebook_embed_dim=8,  # 8
                 codebook_l2_norm=True,  # True
                 codebook_show_usage=True,  # True
                 commit_loss_beta=0.25,  # 0.25
                 entropy_loss_ratio=0.0,  # 0.0
                 encoder_ch_mult=[1, 1, 2, 2, 4],  # [1,1,2,2,4]
                 decoder_ch_mult=[1, 1, 2, 2, 4],  # [1,1,2,2,4]
                 z_channels=256,
                 dropout_p=0.0):
        super().__init__()
        # encoder和decoder就是经典的U-Net架构
        self.encoder = LlamaEncoder(ch_mult=encoder_ch_mult, z_channels=z_channels, dropout=dropout_p)
        self.decoder = LlamaDecoder(ch_mult=decoder_ch_mult, z_channels=z_channels, dropout=dropout_p)

        self.quantize = LlamaVectorQuantizer(codebook_size, codebook_embed_dim,
                                             commit_loss_beta, entropy_loss_ratio,
                                             codebook_l2_norm, codebook_show_usage)
        # z > q
        self.quant_conv = nn.Conv2d(z_channels, codebook_embed_dim, 1)
        # q > z
        self.post_quant_conv = nn.Conv2d(codebook_embed_dim, z_channels, 1)
        self.codebook_embed_dim = codebook_embed_dim

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_indices(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return info[2]

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def decode_codes_to_img(self, codes, tgt_size):
        qz_shape = (
            codes.shape[0], # B
            self.codebook_embed_dim,    # 8
            int(codes.shape[1] ** 0.5), # √256 = 16
            int(codes.shape[1] ** 0.5)
        )
        results = self.decode_code(codes, qz_shape)
        # 经过decoder的多次上采样，会和原始图像一致大小
        if results.shape[-1] != tgt_size:
            results = F.interpolate(results, size=(tgt_size, tgt_size), mode="bicubic")
        # 当 x = -1 时，-1 * 127.5 + 128 = 0.5
        # 当 x = 1 时，1 * 127.5 + 128 = 255.5
        # 但实际上，1) 训练数据（token id）并非原始图像因此没有调整到[-1,1]（通过token id > embedding）
        # 2) randar和decoder的输出也没有限制在[-1, 1]
        imgs = results.detach() * 127.5 + 128   # 默认results的输出范围[-1, 1]，随后转变为 [0.5, 255.5]
        imgs = torch.clamp(imgs, 0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()  # NCHW > NHWC
        return imgs



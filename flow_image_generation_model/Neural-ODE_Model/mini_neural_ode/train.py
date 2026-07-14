from torch import Tensor
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torchvision
import torch
import math

use_cuda = torch.cuda.is_available()


# =============== 模型架构 ===============
# 1. 设置欧拉法逻辑
def ode_solve(z0, t0, t1, f):                                   # 欧拉法迭代，既用于前向计算，也用于反向的伴随灵敏度计算
    """
    Simplest Euler ODE initial value solver
    """
    h_max = 0.05
    n_steps = math.ceil((abs(t1 - t0)/h_max).max().item())      # 计算当前时间步要拆分多少小步

    h = (t1 - t0)/n_steps           # 计算步长
    t = t0
    z = z0

    for i_step in range(n_steps):
        z = z + h * f(z, t)         # 欧拉法更新z_t
        t = t + h                   # 欧拉法更新t
    return z

# 2. 设置前向和反向的具体实现，由欧拉法替代forward，由伴随灵敏度法替代backward
class ODEAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z0, t, flat_parameters, func):
        """
        ctx: 上下文管理器
        z0: 原始输入    [B, C, H, W]
        t: 时间步       [T, B]
        flat_parameters: 拉平的模型参数
        func: 模型
        """
        assert isinstance(func, ODEF)
        bs, *z_shape = z0.size()
        time_len = t.size(0)

        with torch.no_grad():
            z = torch.zeros(time_len, bs, *z_shape).to(z0)  # [T, B, C, H, W]
            z[0] = z0                                       # 记录初始输入
            for i_t in range(time_len - 1):
                z0 = ode_solve(z0, t[i_t], t[i_t+1], func)  # 欧拉法迭代，调用func默认执行func的forward
                z[i_t+1] = z0                               # 记录所有时间步的输出

        ctx.func = func                                         # 保存到上下文管理器
        ctx.save_for_backward(t, z.clone(), flat_parameters)
        return z                                            # 返回所有时间步的输出，因此pytorch会计算Loss与每个时间步的梯度，[T, B, C, H, W]

    @staticmethod
    def backward(ctx, dLdz):
        """
        dLdz shape: time_len, batch_size, *z_shape
        """
        func = ctx.func
        t, z, flat_parameters = ctx.saved_tensors
        time_len, bs, *z_shape = z.size()           # [T, B, C, H, W]
        n_dim = np.prod(z_shape)                    # C*H*W
        n_params = flat_parameters.size(0)          # 参数数量

        # Dynamics of augmented system to be calculated backwards in time
        def augmented_dynamics(aug_z_i, t_i):
            """
            tensors here are temporal slices
            t_i - is tensor with size: bs, 1    是维度[B, 1]的tensor，是ODE中t+h的具体时间步
            aug_z_i - is tensor with size: bs, n_dim*2 + n_params + 1   是维度[B, C*H*W*2 + 参数量 + 1]tensor，是ODE中z + h * augmented_dynamics()的具体增广状态
            """
            # 时间步t对应z  和z的伴随状态
            z_i, a = aug_z_i[:, :n_dim], aug_z_i[:, n_dim:2*n_dim]  # ignore parameters and time

            # Unflatten z and a
            z_i = z_i.view(bs, *z_shape)            # z_i: 时间步t对应z [B, C, H, W]
            a = a.view(bs, *z_shape)                # a: 时间步t对应z的伴随状态 [B, C, H, W]
            with torch.set_grad_enabled(True):
                t_i = t_i.detach().requires_grad_(True)
                z_i = z_i.detach().requires_grad_(True)
                func_eval, adfdz, adfdt, adfdp = func.forward_with_grad(z_i, t_i, grad_outputs=a)  # 执行ODEF的forward_with_grad
                adfdz = adfdz.to(z_i) if adfdz is not None else torch.zeros(bs, *z_shape).to(z_i)
                adfdp = adfdp.to(z_i) if adfdp is not None else torch.zeros(bs, n_params).to(z_i)
                adfdt = adfdt.to(z_i) if adfdt is not None else torch.zeros(bs, 1).to(z_i)

            # Flatten f and adfdz
            func_eval = func_eval.view(bs, n_dim)
            adfdz = adfdz.view(bs, n_dim) 
            return torch.cat((func_eval, -adfdz, -adfdp, -adfdt), dim=1)    # 与增广状态的顺序一致

        dLdz = dLdz.view(time_len, bs, n_dim)                   # dLdz是pytorch的损失自动求导，即dLoss/dz。z是包含所有时间步，因此dLdz是[T, B, C*H*W]
        with torch.no_grad():
            ## Create placeholders for output gradients
            # Prev computed backwards adjoints to be adjusted by direct gradients
            adj_z = torch.zeros(bs, n_dim).to(dLdz)             # adj_z是z_t的完整梯度
            adj_p = torch.zeros(bs, n_params).to(dLdz)          # adj_p是参数θ的完整梯度
            # In contrast to z and p we need to return gradients for all times
            adj_t = torch.zeros(time_len, bs, 1).to(dLdz)       # adj_t是时间t的完整梯度

            for i_t in range(time_len-1, 0, -1):
                z_i = z[i_t]                                # 反向遍历，获取时间步t对应z    [B, C, H, W]
                t_i = t[i_t]                                # 反向遍历，获取时间步t [B]
                f_i = func(z_i, t_i).view(bs, n_dim)        # 模型输入z和t得到f，[B, C*H*W]

                dLdz_i = dLdz[i_t]                          # 反向遍历，获取时间步t对应的dLdz_i，[B, C*H*W]
                # 反向遍历，获取时间步t对应的dLdt_i
                # ∵ 在残差模块中，z_t+1 = z + f(z, θ, t) => dz/dt = f(z, θ, t)
                # ∴ dL/dt = dL/dz * dz/dt = dLdz_i * f 
                dLdt_i = torch.bmm(torch.transpose(dLdz_i.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]    # [B, 1, C*H*W] @ [B, C*H*W, 1] = [B, 1, 1] [:, 0] = [B, 1]

                # 首次迭代时，dLdz_i赋值给adj_z✅️，因为最开始的损失==dLdz_i
                # 后续迭代时，dLdz_i累加至adj_z✅️，因为forward返回的z是所有时间步，那么z_t-1的损失包含两部分：z_t通过伴随法反向求出至z_t-1的梯度，z_t-1原本的梯度
                adj_z += dLdz_i                         
                adj_t[i_t] = adj_t[i_t] - dLdt_i    # -dL/dt_i，感觉是因为时间往前退，积分区域减少，所以取负号？

                # 组装增广状态，参考公式(13)下方的figure2
                # [z_i, dLoss/dz_i, dLoss/dθ_i, -dLoss/dt_i]
                # ∵ 计算z_i仅需要θ_i-1  ∴ dLoss/dθ_i一直都是0矩阵
                aug_z = torch.cat((z_i.view(bs, n_dim), adj_z, torch.zeros(bs, n_params).to(z), adj_t[i_t]), dim=-1)

                # Solve augmented system backwards
                aug_ans = ode_solve(aug_z, t_i, t[i_t-1], augmented_dynamics)   # 得到完成时间间隔t_i - t[i_t-1]的结果

                # Unpack solved backwards augmented system
                adj_z[:] = aug_ans[:, n_dim:2*n_dim]                # 直接赋值，即时间t-1的z_t-1的完整梯度
                adj_p[:] += aug_ans[:, 2*n_dim:2*n_dim + n_params]  # 累加复制，因为参数θ是经历所有时间步，因此所有时间步的梯度累加才能代表本次参数θ对目标结果的影响
                adj_t[i_t-1] = aug_ans[:, 2*n_dim + n_params:]      # 直接赋值，即时间t-1的完整梯度

                del aug_z, aug_ans

            ## Adjust 0 time adjoint with direct gradients
            # Compute direct gradients 
            # ∵ forward返回的z包含所有时间步 & 时间步t0不在for循环中
            # ∴ 需要单独计算时间步t0的dLoss/dt_0
            dLdz_0 = dLdz[0]
            f_0 = func(z[0], t[0]).view(bs, n_dim)
            dLdt_0 = torch.bmm(torch.transpose(dLdz_0.unsqueeze(-1), 1, 2), f_0.unsqueeze(-1))[:, 0]    # ⭐⭐ 已替换

            # Adjust adjoints
            adj_z += dLdz_0
            adj_t[0] = adj_t[0] - dLdt_0
        return adj_z.view(bs, *z_shape), adj_t, adj_p, None

# 3. 基类，把伴随灵敏度需要的单步f的执行以及梯度计算的逻辑提取出来，实际上可以合并到到4.具体子类
class ODEF(nn.Module):
    def forward_with_grad(self, z, t, grad_outputs):
        """Compute f and a df/dz, a df/dp, a df/dt"""
        batch_size = z.shape[0]

        out = self.forward(z, t)    # 计算当前时间步的f

        a = grad_outputs            # a_t，即dL/dz_t
        # 通过torch的自动梯度计算，对out或者说f执行计算关于z_t, t, θ的梯度，即df/dz_t, df/dt, df/dθ
        # 指定grad_outputs=a_t，那么adfdz就是当前时间步的正确伴随状态a_t * df/dz_t，同理adfdt和adfdp
        adfdz, adfdt, *adfdp = torch.autograd.grad(
            (out,), (z, t) + tuple(self.parameters()), grad_outputs=(a),
            allow_unused=True, retain_graph=True
        )
        # grad method automatically sums gradients for batch items, we have to expand them back 
        if adfdp is not None:
            adfdp = torch.cat([p_grad.flatten() for p_grad in adfdp]).unsqueeze(0)  # flatten拉平维度，并且unsqueeze(0)即B维度为1
            adfdp = adfdp.expand(batch_size, -1) / batch_size   # 在B维度扩增为batch_size，并且所有值/batch_size，保持梯度总和不变
        if adfdt is not None:
            adfdt = adfdt.expand(batch_size, 1) / batch_size
        return out, adfdz, adfdt, adfdp

    def flatten_parameters(self):
        p_shapes = []
        flat_parameters = []
        for p in self.parameters():
            p_shapes.append(p.size())
            flat_parameters.append(p.flatten())
        return torch.cat(flat_parameters)

# 4. 具体子类，负责f的具体实现
class ConvODEF(ODEF):
    """
    继承ODEF的两个方法forward_with_grad 和flatten_parameters
    """

    def __init__(self, dim):
        super(ConvODEF, self).__init__()
        self.conv1 = nn.Conv2d(dim + 1, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim + 1, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(dim)

    def forward(self, x, t):
        xt = self.add_time(x, t)
        h = self.norm1(torch.relu(self.conv1(xt)))
        ht = self.add_time(h, t)
        dxdt = self.norm2(torch.relu(self.conv2(ht)))
        return dxdt

    def add_time(self, in_tensor, t):
        bs, c, w, h = in_tensor.shape
        return torch.cat((in_tensor, t.expand(bs, 1, w, h)), dim=1)

# 5. 封装类，把4.的具体子类以及真正的forward（欧拉法）和真正的backward（伴随灵敏度法）组装在一起
class NeuralODE(nn.Module):
    def __init__(self, func):
        super(NeuralODE, self).__init__()
        assert isinstance(func, ODEF)
        self.func = func

    def forward(self, z0, t=Tensor([0., 1.]), return_whole_sequence=False):
        t = t.to(z0)
        # 调用ODEAdjoint外部类的函数，传入z0, t, func的总参数, func
        z = ODEAdjoint.apply(z0, t, self.func.flatten_parameters(), self.func)
        if return_whole_sequence:
            return z
        else:
            return z[-1]

# 6. 最终类，把数据特征下采样 + 封装类最终输出 + 输出的处理， 组装在一起 
class ContinuousNeuralMNISTClassifier(nn.Module):
    def __init__(self, ode):
        super(ContinuousNeuralMNISTClassifier, self).__init__()
        # 下采样 28 - 26 - 13 - 6
        self.downsample = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
        )
        self.feature = ode
        self.norm = nn.BatchNorm2d(64)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        # 前处理-提取特征
        x = self.downsample(x)
        # 神经ODE迭代，即欧拉迭代
        x = self.feature(x)
        # 后处理
        x = self.norm(x)
        x = self.avg_pool(x)
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        x = x.view(-1, shape)
        # 最终预测
        out = self.fc(x)
        return out

# =============== 组装模型 ===============
# func就是欧拉迭代中的f
func = ConvODEF(64)
# 把func赋值到NeuralODE，用于控制前向和反向
ode = NeuralODE(func)
# 把NeuralODE赋值到ContinuousNeuralMNISTClassifier，用于控制数据特征的整条链路
model = ContinuousNeuralMNISTClassifier(ode)

# 关键修复：将模型移到 GPU（如果可用）
if use_cuda:
    model = model.cuda()
    print("Using CUDA")
else:
    print("Using CPU")

# =============== 加载数据 ===============
batch_size = 128
img_std = 0.3081
img_mean = 0.1307
path = "/workspace/data"

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(path, train=True, download=False,
                             transform=torchvision.transforms.Compose([
                                 torchvision.transforms.ToTensor(),
                                 torchvision.transforms.Normalize((img_mean,), (img_std,))
                             ])
    ),
    batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(path, train=False, download=False,
                             transform=torchvision.transforms.Compose([
                                 torchvision.transforms.ToTensor(),
                                 torchvision.transforms.Normalize((img_mean,), (img_std,))
                             ])
    ),
    batch_size=batch_size, shuffle=True
)

# =============== 指定优化器 ===============
optimizer = torch.optim.AdamW(model.parameters())

# =============== 训练循环 ===============
def train(epoch):
    num_items = 0
    train_losses = []

    model.train()
    criterion = nn.CrossEntropyLoss()
    print(f"Training Epoch {epoch}...")
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        if use_cuda:
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()               # 梯度清空
        output = model(data)                # 前向
        loss = criterion(output, target)    # 计算loss
        loss.backward()                     # 反向计算梯度
        optimizer.step()                    # 更新参数

        train_losses += [loss.item()]
        num_items += data.shape[0]
    print('Train loss: {:.5f}'.format(np.mean(train_losses)))
    return train_losses


def test(epoch=0, save_visual=True, num_samples=16):
    accuracy = 0.0
    num_items = 0
    vis_data = {'images': [], 'preds': [], 'targets': []}

    model.eval()
    print(f"Testing...")

    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
            if use_cuda:
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            preds = torch.argmax(output, dim=1)
            accuracy += torch.sum(preds == target).item()
            num_items += data.shape[0]

            # 只收集第一个batch用于可视化
            if save_visual and batch_idx == 0 and len(vis_data['images']) == 0:
                vis_data['images'] = data[:num_samples].cpu()
                vis_data['preds'] = preds[:num_samples].cpu()
                vis_data['targets'] = target[:num_samples].cpu()

    accuracy = accuracy * 100 / num_items
    print("Test Accuracy: {:.3f}%".format(accuracy))

    if save_visual and len(vis_data['images']) > 0:
        try:
            # 反归一化
            images = vis_data['images'] * img_std + img_mean
            images = torch.clamp(images, 0, 1).numpy()  # 裁剪到[0,1]防止溢出

            n_samples = len(images)
            n_cols = math.ceil(math.sqrt(n_samples))
            n_rows = math.ceil(n_samples / n_cols)

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2.3))
            fig.suptitle(f'Epoch {epoch} - Test Predictions (Green=Correct, Red=Wrong)',
                         fontsize=14, y=0.98)

            if n_samples == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)

            for idx in range(n_samples):
                row = idx // n_cols
                col = idx % n_cols
                ax = axes[row, col]

                img = images[idx].squeeze(0)
                ax.imshow(img, cmap='gray', vmin=0, vmax=1) # 将图像信息渲染到该画布子区域中

                pred = vis_data['preds'][idx].item()
                true = vis_data['targets'][idx].item()
                is_correct = (pred == true)

                color = '#2ca02c' if is_correct else '#d62728'
                ax.set_title(f'Pred: {pred}\nTrue: {true}',
                             color=color, fontsize=10, fontweight='bold')
                ax.axis('off')

            # 隐藏多余子图
            for idx in range(n_samples, n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                axes[row, col].axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            save_path = f'test_epoch_{epoch:03d}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Saved visualization: {save_path}")
            plt.close()
        except Exception as e:
            print(f"Warning: Failed to save visualization: {e}")

    return accuracy


# =============== 开启训练 ===============
n_epochs = 50
train_losses = []
for epoch in range(1, n_epochs + 1):
    train_losses += train(epoch)
    test(epoch=epoch, save_visual=True, num_samples=16)


plt.figure(figsize=(9, 5))
history = pd.DataFrame({"loss": train_losses})
history["cum_data"] = history.index * batch_size
history["smooth_loss"] = history.loss.ewm(halflife=10).mean()

# 绘制曲线
ax = history.plot(x="cum_data", y="smooth_loss", figsize=(12, 5),
                  title="Train Loss (Smoothed)", color='#2ca02c')
ax.set_xlabel("Samples")
ax.set_ylabel("Loss")

# 关键：保存图像（在 show 之前）
save_path = f'training_loss.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Loss curve saved: {save_path}")


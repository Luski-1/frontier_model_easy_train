from torchvision import transforms, datasets
from torchvision.utils import save_image
from config import cfg
from nice import NICE
import torch.optim as optim
import torch
import os

def logit_transform(x, alpha=1e-5):
    x = alpha + (1 - 2 * alpha) * x     # 把 [0,1] 收缩到 [alpha, 1-alpha]，避免 log(0)
    return torch.log(x) - torch.log(1 - x)  # logit: [0,1] → (-∞, +∞)

def sigmoid_transform(x, alpha=1e-5):
    x = torch.sigmoid(x)                         # (-∞, +∞) → (0, 1)
    return (x - alpha) / (1 - 2 * alpha)         # 映射回精确的 [0, 1]


# 1. 加载数据
transform = transforms.ToTensor()   # 把数据转换为[0, 1]
dataset = datasets.MNIST(
    root="/workspace/data", train=True, transform=transform, download=False
)
dataloader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=cfg["TRAIN_BATCH_SIZE"], shuffle=True, pin_memory=True
)

# 2. 加载模型
model = NICE(data_dim=784, num_coupling_layers=cfg["NUM_COUPLING_LAYERS"])
if cfg["USE_CUDA"]:
    device = torch.device("cuda")
    model = model.to(device)

model.train()

# 4. 优化器设置：降低学习率 + 增大 eps + 正则化参数
opt = optim.Adam(model.parameters(), lr=1e-4, eps=1e-4, weight_decay=1e-5)

for epoch in range(cfg["TRAIN_EPOCHS"]):
    mean_likelihood = 0.0
    num_minibatches = 0

    for batch_id, (x, _) in enumerate(dataloader):

        if cfg["USE_CUDA"]:
            # 增加随机的微小数值，有助于把离散数据演化为连续数据
            x = x.view(-1, 784).cuda() + torch.rand(784, device='cuda') / 256.0
        else:
            x = x.view(-1, 784) + torch.rand(784) / 256.0
        # 截断，保证在0~1区间
        x = torch.clamp(x, 0, 1)

        # 新增逻辑，让数据处于[-∞, ∞]区间，与高斯分布对应，降低学习难度
        x = logit_transform(x)

        z, likelihood = model(x)

        # 太容易指数爆炸导致梯度NAN，限制一下
        if torch.isnan(likelihood).any() or torch.abs(z).max() > 50:
            print(f"Skip bad batch at epoch {epoch}")
            opt.zero_grad()
            continue

        # 转负，可以对接梯度下降
        loss = -torch.mean(likelihood)

        opt.zero_grad()
        loss.backward()

        # 梯度裁剪是必须的，避免梯度NAN
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        # 获取loss的标量，避免保留graph
        mean_likelihood -= loss.item()
        num_minibatches += 1

    mean_likelihood /= num_minibatches
    print("Epoch {} completed. Log Likelihood: {}".format(epoch, mean_likelihood))

    if (epoch + 1) % 5 == 0:
        # 保存模型（原有代码）
        save_path = os.path.join(cfg["MODEL_SAVE_PATH"], "{}.pt".format(epoch))
        os.makedirs(cfg["MODEL_SAVE_PATH"], exist_ok=True)
        torch.save(model.state_dict(), save_path)

        # 生成并保存图片
        print("开始先验抽样生成")
        model.eval()
        with torch.no_grad():
            # 推理生成，就是逆函数，颠倒过来
            samples = model.sample(4)  # [4, 784]

            # [-∞, ∞]映射回 [0,1]
            samples = sigmoid_transform(samples)
            
            # reshape 回 [batch, channel, height, width]
            samples = samples.view(4, 1, 28, 28)
            
            # 裁剪到 [0,1] 范围
            samples = torch.clamp(samples, 0, 1)
            
            # 保存为 2x2 的图片网格
            img_path = os.path.join(cfg["MODEL_SAVE_PATH"], "samples_epoch_{}.png".format(epoch))
            save_image(samples, img_path, nrow=2, normalize=False)
            print(f"已保存生成样本到: {img_path}")
        
        model.train()


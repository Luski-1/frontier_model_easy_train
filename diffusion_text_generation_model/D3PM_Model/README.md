# 项目
Denoising Diffusion Models in Discrete State-Spaces

## 主要内容
1. reference为Github的大佬关于D3PM实现的pytorch版本代码（非官方作者），本人仅对其中MNIST分支的训练过程/推理过程的代码中增加方便理解的注释和知识点
2. pytorch版本代码已经非常精简并且注释非常详细，因此不再需要精简化重写。可以直接启动训练，修改MNIST的路径即可
3. 增加了conTranpose2d的知识讲解

## 疑惑

KL散度的损失权重默认设置为0，后续本人已经修改为1.0，也能够正常收敛并且生成图像

## 代码文件
```text
D3PM_Model/
│
├── convTranspose2d.md             # 转置卷积原理笔记
├── d3pm.md
├── README.md
├── reference/                     # pytorch版本实现
│   ├── d3pm_runner.py             # MNIST
│   ├── d3pm_runner_cifar10.py     # CIFAR-10
│   ├── dit.py
│   └── readme.md
└── train_result/				   # 本人训练结果
    └── sample_100.gif
```

## 参考资料
1. D3PM文献: https://arxiv.org/pdf/2107.03006
2. D3PM的pytorch代码实现（非官方作者）: https://github.com/cloneofsimo/d3pm/tree/main
4. D3PM模型理论讲解视频👍👍👍：https://www.bilibili.com/video/BV16J4nz5Eq5/?spm_id_from=333.1387.search.video_card.click
5. D3PM模型理论讲解视频（基于pytorch代码版本）👍👍👍: https://www.bilibili.com/video/BV1ewAHzGEJN/?spm_id_from=333.1387.search.video_card.click&vd_source=401f1f7d80fdd51bba1fc24cf7961ff9
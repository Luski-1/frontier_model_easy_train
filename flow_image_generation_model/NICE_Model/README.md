# 项目
NICE: Non-linear Independent Components Estimation

## 主要内容
1. reference应该并非官方项目代码，而是取自Github大佬的实现：https://github.com/DakshIdnani/pytorch-nice；本人对该项目增加详细的人工注释以及知识点

2. 这次没有精简化的重写项目，因为该项目的代码已经非常简单。在该项目的基础上，添加了以下内容

    - 高斯分布的实现
    - x进入模型前的范围转换
    - 交替耦合的行列式解释
    - 避免梯度爆炸💥的适当限制

3. 该项目能否训练？
    - 可以开启训练
        ```text
        python train.py
        ```
    - 训练前要求
        - 请提前下载MNIST数据集，可通过datasets方法或者通过kaggle下载
        ```text
        from torchvision import datasets
        
        dataset = datasets.MNIST(
            root='./data',
            split="all",
            download=True,
        )  
        ```

4. 该项目能否推理？
    - 默认在训练过程中，会通sample方法进行先验抽样生成，如果需要编写推理方法，参考该方法编写对应代码即可

## 代码文件
```text
NICE_Model/
├── NICE.md
├── README.md
├── reference/						# 原项目
│   ├── config.py
│   ├── LICENSE
│   ├── modules.py
│   ├── nice.py
│   ├── README.md
│   ├── samples/
│   │   ├── samples1.png
│   │   ├── samples2.png
│   │   ├── samples3.png
│   │   └── samples4.png
│   └── train.py
└── train_result/					# 训练结果
    └── samples_epoch_199.png
```

## 参考资料
1. NICE文献: https://arxiv.org/pdf/1410.8516
2. NICE的复现（非官方）Github: https://github.com/DakshIdnani/pytorch-nice
3. NICE模型理论讲解视频👍👍👍：https://www.bilibili.com/video/BV1gj1xBqEgR/?spm_id_from=333.1387.search.video_card.click
5. NICE代码讲解视频[讲解的代码是非官方复现]👍👍👍:https://www.bilibili.com/video/BV1z1c4z8EPY/?spm_id_from=333.1387.search.video_card.click
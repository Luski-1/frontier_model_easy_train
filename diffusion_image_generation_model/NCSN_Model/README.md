# 项目
NCSN_Model

## 主要内容
1. reference为原项目代码，本人对训练过程/推理过程的代码中增加方便理解的注释和知识点
2. mini_ncsn为重写项目的精简化代码
   - 不修改原项目的核心思路的基础上，支持训练过程选择MNIST或者CELEBA，有推理方法
   - 删除分布式，精简参数，精简日志打印
   - 相同模块内容的单py文件化
   - 保留增加的对应注释
3. mini_ncsn能否训练？
    - 可以开启训练
        ```text
        python train.py
        ```
    - 训练前要求
        - 请提前下载CELEBA数据集或者MNIST数据，可通过datasets方法或者通过kaggle下载
        ```text
        from torchvision import datasets
        dataset = datasets.CelebA(
            root="./data",
            split="all",
            download=True,  # 关键：自动下载
            target_type=["attr", "identity"]
        )

        dataset = MNIST(
            root='./data',
            split="all",
            download=True,
        )   
        ```

        - 设置train.py中data_root参数，指定为数据集地址，并且根据数据集调整channels
        - 根据显存大小，调整per_device_train_batch_size
4. mini_ncsn能否推理？
    - 默认在训练过程中，trainer会通utils.py中SampleCallback回调对象的方法anneal_Langevin_dynamics进行朗之万动力学采样，如果需要编写推理方法，参考该方法编写对应代码即可

## 代码文件
```text
NCSN_Model/
├── README.md
├── ncsn.md [项目流程简要概括]
│
├── mini_ncsn/ [重写项目]
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
│
├── reference/ [原项目]
│   ├── LICENSE
│   ├── README.md
│   ├── main.py
│   ├── assets/
│   │   ├── celeba_large.gif
│   │   ├── celeba_small.gif
│   │   ├── cifar10_large.gif
│   │   ├── cifar10_small.gif
│   │   ├── mnist_large.gif
│   │   └── mnist_small.gif
│   ├── configs/
│   │   ├── anneal.yml
│   │   ├── baseline.yml
│   │   ├── scorenet.yml
│   │   └── toy.yml
│   ├── datasets/
│   │   ├── celeba.py
│   │   ├── utils.py
│   │   └── vision.py
│   ├── losses/
│   │   ├── dsm.py
│   │   └── sliced_sm.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cond_refinenet_dilated.py
│   │   ├── gmm.py
│   │   ├── inception.py
│   │   ├── pix2pix.py
│   │   ├── refinenet_dilated_baseline.py
│   │   └── scorenet.py
│   └── runners/
│       ├── __init__.py
│       ├── anneal_runner.py
│       ├── baseline_runner.py
│       ├── scorenet_runner.py
│       └── toy_runner.py
│
└── train_result/ [训练结果]
    └── sample_epoch50.png
```

## 参考资料
1. NCSN文献: https://arxiv.org/abs/1907.05600
2. NCSN原项目Github: https://github.com/ermongroup/ncsn
3. NCSN模型理论讲解视频的推荐👍👍👍：https://www.bilibili.com/video/BV19cBMBMEzp/?spm_id_from=333.1387.search.video_card.click&vd_source=401f1f7d80fdd51bba1fc24cf7961ff9
4. NCSN模型理论讲解视频的推荐👍：https://www.bilibili.com/video/BV1hFskzSE6f/?spm_id_from=333.1387.search.video_card.click&vd_source=401f1f7d80fdd51bba1fc24cf7961ff9
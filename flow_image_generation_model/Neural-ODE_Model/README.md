# 项目
Neural-ODE

## 主要内容
1. reference并非原项目代码，而是俄罗斯大佬的简化实现：https://msurtsukov.github.io/Neural-ODE/ ；本人主要对简化实现的Neural_ODEs.ipynb增加详细的人工注释以及疑问

2. mini_neural_ode是对Neural_ODEs.ipynb进行合并的重写，基本内容与Neural_ODEs.ipynb别无二致，保留详细的人工注释的情况下修改了一部分代码

3. mini_neural_ode能否训练？
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

        - 设置train.py中path参数，指定为数据集的目录
        - 根据显存大小，调整batch_size

4. mini_neural_ode能否推理？
    - 默认在训练过程中，会通test方法进行先验抽样生成，如果需要编写推理方法，参考该方法编写对应代码即可

## 疑问

1. 计算dL/dt_0时，理应使用f(a_0, t0)，但是俄罗斯大佬的实现中直接使用for循环最终残留的f，那么该f就是(a_1, t_1)，有点不合理
2. dL/dt是负的，也是比较难理解。但文章作者好像说因为倒退会让积分区域减少，所以dL/dt就是负的？详细查看https://github.com/rtqichen/torchdiffeq/issues/218

## 代码文件
```text
Neural-ODE_Model/
├── README.md                        
├── mini_neural_ode/				 # 重写项目
│   └── train.py
├── reference/
│   ├── .gitignore
│   ├── README.md
│   ├── Neural_ODEs.ipynb            # 英文版 Notebook	是增加人工注释的主要文件
│   ├── Neural ODEs (Russian).ipynb  # 俄文版 Notebook
│   ├── Neural_ODEs.py               # Notebook 导出的 Python 文件
│   └── assets/
│       ├── backprop.png
│       ├── CNF_NF_comp.png
│       ├── comp_result.png
│       ├── linear_learning.gif
│       ├── methods_compare.png
│       ├── mnist_example.png
│       ├── ode_rnn_comp.png
│       ├── ode_solver_attrs.png
│       ├── pseudocode.png
│       ├── Screenshot_2019-01-16 1806 07366 pdf.png
│       ├── spirals.png
│       ├── spirals_examples.png
│       ├── spirals_homotopy.png
│       ├── spirals_reconstructed.png
│       ├── train_error.png
│       ├── vae_model.png
│       └── imgs/
│           └── linear/
└── train_result/
    └── epoch_030.png                # 第 30 epoch 训练结果图
```

## 参考资料
1. Neural-ODE文献: https://arxiv.org/pdf/1806.07366
2. Neural-ODE原项目Github: https://github.com/rtqichen/torchdiffeq
3. Neural-ODE俄罗斯大佬的简化复现[即reference目录内容]: https://msurtsukov.github.io/Neural-ODE/
4. Neural-ODE模型理论讲解视频👍👍👍：https://www.bilibili.com/video/BV1haAozBEjQ/?spm_id_from=333.337.search-card.all.click
5. Neural-ODE模型代码讲解视频[讲解的代码是俄罗斯大佬的简化复现]👍👍👍:https://www.bilibili.com/video/BV1ZMcBzsEvT/?spm_id_from=333.337.search-card.all.click
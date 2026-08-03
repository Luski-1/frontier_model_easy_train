# 项目
SDE: Score-Based Generative Modeling through Stochastic Differential Equations

## 主要内容
1. reference为原项目的pytorch版本，本人对训练过程/推理过程的代码中增加方便理解的注释和知识点，增加人工注释的训练路线有VESDE的cifar10_ncsnpp_continuous和VPSDE的cifar10_ddpmpp_continuous。
2. mini_sde为重写项目的精简化代码，相对于原项目而言，⭐**逻辑更加直观，易于学习SDE原理，非常推荐想快速理解代码的新手**⭐，包含了DDPM和NCSN在CIFAR10数据的两种版本。
   - 不修改原项目的核心思路的基础上，支持VPSDE的DDPM和VESDE的NCSN
   - 不修改原项目的核心思路的基础上，反向求解支持predict-correct方法和ode方法
   - 相同模块内容的单py文件化
   - 保留增加的对应注释
3. mini_sde能否训练？
    - 可以开启训练（单卡也可以直接启动）
        ```text
        accelerate launch --num_processes ? train.py
        ```
    - 训练前要求
        - 请提前下载CIFAR-10数据集，可通过datasets方法或者通过 [魔塔社区](https://modelscope.cn/datasets/EFate1006/CIFAR-10)下载
        ```python
        import torchvision
        
        # download=True 自动下载，存到 ./data 文件夹
        train_set = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True
        )
        ```
        
        - 设置xxx_config.yaml中dataset_path参数，指定为数据集地址
        - 根据显存大小，调整per_device_batch_size
4. mini_sde能否推理？
    - 默认在训练过程中，会通过get_ode_solve_result方法或get_pc_solve_result方法进行先验抽样生成，如果需要编写推理方法，自行编写模型权重加载代码并且调用上述方法即可

## 代码文件
```text
SDE/
├── README.md
├── sde.md							# SDE项目的概括理论
├── mini_sde/						# 重写项目
│   ├── ema.py		
│   ├── loss.py
│   ├── model.py
│   ├── train.py
│   ├── utils.py
│   ├── config/						# 配置文件夹
│   │   ├── ddpm_config.yaml
│   │   └── ncsn_config.yaml
│   ├── reverse_solver/				# 反向求解（推理采样）
│   │   ├── ode.py
│   │   └── predict_correct.py
│   └── sde/						# SDE管理
│       ├── rsde.py
│       ├── vesde.py
│       └── vpsde.py
├── reference/						# 原项目
│   ├── .gitignore
│   ├── LICENSE
│   ├── README.md
│   ├── Score_SDE_demo_PyTorch.ipynb
│   ├── controllable_generation.py
│   ├── datasets.py
│   ├── debug.py
│   ├── evaluation.py
│   ├── likelihood.py
│   ├── losses.py
│   ├── main.py
│   ├── requirements.txt
│   ├── run_lib.py
│   ├── sampling.py
│   ├── sde_lib.py
│   ├── utils.py
│   ├── assets/
│   │   ├── bedroom.jpeg
│   │   ├── celebahq_256.jpg
│   │   ├── church.jpeg
│   │   ├── ffhq_1024.jpeg
│   │   ├── ffhq_256.jpg
│   │   ├── ffhq_samples.jpg
│   │   └── schematic.jpg
│   ├── configs/					# 训练的配置文件路径
│   │   ├── default_celeba_configs.py
│   │   ├── default_cifar10_configs.py
│   │   ├── default_lsun_configs.py
│   │   ├── subvp/
│   │   │   ├── cifar10_ddpm_continuous.py
│   │   │   ├── cifar10_ddpmpp_continuous.py
│   │   │   ├── cifar10_ddpmpp_deep_continuous.py
│   │   │   ├── cifar10_ncsnpp_continuous.py
│   │   │   └── cifar10_ncsnpp_deep_continuous.py
│   │   ├── ve/
│   │   │   ├── bedroom_ncsnpp_continuous.py
│   │   │   ├── celeba_ncsnpp.py
│   │   │   ├── celebahq_256_ncsnpp_continuous.py
│   │   │   ├── celebahq_ncsnpp_continuous.py
│   │   │   ├── church_ncsnpp_continuous.py
│   │   │   ├── cifar10_ddpm.py
│   │   │   ├── cifar10_ncsnpp.py
│   │   │   ├── cifar10_ncsnpp_continuous.py
│   │   │   ├── cifar10_ncsnpp_deep_continuous.py
│   │   │   ├── ffhq_256_ncsnpp_continuous.py
│   │   │   ├── ffhq_ncsnpp_continuous.py
│   │   │   ├── ncsn/
│   │   │   │   ├── celeba.py
│   │   │   │   ├── celeba_124.py
│   │   │   │   ├── celeba_1245.py
│   │   │   │   ├── celeba_5.py
│   │   │   │   ├── cifar10.py
│   │   │   │   ├── cifar10_124.py
│   │   │   │   ├── cifar10_1245.py
│   │   │   │   └── cifar10_5.py
│   │   │   └── ncsnv2/
│   │   │       ├── bedroom.py
│   │   │       ├── celeba.py
│   │   │       └── cifar10.py
│   │   └── vp/
│   │       ├── cifar10_ddpmpp.py
│   │       ├── cifar10_ddpmpp_continuous.py
│   │       ├── cifar10_ddpmpp_deep_continuous.py
│   │       ├── cifar10_ncsnpp.py
│   │       ├── cifar10_ncsnpp_continuous.py
│   │       ├── cifar10_ncsnpp_deep_continuous.py
│   │       └── ddpm/
│   │           ├── bedroom.py
│   │           ├── celebahq.py
│   │           ├── church.py
│   │           ├── cifar10.py
│   │           ├── cifar10_continuous.py
│   │           └── cifar10_unconditional.py
│   ├── models/						# 模型架构
│   │   ├── __init__.py
│   │   ├── ddpm.py
│   │   ├── ema.py
│   │   ├── layers.py
│   │   ├── layerspp.py
│   │   ├── ncsnpp.py
│   │   ├── ncsnv2.py
│   │   ├── normalization.py
│   │   ├── up_or_down_sampling.py
│   │   └── utils.py
│   └── op/							# 反向求解
│       ├── __init__.py
│       ├── fused_act.py
│       ├── fused_bias_act.cpp
│       ├── fused_bias_act_kernel.cu
│       ├── upfirdn2d.cpp
│       ├── upfirdn2d.py
│       └── upfirdn2d_kernel.cu
└── train_result/					# 重写项目的训练结果
    ├── global_step_320000_vesde_ncsn_ema_pc.png
    └── global_step_320000_vpsde_ddpm_ema_pc.png
```

## 参考资料
1. SDE文献: https://arxiv.org/pdf/2011.13456
2. SDE原项目Github: https://github.com/yang-song/score_sde/tree/main
3. SDE模型理论讲解视频👍👍👍：https://www.bilibili.com/video/BV1ZT421v7bx/?spm_id_from=333.1387.search.video_card.click
4. SDE模型代码讲解视频👍👍👍: https://www.bilibili.com/video/BV1jGFCzPEFc/?spm_id_from=333.1387.search.video_card.click
5. 完成SDE的学习，建议继续学习EDM，推荐一套SDE-EDM理论讲解视频👍👍👍：[SDE-1](https://www.bilibili.com/video/BV1y1YpejEB4/?spm_id_from=333.1387.search.video_card.click&vd_source=401f1f7d80fdd51bba1fc24cf7961ff9) | [SDE-2](https://www.bilibili.com/video/BV1e7pneaEki?spm_id_from=333.788.videopod.sections&vd_source=401f1f7d80fdd51bba1fc24cf7961ff9) | [SDE-3](https://www.bilibili.com/video/BV1WstpeAEcT?spm_id_from=333.788.videopod.sections&vd_source=401f1f7d80fdd51bba1fc24cf7961ff9) | [EDM-1](https://www.bilibili.com/video/BV1Uqx3eqEWM?spm_id_from=333.788.videopod.sections&vd_source=401f1f7d80fdd51bba1fc24cf7961ff9) | [EDM-2](https://www.bilibili.com/video/BV1Swx4e2E5v?spm_id_from=333.788.videopod.sections&vd_source=401f1f7d80fdd51bba1fc24cf7961ff9) | [EDM-3](https://www.bilibili.com/video/BV14d27YHEVr?spm_id_from=333.788.videopod.sections&vd_source=401f1f7d80fdd51bba1fc24cf7961ff9) | [EDM-4](https://www.bilibili.com/video/BV1jb2mYpE2G?spm_id_from=333.788.videopod.sections&vd_source=401f1f7d80fdd51bba1fc24cf7961ff9) | [EDM-5](https://www.bilibili.com/video/BV1hFyGYqExX?spm_id_from=333.788.videopod.sections&vd_source=401f1f7d80fdd51bba1fc24cf7961ff9) | [EDM-6](https://www.bilibili.com/video/BV1yYypYRELM?spm_id_from=333.788.videopod.sections&vd_source=401f1f7d80fdd51bba1fc24cf7961ff9) | [EDM-7](https://www.bilibili.com/video/BV1Sn17YHE3B?spm_id_from=333.788.videopod.sections&vd_source=401f1f7d80fdd51bba1fc24cf7961ff9) | [EDM-8](https://www.bilibili.com/video/BV1FWSoYtEtW?spm_id_from=333.788.videopod.sections&vd_source=401f1f7d80fdd51bba1fc24cf7961ff9)
# 项目
DDPM_Model

## 主要内容
1. reference为原项目代码，本人对训练过程/推理过程的代码中增加方便理解的注释和知识点
2. mini_ddpm为重写项目的精简化代码，但**DDPM原项目的代码已经很简洁明了**，建议读者直接阅读源码即可，但是老规矩还是奉上重写的代码
   - 不修改原项目的核心思路的基础上，支持训练过程选择MNIST或者CELEBA，有推理方法
   - 相同模块内容的单py文件化
   - 保留增加的对应注释
3. mini_ddpm能否训练？
    - 可以开启训练
        ```text
        python train.py 或者 accelerate launch --num_processes ? train_accelerate.py
        ```
    - 训练前要求
        - 请提前下载CELEBA数据集，可通过datasets方法或者通过kaggle下载
        ```text
        from torchvision import datasets
        dataset = datasets.CelebA(
            root="./data",
            split="all",
            download=True,
            target_type=["attr", "identity"]
        )   
        ```

        - 设置train.py或train_accelerate.py中data_root参数，指定为数据集地址
        - 根据显存大小，调整per_device_train_batch_size和gradient_accumulation_steps
4. mini_ncsn能否推理？
    - 默认在训练过程中，trainer会通prediction_step方法进行先验抽样生成，如果需要编写推理方法，参考该方法编写对应代码即可

## 代码文件
```text
DDPM_Model/
├── ddpm.md                          # DDPM 简要流程概括
├── README.md                        
│
├── mini_ddpm/                       # [重写项目]
│   ├── config.py                    
│   ├── dataset.py                   
│   ├── ema.py                       
│   ├── model.py                     
│   ├── train.py                     
│   ├── train_accelerate.py          
│   └── utils.py                     
│
├── reference/                       # [原项目]
│   ├── .gitignore
│   ├── README.md
│   ├── setup.py
│   ├── ddpm/                        
│   │   ├── __init__.py
│   │   ├── diffusion.py             
│   │   ├── ema.py                   
│   │   ├── script_utils.py          
│   │   ├── unet.py                  
│   │   └── utils.py                 
│   └── scripts/                     
│       ├── sample_images.py         
│       └── train_cifar.py           
│
└── train_result/                    # [训练结果]
    └── epoch300.0_ema.png           
```

## 参考资料
1. DDPM文献: https://arxiv.org/abs/2006.11239
2. DDPM原项目Github: https://github.com/hojonathanho/diffusion
3. DDPM模型理论讲解文章👍👍👍[知乎UP:猛猿]：https://zhuanlan.zhihu.com/p/637815071  |  https://zhuanlan.zhihu.com/p/650394311  | https://zhuanlan.zhihu.com/p/655568910
4. DDPM模型理论讲解视频👍👍👍：https://www.bilibili.com/video/BV16ZsPz4ECF/?spm_id_from=333.1387.search.video_card.click  | https://www.bilibili.com/video/BV11KsPzwE2m?spm_id_from=333.788.videopod.sections&vd_source=401f1f7d80fdd51bba1fc24cf7961ff9
5. DDPM模型理论讲解视频👍👍👍: https://www.bilibili.com/video/BV1qH4y1g7Pe/?spm_id_from=333.1387.search.video_card.click

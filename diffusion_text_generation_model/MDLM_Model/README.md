# 项目
MDLM: Masked Diffusion Language Models

## 主要内容
1. reference为官方实现的原项目，本人对训练过程/推理过程的代码中增加方便理解的注释和知识点，增加人工注释的训练路线只有MDLM分支（AR、D3PM、SEDD分支没有注释），噪声调度器只有LogLinear分支，模型架构只有DIT分支
2. mini_mdlm为重写项目的精简化代码，相对于原项目而言，⭐**噪声调度器的逻辑更加直观，是直接适配LogLinear，避免原项目为了覆盖多种调度器所作出的繁杂适配**⭐。
   - 不修改原项目的核心思路的基础上，支持LogLinear噪声调度器的MDLM训练
   - 不修改原项目的核心思路的基础上，推理解码直接原生解码方式和半自回归解码方式
   - 相同模块内容的单py文件化
   - 保留增加的对应注释
3. mini_mdlm能否训练？
    - 可以开启训练（单卡也可以直接启动）
        ```python
        accelerate launch --num_processes ? train.py
        ```
    - 训练前要求
        - 请提前下载数据集，mini_mdlm使用的是miniPile数据集，可通过datasets下载或者前往[huggingface](https://huggingface.co/datasets/JeanKaddour/minipile/tree/main/data)下载
        ```python
        import datasets
        
        dataset = datasets.load_dataset(            # 通过huggingface的datasets下载
            'JeanKaddour/minipile',
            cache_dir="/workspace/data/cache_dir"
        )
        dataset.save_to_disk("/worksapce/data/miniPile")
        ```
        
        - 请提前下载tokenizer，mini_mdlm使用的是谷歌的bert-base-uncased，可通过[魔塔社区](https://modelscope.cn/models/google-bert/bert-base-uncased)下载，实际上不需要模型权重，只需要tokenizer相关配置文件即可
        - 设置config.yaml中dataset_path参数，指定为数据集地址；设置config.yaml中packed_dataset_path参数，指定为数据集处理后的保存地址
        - 根据显存大小，调整per_device_batch_size
4. mini_mdlm能否推理？
    - 默认在训练过程中，会通过semi_ar_sample方法或default_sample方法进行半自回归推理方法和默认推理方法，如果需要编写推理方法，自行编写模型权重加载代码并且调用上述方法即可

## 代码文件
```text
MDLM_Model/
├── README.md
├── mdlm.md							  # 原项目概括
├── mini_mdlm/                        # 重写项目
│   ├── config.yaml
│   ├── data.py
│   ├── model.py
│   ├── noise.py
│   ├── sampler.py					  # 分布式Sampler代码在重写项目中没使用，仅做学习参考
│   ├── train.py
│   └── utils.py					  # 保存gif的代码由AI撰写
├── reference/                        # 原项目官方实现
│   ├── CITATION.cff
│   ├── LICENSE
│   ├── README.md
│   ├── requirements.yaml
│   ├── main.py                       # 训练/采样入口（Hydra）
│   ├── diffusion.py                  # 扩散过程 + 损失（SUBS/SEDD/D3PM）
│   ├── noise_schedule.py             # 噪声调度（linear/loglinear/polynomial）
│   ├── dataloader.py                 # 数据加载与 tokenization
│   ├── utils.py                      
│   ├── configs/
│   │   ├── config.yaml
│   │   ├── model/                     
│   │   ├── data/                      
│   │   ├── noise/                     
│   │   ├── lr_scheduler/              
│   │   ├── strategy/                  
│   │   └── callbacks/                 
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dit.py                     # 默认主干：Diffusion Transformer
│   │   ├── dimamba.py                 
│   │   ├── autoregressive.py          
│   │   └── ema.py                     
│   └── scripts/                      
│       ├── train_owt_mdlm.sh		  # 默认训练脚本
│       ├── train_owt_sedd.sh
│       ├── train_lm1b_d3pm.sh
│       ├── train_lm1b_ar.sh
│       ├── eval_owt_T_mdlm.sh
│       ├── eval_owt_sedd.sh
│       └── eval_owt_ar.sh

└── train_result/                     # 重写项目的训练结果
    ├── step124000_default.gif
    └── step124000_semi_ar.gif
```

## 参考资料
1. MDLM文献: https://arxiv.org/pdf/2406.07524
2. MDLM原项目Github: https://github.com/kuleshov-group/mdlm
3. MDLM模型理论讲解视频👍👍👍：https://www.bilibili.com/video/BV17usuzvEH4/?spm_id_from=333.1387.search.video_card.click
4. MDLM模型原代码讲解视频👍👍👍: https://www.bilibili.com/video/BV1KbANzSE3j?spm_id_from=333.788.videopod.sections&vd_source=401f1f7d80fdd51bba1fc24cf7961ff9
# 项目
Rand_AR_Model

## 主要内容
1. reference为原项目代码，本人对训练过程/推理过程的代码中增加方便理解的注释
2. mini_RandAR为重写项目的精简化代码
   - 不修改原项目的核心思路的基础上，仅保留训练过程的默认分支路线以及推理方法
   - 删除分布式，删除benchmark评测相关内容，精简参数，精简日志打印
   - 相同模块内容的单py文件化
   - 保留增加的对应注释
3. mini_RandAR能否开启训练？
   - 可以开启训练，可使用train.sh
      ```text
      python new_train.py \
          --config new_configs/randar_xl_0.7b.yaml \
          --exp-name randar_xl_0.7b \
          --data-path /workspace/data/imagenet-llamagen-adm-256_codes \
          --vq-ckpt /workspace/model/vq_ds16_c2i.pt
      ```
   - 参数解释
      - imagenet-llamagen-adm-256_codes是训练数据文件[ImageNet图像的离散化数据]，可以参考原项目MD文档进行下载，也可以参考原项目extracr_latent_codes.py对ImageNet图像进行离散化
      - vq_ds16_c2i.pt是tokenzier文件[VQ模型]，用于解码RandAR的预测图像ID，可以参考原项目MD文档进行下载
        
## 疑惑
1. 不理解原项目为什么对Class token的旋转矩阵设置为零向量矩阵，class toekn embedding经过旋转后成为zero vector > Attention过程中后续image token的q与class toekn k的注意力分数永远都为0 > 永远无法获取class token的信息 > 无法实现CFG
2. 个人愚见，可以在预计算旋转向量的函数precompute_freqs_cis_2d中，对class token对应的旋转矩阵设置为[0, 1]，即旋转0°的矩阵

## 代码文件
```text
├─mini_RandAR [重写项目]
│  │  new_dataset.py
│  │  new_model.py
│  │  new_tokenizer.py
│  │  new_train.py
│  │  new_utils.py
│  │  train.sh
│  │  
│  └─new_configs
│          randar_xl_0.7b.yaml
│ 
├─train_result [本人训练结果]
│          
└─reference [原项目]
    │  .gitignore
    │  documentation.md
    │  LICENSE
    │  README.md
    │  sample_c2i.py
    │  train_c2i.py
    │  
    ├─configs
    │  ├─randar
    │  │      randar_l_0.3b_llamagen.yaml
    │  │      randar_l_0.3b_maskgit.yaml
    │  │      randar_xl_0.7b_llamagen.yaml
    │  │      
    │  └─tokenization
    │          llamagen.yaml
    │          maskgit.yaml
    │          
    ├─imgs
    │      teaser.png
    │      
    ├─RandAR
    │  │  util.py
    │  │  __init__.py
    │  │  
    │  ├─applications
    │  │      resolution_extrapolation.py
    │  │      __init__.py
    │  │      
    │  ├─dataset
    │  │      augmentation.py
    │  │      builder.py
    │  │      imagenet.py
    │  │      __init__.py
    │  │      
    │  ├─eval
    │  │      fid.py
    │  │      __init__.py
    │  │      
    │  ├─model
    │  │      generate.py
    │  │      llamagen_gpt.py
    │  │      maskgit_tokenizer.py
    │  │      randar_gpt.py
    │  │      tokenizer.py
    │  │      utils.py
    │  │      __init__.py
    │  │      
    │  └─utils
    │          distributed.py
    │          logger.py
    │          lr_scheduler.py
    │          visualization.py
    │          __init__.py
    │          
    ├─slurm_scripts
    │      randar_0.3b_llamagen_360k.sh
    │      randar_0.3b_maskgit_360k.sh
    │      randar_0.7b_llamagen_360k.sh
    │      
    └─tools
            extracr_latent_codes.py
            resolution_extrapolation.py
            search_cfg_weights.py
```
   
## 参考资料
1. 原项目github：https://github.com/ziqipang/RandAR
2. 原项目文献：https://arxiv.org/abs/2412.01827


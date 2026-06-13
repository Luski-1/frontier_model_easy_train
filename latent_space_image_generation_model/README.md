# 项目
VAE_Model

## 主要内容
1. 精简版本的VAE模型训练代码，包含单卡训练版本(train_face.py)以及多卡训练版本(train_face_accelerator.py)
2. 能否训练？
    - 可以开启训练
        ```text
        python train_face.py or accelerate launch --num_processes ? train_face_accelerator.py
        ```
    - 训练前要求
        - 请提前下载CELEBA数据集，可通过datasets方法或者通过kaggle下载
        ```text
        from torchvision import datasets
        dataset = datasets.CelebA(
            root="./data",
            split="all",
            download=True,  # 关键：自动下载
            target_type=["attr", "identity"]
        )   
        ```
        - 设置train_face.py或train_face_accelerator.py中data_root参数，指定为数据集地址
        - 根据显存大小，调整per_device_train_batch_size
3. 能否推理？
    - 默认在训练过程中，trainer会通过_save_eval_images方法以eval方式保留三种类型的图像生成，如果需要编写推理方法，参考_save_eval_images方法编写对应代码即可

## 代码文件
```text
VAE_Model/
├── config_face.py              # VAEConfig：模型所有超参定义
├── model_face.py               # ConvVAE 模型主体（编码器/解码器/损失）
├── dataset_face.py             # CelebA Dataset（预处理、归一化）
├── train_face.py               # 单卡训练入口（HF Trainer）
├── train_face_accelerator.py   # 多卡训练入口（Accelerate/DDP）
├── vae.md                      # [项目流程简要概括]
├── README.md
├── 维度转换.md                  # 张量维度变换说明
└── train_result/               # [训练结果-140 epoch]
    ├── recon_compare_step024696.png   # 重建对比图
    ├── prior_sample_step024696.png    # 先验采样图
    └── interp_step024696.png          # 隐空间插值图
```

## 参考资料
1. VAE文献：https://arxiv.org/abs/1312.6114
2. VAE模型理论讲解视频的推荐👍：https://www.bilibili.com/video/BV1op421S7Ep/?spm_id_from=333.1387.search.video_card.click
3. VAE模型理论讲解视频的推荐👍：https://www.bilibili.com/video/BV1xFxMz1EMS/?spm_id_from=333.1387.search.video_card.click


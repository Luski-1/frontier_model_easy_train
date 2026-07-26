# 项目
CFG

## 主要内容
1. CFG属于可插拔的模块，不属于特定模型，因此没有做官方项目的人工注释

3. 该项目能否训练？
    - 可以开启训练
        ```python
        python train.py 或者 accelerate launch --num_processes ? train_accelerate.py
        ```
    - 训练前要求
        - 请提前下载类似ImageNet（多种类别，并且以文件夹作为分类）的数据集，可通过datasets方法或者通过kaggle下载。本人使用的是在kaggle下载的ImageNet-256数据集 
        - 设置train.py或train_accelerate.py中train_root和eval_root参数，指定为数据集地址
        - 根据显存大小，调整per_device_train_batch_size和gradient_accumulation_steps
    
4. 该项目能否推理？
    - 默认在训练过程中，trainer会通prediction_step方法进行先验抽样生成，如果需要编写推理方法，参考该方法编写对应代码即可

## 代码文件
```text
CFG/
├── config.py
├── model.py
├── train.py
├── train_accelerate.py
├── ema.py
├── utils.py
├── cfg.md
├── README.md
├── epoch299.0_ema.png
├── epoch300.0_ema.png
```

## 参考资料
1. CFG文献: https://arxiv.org/pdf/2208.11970
4. CFG理论讲解视频👍👍👍：https://www.bilibili.com/video/BV1BS411P7dW/?spm_id_from=333.1387.search.video_card.click&vd_source=401f1f7d80fdd51bba1fc24cf7961ff9
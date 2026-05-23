# 项目
SigLip

## 主要内容
1. 增加对应注释以及对应代码调整
2. 该项目能否开启训练
   - 可以
     - 提前下载roberta-large和dinov3-vitl16-pretrain-lvd1689m模型
     - 下载数据：python3 download.py
     - 训练命令：python3 train.py

## 代码文件
```text
siglip/
├── train.py
├── dataset.py
├── model.py
├── download.py
└── result/ [本人训练结果]
    ├── predict_example.jpg (训练6000step的结果展示)
    └── data_example.md (训练数据的格式展示)
```

## 参考资料
1. 原项目github：https://github.com/wyf3/llm_related/tree/main/train_siglip_from_scratch；
2. 原项目作者的讲解视频：https://www.bilibili.com/video/BV1i6kBYEELj/?spm_id_from=333.1387.search.video_card.click&vd_source=401f1f7d80fdd51bba1fc24cf7961ff9
3. siglip文献：https://arxiv.org/abs/2303.15343

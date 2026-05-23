# 项目
SigLip

## 主要内容
1. 增加部分注释以及修改部分模型代码
2. 增加训练数据下载以及查看的代码
3. 能否开启训练
   - 可以
     - 需要提前下载图像侧模型dinov3-vitl16-pretrain-lvd1689m
     - 需要提前下载数据 python3 download.py
     - 训练命令 python3 train.py

## 代码文件
```text
SigLip
│
│  README.md
│  data_example.md
│  dataset.py
│  download.py
│  model.py
│  train.py
│
├─train_result [本人训练结果]
│  │  predict_example.jpg
```

## 参考资料
1. 原项目github:https://github.com/wyf3/llm_related/tree/main/train_siglip_from_scratch
2. 原项目作者讲解视频：https://www.bilibili.com/video/BV1i6kBYEELj/?spm_id_from=333.1387.search.video_card.click
3. SigLip模型文献：https://arxiv.org/abs/2303.15343

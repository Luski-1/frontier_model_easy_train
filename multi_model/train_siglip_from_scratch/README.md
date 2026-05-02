# 项目
SigLip

## 项目架构
1. train.py
2. dataset.py
3. model.py
4. result/predict_example.jpg
5. download.py
6. result/data_example.md

## 参考资料
1. 主要参考项目：https://github.com/wyf3/llm_related/tree/main/train_siglip_from_scratch；
2. 主要参考项目的up主讲解视频：https://www.bilibili.com/video/BV1i6kBYEELj/?spm_id_from=333.1387.search.video_card.click&vd_source=401f1f7d80fdd51bba1fc24cf7961ff9
3. 该模型文献：https://arxiv.org/abs/2303.15343

## 主要修改内容
1. 增加对应注释以及对应代码调整

## 该项目能否直接开启训练：能
1. 提前下载roberta-large和dinov3-vitl16-pretrain-lvd1689m模型
2. 下载数据：python3 download.py
3. 开启训练：python3 train.py
- PS：可能需要注意数据集保存/读取位置是否正确匹配

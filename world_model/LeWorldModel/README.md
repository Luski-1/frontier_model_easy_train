# 项目
LeWorldModel

## 项目架构
1. download.py
2. train.py
3. pusht.yaml (config/train/data/pusht.yaml)
4. dataset.py (stable_worldmodel/data/dataset.py，仅截取源项目所使用部分，非完整stable_worldmodel/data/dataset.py)
5. utils.py

- PS：源项目的eval.py涉及gymnasium库，较复杂不做详细注释

## 参考资料
1. 源项目：https://github.com/lucas-maes/le-wm
2. 该模型文献：https://arxiv.org/pdf/2603.19312v1
3. 源项目公开数据：https://huggingface.co/collections/quentinll/lewm       可自行下载或使用download.py

## 主要修改内容
1. 增加对应注释

## 该项目能否直接开启训练
- 不能，该项目仅对源项目部分核心代码增加注释和参考文档，请使用源项目开展训练和推理

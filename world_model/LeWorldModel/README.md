# 项目
LeWorldModel

## 代码文件
```text
# 项目文件结构
LeWorldModel/
├── download.py
├── train.py
├── config/
│   └── train/
│       └── data/
│           └── pusht.yaml
├── stable_worldmodel/
│   └── data/
│       └── dataset.py        (节选stable_worldmodel库中源项目涉及调用的代码，非完整文件)
├── stable_pretraining/
│   └── data/
│       └── transforms.py     (节选stable_pretraining库中原项目涉及调用的代码，非完整文件)
├── utils.py
├── module.py
├── result/
│   └── data_example.md       (训练数据展示)
│   └── result.jpg            (推理结果)
└── ref/
    ├── SIGReg代码.md
    ├── SIGReg理论.md
    └── 推理阶段.md           (AI生成)
```


## 参考资料
1. 源项目：https://github.com/lucas-maes/le-wm
2. 该模型文献：https://arxiv.org/pdf/2603.19312v1
3. 源项目公开数据：https://huggingface.co/collections/quentinll/lewm       可自行下载或使用download.py

## 主要修改内容
1. 源项目的推理部分比较复杂并且涉及gymnasium库，难以重写重写整个项目
2. 仅对源项目的训练部分增加对应注释
3. 推理部分没有注释，可以参考ref的md文档

## 该项目能否直接开启训练
- 不能，该项目仅对源项目部分核心代码增加注释和参考文档，请使用源项目开展训练和推理

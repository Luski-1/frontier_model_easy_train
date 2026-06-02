# 项目
LEMW_Model

## 主要内容
1. reference为原项目代码，本人对训练过程的代码中增加方便理解的注释和知识点
   - 其中把原项目训练过程中所涉及的依赖库中的stable_worldmodel.data.dataset.py和stable_worldmodel.data.formats.hdf5.py摘出来添加对应的注释
   - 因仿真环境相关代码比较复杂，推理过程没有对应注释，推理要点可以参考lewm.md
   - stable_worldmodel的版本是0.1.0；stable-pretraining的版本是0.1.7
   - 原项目的相关日期是2026.5.31      
2. mini_lewm为重写项目的精简化代码
   - 不修改原项目的核心思路的基础上，仅保留训练过程的默认路线(pushT)
   - ⭐⭐训练过程取消lightning训练框架，训练过程取消stable_worldmodel库的所有依赖（数据加载、优化器加载、调度器加载...），降低学习成本
   - 简化训练过程的yaml配置文件
   - 相同模块内容的单py文件化
   - 保留增加的对应注释
   - 推理过程基本与原项目一致，但简化了推理过程的yaml配置文件，可以直接通过配置文件指定的模型目录和数据集目录，不再需要指定环境变量export STABLEWM_HOME
3. mini_lewm能否开启训练？
   - 可以开启训练
     ```text
     python train.py
     ```
   - 必须配置train.yaml中h5_path（数据集位置）和output_dir（训练文件保存位置）
   - 必须提前下载pushT数据集，请参考原项目的README.md
4. mini_lewm能否开启推理？
   - 可以开启推理
     ```text
     python eval.py
     ```
   - 必须配置eval.yaml中policy_path（模型位置）和cache_dir（数据集位置）和results_dir（推理文件保存位置）
   - 必须提前安装stable_worldmodel库，请参考原项目的README.md


## 代码文件
```text
\LEWM_MODEL
│  DATA_EXAMPLE.md [pushT数据集展示]
│  lewm.md [原项目流程简要概括]
│  README.md
│  SIGReg代码.md [原项目SIGReg Loss的代码分析]
│  SIGReg理论.md [原项目SIGReg Loss的理论分析]
│  
├─mini_lewm [重写项目]
│  │  dataset.py
│  │  eval.py
│  │  loss.py
│  │  model.py
│  │  train.py
│  │  utils.py
│  │  
│  └─configs
│          eval.yaml
│          train.yaml
│          
├─reference [原项目]
│  │  eval.py
│  │  jepa.py
│  │  LICENSE
│  │  module.py
│  │  README.md
│  │  stable_worldmodel.data.dataset.py [新增文件: 原项目中stable_worldmodel依赖库的相关数据集代码]
│  │  stable_worldmodel.data.formats.hdf5.py [新增文件: 原项目中stable_worldmodel依赖库的相关数据集代码]
│  │  train.py
│  │  utils.py
│  │  
│  ├─assets
│  │      lewm.gif
│  │      
│  └─config
│      ├─eval
│      │  │  cube.yaml
│      │  │  pusht.yaml
│      │  │  reacher.yaml
│      │  │  tworoom.yaml
│      │  │  
│      │  ├─launcher
│      │  │      local.yaml
│      │  │      
│      │  └─solver
│      │          adam.yaml
│      │          cem.yaml
│      │          
│      └─train
│          │  lewm.yaml
│          │  
│          ├─data
│          │      dmc.yaml
│          │      ogb.yaml
│          │      pusht.yaml
│          │      tworoom.yaml
│          │      
│          ├─launcher
│          │      local.yaml
│          │      
│          └─model
│                  lewm.yaml
│                  
└─train_result [训练结果]
        env_49.mp4
        pusht_results.txt
```

## 参考资料
1. 原项目github：https://github.com/lucas-maes/le-wm
2. 原项目文献：https://arxiv.org/pdf/2603.19312v1

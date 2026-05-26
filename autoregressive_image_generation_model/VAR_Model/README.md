# 项目
VAR_Model

## 主要内容
1. reference为原项目代码，本人对训练过程/推理过程的代码中增加方便理解的注释和知识点
2. mini_var为重写项目的精简化代码
   - 不修改原项目的核心思路的基础上，仅保留训练过程的默认分支路线(256*256)以及推理方法
   - 删除分布式，精简参数，精简日志打印
   - 相同模块内容的单py文件化
   - 保留增加的对应注释
3. mini_var能否开启训练？
   - 可以开启训练
      ```text
      python new_train.py \
          --bs 32 \
          --depth 16 \
          --data_path /path/to/ImageNet \
          --vae_ckpt /path/to/vae_ch160v4096z32.pth
      ```
   - 前置要求
      - 数据集默认是ImageNet，mini_VAR自动设置为256*256（降低显存），并且要求拆分为train和eval。如果本地数据集没有提前拆分train和eval，可以使用split_dataset.py原地拆分
      - vae_ch160v4096z32.pth是tokenzier文件[VQVAE模型]，用于解码VAR的预测图像ID，可以参考原项目MD文档进行下载
4. mini_var能否开启推理？
    - 可以开启推理
        ```text
        python new_sample.py \
            --vae_ckpt /path/to/vae_ch160v4096z32.pth
        ```


## 代码文件
```text
VAR_Model
│  README.md
│  split_dataset.py [可用于拆分ImageNet]
│  var.md [原项目流程简要概括]
│  
├─mini_var [重写项目]
│      new_args.py
│      new_data.py
│      new_sample.py
│      new_train.py
│      new_utils.py
│      new_var.py
│      new_vqvae.py
│      
├─reference [原项目]
│  │  .gitignore
│  │  demo_sample.ipynb
│  │  demo_zero_shot_edit.ipynb
│  │  dist.py
│  │  LICENSE
│  │  README.md
│  │  requirements.txt
│  │  train.py
│  │  trainer.py
│  │  
│  ├─models
│  │      basic_vae.py
│  │      basic_var.py
│  │      helpers.py
│  │      quant.py
│  │      var.py
│  │      vqvae.py
│  │      __init__.py
│  │      
│  └─utils
│          amp_sc.py
│          arg_util.py
│          data.py
│          data_sampler.py
│          lr_control.py
│          misc.py
│          
└─train_result
        generated_result.png
```
   
## 参考资料
1. 原项目github：https://github.com/FoundationVision/VAR
2. 原项目文献：https://arxiv.org/abs/2404.02905
3. VAR模型理论讲解视频的推荐：https://www.bilibili.com/video/BV1FsoiYQEif/?spm_id_from=333.1387.search.video_card.click


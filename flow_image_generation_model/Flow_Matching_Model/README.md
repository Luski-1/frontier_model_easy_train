# 项目
Flow_Matching

## 主要内容
1. reference为原项目代码

    - 本人以reference.example.image.train作为入口，所涉及的代码中增加方便理解的人工注释和知识点
    - 对reference.example.2d_flow_matching.ipynb/2d_discrete_flow_matching.ipynb增加人工注释。如果是想快速理解训练逻辑，可以先学习2d_flow_matching.ipynb。
    - 2d_discrete_flow_matching.ipynb是离散扩散文本模型，虽然增加了部分人工注释，但未注释的内容可以等到离散扩散文本模型MDLM或LLADA项目再学习

2. mini_flow_matching是为重写项目的精简化代码，ODE求解器采用更加简单的实现逻辑，同时包含EDM的时间t采样或者常规等差的时间t采样

3. mini_flow_matching能否训练？
    - 可以开启训练
        
        ```python
        python train.py 或者 accelerate launch --num_processes ? train_accelerator.py
        ```
    - 训练前要求
        - 请提前下载CELEBA数据集，可通过datasets方法或者通过kaggle下载
        ```python
        from torchvision import datasets
        dataset = datasets.CelebA(
            root="./data",
            split="all",
            download=True,
            target_type=["attr", "identity"]
        )  
        ```

        - 设置config.py中data-root参数，或者启动训练时指定--data-root参数，为数据集的目录
        - 根据显存大小，调整batch-size

4. mini_flow_matching能否推理？
    - 可以开启推理

    ```python
    python eval.py
    ```

## 代码文件
```text
Flow_Matching_Model/
│
├── Flow_Matching.md
├── README.md
│
├── mini_flow_matching/			# 重写项目
│   ├── config.py
│   ├── dataset.py
│   ├── eval.py
│   ├── model.py
│   ├── train.py
│   ├── train_accelerator.py
│   └── utils.py
│
├── reference/
│   ├── CHANGELOG.md
│   ├── CODE_OF_CONDUCT.md
│   ├── CONTRIBUTING.md
│   ├── LICENSE
│   ├── README.md
│   ├── RELEASE.md
│   ├── environment.yml
│   ├── setup.py
│   ├── .flake8
│   ├── .gitignore
│   ├── .pre-commit-config.yaml
│   │
│   ├── assets/
│   │   ├── arXiv-2412.06264-red.svg
│   │   ├── License-CC_BY--NC_4.0-lightgrey.svg
│   │   └── teaser.png
│   │
│   ├── docs/
│   │   ├── Makefile
│   │   ├── README.md
│   │   ├── custom_directives.py
│   │   ├── deploy.py
│   │   ├── deps.yml
│   │   ├── server.py
│   │   ├── _templates/classtemplate.rst
│   │   └ source/
│   │       ├── conf.py
│   │       ├── dummy.rst
│   │       ├── index.rst
│   │       ├── installation.rst
│   │       ├── modules.rst
│   │       ├── notebooks.rst
│   │       ├── references.rst
│   │       ├── refs.bib
│   │       ├── _images/ (discrete.png, riemannian_sphere.png, riemannian_torus.png, standalone.png)
│   │       ├── _static/css/custom.css
│   │       ├── _templates/classtemplate.rst
│   │       └ flow_matching.loss.rst
│   │       ├── flow_matching.path.rst
│   │       ├── flow_matching.path.scheduler.rst
│   │       ├── flow_matching.solver.rst
│   │       ├── flow_matching.utils.manifolds.rst
│   │       └ flow_matching.utils.model_wrapper.rst
│   │
│   ├── examples/											# 官方推荐的案例
│   │   ├── README.md
│   │   ├── 2d_cnf_maximum_likelihood.ipynb
│   │   ├── 2d_discrete_flow_matching.ipynb					# 添加了部分人工注释，剩余内容建议等到MDLM或LLADA再学习
│   │   ├── 2d_flow_matching.ipynb							# 添加了详细人工注释
│   │   ├── 2d_riemannian_flow_matching_flat_torus.ipynb
│   │   ├── 2d_riemannian_flow_matching_sphere.ipynb
│   │   ├── standalone_discrete_flow_matching.ipynb
│   │   ├── standalone_flow_matching.ipynb
│   │   │
│   │   └ image/
│   │   │   ├── README.md
│   │   │   ├── requirements.txt
│   │   │   ├── train.py									# 训练图像的主要入口，除模型架构以外都添加详细人工注释
│   │   │   ├── submitit_train.py
│   │   │   ├── train_arg_parser.py
│   │   │   ├── load_model_checkpoint.ipynb
│   │   │   │
│   │   │   ├── models/
│   │   │   │   ├── discrete_unet.py
│   │   │   │   ├── ema.py
│   │   │   │   ├── model_configs.py
│   │   │   │   ├── nn.py
│   │   │   │   └ unet.py
│   │   │   │
│   │   │   └ training/
│   │   │       ├── data_transform.py
│   │   │       ├── distributed_mode.py
│   │   │       ├── edm_time_discretization.py
│   │   │       ├── eval_loop.py
│   │   │       ├── grad_scaler.py
│   │   │       ├── load_and_save.py
│   │   │       └ train_loop.py
│   │   │
│   │   └ text/
│   │       ├── README.md
│   │       ├── environment.yml
│   │       ├── run_train.py
│   │       ├── train.py
│   │       │
│   │       ├── configs/
│   │       │   └ config.yaml
│   │       │
│   │       ├── data/
│   │       │   ├── __init__.py
│   │       │   ├── data.py
│   │       │   ├── tokenizer.py
│   │       │   └ utils.py
│   │       │
│   │       ├── logic/
│   │       │   ├── __init__.py
│   │       │   ├── evaluate.py
│   │       │   ├── flow.py
│   │       │   ├── generate.py
│   │       │   ├── state.py
│   │       │   └ training.py
│   │       │
│   │       ├── model/
│   │       │   ├── __init__.py
│   │       │   ├── rotary.py
│   │       │   └ transformer.py
│   │       │
│   │       ├── scripts/
│   │       │   ├── eval.py
│   │       │   └ run_eval.py
│   │       │
│   │       └ utils/
│   │           ├── __init__.py
│   │           ├── checkpointing.py
│   │           └ logging.py
│   │
│   ├── flow_matching/
│   │   ├── __init__.py
│   │   │
│   │   ├── loss/
│   │   │   ├── __init__.py
│   │   │   └ generalized_loss.py
│   │   │
│   │   ├── path/
│   │   │   ├── __init__.py
│   │   │   ├── affine.py
│   │   │   ├── geodesic.py
│   │   │   ├── mixture.py
│   │   │   ├── path.py
│   │   │   ├── path_sample.py
│   │   │   │
│   │   │   └ scheduler/
│   │   │       ├── __init__.py
│   │   │       ├── scheduler.py
│   │   │       └ schedule_transform.py
│   │   │
│   │   ├── solver/
│   │   │   ├── __init__.py
│   │   │   ├── discrete_solver.py
│   │   │   ├── ode_solver.py
│   │   │   ├── riemannian_ode_solver.py
│   │   │   ├── solver.py
│   │   │   └ utils.py
│   │   │
│   │   └ utils/
│   │       ├── __init__.py
│   │       ├── categorical_sampler.py
│   │       ├── model_wrapper.py
│   │       ├── utils.py
│   │       │
│   │       └ manifolds/
│   │           ├── __init__.py
│   │           ├── manifold.py
│   │           ├── sphere.py
│   │           ├── torus.py
│   │           └ utils.py
│   │
│   ├── tests/
│   │   ├── __init__.py
│   │   │
│   │   ├── path/
│   │   │   ├── __init__.py
│   │   │   ├── test_path.py
│   │   │   ├── test_schedule_transform.py
│   │   │   └ test_scheduler.py
│   │   │
│   │   ├── solver/
│   │   │   ├── __init__.py
│   │   │   ├── test_discrete_solver.py
│   │   │   ├── test_ode_solver.py
│   │   │   └ test_riemannian_ode_solver.py
│   │   │
│   │   └ utils/
│   │       ├── __init__.py
│   │       └ test_utils.py
│   │
│   ├── .github/
│   │   ├── ISSUE_TEMPLATE/
│   │   │   ├── bug_report.md
│   │   │   └ feature_request.md
│       └ workflows/
│           ├── ci.yaml
│           └ notebooks.yaml
│
├── train_result/
    ├── edm_sample_epoch_99.png				# 使用EDM提及的时间t采样方法
    └ sample_epoch_99.png					# 使用平局间隔的时间t采样方法
```

## 参考资料
1. Flow-Matching文献: https://arxiv.org/pdf/2412.06264
2. Flow-Matching原项目Github: https://github.com/facebookresearch/flow_matching
3. Flow-Matching模型理论讲解视频👍👍👍：https://www.bilibili.com/video/BV1i3PszdEWF/?spm_id_from=333.1387.search.video_card.click
4. Flow-Matching原项目代码讲解视频👍👍👍：https://www.bilibili.com/video/BV1xZcEznEYX/?spm_id_from=333.1387.search.video_card.click&vd_source=401f1f7d80fdd51bba1fc24cf7961ff9
5. Flow-Matching模型理论讲解视频👍👍👍:https://www.bilibili.com/video/BV142C2BSEbm/?spm_id_from=333.1387.upload.video_card.click&vd_source=401f1f7d80fdd51bba1fc24cf7961ff9  |  https://www.bilibili.com/video/BV1K2C2BSEZB/?spm_id_from=333.1387.upload.video_card.click&vd_source=401f1f7d80fdd51bba1fc24cf7961ff9
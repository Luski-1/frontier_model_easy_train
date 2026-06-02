# LeWM — 概览

## 模型架构

LeWM 是一个基于 JEPA（Joint-Embedding Predictive Architecture）的潜在世界模型，用于机器人操控任务的视觉预测和动作规划。

核心组件：

```
JEPA
├── encoder      — ViT（HuggingFace），将图像编码为 CLS token embedding
├── action_encoder — Embedder，将动作序列编码为 embedding
├── predictor    — ARPredictor（AdaLN-zero 条件 Transformer），自回归预测未来 embedding
├── projector    — MLP，对 encoder 输出做投影
└── pred_proj    — MLP，对 predictor 输出做投影
```

训练时：encoder 提取图像 embedding → predictor 根据历史 embedding + 动作预测未来 embedding → 用真实未来 embedding 计算损失。

推理时：encoder 编码当前观测和目标图像 → predictor 在潜在空间自回归推演 → 用 CEM 优化器搜索使预测最接近目标的动作序列。

---

## 训练流程

```
1. 加载 HDF5 数据集（pusht 机器人推箱子任务）
   ↓
2. 按帧抽样（frameskip=5）+ Z-Score 归一化 + 组装时间窗口
   ↓
3. 构建模型（encoder + predictor + action_encoder + projector + pred_proj）
   ↓
4. 训练循环（每个 epoch）：
   a. encode：图像 → ViT → CLS token → projector → pixel embedding
   b. encode：动作 → Embedder → action embedding
   c. predict：取 history_size 步的 pixel + action embedding → ARPredictor → 预测 embedding
   d. 损失 = MSE(预测 embedding, 真实未来 embedding) + λ × SIGReg(全体 embedding)
   e. 反向传播 + 梯度裁剪 + AdamW 更新
   ↓
5. 保存 checkpoint（state_dict.pt + config.json，兼容原始 eval.py）
```

---

## 推理流程(AI解读)

推理阶段的核心是 **CEM（Cross-Entropy Method，交叉熵方法）** 动作规划器，在潜在空间搜索最优动作序列，全程不执行真实环境交互。

整体流程：

```
1. 加载训练好的 checkpoint
   load_pretrained(cfg.policy) 读取 config.json → hydra 实例化 JEPA 模型
   → 加载 state_dict.pt → 模型设为 eval 模式，冻结梯度
   ↓
2. 每个评估回合（共 num_eval=50 个起点）：
   a. 从 HDF5 数据集采样一个起始帧 + 目标帧（goal_offset_steps=25 帧后）
   b. encode 当前观测图像 → 初始 embedding（ctx_emb）
   c. encode 目标图像 → goal_emb（规划的目标 anchor）
   ↓
3. CEM 规划循环（horizon=5 步，共 2 轮 rollout）：
   ┌─ 迭代 1~30 次 ─────────────────────────────────────┐
   │  a. 采样：从 N(μ, Σ) 采样 num_samples=300 条候选  │
   │     动作序列，形状 (B, 300, horizon, action_dim)     │
   │  b. rollout：对每条候选序列，用 predictor 自回归     │
   │     推演 horizon 步，得到预测的 embedding 轨迹         │
   │     （每步：action_encoder → predictor → pred_emb）   │
   │  c. 计算 cost：MSE(最终预测 emb, goal_emb)           │
   │     得到每条候选的标量 cost，(B, 300)                 │
   │  d. 选精英：取 cost 最低的 num_elites=30 条          │
   │  e. 重拟合：用精英集重新估计 μ 和 Σ                │
   │  f. 下一轮迭代用新的 N(μ_new, Σ_new) 采样          │
   └───────────────────────────────────────────────────────┘
   ↓
4. 输出：CEM 收敛后，取最优候选动作序列的第一步执行
   （warm_start：下一轮规划用上一轮剩余动作初始化分布）
   ↓
5. 在真实环境中执行一步 → 观测新状态 → 回到步骤 2 继续规划
```

CEM 关键超参（来自 `config/eval/solver/cem.yaml`）：

| 参数 | 值 | 含义 |
|------|-----|------|
| horizon | 5 | 每次规划往前看多少步 |
| action_block | 5 | 每条候选动作被切分成几块 |
| num_samples | 300 | 每轮采样的候选动作序列数 |
| num_elites | 30 | 每轮保留的精英候选数 |
| iterations | 30 | CEM 内循环迭代次数 |
| eval_budget | 50 | 每回合最多预测 50 步 |

注意：`horizon × action_block ≤ eval_budget`，所以 pusht 配置下需要 2 轮 rollout 才能覆盖完整预算。

计算量瓶颈：`get_cost` 每调用一次 = iterations × num_samples × horizon 次模型前向传播。
以默认配置计算：30 × 300 × 5 = 45,000 次前向传播/每步规划。

---

## 关键参数

| 类别 | 参数 | 值 | 说明 |
|------|------|-----|------|
| **数据** | frameskip | 5 | 原始数据每隔 5 帧取 1 帧 |
| | history_size | 3 | 上下文窗口步数（输入 3 步历史） |
| | num_preds | 1 | 预测步数（预测 1 步未来） |
| | img_size | 224 | 图像输入分辨率 |
| **模型** | embed_dim | 192 | embedding 维度（ViT-tiny） |
| | predictor_depth | 6 | predictor Transformer 层数 |
| | predictor_heads | 16 | 注意力头数 |
| **训练** | lr | 5e-5 | 学习率 |
| | batch_size | 128 | 批大小 |
| | max_epochs | 100 | 训练轮数 |
| | warmup_epochs | 10 | warmup 轮数 |
| | sigreg_weight | 0.09 | SIGReg 正则化权重 λ |
| **精度** | precision | bf16 | 混合精度训练 |

---

## 使用方法

### 训练
```bash
cd C:\Users\vincentliang1\Desktop\le-wm-main
python mini_lewm/train.py
```

### 评估
```bash
cd C:\Users\vincentliang1\Desktop\le-wm-main
python mini_lewm/eval.py
```

评估前需修改 `mini_lewm/configs/eval.yaml` 中的 `policy` 路径，指向训练保存的 checkpoint 目录。

依赖说明：`mini_lewm/eval.py` 保留了 `stable_worldmodel` 的 World/CEM/Policy 评估基础设施（gymnasium 环境相关，不重写），仅替换了模型加载（`load_model` 替代 `swm.wm.utils.load_pretrained`）和配置系统（`yaml.safe_load` 替代 `@hydra.main`）。

import os

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import numpy as np
import torch
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

from utils import load_config, load_model

# ImageNet 归一化参数，替代 spt.data.dataset_stats.ImageNet
IMAGENET_STATS = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**IMAGENET_STATS),
            transforms.Resize(size=cfg["eval"]["img_size"]),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    import stable_worldmodel as swm

    dataset_path = Path(cfg["cache_dir"])
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg["dataset"]["keys_to_cache"],
        cache_dir=dataset_path,
    )
    return dataset


def run():
    """评估入口。"""

    # ---- 加载配置 ----
    config_path = os.path.join(os.path.dirname(__file__), "configs", "eval.yaml")
    cfg = load_config(config_path)

    # 预测轮数 * 每一轮的行动数 <= 预测预算
    assert (
        cfg["plan_config"]["horizon"] * cfg["plan_config"]["action_block"] <= cfg["eval"]["eval_budget"]
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # ---- create world environment ----
    import stable_worldmodel as swm

    cfg["world"]["max_episode_steps"] = 2 * cfg["eval"]["eval_budget"]  # 最大预算
    cfg["world"]["num_envs"] = cfg["eval"]["num_eval"]
    world = swm.World(**cfg["world"])

    # ---- create the transform ----
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg["eval"]["dataset_name"])
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg["dataset"]["keys_to_cache"]:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # ---- run evaluation ----
    policy = cfg.get("policy", "random")

    if policy != "random":
        # 用 mini_lewm.utils.load_model 替代 swm.wm.utils.load_pretrained
        model = load_model(cfg["policy_path"])
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True

        config = swm.PlanConfig(**cfg["plan_config"])

        # 手动构造 CEMSolver，替代 hydra.utils.instantiate(cfg.solver, model=model)
        solver_cfg = cfg["solver"]
        from stable_worldmodel.solver import CEMSolver
        solver = CEMSolver(
            model=model,
            batch_size=solver_cfg.get("batch_size", 1),
            num_samples=solver_cfg["num_samples"],
            var_scale=solver_cfg["var_scale"],
            n_steps=solver_cfg["n_steps"],
            topk=solver_cfg["topk"],
            device=solver_cfg.get("device", "cuda"),
            seed=cfg["seed"],
        )

        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    # results_path 从配置读取，不再依赖 STABLEWM_HOME 默认值
    results_path = Path(cfg["output"]["results_dir"])

    # ---- sample the episodes and the starting indices ----
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg["eval"]["goal_offset_steps"] - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    # Map each dataset row's episode_idx to its max_start_idx
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg["seed"])
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg["eval"]["num_eval"], replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg["eval"]["num_eval"]:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    results_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg["eval"]["goal_offset_steps"],
        eval_budget=cfg["eval"]["eval_budget"],
        episodes_idx=eval_episodes.tolist(),
        callables=cfg["eval"].get("callables"),
        video=results_path,
    )
    end_time = time.time()

    print(metrics)

    results_path = results_path / cfg["output"]["filename"]
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        import yaml as _yaml
        f.write(_yaml.dump(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()

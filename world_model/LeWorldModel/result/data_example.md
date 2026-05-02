# 整体情况
1. 该数据类型为视频类型，包含每帧动作、状态等相关信息

# 数据结构
1. 通过python读取
```python
import h5py

# 1. 修改为你的 .h5 文件路径，需要提前解压
HDF5_FILE_PATH = "pusht_expert_train.h5"

# 2. 安全打开HDF5文件，遍历所有数据集
with h5py.File(HDF5_FILE_PATH, "r") as f:
    print("=" * 50)
    print(f"HDF5 文件: {HDF5_FILE_PATH}")
    print("=" * 50)
    
    # 遍历文件中所有的键（字段）
    for key in f.keys():
        dataset = f[key]
        print(f"\n字段名: {key}")
        print(f"   维度形状: {dataset.shape}")
        print(f"   数据类型: {dataset.dtype}")
        print(f"   是否为数据集: {isinstance(dataset, h5py.Dataset)}")

print("\n✅ 查看完成！仅展示元数据，未打印任何真实数据")
```
2. 打印结果
```text
文件: pusht_expert_train.h5
==================================================

字段名: action
   维度形状: (2336736, 2)
   数据类型: float32
   是否为数据集: True

字段名: ep_len
   维度形状: (18685,)
   数据类型: int32
   是否为数据集: True

字段名: ep_offset
   维度形状: (18685,)
   数据类型: int64
   是否为数据集: True

字段名: episode_idx
   维度形状: (2336736,)
   数据类型: int64
   是否为数据集: True

字段名: pixels
   维度形状: (2336736, 224, 224, 3)
   数据类型: uint8
   是否为数据集: True

字段名: proprio
   维度形状: (2336736, 4)
   数据类型: float32
   是否为数据集: True

字段名: state
   维度形状: (2336736, 7)
   数据类型: float32
   是否为数据集: True

字段名: step_idx
   维度形状: (2336736,)
   数据类型: int64
   是否为数据集: True
```

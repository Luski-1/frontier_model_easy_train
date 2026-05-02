```python
import h5py

# 1. 修改为你的 .h5 文件路径，需要提前解压
HDF5_FILE_PATH = "pusht_expert_train.h5"

# 2. 安全打开HDF5文件，遍历所有数据集
with h5py.File(HDF5_FILE_PATH, "r") as f:
    print("=" * 50)
    print(f"📂 HDF5 文件: {HDF5_FILE_PATH}")
    print("=" * 50)
    
    # 遍历文件中所有的键（字段）
    for key in f.keys():
        dataset = f[key]
        print(f"\n🔑 字段名: {key}")
        print(f"   维度形状: {dataset.shape}")
        print(f"   数据类型: {dataset.dtype}")
        print(f"   是否为数据集: {isinstance(dataset, h5py.Dataset)}")

print("\n✅ 查看完成！仅展示元数据，未打印任何真实数据")
```

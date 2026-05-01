# 设置HF镜像源（除非服务器可以访问国内镜像）
from datasets import load_dataset, load_from_disk, Dataset
import pandas as pd
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 国内镜像
# 设置自定义缓存目录
os.environ['HF_HOME'] = './RealSyn15M'


# 1. -------------------- 下载数据 --------------------
def download_raw_data():
    # 流式模式，不下载完整数据集，默认不保存，需要自己手动保存
    dataset = load_dataset("Kaichengalex/RealSyn15M", streaming=True)
    train_data = dataset['train']

    # 只下载前500000条
    subset = list(train_data.take(500000))
    print(f"已加载 {len(subset)} 条数据")

    dataset_1000 = Dataset.from_list(list(subset))

    # 手动保存到磁盘
    dataset_1000.save_to_disk('./')


# 2. -------------------- 转换数据 --------------------
def transform_data():
    """准备数据：将Arrow转换为Parquet"""
    print("步骤1: 加载数据集...")
    dataset = load_from_disk('./RealSyn15M')
    df = dataset.to_pandas()

    print(f"原始列名: {df.columns.tolist()}")

    # 重命名URL列
    url_df = df.rename(columns={'raw_image_url': 'URL'})
    print(f"转换后列名: {url_df.columns.tolist()}")

    # 验证syn_text列存在
    if 'syn_text' not in url_df.columns:
        raise ValueError(f"❌ 'syn_text'列不存在！实际列: {url_df.columns.tolist()}")

    # 保存
    output_path = './RealSyn15M/realsyn15m_urls.parquet'
    url_df.to_parquet(output_path, index=False)
    print(f"✅ 转换成功，文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    # 验证文件可读取
    test_df = pd.read_parquet(output_path)
    print(f"验证读取: 列名 {test_df.columns.tolist()}, 行数 {len(test_df)}")
    return output_path


# 3. -------------------- 转换数据 --------------------
def download_images(parquet_path):
    """下载图像"""
    print("\n步骤2: 开始下载...")

    from img2dataset import download

    download(
        url_list=parquet_path,
        input_format="parquet",
        url_col="URL",
        caption_col="syn_text",  # 主要保留的文本字段
        output_format="webdataset",
        output_folder="C:/PythonData/realsyn15m_images",
        processes_count=8,
        thread_count=16,
        image_size=224,
        resize_mode="center_crop",
        enable_wandb=False,
        number_sample_per_shard=10000,
        save_additional_columns=["text1", "text2", "text3"],  # 可以不增加额外字段
    )
    print("✅ 下载完成")


if __name__ == '__main__':
    """
    1. 该数据集的图像仅保存url，需要自行根据url下载对应真实图像
    2. 下载图像默认使用img2dataset库
    """

    # 下载原始数据
    download_raw_data()

    # 转换数据
    transform_data()

    # 下载图像
    download_images('./RealSyn15M/realsyn15m_urls.parquet')

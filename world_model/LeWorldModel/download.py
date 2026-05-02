from huggingface_hub import snapshot_download
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 下载到自定义目录
snapshot_download(
    repo_id="quentinll/lewm-pusht",
    repo_type="dataset",
    local_dir="./lewm-pusht",  # 自定义保存路径
    resume_download=True,        # 断点续传（网络断了可继续）
    max_workers=2              # 多线程下载，更快
)

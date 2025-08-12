#!/usr/bin/env python3
"""
预缓存所需模型到 $HF_HOME 下：
  - Qwen/Qwen3-1.7B
  - facebook/esm2_t12_35M_UR50D

用法：
  export HF_HOME=/home/x-cli32/chaohan/huggingface
  python precache_models.py
"""

import os
from huggingface_hub import snapshot_download

def main():
    # 1. 读取 HF_HOME（或使用默认 ~/.cache/huggingface）
    cache_dir = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    os.makedirs(cache_dir, exist_ok=True)
    print(f"🔍 使用 HuggingFace 缓存目录: {cache_dir}")

    # 2. 要预缓存的 repo 列表
    repos = [
        "Qwen/Qwen3-1.7B",
        "facebook/esm2_t12_35M_UR50D",
    ]

    # 3. 逐个下载
    for repo_id in repos:
        print(f"\n➡️ 正在预缓存 {repo_id} ...")
        snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_files_only=False,   # 允许在线拉取
            resume_download=True,     # 支持断点续传
        )
        print(f"✅ {repo_id} 已缓存到 {cache_dir}/hub/models--{repo_id.replace('/', '--')}")

    print("\n🎉 模型预缓存完成，请用 local_files_only=True 启动脚本或设置 TRANSFORMERS_OFFLINE=1 进行离线加载。")

if __name__ == "__main__":
    main()

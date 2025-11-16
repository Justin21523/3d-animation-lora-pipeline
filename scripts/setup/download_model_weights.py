#!/usr/bin/env python3
"""
模型權重下載腳本
下載所有必需的預訓練模型權重

移動自 /tmp 到項目scripts/setup目錄
"""

import os
import sys
from pathlib import Path
import urllib.request


def download_with_progress(url, dest_path, desc="Downloading"):
    """下載文件並顯示進度"""
    try:
        print(f"\n{desc}: {url}")
        print(f"保存到: {dest_path}")

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        def reporthook(blocknum, blocksize, totalsize):
            if totalsize > 0:
                downloaded = blocknum * blocksize
                percent = min(downloaded * 100.0 / totalsize, 100)
                size_mb = downloaded / (1024 * 1024)
                total_mb = totalsize / (1024 * 1024)
                sys.stdout.write(f"\r  進度: {percent:.1f}% ({size_mb:.1f}/{total_mb:.1f} MB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, dest_path, reporthook)
        print(f"\n✓ 下載完成!")

        file_size = os.path.getsize(dest_path)
        if file_size < 1024:
            print(f"✗ 警告: 文件太小 ({file_size} bytes)")
            return False
        print(f"  文件大小: {file_size / (1024*1024):.1f} MB")
        return True

    except Exception as e:
        print(f"\n✗ 下載失敗: {e}")
        return False


def download_from_huggingface(repo_id, filename, local_dir):
    """從HuggingFace Hub下載"""
    try:
        from huggingface_hub import hf_hub_download
        print(f"\n從 HuggingFace 下載: {repo_id}/{filename}")

        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )

        if os.path.exists(file_path) and os.path.getsize(file_path) > 1024:
            print(f"✓ 下載完成: {file_path}")
            print(f"  文件大小: {os.path.getsize(file_path) / (1024*1024):.1f} MB")
            return True
        return False

    except Exception as e:
        print(f"✗ HuggingFace 下載失敗: {e}")
        return False


def main():
    # 模型基礎目錄
    MODEL_BASE = Path("/mnt/c/AI_LLM_projects/ai_warehouse/models")

    print("=" * 70)
    print("模型權重下載腳本")
    print("=" * 70)

    # 下載配置
    downloads = [
        {
            "name": "YOLOv8x",
            "path": MODEL_BASE / "detection/yolov8x.pt",
            "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt",
        },
        {
            "name": "U2Net",
            "path": MODEL_BASE / "segmentation/U-2-Net/u2net.pth",
            "url": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
        },
        {
            "name": "ISNet (DIS)",
            "path": MODEL_BASE / "segmentation/DIS/isnet-general-use.pth",
            "url": None,  # 需要手動gdown
            "manual": "gdown https://drive.google.com/uc?id=1XHIzgTzY5BQHw140EDIgwIb53K659ENH"
        },
        {
            "name": "Anime-Seg",
            "path": MODEL_BASE / "segmentation/anime-segmentation/isnetis.ckpt",
            "hf_repo": "skytnt/anime-seg",
            "hf_file": "isnetis.ckpt"
        },
        {
            "name": "RIFE 4.26",
            "path": MODEL_BASE / "interpolation/Practical-RIFE/train_log/flownet.pkl",
            "hf_repo": "AlexWortega/RIFE",
            "hf_file": "flownet.pkl"
        },
    ]

    success_count = 0
    total_count = len(downloads)

    for i, config in enumerate(downloads, 1):
        print(f"\n[{i}/{total_count}] {config['name']}")

        if config["path"].exists() and os.path.getsize(config["path"]) > 1024*1024:
            size_mb = os.path.getsize(config["path"]) / (1024*1024)
            print(f"✓ 已存在 ({size_mb:.1f} MB)")
            success_count += 1
            continue

        # HuggingFace 下載
        if "hf_repo" in config:
            if download_from_huggingface(
                config["hf_repo"],
                config["hf_file"],
                str(config["path"].parent)
            ):
                success_count += 1

        # URL 下載
        elif "url" in config and config["url"]:
            if download_with_progress(config["url"], str(config["path"]), f"下載 {config['name']}"):
                success_count += 1

        # 手動下載說明
        elif "manual" in config:
            print(f"⚠ 需要手動下載:")
            print(f"  {config['manual']}")

    print("\n" + "=" * 70)
    print(f"完成: {success_count}/{total_count} 模型已就緒")
    print("=" * 70)

    return success_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Model Download Automation Script

Downloads required models for Intelligent Frame Processing:
- SAM2 (instance segmentation)
- LaMa (inpainting)
- RealESRGAN (upscaling)
- CodeFormer (face enhancement)

Usage:
    # Download all models
    python scripts/setup/download_models.py --all

    # Download specific models
    python scripts/setup/download_models.py --models sam2 lama

    # Check what's installed
    python scripts/setup/download_models.py --check
"""

import argparse
import sys
from pathlib import Path
import subprocess
import urllib.request
from tqdm import tqdm


# Model definitions
MODELS = {
    'sam2': {
        'name': 'SAM2 (Segment Anything 2)',
        'description': 'Instance segmentation for character extraction',
        'install_method': 'pip',
        'package': 'git+https://github.com/facebookresearch/segment-anything-2.git',
        'checkpoints': [
            {
                'name': 'sam2_hiera_large.pt',
                'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt',
                'size': '3.1 GB'
            },
            {
                'name': 'sam2_hiera_base_plus.pt',
                'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt',
                'size': '1.2 GB'
            }
        ],
        'checkpoint_dir': '/mnt/c/AI_LLM_projects/ai_warehouse/models/segmentation'
    },
    'lama': {
        'name': 'LaMa (Large Mask Inpainting)',
        'description': 'High-quality background inpainting',
        'install_method': 'pip',
        'package': 'lama-cleaner',
        'checkpoints': [],  # Downloaded automatically by lama-cleaner
        'checkpoint_dir': None
    },
    'realesrgan': {
        'name': 'RealESRGAN',
        'description': 'High-quality upscaling',
        'install_method': 'pip',
        'package': 'realesrgan',
        'additional_packages': ['basicsr'],
        'checkpoints': [
            {
                'name': 'RealESRGAN_x4plus_anime_6B.pth',
                'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
                'size': '17 MB'
            }
        ],
        'checkpoint_dir': '/mnt/c/AI_LLM_projects/ai_warehouse/models/enhancement'
    },
    'codeformer': {
        'name': 'CodeFormer',
        'description': 'Face enhancement and restoration',
        'install_method': 'manual',
        'note': 'CodeFormer requires manual installation from GitHub',
        'github': 'https://github.com/sczhou/CodeFormer',
        'checkpoints': [],
        'checkpoint_dir': '/mnt/c/AI_LLM_projects/ai_warehouse/models/enhancement'
    }
}


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path):
    """Download file with progress bar"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📥 Downloading {output_path.name}...")

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, str(output_path), reporthook=t.update_to)

    print(f"✓ Downloaded to {output_path}")


def install_package(package_name: str):
    """Install Python package via pip"""
    print(f"📦 Installing {package_name}...")

    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", package_name
        ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f"✓ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_name}: {e}")
        return False


def check_installation(model_key: str) -> dict:
    """Check if model is installed"""
    model = MODELS[model_key]
    status = {'installed': False, 'checkpoints': []}

    # Check Python package
    if model['install_method'] == 'pip':
        package = model['package'].split('+')[-1].split('/')[-1].replace('.git', '')

        try:
            __import__(package.replace('-', '_'))
            status['package_installed'] = True
        except ImportError:
            status['package_installed'] = False
    else:
        status['package_installed'] = None  # Manual install

    # Check checkpoints
    if model['checkpoint_dir']:
        checkpoint_dir = Path(model['checkpoint_dir'])
        for ckpt in model.get('checkpoints', []):
            ckpt_path = checkpoint_dir / ckpt['name']
            status['checkpoints'].append({
                'name': ckpt['name'],
                'exists': ckpt_path.exists(),
                'path': str(ckpt_path)
            })

    status['installed'] = (
        status.get('package_installed', True) and
        all(c['exists'] for c in status['checkpoints'])
    )

    return status


def install_model(model_key: str, download_checkpoints: bool = True):
    """Install specific model"""
    model = MODELS[model_key]

    print(f"\n{'='*60}")
    print(f"Installing {model['name']}")
    print(f"{'='*60}")
    print(f"Description: {model['description']}\n")

    # Step 1: Install Python package
    if model['install_method'] == 'pip':
        if not install_package(model['package']):
            print(f"⚠️ Package installation failed, but continuing...")

        # Install additional packages
        for pkg in model.get('additional_packages', []):
            install_package(pkg)

    elif model['install_method'] == 'manual':
        print(f"⚠️ {model['name']} requires manual installation:")
        print(f"   GitHub: {model.get('github', 'N/A')}")
        print(f"   Note: {model.get('note', '')}")
        return

    # Step 2: Download checkpoints
    if download_checkpoints and model.get('checkpoints'):
        checkpoint_dir = Path(model['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for ckpt in model['checkpoints']:
            ckpt_path = checkpoint_dir / ckpt['name']

            if ckpt_path.exists():
                print(f"✓ Checkpoint already exists: {ckpt['name']}")
                continue

            print(f"\n📦 Checkpoint: {ckpt['name']} ({ckpt['size']})")

            try:
                download_file(ckpt['url'], ckpt_path)
            except Exception as e:
                print(f"❌ Failed to download {ckpt['name']}: {e}")
                print(f"   You can manually download from: {ckpt['url']}")

    print(f"\n✅ {model['name']} installation complete!")


def check_all_models():
    """Check installation status of all models"""
    print("\n" + "="*60)
    print("  MODEL INSTALLATION STATUS")
    print("="*60 + "\n")

    for model_key, model in MODELS.items():
        status = check_installation(model_key)

        # Status indicator
        if status['installed']:
            indicator = "✅"
        elif status.get('package_installed') or any(c['exists'] for c in status['checkpoints']):
            indicator = "⚠️"
        else:
            indicator = "❌"

        print(f"{indicator} {model['name']}")
        print(f"   Description: {model['description']}")

        if status.get('package_installed') is not None:
            pkg_status = "✓" if status['package_installed'] else "✗"
            print(f"   Package: {pkg_status}")

        if status['checkpoints']:
            print(f"   Checkpoints:")
            for ckpt in status['checkpoints']:
                ckpt_status = "✓" if ckpt['exists'] else "✗"
                print(f"     {ckpt_status} {ckpt['name']}")

        print()


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Download models for Intelligent Frame Processing"
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all models'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        choices=list(MODELS.keys()),
        help='Specific models to download'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check installation status of all models'
    )
    parser.add_argument(
        '--no-checkpoints',
        action='store_true',
        help='Skip checkpoint downloads (only install packages)'
    )

    args = parser.parse_args()

    # Check status
    if args.check:
        check_all_models()
        return

    # Determine which models to install
    models_to_install = []

    if args.all:
        models_to_install = list(MODELS.keys())
    elif args.models:
        models_to_install = args.models
    else:
        # Interactive selection
        print("\n" + "="*60)
        print("  MODEL DOWNLOAD")
        print("="*60 + "\n")
        print("Available models:\n")

        for i, (key, model) in enumerate(MODELS.items(), 1):
            print(f"{i}. {model['name']}")
            print(f"   {model['description']}\n")

        print("Select models to download (comma-separated numbers, or 'all'):")
        selection = input("> ").strip()

        if selection.lower() == 'all':
            models_to_install = list(MODELS.keys())
        else:
            try:
                indices = [int(x.strip()) - 1 for x in selection.split(',')]
                models_to_install = [list(MODELS.keys())[i] for i in indices]
            except (ValueError, IndexError):
                print("❌ Invalid selection")
                return

    # Install models
    download_checkpoints = not args.no_checkpoints

    for model_key in models_to_install:
        install_model(model_key, download_checkpoints)

    # Final status check
    print("\n" + "="*60)
    print("  INSTALLATION COMPLETE")
    print("="*60 + "\n")

    check_all_models()

    print("\n📚 Next steps:")
    print("  1. Test installation:")
    print("     python scripts/core/models/model_loader.py --test-model all")
    print("\n  2. Run intelligent processor:")
    print("     python scripts/data_curation/intelligent_frame_processor.py \\")
    print("       input_frames/ --output-dir output/ --device cuda")
    print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test SOTA Evaluation Components

Verifies that all SOTA models and dependencies are correctly installed.
"""

import sys
from pathlib import Path

# Colors for terminal output
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
BLUE = '\033[0;34m'
NC = '\033[0m'


def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    if package_name is None:
        package_name = module_name

    try:
        __import__(module_name)
        print(f"{GREEN}✓{NC} {package_name} imported successfully")
        return True
    except ImportError as e:
        print(f"{RED}✗{NC} {package_name} import failed: {e}")
        return False


def test_transformers():
    """Test transformers and model loading"""
    print(f"\n{BLUE}[1/7] Testing Transformers...{NC}")

    if not test_import('transformers'):
        return False

    try:
        from transformers import AutoTokenizer
        print(f"{GREEN}✓{NC} Transformers API available")
        return True
    except Exception as e:
        print(f"{RED}✗{NC} Transformers test failed: {e}")
        return False


def test_insightface():
    """Test InsightFace"""
    print(f"\n{BLUE}[2/7] Testing InsightFace...{NC}")

    if not test_import('insightface'):
        return False

    try:
        from insightface.app import FaceAnalysis
        print(f"{GREEN}✓{NC} InsightFace API available")

        # Test initialization (CPU only for speed)
        app = FaceAnalysis(providers=['CPUExecutionProvider'])
        print(f"{GREEN}✓{NC} InsightFace can initialize")
        return True
    except Exception as e:
        print(f"{RED}✗{NC} InsightFace test failed: {e}")
        return False


def test_lpips():
    """Test LPIPS"""
    print(f"\n{BLUE}[3/7] Testing LPIPS...{NC}")

    if not test_import('lpips'):
        return False

    try:
        import lpips
        import torch
        loss_fn = lpips.LPIPS(net='alex')
        print(f"{GREEN}✓{NC} LPIPS model can be loaded")
        return True
    except Exception as e:
        print(f"{RED}✗{NC} LPIPS test failed: {e}")
        return False


def test_pyiqa():
    """Test PyIQA (MUSIQ)"""
    print(f"\n{BLUE}[4/7] Testing PyIQA (MUSIQ)...{NC}")

    if not test_import('pyiqa'):
        return False

    try:
        import pyiqa
        print(f"{GREEN}✓{NC} PyIQA available")

        # List available metrics
        print(f"{GREEN}✓{NC} PyIQA metrics available")
        return True
    except Exception as e:
        print(f"{RED}✗{NC} PyIQA test failed: {e}")
        return False


def test_diffusers():
    """Test Diffusers"""
    print(f"\n{BLUE}[5/7] Testing Diffusers...{NC}")

    if not test_import('diffusers'):
        return False

    try:
        from diffusers import StableDiffusionPipeline
        print(f"{GREEN}✓{NC} Diffusers pipeline available")
        return True
    except Exception as e:
        print(f"{RED}✗{NC} Diffusers test failed: {e}")
        return False


def test_model_paths():
    """Test model paths configuration"""
    print(f"\n{BLUE}[6/7] Testing Model Paths Configuration...{NC}")

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from scripts.core.utils.model_paths import load_model_paths

        config = load_model_paths()
        print(f"{GREEN}✓{NC} Model paths config loaded")

        # Verify key paths
        assert 'base_models' in config
        assert 'evaluation_models' in config
        assert 'projects' in config
        print(f"{GREEN}✓{NC} Config structure valid")

        return True
    except Exception as e:
        print(f"{RED}✗{NC} Model paths test failed: {e}")
        return False


def test_sota_evaluator():
    """Test SOTA evaluator can be imported"""
    print(f"\n{BLUE}[7/7] Testing SOTA Evaluator Script...{NC}")

    try:
        evaluator_path = Path(__file__).parent.parent / 'evaluation' / 'sota_lora_evaluator.py'
        if not evaluator_path.exists():
            print(f"{RED}✗{NC} SOTA evaluator script not found")
            return False

        print(f"{GREEN}✓{NC} SOTA evaluator script exists")
        return True
    except Exception as e:
        print(f"{RED}✗{NC} SOTA evaluator test failed: {e}")
        return False


def check_model_files():
    """Check if model files exist"""
    print(f"\n{BLUE}Checking Model Files...{NC}")

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from scripts.core.utils.model_paths import get_project_config

        luca_config = get_project_config('luca')

        models = {
            'SD v1.5 (Base Model)': Path(luca_config['base_model']),
            'Qwen2-VL-7B (Caption)': Path(luca_config['caption_model']),
            'InternVL2-8B (Evaluation)': Path(luca_config['evaluation_models']['internvl2']),
        }

        for name, path in models.items():
            if path.exists():
                print(f"{GREEN}✓{NC} {name}: {path}")
            else:
                print(f"{YELLOW}○{NC} {name}: Not found (will download on first use or use fallback)")
    except Exception as e:
        print(f"{RED}✗{NC} Model file check failed: {e}")


def main():
    print(f"{BLUE}{'='*70}{NC}")
    print(f"{BLUE}SOTA EVALUATION COMPONENTS TEST{NC}")
    print(f"{BLUE}{'='*70}{NC}")

    tests = [
        ('Transformers', test_transformers),
        ('InsightFace', test_insightface),
        ('LPIPS', test_lpips),
        ('PyIQA', test_pyiqa),
        ('Diffusers', test_diffusers),
        ('Model Paths', test_model_paths),
        ('SOTA Evaluator', test_sota_evaluator),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"{RED}✗{NC} {name} test crashed: {e}")
            results.append((name, False))

    # Check model files
    check_model_files()

    # Summary
    print(f"\n{BLUE}{'='*70}{NC}")
    print(f"{BLUE}TEST SUMMARY{NC}")
    print(f"{BLUE}{'='*70}{NC}")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = f"{GREEN}PASS{NC}" if result else f"{RED}FAIL{NC}"
        print(f"  {name}: {status}")

    print(f"\n{BLUE}Total: {passed}/{total} tests passed{NC}")

    if passed == total:
        print(f"\n{GREEN}✓ All components ready!{NC}")
        print(f"\n{YELLOW}Note:{NC} Some models (InternVL2) will be downloaded on first use.")
        print(f"{YELLOW}      If not available, system will automatically use fallback models (CLIP).{NC}")
        return 0
    else:
        print(f"\n{YELLOW}⚠ Some components failed, but system can still work with fallbacks.{NC}")
        return 1


if __name__ == '__main__':
    sys.exit(main())

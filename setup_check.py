#!/usr/bin/env python3
"""
Setup verification script.
Checks all dependencies and data paths.
"""
import sys
import importlib
from pathlib import Path


def check_package(name, import_name=None):
    """Check if package is installed."""
    if import_name is None:
        import_name = name

    try:
        importlib.import_module(import_name)
        print(f"✓ {name}")
        return True
    except ImportError:
        print(f"✗ {name} - NOT FOUND")
        return False


def main():
    print("="*60)
    print("Setup Verification")
    print("="*60)

    print("\nChecking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} - Need 3.8+")

    print("\nChecking required packages...")
    packages = [
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('peft', 'peft'),
        ('bitsandbytes', 'bitsandbytes'),
        ('accelerate', 'accelerate'),
        ('datasets', 'datasets'),
        ('pillow', 'PIL'),
        ('numpy', 'numpy'),
        ('opencv-python', 'cv2'),
        ('shapely', 'shapely'),
        ('matplotlib', 'matplotlib'),
        ('tqdm', 'tqdm'),
        ('scikit-learn', 'sklearn'),
        ('scipy', 'scipy'),
    ]

    all_ok = True
    for name, import_name in packages:
        if not check_package(name, import_name):
            all_ok = False

    print("\nChecking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠ CUDA not available - will run on CPU (very slow)")
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")

    print("\nChecking project structure...")
    dirs = ['data', 'model', 'train', 'inference', 'eval', 'configs']
    for dirname in dirs:
        path = Path(dirname)
        if path.exists():
            print(f"✓ {dirname}/")
        else:
            print(f"✗ {dirname}/ - NOT FOUND")
            all_ok = False

    print("\nChecking config...")
    try:
        from configs.config import CONFIG
        print(f"✓ Config loaded")
        print(f"  Data path: {CONFIG['ocid_grasp_path']}")
        print(f"  Model: {CONFIG['model_name']}")

        # Check if data exists
        data_path = Path(CONFIG['ocid_grasp_path'])
        if data_path.exists():
            print(f"✓ Data directory exists")

            # Check for splits
            for split in ['train.pkl', 'val.pkl', 'test.pkl']:
                if (data_path / split).exists():
                    print(f"  ✓ {split}")
                else:
                    print(f"  ✗ {split} - NOT FOUND")
        else:
            print(f"⚠ Data directory not found: {data_path}")
            print(f"  Update 'ocid_grasp_path' in configs/config.py")

    except Exception as e:
        print(f"✗ Error loading config: {e}")
        all_ok = False

    print("\n" + "="*60)
    if all_ok:
        print("✓ All checks passed!")
        print("\nNext steps:")
        print("  1. python data/test_loader.py          # Test data loading")
        print("  2. python model/test_overfit.py        # Sanity check")
        print("  3. python train/train_grasp_lora.py    # Start training")
    else:
        print("✗ Some checks failed")
        print("\nPlease install missing packages:")
        print("  pip install -r requirements.txt")

    print("="*60)


if __name__ == '__main__':
    main()

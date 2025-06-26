#!/usr/bin/env python3
"""Check if all required dependencies are installed."""

import importlib
import sys
from pathlib import Path

# Core dependencies
REQUIRED_PACKAGES = {
    "torch": "PyTorch",
    "transformers": "Transformers",
    "datasets": "Datasets",
    "accelerate": "Accelerate",
    "numpy": "NumPy",
    "scipy": "SciPy",
    "sklearn": "Scikit-learn",
    "tqdm": "TQDM",
    "yaml": "PyYAML",
}

# Optional packages
OPTIONAL_PACKAGES = {"wandb": "Weights & Biases", "tensorboard": "TensorBoard"}


def check_package(package_name: str, display_name: str, required: bool = True):
    """Check if a package is installed."""
    try:
        importlib.import_module(package_name)
        print(f"✓ {display_name} ({package_name})")
        return True
    except ImportError:
        if required:
            print(f"✗ {display_name} ({package_name}) - REQUIRED")
        else:
            print(f"- {display_name} ({package_name}) - optional")
        return False


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch

        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print("- CUDA not available (CPU training only)")
    except:
        print("✗ Cannot check CUDA (PyTorch not installed)")


def main():
    print("Checking OLMo dependencies...")
    print("=" * 50)

    # Check required packages
    print("\nRequired packages:")
    missing_required = []
    for pkg, name in REQUIRED_PACKAGES.items():
        if not check_package(pkg, name, required=True):
            missing_required.append(pkg)

    # Check optional packages
    print("\nOptional packages:")
    for pkg, name in OPTIONAL_PACKAGES.items():
        check_package(pkg, name, required=False)

    # Check CUDA
    print("\nGPU Support:")
    check_cuda()

    # Check if OLMo modules are importable
    print("\nOLMo modules:")
    olmo_path = Path(__file__).parent
    sys.path.insert(0, str(olmo_path))

    olmo_modules = ["olmo.models.olmo_model", "olmo.data.data_curation", "olmo.training.trainer"]

    for module in olmo_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module} - {str(e)}")

    # Summary
    print("\n" + "=" * 50)
    if missing_required:
        print(f"❌ Missing {len(missing_required)} required packages:")
        for pkg in missing_required:
            print(f"   - {pkg}")
        print("\nInstall with: pip install " + " ".join(missing_required))
        sys.exit(1)
    else:
        print("✅ All required dependencies are installed!")
        print("\nYou can now run: ./run_minimal_test.sh")


if __name__ == "__main__":
    main()

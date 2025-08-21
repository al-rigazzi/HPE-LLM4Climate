#!/usr/bin/env python
"""
Test script to verify the training environment setup

This script performs basic checks to ensure all components are working correctly.
"""

import sys
import subprocess
import importlib
from pathlib import Path


def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version >= (3, 8):
        print(f"   ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ✗ Python {version.major}.{version.minor}.{version.micro} - requires 3.8+")
        return False


def check_package(package_name, import_name=None, version_attr=None):
    """Check if a package is installed and optionally get version."""
    import_name = import_name or package_name
    try:
        module = importlib.import_module(import_name)
        if version_attr and hasattr(module, version_attr):
            version = getattr(module, version_attr)
            print(f"   ✓ {package_name} {version}")
        else:
            print(f"   ✓ {package_name}")
        return True
    except ImportError:
        print(f"   ✗ {package_name} not found")
        return False


def check_cuda():
    """Check CUDA availability."""
    print("🚀 Checking CUDA support...")
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            print(f"   ✓ CUDA available with {device_count} GPU(s)")
            print(f"   ✓ Primary GPU: {device_name}")
            return True
        else:
            print("   ⚠️  CUDA not available (CPU only)")
            return False
    except ImportError:
        print("   ✗ PyTorch not installed")
        return False


def check_deepspeed():
    """Check DeepSpeed installation and capabilities."""
    print("⚡ Checking DeepSpeed...")
    try:
        import deepspeed
        print(f"   ✓ DeepSpeed {deepspeed.__version__}")

        # Check DeepSpeed operations
        try:
            from deepspeed.ops.adam import FusedAdam
            print("   ✓ Fused Adam optimizer available")
        except ImportError:
            print("   ⚠️  Fused optimizers not available")

        return True
    except ImportError:
        print("   ✗ DeepSpeed not installed")
        return False


def check_model_files():
    """Check if AIFS model files exist."""
    print("🏗️  Checking AIFS model files...")

    base_path = Path(__file__).parent.parent.parent.parent
    aifs_dir = base_path / "aifs-single-1.0"

    required_files = [
        "aifs-single-mse-1.0.ckpt",
        "config_finetuning.yaml",
        "config_pretraining.yaml"
    ]

    all_found = True
    for file_name in required_files:
        file_path = aifs_dir / file_name
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"   ✓ {file_name} ({size_mb:.1f} MB)")
        else:
            print(f"   ✗ {file_name} not found")
            all_found = False

    return all_found


def check_config_files():
    """Check if configuration files exist."""
    print("⚙️  Checking configuration files...")

    config_dir = Path(__file__).parent.parent  # Go to training directory
    required_configs = [
        "config.yaml",
        "deepspeed_config.json"
    ]

    all_found = True
    for config_file in required_configs:
        config_path = config_dir / config_file
        if config_path.exists():
            print(f"   ✓ {config_file}")
        else:
            print(f"   ✗ {config_file} not found")
            all_found = False

    return all_found


def run_basic_model_test():
    """Run a basic model loading test."""
    print("🧪 Running basic model test...")

    try:
        # Test basic PyTorch functionality
        import torch
        x = torch.randn(2, 3)
        y = torch.mm(x, x.t())
        print("   ✓ PyTorch tensor operations working")

        # Test transformers (using a smaller model for testing)
        from transformers import AutoTokenizer
        # Use a lightweight tokenizer for testing
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2", trust_remote_code=True)
            tokens = tokenizer("Hello world")
            print("   ✓ Transformers tokenization working")
        except Exception:
            # Fallback for basic tokenization test
            from transformers import PreTrainedTokenizerFast
            print("   ✓ Transformers library available")

        # Test basic YAML loading
        import yaml
        with open(Path(__file__).parent.parent / "config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        print("   ✓ Configuration file loading working")

        return True

    except Exception as e:
        print(f"   ✗ Model test failed: {str(e)}")
        return False


def main():
    """Run all checks."""
    print("🔍 HPE LLM4Climate Training Environment Test")
    print("=" * 50)

    checks = [
        ("Python Version", check_python_version),
        ("Core Packages", lambda: all([
            check_package("torch", version_attr="__version__"),
            check_package("transformers", version_attr="__version__"),
            check_package("numpy", version_attr="__version__"),
            check_package("yaml", "yaml"),
            # wandb disabled for now
            # check_package("wandb", version_attr="__version__"),
        ])),
        ("CUDA Support", check_cuda),
        ("DeepSpeed", check_deepspeed),
        ("Model Files", check_model_files),
        ("Config Files", check_config_files),
        ("Basic Functionality", run_basic_model_test),
    ]

    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"   ✗ Error: {str(e)}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"   {status:<8} {name}")

    print(f"\n🎯 Overall: {passed}/{total} checks passed")

    if passed == total:
        print("🎉 Environment is ready for training!")
        return 0
    else:
        print("⚠️  Some issues found. Please check the failed tests above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

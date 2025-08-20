#!/usr/bin/env python3
"""
AIFS Checkpoint Inspector

A utility to inspect the structure of ECMWF AIFS PyTorch checkpoints
without requiring the full anemoi framework dependencies.
"""

import io
import os
import zipfile
from pathlib import Path

import torch


def inspect_aifs_checkpoint(checkpoint_path: str):
    """
    Inspect AIFS checkpoint structure with detailed analysis.

    Args:
        checkpoint_path: Path to the .ckpt file
    """
    print(f"🔍 AIFS Checkpoint Inspector")
    print(f"=" * 50)
    print(f"File: {checkpoint_path}")
    print(
        f"Size: {os.path.getsize(checkpoint_path):,} bytes ({os.path.getsize(checkpoint_path)/1024/1024:.1f} MB)"
    )

    # Check if it's a zip file (PyTorch format)
    try:
        with zipfile.ZipFile(checkpoint_path, "r") as zf:
            print(f"\n📦 Archive Contents:")
            files = zf.namelist()
            for file in files[:10]:  # Show first 10 files
                info = zf.getinfo(file)
                print(f"  📄 {file} ({info.file_size:,} bytes)")
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more files")

            # Try to read some metadata files
            for metadata_file in ["version", "data.pkl"]:
                if metadata_file in files:
                    try:
                        with zf.open(metadata_file) as f:
                            content = f.read()
                            print(f"\n📋 {metadata_file}:")
                            if metadata_file == "version":
                                print(f"  {content.decode('utf-8')}")
                            else:
                                print(f"  Size: {len(content):,} bytes")
                    except Exception as e:
                        print(f"  ⚠️  Could not read {metadata_file}: {e}")

    except zipfile.BadZipFile:
        print("⚠️  Not a zip file - might be a different PyTorch format")

    # Attempt to load with torch (will likely fail but give useful info)
    print(f"\n🔧 PyTorch Loading Attempt:")
    try:
        # Try with pickle loading disabled first
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            print(f"✅ Loaded with weights_only=True")
        except Exception:
            # Try with full loading
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            print(f"✅ Loaded with weights_only=False")

        print(f"📊 Checkpoint type: {type(checkpoint)}")

        if isinstance(checkpoint, dict):
            print(f"🔑 Top-level keys: {list(checkpoint.keys())}")

            # Analyze each key
            for key in checkpoint.keys():
                value = checkpoint[key]
                if isinstance(value, dict):
                    print(f"\n📁 {key} (dict with {len(value)} keys):")
                    # Show sample keys for dictionaries
                    sample_keys = list(value.keys())[:5]
                    for subkey in sample_keys:
                        subvalue = value[subkey]
                        if hasattr(subvalue, "shape"):
                            print(f"  🧮 {subkey}: tensor {subvalue.shape} ({subvalue.dtype})")
                        elif isinstance(subvalue, (int, float, str)):
                            print(
                                f"  📝 {subkey}: {type(subvalue).__name__} = {str(subvalue)[:50]}"
                            )
                        else:
                            print(f"  ❓ {subkey}: {type(subvalue).__name__}")
                    if len(value) > 5:
                        print(f"  ... and {len(value) - 5} more items")

                elif hasattr(value, "shape"):
                    print(f"\n🧮 {key}: tensor {value.shape} ({value.dtype})")

                elif isinstance(value, (int, float, str)):
                    print(f"\n📝 {key}: {type(value).__name__} = {str(value)[:100]}")

                else:
                    print(f"\n❓ {key}: {type(value).__name__}")
                    if hasattr(value, "__dict__"):
                        attrs = [attr for attr in dir(value) if not attr.startswith("_")][:3]
                        if attrs:
                            print(f"  Available attributes: {attrs}...")

    except Exception as e:
        print(f"❌ Loading failed: {e}")

        # Try to get more specific error information
        error_str = str(e)
        if "anemoi" in error_str.lower():
            print(f"💡 This checkpoint requires the 'anemoi' framework from ECMWF")
            print(f"   Install with: pip install anemoi-models")
        elif "module" in error_str.lower():
            missing_module = error_str.split("'")[1] if "'" in error_str else "unknown"
            print(f"💡 Missing dependency: {missing_module}")

    print(f"\n🎯 Summary:")
    print(f"  • File size: ~{os.path.getsize(checkpoint_path)/1024/1024:.0f} MB")
    print(f"  • Format: PyTorch checkpoint (.ckpt)")
    print(f"  • Source: ECMWF AIFS Single v1.0")
    print(f"  • Requirements: anemoi framework for full loading")


if __name__ == "__main__":
    checkpoint_path = "/Users/arigazzi/Documents/DeepLearning/LLM for climate/HPE-LLM4Climate/aifs-single-1.0/aifs-single-mse-1.0.ckpt"

    if os.path.exists(checkpoint_path):
        inspect_aifs_checkpoint(checkpoint_path)
    else:
        print(f"❌ Checkpoint file not found: {checkpoint_path}")
        print("Please ensure the AIFS submodule is properly initialized with Git LFS")

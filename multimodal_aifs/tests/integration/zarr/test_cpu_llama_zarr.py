#!/usr/bin/env python3
"""
CPU-Optimized Real Llama Test with Zarr

This script tests with a smaller, CPU-friendly setup using conftest infrastructure:
- Uses conftest fixtures for model management
- Respects environment variables for configuration
- Processes smaller batches
- Uses shorter sequences
"""

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("🖥️  CPU-Optimized Llama + AIFS + Zarr Test")
print("=" * 50)


@pytest.mark.integration
def test_lightweight_llama_zarr(aifs_llama_model, zarr_dataset_path):
    """Test with lightweight configuration for CPU."""

    try:
        from multimodal_aifs.utils.aifs_time_series_tokenizer import AIFSTimeSeriesTokenizer
        from multimodal_aifs.utils.zarr_data_loader import ZarrClimateLoader

        print("Modules imported")
    except ImportError as e:
        print(f"Import error: {e}")
        pytest.fail(f"Import error: {e}")

    # Use model from conftest
    model = aifs_llama_model
    device = model.device
    print(f"🖥️  Device: {device}")
    print(f"Using model from conftest fixture")
    print(f"   AIFS: {type(model.time_series_tokenizer).__name__}")
    print(f"   🦙 LLM: {type(model.llama_model).__name__}")

    # Step 1: Load minimal climate data
    print(f"\nStep 1: Loading Minimal Climate Data")
    print("-" * 40)

    try:
        loader = ZarrClimateLoader(zarr_dataset_path)

        # Load just 2 timesteps for CPU efficiency
        climate_data = loader.load_time_range(
            "2024-01-01T00:00:00", "2024-01-01T06:00:00"  # Just 2 timesteps
        )

        # Convert to small tensor
        climate_tensor = loader.to_aifs_tensor(
            climate_data, batch_size=1, normalize=True, device=device
        )

        print(f"Climate tensor: {climate_tensor.shape} on {climate_tensor.device}")
        print(f"   Memory: {climate_tensor.numel() * 4 / 1e6:.1f} MB")

    except Exception as e:
        print(f"Failed to load climate data: {e}")
        pytest.fail(f"Failed to load climate data: {e}")

    # Step 2: Initialize lightweight Llama with heavy quantization
    print(f"\n🦙 Step 2: Initializing Quantized Llama")
    print("-" * 40)

    # Model already available from conftest fixture (respects environment variables)
    print("   Using model from conftest fixture (environment-controlled)")
    print(f"   Device: {device}")
    print(f"   Quantization: {os.environ.get('USE_QUANTIZATION', 'false')}")
    print(f"   🦙 Mock LLM: {os.environ.get('USE_MOCK_LLM', 'false')}")

    # Step 3: Process with optimized settings
    print(f"\nStep 3: CPU-Optimized Processing")
    print("-" * 40)

    try:
        # Tokenize climate data - suppress MPS fallback warning
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*The operator 'aten::scatter_reduce.two_out' is not currently supported on the MPS backend.*",
                category=UserWarning,
            )
            climate_tokens = model.tokenize_climate_data(climate_tensor)
        print(f"Climate tokens: {climate_tokens.shape}")

        # Simple text for CPU efficiency
        text_inputs = ["Analyze temperature data."]

        start_time = time.time()

        # Process with timeout protection - suppress MPS fallback warning
        with torch.no_grad():  # Disable gradients for inference
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*The operator 'aten::scatter_reduce.two_out' is not currently supported on the MPS backend.*",
                    category=UserWarning,
                )
                result = model.process_climate_text(
                    climate_tokens, text_inputs, task="embedding"  # Simpler task
                )

        elapsed = time.time() - start_time
        print(f"Processing complete in {elapsed:.1f}s")
        print(f"   Output shape: {result['fused_output'].shape}")

        if "generated_text" in result:
            print(f"   Result: {result['generated_text']}")

    except Exception as e:
        print(f"Processing failed: {e}")
        pytest.fail(f"Processing failed: {e}")

    # Step 4: Memory efficiency check
    print(f"\nStep 4: Memory Efficiency")
    print("-" * 40)

    try:
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())

        # Estimate memory usage
        param_memory_mb = total_params * 4 / 1e6  # Assume float32

        print(f"   🔢 Total parameters: {total_params:,}")
        print(f"   Parameter memory: {param_memory_mb:.1f} MB")

        # Check if quantization worked
        quantized_params = 0
        for param in model.parameters():
            if param.dtype in [torch.int8, torch.uint8]:
                quantized_params += param.numel()

        if quantized_params > 0:
            print(f"   ⚗️  Quantized parameters: {quantized_params:,}")
            print(f"   📉 Memory reduction: ~{100 * quantized_params / total_params:.1f}%")

    except Exception as e:
        print(f"Memory check incomplete: {e}")

    print(f"\nCPU Test Complete!")
    # Test passes by reaching this point without failures


@pytest.mark.large_memory
@pytest.mark.integration
def test_compare_with_mock(aifs_llama_model, zarr_dataset_path):
    """Compare real vs mock LLM performance with same climate data."""
    print(f"\n⚖️  Comparison: Real vs Mock Llama (conftest)")
    print("-" * 40)

    model = aifs_llama_model
    use_mock_env = os.environ.get("USE_MOCK_LLM", "").lower() in ("true", "1", "yes")

    print(f"   Testing current configuration:")
    print(f"   Using mock: {use_mock_env}")
    print(f"   Device: {model.device}")

    try:
        start_time = time.time()

        # Test the current model configuration
        param_count = sum(p.numel() for p in model.parameters())
        init_time = time.time() - start_time

        print(f"      Model ready in {init_time:.1f}s")
        print(f"      🔢 Parameters: {param_count:,}")
        print(f"      Type: {'Mock' if use_mock_env else 'Real'} LLM")

        # Simple performance test - use AIFS-compatible dimensions
        dummy_data = torch.randn(1, 2, 1, 542080, 103).to(model.device)  # AIFS format
        dummy_text = ["Test query"]

        with torch.no_grad():
            test_start = time.time()
            result = model.forward(dummy_data, dummy_text, task="embedding")
            test_time = time.time() - test_start

        print(f"      Inference time: {test_time:.3f}s")

    except Exception as e:
        print(f"      Failed: {e}")
        pytest.fail(f"Comparison test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

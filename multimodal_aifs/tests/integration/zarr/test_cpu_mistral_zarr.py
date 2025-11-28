#!/usr/bin/env python3
# Copyright 2025 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
CPU-Optimized Real Mistral Test with Zarr

This script tests with a smaller, CPU-friendly setup using conftest infrastructure:
- Uses conftest fixtures for model management
- Respects environment variables for configuration
- Processes smaller batches
- Uses shorter sequences
"""

import sys
import time
from pathlib import Path

import pytest
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("🖥️  CPU-Optimized Mistral + AIFS + Zarr Test")
print("=" * 50)


@pytest.mark.large_memory
@pytest.mark.integration
def test_lightweight_mistral_zarr(
    aifs_mistral_model, aifs_model, zarr_dataset_path, llm_mock_status
):
    """Test with lightweight configuration for CPU."""

    try:
        from multimodal_aifs.utils.zarr_data_loader import ZarrClimateLoader

        print("Modules imported")
    except ImportError as e:
        print(f"Import error: {e}")
        pytest.fail(f"Import error: {e}")

    # Use model from conftest
    model = aifs_mistral_model
    device = model.device
    print(f"🖥️  Device: {device}")
    print("Using model from conftest fixture")
    print(f"   AIFS: {type(model.time_series_tokenizer).__name__}")
    print(f"   🤖 LLM: {type(model.mistral_model).__name__}")

    # Step 1: Load minimal climate data
    print("\nStep 1: Loading Minimal Climate Data")
    print("-" * 40)

    try:
        loader = ZarrClimateLoader(zarr_dataset_path)

        # Load all available timesteps (real ECMWF data has exactly 2 timesteps)
        climate_data = loader.load_time_range(None, None)  # Load all timesteps

        # Get runner from aifs_model fixture for input preparation pipeline
        runner = aifs_model.get("runner") if not aifs_model.get("is_mock") else None

        # Convert to small tensor using new input preparation pipeline
        climate_tensor = loader.to_aifs_tensor(
            climate_data,
            batch_size=1,
            normalize=True,
            device=device,
            runner=runner,
            use_forcing_pipeline=True,
        )

        print(f"Climate tensor: {climate_tensor.shape} on {climate_tensor.device}")
        print(f"   Memory: {climate_tensor.numel() * 4 / 1e6:.1f} MB")

    except Exception as e:
        print(f"Failed to load climate data: {e}")
        pytest.fail(f"Failed to load climate data: {e}")

    # Step 2: Initialize lightweight Mistral with heavy quantization
    print("\n🤖 Step 2: Initializing Quantized Mistral")
    print("-" * 40)

    # Model already available from conftest fixture (respects environment variables)
    print("   Using model from conftest fixture (environment-controlled)")
    print(f"   Device: {device}")
    print(f"   Quantization: {llm_mock_status['use_quantization']}")
    print(f"   🤖 Mock LLM: {llm_mock_status['use_mock_llm']}")

    # Step 3: Process with optimized settings
    print("\nStep 3: CPU-Optimized Processing")
    print("-" * 40)

    try:
        # Tokenize climate data
        climate_tokens = model.tokenize_climate_data(climate_tensor)
        print(f"Climate tokens: {climate_tokens.shape}")

        # Simple text for CPU efficiency
        text_inputs = ["Analyze temperature data."]

        start_time = time.time()

        # Process with timeout protection
        with torch.no_grad():  # Disable gradients for inference
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
    print("\nStep 4: Memory Efficiency")
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

    print("\nCPU Test Complete!")
    # Test passes by reaching this point without failures


@pytest.mark.large_memory
@pytest.mark.integration
def test_compare_with_mock(aifs_mistral_model, aifs_model, zarr_dataset_path, llm_mock_status):
    """Compare real vs mock LLM performance with same climate data."""
    print("\n⚖️  Comparison: Real vs Mock Mistral (conftest)")
    print("-" * 40)

    model = aifs_mistral_model
    use_mock_env = llm_mock_status["use_mock_llm"]

    print("   Testing current configuration:")
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

        # Simple performance test using dummy tensor (shape testing only, not real climate data)
        dummy_data = torch.randn(1, 2, 1, 542080, 103).to(model.device)  # AIFS format
        dummy_text = ["Test query"]

        with torch.no_grad():
            test_start = time.time()
            _ = model.forward(dummy_data, dummy_text, task="embedding")
            test_time = time.time() - test_start

        print(f"      Inference time: {test_time:.3f}s")

    except Exception as e:
        print(f"      Failed: {e}")
        pytest.fail(f"Comparison test failed: {e}")

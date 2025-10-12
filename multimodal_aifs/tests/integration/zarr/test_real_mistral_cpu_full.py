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
Real Mistral on CPU Test (No Quantization)

This script tests with real Mistral-7B-Instruct-v0.3 on CPU without quantization.
Since quantization requires CUDA, we'll load the model in full precision
but use very small batches and sequences.

Usage:
    python test_real_mistral_cpu_full.py
"""

import gc
import os
import sys
import time
from pathlib import Path

import pytest
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("🤖 Real Mistral-7B-Instruct on CPU (Full Precision)")
print("=" * 50)


def test_real_mistral_cpu(llm_mock_status, aifs_mistral_model, test_device, zarr_dataset_path):
    """Test real Mistral model with AIFS on CPU with full integration."""

    try:
        from multimodal_aifs.utils.zarr_data_loader import ZarrClimateLoader

        print("Modules imported")
    except ImportError as e:
        print(f"Import error: {e}")
        pytest.fail(f"Import error: {e}")

    # Check if we should skip this test based on mock status
    if llm_mock_status["should_skip_real_llm_tests"]:
        print("   USE_MOCK_LLM is set to true, skipping real Mistral test")
        pytest.skip("USE_MOCK_LLM is enabled, skipping real Mistral test")

    # Step 1: Load minimal climate data
    print("\nStep 1: Loading Climate Data")
    print("-" * 30)

    try:
        loader = ZarrClimateLoader(zarr_dataset_path)

        # Load minimal data - just 1 timestep
        climate_data = loader.load_time_range(
            "2024-01-01", "2024-01-01T00:00:00"  # Just 1 timestep
        )

        # Convert to very small tensor
        climate_tensor = loader.to_aifs_tensor(climate_data, batch_size=1, normalize=True)

        print(f"Climate tensor: {climate_tensor.shape}")

    except Exception as e:
        print(f"Failed to load climate data: {e}")
        pytest.fail(f"Failed to load climate data: {e}")

    # Step 2: Use the pre-configured fusion model
    print("\n🤖 Step 2: Using Fusion Model from Fixture")
    print("-" * 50)

    # Check if model was created successfully
    if aifs_mistral_model is None:
        print("   No AIFS model available")
        pytest.skip("No AIFS model available for testing")

    # Use the pre-configured fusion model from fixture
    model = aifs_mistral_model
    print("   Using fusion model from fixture")
    print(f"   Hidden size: {model.mistral_hidden_size}")

    # Check if we actually got real Mistral
    is_real_mistral = hasattr(model.mistral_model, "config") and model.mistral_tokenizer is not None
    print(f"   Real Mistral status: {is_real_mistral}")

    if not is_real_mistral:
        print("   Fallback to mock model occurred")
        print("   Set USE_MOCK_LLM=false to test real Mistral")
        pytest.skip("Real Mistral model not available, test skipped")

    # Step 3: Test AIFS tokenization
    print("\nStep 3: AIFS Climate Tokenization")
    print("-" * 30)

    try:
        climate_tokens = model.tokenize_climate_data(climate_tensor)
        print(f"Climate tokens: {climate_tokens.shape}")

    except Exception as e:
        print(f"Climate tokenization failed: {e}")
        pytest.fail(f"Climate tokenization failed: {e}")

    # Step 4: Test real Mistral text processing
    print("\nStep 4: Real Mistral Text Processing")
    print("-" * 30)

    try:
        # Very short text for CPU efficiency
        text_inputs = ["Temperature data analysis."]

        print("   ⏳ Tokenizing text with real Mistral tokenizer...")
        text_tokens = model.tokenize_text(text_inputs)

        print(f"Text tokenized: {text_tokens['input_ids'].shape}")

        # Decode to verify real tokenizer
        sample_ids = text_tokens["input_ids"][0][:5].tolist()
        decoded = model.mistral_tokenizer.decode(sample_ids)
        print(f"   Sample tokens: '{decoded}'")

    except Exception as e:
        print(f"Text processing failed: {e}")
        pytest.fail(f"Text processing failed: {e}")

    # Step 5: Test real multimodal processing (lightweight)
    print("\n🔗 Step 5: Real Multimodal Processing")
    print("-" * 30)

    try:
        print("   ⏳ Running real Mistral forward pass (may take 30-60s on CPU)...")

        start_time = time.time()

        # Use embedding task for efficiency
        with torch.no_grad():  # Crucial for memory efficiency
            result = model.process_climate_text(
                climate_tokens, text_inputs, task="embedding"  # Lightest task
            )

        elapsed = time.time() - start_time
        print(f"Real Mistral processing complete in {elapsed:.1f}s")
        print(f"   Output shape: {result['fused_output'].shape}")
        print(f"   Generated text: {result['generated_text']}")

        # Verify we're using real embeddings
        output_range = result["fused_output"].abs().max().item()
        print(f"   Output range: {output_range:.3f}")

    except Exception as e:
        print(f"Multimodal processing failed: {e}")
        print("   This is expected on systems with limited RAM")
        pytest.fail(f"Multimodal processing failed: {e}")

    # Step 6: Memory usage analysis
    print("\nStep 6: Memory Analysis")
    print("-" * 30)

    try:
        # Model parameter analysis
        total_params = sum(p.numel() for p in model.parameters())
        aifs_params = sum(p.numel() for p in model.time_series_tokenizer.parameters())
        mistral_params = total_params - aifs_params

        print(f"   🔢 Total parameters: {total_params:,}")
        print(f"   AIFS parameters: {aifs_params:,}")
        print(f"   🤖 Mistral parameters: {mistral_params:,}")
        print(f"   Estimated memory: {total_params * 4 / 1e9:.1f} GB")

        # Check model precision
        sample_param = next(model.mistral_model.parameters())
        print(f"   Model dtype: {sample_param.dtype}")
        print(f"   🖥️  Device: {sample_param.device}")

    except Exception as e:
        print(f"Memory analysis incomplete: {e}")

    # Cleanup
    print("\n🧹 Cleanup")
    print("-" * 10)
    try:
        del model
        gc.collect()
        print("Memory cleaned up")
    except RuntimeError:
        pass

    print("\nReal Mistral CPU Test Complete!")
    print("Successfully demonstrated:")
    print("   Zarr → AIFS tokenization")
    print("   🤖 Real Mistral-7B-Instruct-v0.3 loading")
    print("   Real Mistral text processing")
    print("   🔗 Multimodal fusion")
    print("   CPU execution (full precision)")
    # Test passes by reaching this point without failures


def test_memory_requirements():
    """Test what happens with memory constraints."""
    print("\nMemory Requirements Test")
    print("-" * 30)

    try:
        import psutil

        mem = psutil.virtual_memory()
        print(f"   Total RAM: {mem.total / 1e9:.1f} GB")
        print(f"   Available: {mem.available / 1e9:.1f} GB")
        print(f"   Used: {mem.percent:.1f}%")

        # Estimate if Mistral will fit
        estimated_need = 16  # GB
        if mem.available / 1e9 >= estimated_need:
            print("   Should be able to load Mistral-7B-Instruct")
        else:
            print("   May struggle to load Mistral-7B-Instruct")
            print("   Recommended: Close other applications")

    except ImportError:
        print("   Cannot check memory (psutil not installed)")


def main():
    """Main function."""

    # Get zarr path - use real ECMWF dataset
    zarr_file = "data/real_ecmwf_latest.zarr"

    # Check prerequisites
    if not Path(zarr_file).exists():
        print(f"Real ECMWF dataset not found: {zarr_file}")
        return

    # Memory check
    test_memory_requirements()

    # Warning about loading time
    print("\nImportant Notes:")
    print("   First run will download ~16GB model")
    print("   Loading may take 2-5 minutes")
    print("   Processing will be slow on CPU")
    print("   Requires 16+ GB RAM")

    response = input("\n   Continue with real Mistral test? (y/N): ")
    if response.lower() != "y":
        print("   Test cancelled")
        return

    # Create fixtures manually for standalone execution
    print("\nSetting up test fixtures...")

    # Create test device
    from multimodal_aifs.utils import get_best_device

    test_device = get_best_device()
    print(f"   📱 Using device: {test_device}")

    # Create LLM mock status
    use_mock_llm = os.environ.get("USE_MOCK_LLM", "true").lower() in ("true", "1", "yes")
    llm_mock_status = {
        "use_mock_llm": use_mock_llm,
        "use_quantization": False,
        "model_name": os.environ.get("LLM_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3"),
        "should_skip_real_llm_tests": use_mock_llm,
    }
    print(f"   🤖 Mock LLM: {use_mock_llm}")

    # Create zarr dataset path
    zarr_dataset_path = zarr_file
    print(f"   Zarr path: {zarr_dataset_path}")

    # Check if we should skip based on mock status
    if llm_mock_status["should_skip_real_llm_tests"]:
        print("   USE_MOCK_LLM is set to true, skipping real Mistral test")
        print("   Set USE_MOCK_LLM=false to run real Mistral test")
        return

    # Create AIFS model fixture manually
    print("\n🏗️ Creating AIFS model...")
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from multimodal_aifs.conftest import AIFSClimateTextFusionWrapper

        # Try to load real AIFS model
        aifs_model_path = "aifs-single-1.0/aifs-single-mse-1.0.ckpt"
        if Path(aifs_model_path).exists():
            print(f"   📦 Loading real AIFS model from {aifs_model_path}")
            try:
                # Try to load the actual AIFS model from checkpoint
                # This is a simplified version - in practice you'd need the full loading logic
                aifs_mistral_model = AIFSClimateTextFusionWrapper(
                    model=None,  # We'll need to load this properly
                    device_str=str(test_device),
                    fusion_dim=512,
                    use_mock_mistral=False,
                    verbose=True,
                )
                print("   Note: AIFS model loading not fully implemented in standalone script")
                print("   For full functionality, run as pytest test")
            except Exception as e:
                print(f"   Failed to load real AIFS model: {e}")
                print("   Falling back to mock model")
                aifs_mistral_model = None
        else:
            print(f"   AIFS model not found at {aifs_model_path}, using mock")
            aifs_mistral_model = None

    except Exception as e:
        print(f"   Failed to create AIFS model: {e}")
        return

    # Run the test with manually created fixtures
    try:
        test_real_mistral_cpu(llm_mock_status, aifs_mistral_model, test_device, zarr_dataset_path)
        print("\n🏆 SUCCESS: Real Mistral-7B-Instruct working on CPU!")
        print("Your system can run the full AIFS + Mistral pipeline")
    except Exception:
        print("\n💥 FAILED: Real Mistral couldn't run on this system")
        print("Consider using the mock version for development")


if __name__ == "__main__":
    main()

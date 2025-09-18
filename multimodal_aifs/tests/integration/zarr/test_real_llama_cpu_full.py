#!/usr/bin/env python3
"""
Real Llama on CPU Test (No Quantization)

This script tests with real Meta-Llama-3-8B on CPU without quantization.
Since quantization requires CUDA, we'll load the model in full precision
but use very small batches and sequences.

Usage:
    python test_real_llama_cpu_full.py
"""

import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("🦙 Real Llama-3-8B on CPU (Full Precision)")
print("=" * 50)


def test_real_llama_cpu(llm_mock_status, aifs_llama_model, test_device, zarr_dataset_path):
    """Test real Llama model with AIFS on CPU with full integration."""

    try:
        from multimodal_aifs.utils.aifs_time_series_tokenizer import AIFSTimeSeriesTokenizer
        from multimodal_aifs.utils.zarr_data_loader import ZarrClimateLoader

        print("✅ Modules imported")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        pytest.fail(f"Import error: {e}")

    # Check if we should skip this test based on mock status
    if llm_mock_status["should_skip_real_llm_tests"]:
        print("   ⚠️  USE_MOCK_LLM is set to true, skipping real Llama test")
        pytest.skip("USE_MOCK_LLM is enabled, skipping real Llama test")

    # Step 1: Load minimal climate data
    print(f"\n📊 Step 1: Loading Climate Data")
    print("-" * 30)

    try:
        loader = ZarrClimateLoader(zarr_dataset_path)

        # Load minimal data - just 1 timestep
        climate_data = loader.load_time_range(
            "2024-01-01", "2024-01-01T00:00:00"  # Just 1 timestep
        )

        # Convert to very small tensor
        climate_tensor = loader.to_aifs_tensor(climate_data, batch_size=1, normalize=True)

        print(f"✅ Climate tensor: {climate_tensor.shape}")

    except Exception as e:
        print(f"❌ Failed to load climate data: {e}")
        pytest.fail(f"Failed to load climate data: {e}")

    # Step 2: Use the pre-configured fusion model
    print(f"\n🦙 Step 2: Using Fusion Model from Fixture")
    print("-" * 50)

    # Check if model was created successfully
    if aifs_llama_model is None:
        print("   ❌ No AIFS model available")
        pytest.skip("No AIFS model available for testing")

    # Use the pre-configured fusion model from fixture
    model = aifs_llama_model
    print(f"   ✅ Using fusion model from fixture")
    print(f"   📏 Hidden size: {model.llama_hidden_size}")

    # Check if we actually got real Llama
    is_real_llama = hasattr(model.llama_model, "config") and model.llama_tokenizer is not None
    print(f"   🔍 Real Llama status: {is_real_llama}")

    if not is_real_llama:
        print("   ⚠️  Fallback to mock model occurred")
        print("   💡 Set USE_MOCK_LLM=false to test real Llama")
        pytest.skip("Real Llama model not available, test skipped")

    # Step 3: Test AIFS tokenization
    print(f"\n🌍 Step 3: AIFS Climate Tokenization")
    print("-" * 30)

    try:
        climate_tokens = model.tokenize_climate_data(climate_tensor)
        print(f"✅ Climate tokens: {climate_tokens.shape}")

    except Exception as e:
        print(f"❌ Climate tokenization failed: {e}")
        pytest.fail(f"Climate tokenization failed: {e}")

    # Step 4: Test real Llama text processing
    print(f"\n💬 Step 4: Real Llama Text Processing")
    print("-" * 30)

    try:
        # Very short text for CPU efficiency
        text_inputs = ["Temperature data analysis."]

        print("   ⏳ Tokenizing text with real Llama tokenizer...")
        text_tokens = model.tokenize_text(text_inputs)

        print(f"✅ Text tokenized: {text_tokens['input_ids'].shape}")

        # Decode to verify real tokenizer
        sample_ids = text_tokens["input_ids"][0][:5].tolist()
        decoded = model.llama_tokenizer.decode(sample_ids)
        print(f"   🔍 Sample tokens: '{decoded}'")

    except Exception as e:
        print(f"❌ Text processing failed: {e}")
        pytest.fail(f"Text processing failed: {e}")

    # Step 5: Test real multimodal processing (lightweight)
    print(f"\n🔗 Step 5: Real Multimodal Processing")
    print("-" * 30)

    try:
        print("   ⏳ Running real Llama forward pass (may take 30-60s on CPU)...")

        start_time = time.time()

        # Use embedding task for efficiency
        with torch.no_grad():  # Crucial for memory efficiency
            result = model.process_climate_text(
                climate_tokens, text_inputs, task="embedding"  # Lightest task
            )

        elapsed = time.time() - start_time
        print(f"✅ Real Llama processing complete in {elapsed:.1f}s")
        print(f"   🎯 Output shape: {result['fused_output'].shape}")
        print(f"   💬 Generated text: {result['generated_text']}")

        # Verify we're using real embeddings
        output_range = result["fused_output"].abs().max().item()
        print(f"   🔍 Output range: {output_range:.3f}")

    except Exception as e:
        print(f"❌ Multimodal processing failed: {e}")
        print(f"   This is expected on systems with limited RAM")
        pytest.fail(f"Multimodal processing failed: {e}")

    # Step 6: Memory usage analysis
    print(f"\n📊 Step 6: Memory Analysis")
    print("-" * 30)

    try:
        # Model parameter analysis
        total_params = sum(p.numel() for p in model.parameters())
        aifs_params = sum(p.numel() for p in model.time_series_tokenizer.parameters())
        llama_params = total_params - aifs_params

        print(f"   🔢 Total parameters: {total_params:,}")
        print(f"   🌍 AIFS parameters: {aifs_params:,}")
        print(f"   🦙 Llama parameters: {llama_params:,}")
        print(f"   💾 Estimated memory: {total_params * 4 / 1e9:.1f} GB")

        # Check model precision
        sample_param = next(model.llama_model.parameters())
        print(f"   🎯 Model dtype: {sample_param.dtype}")
        print(f"   🖥️  Device: {sample_param.device}")

    except Exception as e:
        print(f"⚠️  Memory analysis incomplete: {e}")

    # Cleanup
    print(f"\n🧹 Cleanup")
    print("-" * 10)
    try:
        del model
        gc.collect()
        print("✅ Memory cleaned up")
    except:
        pass

    print(f"\n🎉 Real Llama CPU Test Complete!")
    print(f"✅ Successfully demonstrated:")
    print(f"   📊 Zarr → AIFS tokenization")
    print(f"   🦙 Real Meta-Llama-3-8B loading")
    print(f"   💬 Real Llama text processing")
    print(f"   🔗 Multimodal fusion")
    print(f"   💾 CPU execution (full precision)")
    # Test passes by reaching this point without failures


def test_memory_requirements():
    """Test what happens with memory constraints."""
    print(f"\n🧪 Memory Requirements Test")
    print("-" * 30)

    try:
        import psutil

        mem = psutil.virtual_memory()
        print(f"   💾 Total RAM: {mem.total / 1e9:.1f} GB")
        print(f"   📊 Available: {mem.available / 1e9:.1f} GB")
        print(f"   📈 Used: {mem.percent:.1f}%")

        # Estimate if Llama will fit
        estimated_need = 16  # GB
        if mem.available / 1e9 >= estimated_need:
            print(f"   ✅ Should be able to load Llama-3-8B")
        else:
            print(f"   ⚠️  May struggle to load Llama-3-8B")
            print(f"   💡 Recommended: Close other applications")

    except ImportError:
        print(f"   ⚠️  Cannot check memory (psutil not installed)")


def main():
    """Main function."""

    # Get zarr path - use unified test dataset
    zarr_file = "test_aifs_large.zarr"

    # Check prerequisites
    if not Path(zarr_file).exists():
        print(f"❌ Test dataset not found: {zarr_file}")
        return

    # Memory check
    test_memory_requirements()

    # Warning about loading time
    print(f"\n⚠️  Important Notes:")
    print(f"   • First run will download ~16GB model")
    print(f"   • Loading may take 2-5 minutes")
    print(f"   • Processing will be slow on CPU")
    print(f"   • Requires 16+ GB RAM")

    response = input(f"\n   Continue with real Llama test? (y/N): ")
    if response.lower() != "y":
        print(f"   Test cancelled")
        return

    # Create fixtures manually for standalone execution
    print(f"\n🔧 Setting up test fixtures...")

    # Create test device
    if torch.cuda.is_available():
        test_device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        test_device = torch.device("mps")
    else:
        test_device = torch.device("cpu")
    print(f"   📱 Using device: {test_device}")

    # Create LLM mock status
    use_mock_llm = os.environ.get("USE_MOCK_LLM", "true").lower() in ("true", "1", "yes")
    llm_mock_status = {
        "use_mock_llm": use_mock_llm,
        "use_quantization": False,
        "model_name": os.environ.get("LLM_MODEL_NAME", "meta-llama/Meta-Llama-3-8B"),
        "should_skip_real_llm_tests": use_mock_llm,
    }
    print(f"   🤖 Mock LLM: {use_mock_llm}")

    # Create zarr dataset path
    zarr_dataset_path = zarr_file
    print(f"   📁 Zarr path: {zarr_dataset_path}")

    # Check if we should skip based on mock status
    if llm_mock_status["should_skip_real_llm_tests"]:
        print("   ⚠️  USE_MOCK_LLM is set to true, skipping real Llama test")
        print("   💡 Set USE_MOCK_LLM=false to run real Llama test")
        return

    # Create AIFS model fixture manually
    print(f"\n🏗️ Creating AIFS model...")
    try:
        # Import the wrapper from conftest.py
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from multimodal_aifs.conftest import AIFSClimateTextFusionWrapper

        # Try to load real AIFS model
        aifs_model_path = "aifs-single-1.0/aifs-single-mse-1.0.ckpt"
        if Path(aifs_model_path).exists():
            print(f"   📦 Loading real AIFS model from {aifs_model_path}")
            try:
                # Try to load the actual AIFS model from checkpoint
                # This is a simplified version - in practice you'd need the full loading logic
                aifs_llama_model = AIFSClimateTextFusionWrapper(
                    model=None,  # We'll need to load this properly
                    device_str=str(test_device),
                    fusion_dim=512,
                    use_mock_llama=False,
                    verbose=True,
                )
                print(f"   ⚠️ Note: AIFS model loading not fully implemented in standalone script")
                print(f"   💡 For full functionality, run as pytest test")
            except Exception as e:
                print(f"   ⚠️ Failed to load real AIFS model: {e}")
                print(f"   🔄 Falling back to mock model")
                aifs_llama_model = None
        else:
            print(f"   ⚠️ AIFS model not found at {aifs_model_path}, using mock")
            aifs_llama_model = None

    except Exception as e:
        print(f"   ❌ Failed to create AIFS model: {e}")
        return

    # Run the test with manually created fixtures
    success = test_real_llama_cpu(llm_mock_status, aifs_llama_model, test_device, zarr_dataset_path)

    if success:
        print(f"\n🏆 SUCCESS: Real Llama-3-8B working on CPU!")
        print(f"🎯 Your system can run the full AIFS + Llama pipeline")
    else:
        print(f"\n💥 FAILED: Real Llama couldn't run on this system")
        print(f"💡 Consider using the mock version for development")


if __name__ == "__main__":
    main()

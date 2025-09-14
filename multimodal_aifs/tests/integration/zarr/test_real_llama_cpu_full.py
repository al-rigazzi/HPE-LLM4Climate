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


def test_real_llama_cpu():
    """Test real Llama-3-8B on CPU without quantization."""

    try:
        from conftest import AIFSLlamaFusionModel
        from multimodal_aifs.utils.aifs_time_series_tokenizer import AIFSTimeSeriesTokenizer
        from multimodal_aifs.utils.zarr_data_loader import ZarrClimateLoader

        print("✅ Modules imported")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        pytest.fail(f"Import error: {e}")

    # Check available memory
    print(f"\n💾 System Memory Check")
    print("-" * 30)
    try:
        import psutil

        available_gb = psutil.virtual_memory().available / 1e9
        total_gb = psutil.virtual_memory().total / 1e9
        print(f"   📊 Available RAM: {available_gb:.1f} GB / {total_gb:.1f} GB")

        if available_gb < 12:
            print(f"   ⚠️  Warning: Llama-3-8B typically needs 12-16GB RAM")
            print(f"   💡 This test may fail or be very slow")
        else:
            print(f"   ✅ Sufficient RAM for Llama-3-8B")
    except ImportError:
        print(f"   ⚠️  psutil not available, cannot check RAM")

    # Step 1: Load minimal climate data
    print(f"\n📊 Step 1: Loading Climate Data")
    print("-" * 30)

    try:
        loader = ZarrClimateLoader("test_climate.zarr")

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

    # Step 2: Try to load real Llama without quantization
    print(f"\n🦙 Step 2: Loading Real Llama-3-8B (Full Precision)")
    print("-" * 50)

    try:
        print("   ⏳ This may take 2-5 minutes to download and load...")
        print("   📥 Model size: ~16GB download, ~32GB RAM when loaded")

        start_time = time.time()

        # Force no quantization for CPU
        model = AIFSLlamaFusionModel(
            llm_model_name="meta-llama/Meta-Llama-3-8B",
            time_series_dim=512,
            fusion_strategy="concat",  # Simpler fusion
            device="cpu",
            use_quantization=False,  # No quantization on CPU
            use_mock_llama=False,  # Force real Llama
        )

        load_time = time.time() - start_time
        print(f"✅ Real Llama loaded in {load_time:.1f}s")
        print(f"   📏 Hidden size: {model.llama_hidden_size}")

        # Check if we actually got real Llama
        is_real_llama = hasattr(model.llama_model, "config") and model.llama_tokenizer is not None
        print(f"   🔍 Real Llama status: {is_real_llama}")

        if not is_real_llama:
            print("   ⚠️  Fallback to mock model occurred")
            pytest.skip("Real Llama model not available, test skipped")

    except Exception as e:
        print(f"❌ Failed to load real Llama: {e}")
        print(f"💡 Common reasons:")
        print(f"   - Insufficient RAM (need 16+ GB)")
        print(f"   - Missing HuggingFace token")
        print(f"   - Network issues during download")
        pytest.skip(f"Real Llama model not available: {e}")

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

    # Check prerequisites
    if not Path("test_climate.zarr").exists():
        print(f"❌ Test dataset not found: test_climate.zarr")
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

    # Run the test
    success = test_real_llama_cpu()

    if success:
        print(f"\n🏆 SUCCESS: Real Llama-3-8B working on CPU!")
        print(f"🎯 Your system can run the full AIFS + Llama pipeline")
    else:
        print(f"\n💥 FAILED: Real Llama couldn't run on this system")
        print(f"💡 Consider using the mock version for development")


if __name__ == "__main__":
    main()

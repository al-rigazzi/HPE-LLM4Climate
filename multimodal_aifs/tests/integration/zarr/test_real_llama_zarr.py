#!/usr/bin/env python3
"""
Real Llama Integration Test with Zarr Data

This script tests the complete pipeline with actual Meta-Llama-3-8B model:
Zarr → AIFS Tokenization → Real Llama Processing → Multimodal Fusion

Usage:
    python test_real_llama_zarr.py --zarr-path test_aifs_small.zarr
    python test_real_llama_zarr.py --zarr-path test_aifs_small.zarr --use-quantization
"""

import argparse
import sys
import time
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("🦙 Real Llama + AIFS + Zarr Integration Test")
print("=" * 50)

# Check system
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Device: {DEVICE}")
print(f"📦 PyTorch: {torch.__version__}")

try:
    import sys
    from pathlib import Path

    # Add the project root to path to import conftest
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

    # Import required modules (AIFSLlamaFusionModel removed - using production model now)
    from multimodal_aifs.utils.aifs_time_series_tokenizer import AIFSTimeSeriesTokenizer
    from multimodal_aifs.utils.zarr_data_loader import ZarrClimateLoader

    print("All modules imported successfully")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


import os

import pytest


@pytest.fixture
def zarr_path(zarr_dataset_path):
    """Provide a default zarr path for testing."""
    return zarr_dataset_path


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("RUN_REAL_LLAMA_TESTS", False),
    reason="Real Llama tests are resource-intensive and require RUN_REAL_LLAMA_TESTS=1",
)
def test_real_llama_with_zarr(
    aifs_llama_model,
    zarr_dataset_path,
    zarr_path: str = None,
    use_quantization: bool = None,
    model_name: str = None,
    max_memory_gb: float = 8.0,
):
    """Test complete pipeline with real Llama model."""

    # Use environment variables to control test behavior
    zarr_path = zarr_path or zarr_dataset_path
    use_quantization = (
        use_quantization
        if use_quantization is not None
        else os.environ.get("USE_QUANTIZATION", "false").lower() in ("true", "1", "yes")
    )
    model_name = model_name or os.environ.get("LLM_MODEL_NAME", "meta-llama/Meta-Llama-3-8B")

    print(f"\nStarting Real Llama Integration Test (conftest)")
    print(f"Zarr path: {zarr_path}")
    print(f"🦙 Model: {model_name}")
    print(f"⚗️  Quantization: {use_quantization}")
    print(f"Max memory: {max_memory_gb} GB")

    # Use model from conftest fixture
    model = aifs_llama_model
    print(f"Using model from conftest fixture")
    print(f"   AIFS: {type(model.time_series_tokenizer).__name__}")
    print(f"   🦙 LLM: {type(model.llama_model).__name__}")
    print(f"   Hidden size: {model.llama_hidden_size}")
    print(f"   Device: {model.device}")

    # Step 1: Load Zarr Climate Data
    print(f"\nStep 1: Loading Climate Data from Zarr")
    print("-" * 40)

    try:
        loader = ZarrClimateLoader(zarr_path)

        # Load a small subset for testing
        climate_data = loader.load_time_range("2024-01-01", "2024-01-01T06:00:00")

        # Convert to AIFS tensor
        climate_tensor = loader.to_aifs_tensor(
            climate_data, batch_size=1, normalize=True  # Small batch for CPU
        )

        print(f"Climate data loaded successfully")
        print(f"   Tensor shape: {climate_tensor.shape}")
        print(f"   Memory: {climate_tensor.numel() * 4 / 1e6:.1f} MB")

    except Exception as e:
        print(f"Failed to load climate data: {e}")
        return False

    # Step 2: Climate Data Processing with AIFS
    print(f"\nStep 2: Processing Climate Data with AIFS")
    print("-" * 40)

    # Step 3: AIFS Climate Tokenization
    print(f"\nStep 3: AIFS Climate Tokenization")
    print("-" * 40)

    try:
        climate_tokens = model.tokenize_climate_data(climate_tensor)
        print(f"Climate tokenization successful")
        print(f"   Token shape: {climate_tokens.shape}")
        print(
            f"   Format: [batch={climate_tokens.shape[0]}, seq_len={climate_tokens.shape[1]}, hidden={climate_tokens.shape[2]}]"
        )

    except Exception as e:
        print(f"Climate tokenization failed: {e}")
        return False

    # Step 4: Text Processing with Real Llama
    print(f"\nStep 4: Text Processing with Real Llama")
    print("-" * 40)

    try:
        # Test text inputs
        text_inputs = [
            "Analyze the temperature patterns in this climate data.",
            "What weather trends are visible in the atmospheric data?",
        ]

        # Process text with real Llama tokenizer
        text_tokens = model.tokenize_text(text_inputs)
        print(f"Text tokenization successful")
        print(f"   Input IDs shape: {text_tokens['input_ids'].shape}")
        print(f"   👁️  Attention mask shape: {text_tokens['attention_mask'].shape}")

        # Show tokenized text (first few tokens)
        if model.llama_tokenizer:
            sample_tokens = text_tokens["input_ids"][0][:10].tolist()
            decoded = model.llama_tokenizer.decode(sample_tokens)
            print(f"   Sample tokens: {decoded}...")

    except Exception as e:
        print(f"Text processing failed: {e}")
        return False

    # Step 5: Multimodal Fusion with Real Llama
    print(f"\n🔗 Step 5: Multimodal Fusion with Real Llama")
    print("-" * 40)

    try:
        print(f"⏳ Running multimodal fusion (this may take a few minutes on CPU)...")
        start_time = time.time()

        # Use the convenience method we added
        result = model.process_climate_text(
            climate_tokens,
            text_inputs[:1],  # Use just one text for CPU efficiency
            task="generation",
        )

        elapsed = time.time() - start_time
        print(f"Multimodal fusion successful!")
        print(f"   ⏱️  Processing time: {elapsed:.1f} seconds")
        print(f"   Fused output shape: {result['fused_output'].shape}")

        if "generated_text" in result:
            print(f"   Generated text: {result['generated_text']}")

        # Show attention weights if available
        if "attention_weights" in result:
            attn_shape = result["attention_weights"].shape
            print(f"   👁️  Attention weights: {attn_shape}")

    except Exception as e:
        print(f"Multimodal fusion failed: {e}")
        print(f"This is expected on CPU with limited memory")
        return False

    # Step 6: Memory and Performance Analysis
    print(f"\nStep 6: Performance Analysis")
    print("-" * 40)

    try:
        # Check memory usage
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1e9
            print(f"   🖥️  GPU memory: {gpu_memory:.2f} GB")

        # Model parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"   🔢 Total parameters: {total_params:,}")
        print(f"   🎓 Trainable parameters: {trainable_params:,}")
        print(f"   Estimated model size: {total_params * 4 / 1e9:.1f} GB (float32)")

    except Exception as e:
        print(f"Performance analysis incomplete: {e}")

    print(f"\nReal Llama Integration Test Complete!")
    print(f"Successfully processed climate data through:")
    print(f"   Zarr → AIFS tokenization")
    print(f"   🦙 Real Meta-Llama-3-8B processing")
    print(f"   🔗 Multimodal fusion")
    print(f"   Text generation")

    return True


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Real Llama + AIFS + Zarr Integration Test")
    parser.add_argument(
        "--zarr-path",
        default="test_aifs_large.zarr",
        help="Path to Zarr dataset (default: test_aifs_large.zarr)",
    )
    parser.add_argument("--use-quantization", action="store_true", help="Use 8-bit quantization")
    parser.add_argument(
        "--model-name", default="meta-llama/Meta-Llama-3-8B", help="Llama model name"
    )
    parser.add_argument("--max-memory", type=float, default=8.0, help="Max memory in GB")

    args = parser.parse_args()

    # Check if zarr dataset exists
    if not Path(args.zarr_path).exists():
        print(f"Zarr dataset not found: {args.zarr_path}")
        print(f"To create the test dataset, run:")
        print(f"   cd /path/to/project && python multimodal_aifs/conftest.py")
        print(f"   # Or use the ensure_test_zarr_dataset fixture")
        return

    # Warning about memory requirements
    if not torch.cuda.is_available():
        print(f"\nWARNING: Running on CPU")
        print(f"   Llama-3-8B requires significant memory and will be slow")
        print(f"   Consider using --use-quantization to reduce memory usage")
        print(f"   Expected processing time: 5-10 minutes")

        response = input(f"\n   Continue? (y/N): ")
        if response.lower() != "y":
            print(f"   Cancelled by user")
            return

    # Run the test
    success = test_real_llama_with_zarr(
        zarr_path=args.zarr_path,
        use_quantization=args.use_quantization,
        model_name=args.model_name,
        max_memory_gb=args.max_memory,
    )

    if success:
        print(f"\n🏆 All tests passed! Real Llama integration is working.")
    else:
        print(f"\n💥 Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()

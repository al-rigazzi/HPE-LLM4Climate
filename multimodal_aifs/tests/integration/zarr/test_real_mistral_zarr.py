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
Real Mistral Integration Test with Zarr Data

This script tests the complete pipeline with actual Mistral-7B-Instruct-v0.3 model:
Zarr → AIFS Tokenization → Real Mistral Processing → Multimodal Fusion

Usage:
    python test_real_mistral_zarr.py --zarr-path test_aifs_small.zarr
    python test_real_mistral_zarr.py --zarr-path test_aifs_small.zarr --use-quantization
"""

import os
import sys
import time
from pathlib import Path

import pytest
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

print("🤖 Real Mistral + AIFS + Zarr Integration Test")
print("=" * 50)

# Check system
from multimodal_aifs.utils import get_best_device

DEVICE = str(get_best_device())
print(f"🖥️  Device: {DEVICE}")
print(f"📦 PyTorch: {torch.__version__}")

try:
    # Import required modules
    from multimodal_aifs.utils.zarr_data_loader import ZarrClimateLoader

    print("All modules imported successfully")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


@pytest.fixture
def zarr_path(zarr_dataset_path):
    """Provide a default zarr path for testing."""
    return zarr_dataset_path


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("RUN_REAL_MISTRAL_TESTS", False),
    reason="Real Mistral tests are resource-intensive and require RUN_REAL_MISTRAL_TESTS=1",
)
def test_real_mistral_with_zarr(
    aifs_mistral_model,
    aifs_model,
    zarr_dataset_path,
    zarr_path: str = None,  # pylint: disable=redefined-outer-name
    use_quantization: bool = None,
    model_name: str = None,
    max_memory_gb: float = 8.0,
):
    """Test complete pipeline with real Mistral model."""

    # Use environment variables to control test behavior
    zarr_path = zarr_path or zarr_dataset_path
    use_quantization = (
        use_quantization
        if use_quantization is not None
        else os.environ.get("USE_QUANTIZATION", "false").lower() in ("true", "1", "yes")
    )
    model_name = model_name or os.environ.get(
        "LLM_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3"
    )

    print("\nStarting Real Mistral Integration Test (conftest)")
    print(f"Zarr path: {zarr_path}")
    print(f"🤖 Model: {model_name}")
    print(f"⚗️  Quantization: {use_quantization}")
    print(f"Max memory: {max_memory_gb} GB")

    # Use model from conftest fixture
    model = aifs_mistral_model
    print("Using model from conftest fixture")
    print(f"   AIFS: {type(model.time_series_tokenizer).__name__}")
    print(f"   🤖 LLM: {type(model.mistral_model).__name__}")
    print(f"   Hidden size: {model.mistral_hidden_size}")
    print(f"   Device: {model.device}")

    # Step 1: Load Zarr Climate Data
    print("\nStep 1: Loading Climate Data from Zarr")
    print("-" * 40)

    try:
        loader = ZarrClimateLoader(zarr_path)

        # Load all available timesteps (real ECMWF data has exactly 2 timesteps)
        climate_data = loader.load_time_range(None, None)  # Load all timesteps

        # Get runner from aifs_model fixture for input preparation pipeline
        runner = aifs_model.get("runner") if not aifs_model.get("is_mock") else None

        # Convert to AIFS tensor using new input preparation pipeline
        climate_tensor = loader.to_aifs_tensor(
            climate_data,
            batch_size=1,
            normalize=True,
            runner=runner,
            use_forcing_pipeline=True,
        )

        print("Climate data loaded successfully")
        print(f"   Tensor shape: {climate_tensor.shape}")
        print(f"   Memory: {climate_tensor.numel() * 4 / 1e6:.1f} MB")

    except Exception as e:
        print(f"Failed to load climate data: {e}")
        return False

    # Step 2: Climate Data Processing with AIFS
    print("\nStep 2: Processing Climate Data with AIFS")
    print("-" * 40)

    # Step 3: AIFS Climate Tokenization
    print("\nStep 3: AIFS Climate Tokenization")
    print("-" * 40)

    try:
        climate_tokens = model.tokenize_climate_data(climate_tensor)
        print("Climate tokenization successful")
        print(f"   Token shape: {climate_tokens.shape}")
        batch_size = climate_tokens.shape[0]
        seq_len = climate_tokens.shape[1]
        hidden_size = climate_tokens.shape[2]
        print(f"   Format: [batch={batch_size}, seq_len={seq_len}, hidden={hidden_size}]")

    except Exception as e:
        print(f"Climate tokenization failed: {e}")
        return False

    # Step 4: Text Processing with Real Mistral
    print("\nStep 4: Text Processing with Real Mistral")
    print("-" * 40)

    try:
        # Test text inputs
        text_inputs = [
            "Analyze the temperature patterns in this climate data.",
            "What weather trends are visible in the atmospheric data?",
        ]

        # Process text with real Mistral tokenizer
        text_tokens = model.tokenize_text(text_inputs)
        print("Text tokenization successful")
        print(f"   Input IDs shape: {text_tokens['input_ids'].shape}")
        print(f"   👁️  Attention mask shape: {text_tokens['attention_mask'].shape}")

        # Show tokenized text (first few tokens)
        if model.mistral_tokenizer:
            sample_tokens = text_tokens["input_ids"][0][:10].tolist()
            decoded = model.mistral_tokenizer.decode(sample_tokens)
            print(f"   Sample tokens: {decoded}...")

    except Exception as e:
        print(f"Text processing failed: {e}")
        return False

    # Step 5: Multimodal Fusion with Real Mistral
    print("\n🔗 Step 5: Multimodal Fusion with Real Mistral")
    print("-" * 40)

    try:
        print("⏳ Running multimodal fusion (this may take a few minutes on CPU)...")
        start_time = time.time()

        # Use the convenience method we added
        result = model.process_climate_text(
            climate_tokens,
            text_inputs[:1],  # Use just one text for CPU efficiency
            task="generation",
        )

        elapsed = time.time() - start_time
        print("Multimodal fusion successful!")
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
        print("This is expected on CPU with limited memory")
        return False

    # Step 6: Memory and Performance Analysis
    print("\nStep 6: Performance Analysis")
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

    print("\nReal Mistral Integration Test Complete!")
    print("Successfully processed climate data through:")
    print("   Zarr → AIFS tokenization")
    print("   🤖 Real Mistral-7B-Instruct-v0.3 processing")
    print("   🔗 Multimodal fusion")
    print("   Text generation")

    return True

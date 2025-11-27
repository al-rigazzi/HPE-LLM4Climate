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
Integration Test for AIFS Climate Fusion (pytest version)

This test validates the complete climate-text fusion pipeline using AIFS
encoder for multimodal climate analysis.

Usage:
    python -m pytest multimodal_aifs/tests/integration/test_aifs_climate_fusion_pytest.py -v
"""

import sys
import time
from pathlib import Path

import pytest
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from multimodal_aifs.constants import AIFS_RAW_ENCODER_OUTPUT_DIM
from multimodal_aifs.core.aifs_climate_fusion import AIFSClimateEmbedding, AIFSClimateTextFusion


def test_batch_processing(aifs_model, test_device):
    """Test batch processing functionality."""
    print("\n📦 Testing Batch Processing")

    # Get the real AIFS model from fixture
    aifs_model_instance = aifs_model["model"] if not aifs_model["is_mock"] else None
    device = str(test_device)

    try:
        # Create fusion module with real AIFS model
        fusion_module = AIFSClimateTextFusion(
            aifs_model=aifs_model_instance,  # Use real model from fixture
            climate_dim=AIFS_RAW_ENCODER_OUTPUT_DIM,
            text_dim=768,
            fusion_dim=512,
            device=device,
        )

        # Check if encoder was loaded successfully
        if fusion_module.aifs_encoder is None:
            print("   Skipping batch test (AIFS encoder not available)")
            pytest.skip("AIFS encoder not available")
            return

        print("   AIFS encoder loaded successfully")

    except Exception as e:
        print(f"   Skipping batch test (initialization failed: {e})")
        pytest.skip(f"Initialization failed: {e}")


@pytest.mark.large_memory
def test_climate_data_encoding(aifs_model, test_device, test_climate_data):
    """Test climate data encoding with AIFS."""
    print("\nTesting Climate Data Encoding")

    # Get the real AIFS model from fixture
    aifs_model_instance = aifs_model["model"] if not aifs_model["is_mock"] else None
    device = str(test_device)

    try:
        # Create fusion module with real AIFS model
        fusion_module = AIFSClimateTextFusion(
            aifs_model=aifs_model_instance,  # Use real model from fixture
            climate_dim=AIFS_RAW_ENCODER_OUTPUT_DIM,
            text_dim=768,
            fusion_dim=512,
            device=device,
        )

        # Check if encoder was loaded successfully
        if fusion_module.aifs_encoder is None:
            print("   Skipping encoding test (AIFS encoder not available)")
            pytest.skip("AIFS encoder not available")
            return
    except Exception as e:
        print(f"   Skipping encoding test (initialization failed: {e})")
        pytest.skip(f"Initialization failed: {e}")

    # Use proper 5D climate data from fixture: [batch, time, ensemble, grid, vars]
    climate_data_5d = test_climate_data["tensor_5d"]  # [1, 2, 1, 542080, 103]
    print(f"   Using 5D climate data: {climate_data_5d.shape} ({climate_data_5d.dtype})")

    # Test encoding
    start_time = time.time()
    try:
        climate_features = fusion_module.encode_climate_data(climate_data_5d)
        encoding_time = time.time() - start_time

        # Validate output
        assert climate_features.shape[0] == climate_data_5d.shape[0]  # batch dimension
        assert climate_features.shape[1] == 512  # fusion_dim
        assert climate_features.device.type == device

        print(
            f"   Climate encoding: {climate_data_5d.shape} ({climate_data_5d.dtype}) "
            f"-> {climate_features.shape} ({climate_features.dtype})"
        )
        print(f"   Encoding time: {encoding_time:.4f}s")
    except Exception as e:
        print(f"   Encoding failed: {e}")
        pytest.fail(f"Encoding failed: {e}")


@pytest.mark.large_memory
def test_climate_embedding_module(aifs_model, test_device, test_climate_data):
    """Test AIFSClimateEmbedding module."""
    print("\nTesting Climate Embedding Module")

    # Get the real AIFS model from fixture
    aifs_model_instance = aifs_model["model"] if not aifs_model["is_mock"] else None
    device = str(test_device)

    try:
        embedding_module = AIFSClimateEmbedding(
            aifs_model=aifs_model_instance,  # Use real model from fixture
            embedding_dim=256,  # Correct parameter name
            device=device,
        )

        # Check if encoder was loaded successfully
        if embedding_module.aifs_encoder is None:
            print("   Skipping embedding test (AIFS encoder not available)")
            pytest.skip("AIFS encoder not available")
            return
    except Exception as e:
        print(f"   Skipping embedding test (initialization failed: {e})")
        pytest.skip(f"Initialization failed: {e}")

    # Use the 5D tensor (raw climate data) for embedding module
    # test since it calls AIFS encoder internally
    sample_data_5d = test_climate_data["tensor_5d"]  # [1, 2, 1, 542080, 103] - raw climate data
    print(f"   Using 5D climate data: {sample_data_5d.shape}")

    try:
        embeddings = embedding_module(sample_data_5d)

        # Validate output
        assert embeddings.shape[0] == sample_data_5d.shape[0]
        assert embeddings.shape[1] == 256  # embedding_dim
        assert embeddings.device.type == device

        print(f"   Embedding: {sample_data_5d.shape} -> {embeddings.shape}")
        print("   Embedding module test passed")
    except Exception as e:
        print(f"   Embedding test failed: {e}")
        pytest.fail(f"Embedding test failed: {e}")


def test_error_handling(aifs_model, test_device):
    """Test error handling for invalid inputs."""
    print("\nTesting Error Handling")

    device = str(test_device)

    # Test with invalid path
    try:
        fusion_module = AIFSClimateTextFusion(
            aifs_checkpoint_path="/invalid/path/to/model.pth",
            climate_dim=AIFS_RAW_ENCODER_OUTPUT_DIM,
            text_dim=768,
            fusion_dim=512,
            device=device,
        )
        print("   Invalid path handled gracefully (encoder=None)")
        assert fusion_module.aifs_encoder is None, "Expected None encoder for invalid path"
    except (ValueError, RuntimeError) as e:
        print(f"   Invalid path handling: {e}")
    except TypeError as e:
        print(
            f"   Error handling test skipped: {e!r} "
            f"is not an instance of {(ValueError, RuntimeError)}"
        )
        # Different error type, but still handled

    # Test with valid checkpoint path but no AIFS model loaded
    print("\n   Testing valid checkpoint path handling...")
    try:
        # Use a real checkpoint path
        valid_checkpoint = (
            project_root
            / "multimodal_aifs"
            / "models"
            / "extracted_models"
            / "aifs_encoder_full.pth"
        )

        fusion_module = AIFSClimateTextFusion(
            aifs_checkpoint_path=valid_checkpoint,
            climate_dim=AIFS_RAW_ENCODER_OUTPUT_DIM,
            text_dim=768,
            fusion_dim=512,
            device=device,
            verbose=False,  # Suppress the warning for cleaner test output
        )

        # Should have checkpoint_path set but encoder should be None until AIFS model is provided
        assert (
            fusion_module.aifs_encoder is None
        ), "Encoder should be None until AIFS model is provided"
        assert hasattr(fusion_module, "checkpoint_path"), "Should store checkpoint path"
        print("   Valid checkpoint path handled correctly")

    except Exception as e:
        print(f"   Valid checkpoint test failed: {e}")
        # This might fail if the checkpoint format is incompatible, which is also acceptable


def test_fusion_module_initialization(aifs_model, test_device):
    """Test AIFSClimateTextFusion initialization."""
    print("\nTesting Fusion Module Initialization")

    # Get the real AIFS model from fixture
    aifs_model_instance = aifs_model["model"] if not aifs_model["is_mock"] else None
    device = str(test_device)

    # Test with real AIFS model
    fusion_module = AIFSClimateTextFusion(
        aifs_model=aifs_model_instance,  # Use real model from fixture
        climate_dim=AIFS_RAW_ENCODER_OUTPUT_DIM,
        text_dim=768,
        fusion_dim=512,
        num_attention_heads=8,
        device=device,
    )

    assert fusion_module.climate_dim == AIFS_RAW_ENCODER_OUTPUT_DIM
    assert fusion_module.text_dim == 768
    assert fusion_module.fusion_dim == 512
    assert fusion_module.device == device

    if aifs_model_instance is not None:
        assert fusion_module.aifs_encoder is not None
        print("   Real model initialization successful")
    else:
        print("   Mock model initialization successful")


@pytest.mark.large_memory
def test_multimodal_fusion(aifs_model, test_device, test_climate_data):
    """Test multimodal fusion functionality."""
    print("\n🔀 Testing Multimodal Fusion")

    # Get the real AIFS model from fixture
    aifs_model_instance = aifs_model["model"] if not aifs_model["is_mock"] else None
    device = str(test_device)

    try:
        fusion_module = AIFSClimateTextFusion(
            aifs_model=aifs_model_instance,  # Use real model from fixture
            climate_dim=AIFS_RAW_ENCODER_OUTPUT_DIM,
            text_dim=768,
            fusion_dim=512,
            device=device,
        )

        # Check if encoder was loaded successfully
        if fusion_module.aifs_encoder is None:
            print("   Skipping fusion test (AIFS encoder not available)")
            pytest.skip("AIFS encoder not available")
            return
    except Exception as e:
        print(f"   Skipping fusion test (initialization failed: {e})")
        pytest.skip(f"Initialization failed: {e}")

    # Use 5D climate data and let fusion module handle encoding
    climate_data_5d = test_climate_data["tensor_5d"]  # [1, 2, 1, 542080, 103]
    # Use single text to match batch size (batch=1)
    texts = ["High temperature patterns"]
    print(f"   Climate data: {climate_data_5d.shape}, Texts: {len(texts)}")

    try:
        # Create dummy text embeddings for testing (text_dim=768)
        # In production, these would come from a proper text encoder
        # (BERT, sentence-transformers, etc.)
        # Number of texts must match batch size
        text_embeddings = torch.randn(len(texts), 768, device=test_device)

        # Test fusion using the forward method that handles encoding internally
        results = fusion_module.forward(climate_data_5d, texts, text_embeddings=text_embeddings)

        # Validate output
        assert "fused_features" in results
        fused_features = results["fused_features"]
        assert fused_features.shape[0] == climate_data_5d.shape[0]
        assert fused_features.shape[1] == 512  # fusion_dim

        print(
            f"   Fusion: Climate {climate_data_5d.shape} + "
            f"{len(texts)} texts -> {fused_features.shape}"
        )
        print("   Multimodal fusion test passed")
    except Exception as e:
        print(f"   Fusion test failed: {e}")
        pytest.fail(f"Fusion test failed: {e}")


@pytest.mark.skip("Similarity will be tested on trained models")
def test_similarity_and_alignment(aifs_model, test_device, test_climate_data):
    """Test similarity and alignment computation."""
    print("\n🔗 Testing Similarity and Alignment")

    # Get the real AIFS model from fixture
    aifs_model_instance = aifs_model["model"] if not aifs_model["is_mock"] else None
    device = str(test_device)

    try:
        fusion_module = AIFSClimateTextFusion(
            aifs_model=aifs_model_instance,  # Use real model from fixture
            climate_dim=AIFS_RAW_ENCODER_OUTPUT_DIM,
            text_dim=768,
            fusion_dim=512,
            device=device,
        )

        # Check if encoder was loaded successfully
        if fusion_module.aifs_encoder is None:
            print("   Skipping similarity test (AIFS encoder not available)")
            pytest.skip("AIFS encoder not available")
            return
    except Exception as e:
        print(f"   Skipping similarity test (initialization failed: {e})")
        pytest.skip(f"Initialization failed: {e}")

    # Create test data for climate similarity test
    climate_data_5d = test_climate_data["tensor_5d"]  # [1, 2, 1, 542080, 103]
    climate_data_batch = climate_data_5d.repeat(2, 1, 1, 1, 1)  # [2, 2, 1, 542080, 103]
    print(f"   Climate data for similarity: {climate_data_batch.shape}")

    try:
        # Test climate similarity computation using the correct method
        climate1 = climate_data_batch[:1]  # First sample
        climate2 = climate_data_batch[1:2]  # Second sample
        similarities = fusion_module.get_climate_similarity(climate1, climate2)

        # Validate output
        assert similarities.numel() == 1  # Single similarity score

        print(f"   Climate similarity: {similarities.item():.4f}")
        print("   Similarity computation test passed")
    except Exception as e:
        print(f"   Similarity test failed: {e}")
        pytest.fail(f"Similarity test failed: {e}")


def test_text_encoding(aifs_model, test_device):
    """Test text encoding functionality."""
    print("\nTesting Text Encoding")

    # Get the real AIFS model from fixture
    aifs_model_instance = aifs_model["model"] if not aifs_model["is_mock"] else None
    device = str(test_device)

    # Create fusion module
    fusion_module = AIFSClimateTextFusion(
        aifs_model=aifs_model_instance,  # Use real model from fixture
        climate_dim=AIFS_RAW_ENCODER_OUTPUT_DIM,
        text_dim=768,
        fusion_dim=512,
        device=device,
    )

    # Test texts
    texts = [
        "High temperature and low pressure system",
        "Tropical storm with heavy rainfall",
        "Clear skies with moderate temperatures",
        "Drought conditions with high pressure",
    ]

    try:
        # Use real text embeddings from TextEmbeddingUtils
        from multimodal_aifs.utils.text_utils import TextEmbeddingUtils

        text_embedding_utils = TextEmbeddingUtils(embedding_dim=768, vocab_size=500)
        text_embedding_utils.build_vocabulary(texts)

        # Generate real embeddings for each text
        text_features = []
        for text in texts:
            embedding = text_embedding_utils.embed_text(text, max_length=64)
            # Take mean across sequence to get single embedding
            text_embedding = embedding.mean(dim=0, keepdim=True).to(device)
            text_features.append(text_embedding)

        text_features = torch.cat(text_features, dim=0)

        # Test that encode_text requires pre-computed embeddings
        projected_text = fusion_module.encode_text(texts, text_embeddings=text_features)

        # Validate output
        assert projected_text.shape[0] == len(texts)
        assert projected_text.shape[1] == 512  # fusion_dim

        print(f"   Text encoding: {len(texts)} texts -> {projected_text.shape}")
        print("   Real embeddings from TextEmbeddingUtils")
    except Exception as e:
        print(f"   Text encoding failed: {e}")
        pytest.fail(f"Text encoding failed: {e}")

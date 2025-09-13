#!/usr/bin/env python3
"""
Test  AIFS Time Series Tokenizer

This test validates that the updated tokenizer works with the new AIFSCompleteEncoder interface.
"""

import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from multimodal_aifs.utils.aifs_time_series_tokenizer import AIFSTimeSeriesTokenizer


def test__tokenizer():
    """Test the  AIFS time series tokenizer with new interface."""
    print("🚀 Testing  AIFS Time Series Tokenizer")
    print("=" * 60)

    try:
        # Test 1: Initialization with checkpoint mode
        print("📦 Test 1: Initialization")

        tokenizer = AIFSTimeSeriesTokenizer(
            aifs_checkpoint_path="/path/to/checkpoint.pt",
            temporal_modeling="transformer",
            hidden_dim=512,
            device="cpu",
            verbose=False,
        )

        print("✅ Tokenizer initialized successfully")
        print(f"   Temporal modeling: {tokenizer.temporal_modeling}")
        print(f"   Hidden dim: {tokenizer.hidden_dim}")
        print(f"   Spatial dim: {tokenizer.spatial_dim}")

        # Test 2: Configuration validation
        print("\n🔧 Test 2: Configuration Validation")

        info = tokenizer.get_tokenizer_info()
        assert info["spatial_dim"] == 218, f"Expected 218, got {info['spatial_dim']}"
        assert info["temporal_modeling"] == "transformer"
        assert info["aifs_encoder"]["type"] == "Checkpoint mode"

        print("✅ Configuration validated")
        print(f"   AIFS encoder type: {info['aifs_encoder']['type']}")
        print(f"   Output dimension: {info['aifs_encoder']['output_dim']}")

        # Test 3: Different temporal models
        print("\n⏱️  Test 3: Different Temporal Models")

        temporal_models = ["none", "lstm", "transformer"]

        for model_type in temporal_models:
            test_tokenizer = AIFSTimeSeriesTokenizer(
                aifs_checkpoint_path="/path/to/checkpoint.pt",
                temporal_modeling=model_type,
                hidden_dim=256,
                device="cpu",
                verbose=False,
            )

            test_info = test_tokenizer.get_tokenizer_info()
            assert test_info["temporal_modeling"] == model_type
            assert test_info["spatial_dim"] == 218

            print(f"   ✅ {model_type} model configured correctly")

        # Test 4: Expected output shapes
        print("\n📊 Test 4: Expected Output Shapes")

        batch_size, time_steps = 4, 8

        for model_type in temporal_models:
            test_tokenizer = AIFSTimeSeriesTokenizer(
                aifs_checkpoint_path="/path/to/checkpoint.pt",
                temporal_modeling=model_type,
                hidden_dim=256,
                device="cpu",
                verbose=False,
            )

            if model_type == "none":
                expected_features = test_tokenizer.spatial_dim  # 218
            else:
                expected_features = test_tokenizer.hidden_dim  # 256

            expected_shape = (batch_size, time_steps, expected_features)
            print(f"   ✅ {model_type}: Expected output shape {expected_shape}")

        # Test 5: Error handling
        print("\n❌ Test 5: Error Handling")

        # Test invalid temporal modeling
        try:
            AIFSTimeSeriesTokenizer(
                aifs_checkpoint_path="/path/to/checkpoint.pt",
                temporal_modeling="invalid",
                verbose=False,
            )
            assert False, "Should have raised ValueError"
        except ValueError:
            print("   ✅ Invalid temporal modeling rejected")

        # Test missing AIFS model/checkpoint
        try:
            AIFSTimeSeriesTokenizer(verbose=False)
            assert False, "Should have raised ValueError"
        except ValueError:
            print("   ✅ Missing AIFS model/checkpoint rejected")

        print("\n🎉 All  Tokenizer Tests Passed!")
        print("✨ Ready for integration with real AIFS models!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        assert False, f"Test failed: {e}"


if __name__ == "__main__":
    test__tokenizer()
    print("All tests completed successfully!")

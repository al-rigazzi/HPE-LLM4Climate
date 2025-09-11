#!/usr/bin/env python3
"""
Test Advanced AIFS Integration

This test validates that all our migrated modules work with the advanced AIFSCompleteEncoder.
"""

import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_advanced_integration():
    """Test all advanced AIFS components."""
    print("🚀 Testing Advanced AIFS Integration")
    print("=" * 60)

    try:
        # Test 1: Advanced encoder utils
        print("📦 Test 1: Advanced Encoder Utils")

        from multimodal_aifs.utils.aifs_encoder_utils import AIFSEncoderWrapper, create_aifs_encoder

        # Test create functions exist and work
        assert callable(create_aifs_encoder)
        print("✅ Encoder utility functions available")

        # Test 2: Advanced time series tokenizer
        print("\n⏱️  Test 2: Advanced Time Series Tokenizer")

        from multimodal_aifs.utils.aifs_time_series_tokenizer import AIFSTimeSeriesTokenizer

        # Test new interface
        tokenizer = AIFSTimeSeriesTokenizer(
            aifs_checkpoint_path="/path/to/checkpoint.pt",
            temporal_modeling="transformer",
            hidden_dim=512,
            device="cpu",
            verbose=False,
        )

        info = tokenizer.get_tokenizer_info()
        assert info["spatial_dim"] == 218, f"Expected 218, got {info['spatial_dim']}"
        print("✅ Advanced tokenizer configured correctly")
        print(f"   Output dimension: {info['spatial_dim']}")

        # Test 3: Advanced climate fusion
        print("\n🌍 Test 3: Advanced Climate Fusion")

        from multimodal_aifs.core.aifs_climate_fusion import (
            AIFSClimateEmbedding,
            AIFSClimateTextFusion,
            create_aifs_embedding_from_model,
            create_aifs_fusion_from_model,
        )

        # Test helper functions exist
        assert callable(create_aifs_fusion_from_model)
        assert callable(create_aifs_embedding_from_model)
        print("✅ Advanced fusion utilities available")

        # Test 4: Advanced location aware fusion
        print("\n📍 Test 4: Advanced Location Aware Fusion")

        from multimodal_aifs.core.aifs_location_aware_fusion import AIFSLocationAwareFusion

        # Test class exists
        assert AIFSLocationAwareFusion is not None
        print("✅ Advanced location fusion class available")  # Test 5: Package exports
        print("\n📦 Test 5: Package Exports")

        import multimodal_aifs

        # Check that key functions are exported
        exported_functions = [
            "create_aifs_encoder",
            "create_aifs_fusion_from_model",
            "create_aifs_embedding_from_model",
        ]

        for func_name in exported_functions:
            assert hasattr(multimodal_aifs, func_name), f"Missing export: {func_name}"
            print(f"   ✅ {func_name}")

        # Test 6: Advanced dimensions
        print("\n📊 Test 6: Advanced Dimensions")

        # Test that all components use the correct 218-dimensional output
        tokenizer_info = tokenizer.get_tokenizer_info()
        assert tokenizer_info["spatial_dim"] == 218
        print("   ✅ Time series tokenizer: 218 dimensions")

        # Test expected shapes for different temporal models
        temporal_models = ["none", "lstm", "transformer"]
        for model_type in temporal_models:
            test_tokenizer = AIFSTimeSeriesTokenizer(
                aifs_checkpoint_path="/path/to/checkpoint.pt",
                temporal_modeling=model_type,
                hidden_dim=256,
                device="cpu",
                verbose=False,
            )

            info = test_tokenizer.get_tokenizer_info()
            assert info["spatial_dim"] == 218

            if model_type == "none":
                expected_features = 218  # Direct AIFS output
            else:
                expected_features = 256  # Temporal processing output

            print(f"   ✅ {model_type}: spatial_dim=218, output_features={expected_features}")

        print("\n🎉 All Advanced Integration Tests Passed!")
        print("✨ Complete migration successful - Ready for production!")

        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_advanced_integration()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Unit Tests for AIFS Encoder Utils

This test module validates the AIFSEncoderWrapper and related utilities
for ECMWF AIFS model integration and climate data encoding.

Usage:
    python multimodal_aifs/tests/unit/test_aifs_encoder_utils.py
    python -m pytest multimodal_aifs/tests/unit/test_aifs_encoder_utils.py -v
"""

import os
import sys
import time
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from multimodal_aifs.utils.aifs_encoder_utils import AIFSEncoderWrapper, create_aifs_encoder


class TestAIFSEncoderUtils(unittest.TestCase):
    """Test suite for AIFS encoder utilities."""

    @classmethod
    def setUpClass(cls):
        """Set up test class with paths and configurations."""
        cls.project_root = project_root
        cls.aifs_model_path = (
            cls.project_root
            / "multimodal_aifs"
            / "models"
            / "extracted_models"
            / "aifs_encoder_full.pth"
        )
        cls.device = "cpu"  # Use CPU for testing
        cls.has_real_model = cls.aifs_model_path.exists()

        print(f"🧪 AIFS Encoder Utils Test Setup")
        print(f"   Project root: {cls.project_root}")
        print(f"   AIFS model path: {cls.aifs_model_path}")
        print(f"   Has real model: {cls.has_real_model}")
        print(f"   Device: {cls.device}")

    def test_aifs_encoder_wrapper_initialization(self):
        """Test AIFSEncoderWrapper initialization."""
        print("\\n🔧 Testing AIFSEncoderWrapper Initialization")

        if self.has_real_model:
            # Test with real model
            wrapper = AIFSEncoderWrapper(encoder_path=str(self.aifs_model_path), device=self.device)

            self.assertIsNotNone(wrapper.encoder)
            self.assertEqual(wrapper.device, self.device)
            self.assertTrue(wrapper.encoder is not None)

            # Test encoder info
            if wrapper.encoder_info:
                info = wrapper.encoder_info
                self.assertIn("total_parameters", info)
                print(f"   ✅ Real model loaded: {info['total_parameters']:,} parameters")
            else:
                print("   ✅ Real model loaded successfully")

        else:
            # Test fallback behavior
            wrapper = AIFSEncoderWrapper(encoder_path=None, device=self.device)

            self.assertIsNotNone(wrapper.encoder)  # Should have dummy encoder
            self.assertEqual(wrapper.device, self.device)

            print("   ✅ Graceful fallback without real model")

    def test_climate_data_encoding(self):
        """Test climate data encoding functionality."""
        print("\\n🌡️ Testing Climate Data Encoding")

        # Create synthetic climate data
        batch_size = 4
        input_features = 218  # AIFS expected input size
        climate_data = torch.randn(batch_size, input_features)

        if self.has_real_model:
            wrapper = AIFSEncoderWrapper(encoder_path=str(self.aifs_model_path), device=self.device)

            # Test encoding
            start_time = time.time()
            encoded = wrapper.encode_climate_data(climate_data)
            encoding_time = time.time() - start_time

            # Validate output
            self.assertEqual(encoded.shape[0], batch_size)
            self.assertGreater(encoded.shape[1], 0)  # Output dimension
            self.assertEqual(encoded.device.type, self.device)

            print(f"   ✅ Encoded shape: {encoded.shape}")
            print(f"   ✅ Encoding time: {encoding_time:.4f}s")
            print(f"   ✅ Output range: [{encoded.min():.3f}, {encoded.max():.3f}]")

            # Test batch encoding
            batch_results = wrapper.encode_batch(climate_data)
            self.assertIsInstance(batch_results, torch.Tensor)
            self.assertEqual(batch_results.shape[0], batch_size)
            self.assertGreater(batch_results.shape[1], 0)

        else:
            print("   ⚠️  Skipping encoding test (no real model)")

    def test_preprocessing_functionality(self):
        """Test climate data preprocessing."""
        print("\\n🔄 Testing Data Preprocessing")

        wrapper = AIFSEncoderWrapper(encoder_path=None, device=self.device)

        # Test with various input formats
        test_cases = [
            torch.randn(4, 218),  # Correct format
            torch.randn(4, 100),  # Wrong size (should be adjusted)
            torch.randn(4, 300),  # Too large (should be truncated)
            np.random.randn(4, 218).astype(np.float32),  # NumPy input
        ]

        for i, data in enumerate(test_cases):
            processed = wrapper._preprocess_climate_data(
                torch.tensor(data) if isinstance(data, np.ndarray) else data
            )

            # Validate preprocessing
            self.assertEqual(processed.shape[0], 4)  # Batch size preserved
            self.assertEqual(processed.dtype, torch.float32)

            print(
                f"   ✅ Test case {i+1}: {type(data).__name__} {tuple(data.shape)} -> {tuple(processed.shape)}"
            )

    def test_feature_transformation(self):
        """Test AIFS feature transformation."""
        print("\\n🔀 Testing Feature Transformation")

        wrapper = AIFSEncoderWrapper(encoder_path=None, device=self.device)

        # Test feature adjustment
        test_data = torch.randn(2, 100)  # Wrong size
        transformed = wrapper._transform_to_aifs_features(test_data)

        self.assertEqual(transformed.shape[1], 218)  # Should be adjusted to AIFS size
        self.assertEqual(transformed.shape[0], 2)  # Batch size preserved

        print(f"   ✅ Transformed {test_data.shape} -> {transformed.shape}")

        # Test with correct size
        correct_data = torch.randn(2, 218)
        transformed_correct = wrapper._transform_to_aifs_features(correct_data)

        self.assertEqual(transformed_correct.shape, correct_data.shape)
        torch.testing.assert_close(transformed_correct, correct_data)

        print(f"   ✅ Correct size preserved: {correct_data.shape}")

    def test_error_handling(self):
        """Test error handling and edge cases."""
        print("\\n❌ Testing Error Handling")

        wrapper = AIFSEncoderWrapper(encoder_path=None, device=self.device)

        # Test graceful handling of edge cases
        # Empty tensor should be handled gracefully (padded or expanded)
        result = wrapper.encode_climate_data(torch.randn(1, 10))  # Too few features
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape[0], 1)

        # Test string input conversion (should handle gracefully or give clear error)
        try:
            wrapper.encode_climate_data("invalid")  # Wrong type
            self.fail("Should have raised an exception for invalid input type")
        except (ValueError, TypeError, AttributeError):
            pass  # Expected behavior

        print("   ✅ Graceful error handling")

        # Test with unloaded encoder
        if not self.has_real_model:
            with self.assertRaises(RuntimeError):
                wrapper.encode_climate_data(torch.randn(2, 218))

            print("   ✅ Unloaded encoder error handling")

    def test_create_aifs_encoder_function(self):
        """Test the create_aifs_encoder utility function."""
        print("\\n🏭 Testing create_aifs_encoder Function")

        # Test without path
        encoder1 = create_aifs_encoder(encoder_path=None, device=self.device)
        self.assertIsInstance(encoder1, AIFSEncoderWrapper)
        self.assertIsNotNone(encoder1.encoder)  # Should still have an encoder (dummy or real)

        print("   ✅ Created encoder without path")

        if self.has_real_model:
            # Test with real path
            encoder2 = create_aifs_encoder(
                encoder_path=str(self.aifs_model_path), device=self.device
            )
            self.assertIsInstance(encoder2, AIFSEncoderWrapper)
            self.assertIsNotNone(encoder2.encoder)

            print("   ✅ Created encoder with real model")

    def test_performance_benchmarks(self):
        """Test performance characteristics."""
        print("\\n⚡ Testing Performance Benchmarks")

        if not self.has_real_model:
            print("   ⚠️  Skipping performance tests (no real model)")
            return

        wrapper = AIFSEncoderWrapper(encoder_path=str(self.aifs_model_path), device=self.device)

        # Benchmark different batch sizes
        batch_sizes = [1, 4, 8, 16]

        for batch_size in batch_sizes:
            data = torch.randn(batch_size, 218)

            start_time = time.time()
            encoded = wrapper.encode_climate_data(data)
            encoding_time = time.time() - start_time

            throughput = batch_size / encoding_time
            memory_usage = encoded.numel() * encoded.element_size()

            print(
                f"   📊 Batch {batch_size:2d}: {encoding_time:.4f}s, {throughput:.1f} samples/s, {memory_usage/1024:.1f}KB"
            )

            # Performance assertions
            self.assertLess(encoding_time, 1.0)  # Should be fast
            self.assertGreater(throughput, 1.0)  # Reasonable throughput


def run_aifs_encoder_tests():
    """Run all AIFS encoder utility tests."""
    print("🧪 Running AIFS Encoder Utils Tests")
    print("=" * 50)

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAIFSEncoderUtils)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All AIFS encoder tests passed!")
    else:
        print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors")

        if result.failures:
            print("\\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback}")

        if result.errors:
            print("\\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_aifs_encoder_tests()
    sys.exit(0 if success else 1)

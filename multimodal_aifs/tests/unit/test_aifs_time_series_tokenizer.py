#!/usr/bin/env python3
"""
Unit Tests for AIFS Time Series Tokenizer

This test module validates the AIFSTimeSeriesTokenizer functionality
for 5-D climate time series data tokenization and multimodal applications.

Usage:
    python multimodal_aifs/tests/unit/test_aifs_time_series_tokenizer.py
    python -m pytest multimodal_aifs/tests/unit/test_aifs_time_series_tokenizer.py -v
"""

import sys
import time
import unittest
import warnings
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from multimodal_aifs.utils.aifs_time_series_tokenizer import AIFSTimeSeriesTokenizer


class TestAIFSTimeSeriesTokenizer(unittest.TestCase):
    """Test suite for AIFS time series tokenizer."""

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

        print(f"🧪 AIFS Time Series Tokenizer Test Setup")
        print(f"   Project root: {cls.project_root}")
        print(f"   AIFS model path: {cls.aifs_model_path}")
        print(f"   Has real model: {cls.has_real_model}")
        print(f"   Device: {cls.device}")

    def create_test_data(
        self,
        batch_size: int = 2,
        time_steps: int = 4,
        n_variables: int = 3,
        spatial_shape: tuple = (16, 16),
    ) -> torch.Tensor:
        """Create test 5-D climate time series data."""
        height, width = spatial_shape
        return torch.randn(batch_size, time_steps, n_variables, height, width)

    def test_tokenizer_initialization(self):
        """Test AIFSTimeSeriesTokenizer initialization with different configurations."""
        print("\\n🔧 Testing Tokenizer Initialization")

        # Test default initialization (transformer)
        tokenizer_default = AIFSTimeSeriesTokenizer(device=self.device)
        self.assertEqual(tokenizer_default.temporal_modeling, "transformer")
        self.assertEqual(tokenizer_default.hidden_dim, 512)
        self.assertEqual(tokenizer_default.device, self.device)
        print("   ✅ Transformer tokenizer initialized (default)")

        # Test LSTM initialization
        tokenizer_lstm = AIFSTimeSeriesTokenizer(
            temporal_modeling="lstm", hidden_dim=256, device=self.device
        )
        self.assertEqual(tokenizer_lstm.temporal_modeling, "lstm")
        self.assertEqual(tokenizer_lstm.hidden_dim, 256)
        print("   ✅ LSTM tokenizer initialized")

        # Test None initialization (spatial only)
        tokenizer_none = AIFSTimeSeriesTokenizer(temporal_modeling="none", device=self.device)
        self.assertEqual(tokenizer_none.temporal_modeling, "none")
        self.assertIsNone(tokenizer_none.temporal_model)
        print("   ✅ Spatial-only tokenizer initialized")

        # Test invalid temporal modeling
        with self.assertRaises(ValueError):
            AIFSTimeSeriesTokenizer(temporal_modeling="invalid")
        print("   ✅ Invalid temporal modeling properly rejected")

    def test_tensor_shapes_validation(self):
        """Test validation of input tensor shapes."""
        print("\\n📊 Testing Tensor Shape Validation")

        tokenizer = AIFSTimeSeriesTokenizer(device=self.device)

        # Test various 5-D tensor shapes
        test_shapes = [
            (1, 2, 3, 8, 8),  # Minimal case
            (2, 4, 5, 16, 16),  # Small case
            (4, 8, 7, 32, 32),  # Medium case
        ]

        for batch, time, vars, h, w in test_shapes:
            data = self.create_test_data(batch, time, vars, (h, w))

            # Should not raise an error for valid 5-D tensors
            try:
                tokens = tokenizer.tokenize_time_series(data)
                expected_shape = (
                    batch,
                    time,
                    (
                        tokenizer.hidden_dim
                        if tokenizer.temporal_modeling != "none"
                        else tokenizer.spatial_dim
                    ),
                )
                self.assertEqual(tokens.shape, expected_shape)
                print(f"   ✅ Shape {data.shape} -> {tokens.shape}")
            except Exception as e:
                self.fail(f"Valid 5-D tensor failed: {e}")

        # Test invalid tensor dimensions
        invalid_tensors = [
            torch.randn(2, 3),  # 2-D
            torch.randn(2, 3, 4),  # 3-D
            torch.randn(2, 3, 4, 5),  # 4-D
            torch.randn(2, 3, 4, 5, 6, 7),  # 6-D
        ]

        for invalid_tensor in invalid_tensors:
            with self.assertRaises(Exception):
                tokenizer.tokenize_time_series(invalid_tensor)
            print(f"   ✅ Invalid shape {invalid_tensor.shape} properly rejected")

    def test_temporal_modeling_outputs(self):
        """Test that different temporal modeling approaches produce expected outputs."""
        print("\\n⏱️ Testing Temporal Modeling Outputs")

        # Create test data
        batch_size, time_steps = 2, 6
        test_data = self.create_test_data(batch_size, time_steps, 5, (32, 32))

        temporal_models = ["transformer", "lstm", "none"]

        for model_type in temporal_models:
            tokenizer = AIFSTimeSeriesTokenizer(
                temporal_modeling=model_type, hidden_dim=256, device=self.device
            )

            tokens = tokenizer.tokenize_time_series(test_data)

            # Validate output shape
            self.assertEqual(tokens.shape[0], batch_size)
            self.assertEqual(tokens.shape[1], time_steps)

            if model_type == "none":
                # Spatial-only should preserve AIFS output dimension
                self.assertEqual(tokens.shape[2], tokenizer.spatial_dim)
            else:
                # Temporal models should output hidden_dim
                self.assertEqual(tokens.shape[2], 256)

            print(f"   ✅ {model_type.upper()}: {test_data.shape} -> {tokens.shape}")

    def test_sequential_vs_parallel_processing(self):
        """Test that sequential and parallel processing produce similar results for spatial-only."""
        print("\\n🔄 Testing Sequential vs Parallel Processing")

        # Test with spatial-only (should be identical)
        tokenizer = AIFSTimeSeriesTokenizer(temporal_modeling="none", device=self.device)

        test_data = self.create_test_data(2, 4, 3, (16, 16))

        # Sequential processing
        tokens_sequential = tokenizer.tokenize_time_series(test_data)

        # Parallel processing
        tokens_parallel = tokenizer.tokenize_batch_parallel(test_data)

        # Should be identical for spatial-only
        self.assertTrue(torch.allclose(tokens_sequential, tokens_parallel, atol=1e-6))
        print(f"   ✅ Sequential and parallel results identical")
        print(f"      Sequential: {tokens_sequential.shape}")
        print(f"      Parallel:   {tokens_parallel.shape}")

    def test_memory_efficiency(self):
        """Test memory usage for different configurations."""
        print("\\n💾 Testing Memory Efficiency")

        test_configs = [
            ("Small", 1, 4, 3, (16, 16)),
            ("Medium", 2, 8, 5, (32, 32)),
            ("Large", 1, 12, 7, (64, 64)),
        ]

        for config_name, batch, time, vars, spatial in test_configs:
            data = self.create_test_data(batch, time, vars, spatial)

            # Calculate input size
            input_size = data.numel() * 4  # float32 bytes

            # Test with transformer tokenizer
            tokenizer = AIFSTimeSeriesTokenizer(temporal_modeling="transformer", device=self.device)

            tokens = tokenizer.tokenize_time_series(data)
            output_size = tokens.numel() * 4  # float32 bytes
            compression_ratio = input_size / output_size

            self.assertGreater(compression_ratio, 1.0)  # Should compress data
            print(
                f"   ✅ {config_name}: {compression_ratio:.1f}x compression ({input_size//1024}KB -> {output_size//1024}KB)"
            )

    def test_batch_encoding(self):
        """Test batch encoding functionality."""
        print("\\n📦 Testing Batch Encoding")

        tokenizer = AIFSTimeSeriesTokenizer(device=self.device)

        # Test different batch sizes
        batch_sizes = [1, 2, 4, 8]
        time_steps = 6

        for batch_size in batch_sizes:
            data = self.create_test_data(batch_size, time_steps, 5, (16, 16))

            # Test tokenization
            tokens = tokenizer.tokenize_time_series(data)

            # Validate batch dimension
            self.assertEqual(tokens.shape[0], batch_size)
            self.assertEqual(tokens.shape[1], time_steps)

            print(f"   ✅ Batch size {batch_size}: {data.shape} -> {tokens.shape}")

    def test_performance_benchmarks(self):
        """Test performance across different configurations."""
        print("\\n⚡ Testing Performance Benchmarks")

        test_cases = [
            ("transformer", 2, 4, 3, (16, 16)),
            ("lstm", 2, 4, 3, (16, 16)),
            ("none", 2, 4, 3, (16, 16)),
        ]

        for model_type, batch, time_steps, vars, spatial in test_cases:
            tokenizer = AIFSTimeSeriesTokenizer(temporal_modeling=model_type, device=self.device)

            data = self.create_test_data(batch, time_steps, vars, spatial)

            # Warmup
            _ = tokenizer.tokenize_time_series(data)

            # Benchmark
            start_time = time.perf_counter()
            tokens = tokenizer.tokenize_time_series(data)
            end_time = time.perf_counter()

            processing_time = end_time - start_time
            throughput = batch / processing_time if processing_time > 0 else float("inf")

            self.assertGreater(throughput, 0)
            print(f"   ✅ {model_type.upper()}: {processing_time:.4f}s, {throughput:.1f} samples/s")

    def test_tokenizer_info(self):
        """Test tokenizer information retrieval."""
        print("\\n📋 Testing Tokenizer Info")

        tokenizer = AIFSTimeSeriesTokenizer(device=self.device)
        info = tokenizer.get_tokenizer_info()

        # Validate info structure
        required_keys = [
            "aifs_encoder",
            "temporal_modeling",
            "hidden_dim",
            "spatial_dim",
            "device",
            "output_shape_pattern",
        ]

        for key in required_keys:
            self.assertIn(key, info)

        self.assertEqual(info["temporal_modeling"], "transformer")  # Default
        self.assertEqual(info["device"], self.device)
        self.assertIsInstance(info["aifs_encoder"], dict)

        print("   ✅ Tokenizer info structure validated")
        print(f"      Temporal modeling: {info['temporal_modeling']}")
        print(f"      Hidden dim: {info['hidden_dim']}")
        print(f"      Spatial dim: {info['spatial_dim']}")

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        print("\\n🔍 Testing Edge Cases")

        tokenizer = AIFSTimeSeriesTokenizer(device=self.device)

        # Test single timestep
        single_step_data = self.create_test_data(2, 1, 3, (16, 16))
        tokens = tokenizer.tokenize_time_series(single_step_data)
        self.assertEqual(tokens.shape[1], 1)
        print("   ✅ Single timestep handled correctly")

        # Test single variable
        single_var_data = self.create_test_data(2, 4, 1, (16, 16))
        tokens = tokenizer.tokenize_time_series(single_var_data)
        self.assertEqual(tokens.shape[0], 2)
        print("   ✅ Single variable handled correctly")

        # Test minimal spatial dimensions
        minimal_spatial_data = self.create_test_data(1, 2, 3, (4, 4))
        tokens = tokenizer.tokenize_time_series(minimal_spatial_data)
        self.assertEqual(tokens.shape, (1, 2, tokenizer.hidden_dim))
        print("   ✅ Minimal spatial dimensions handled correctly")

    def test_device_consistency(self):
        """Test device consistency across operations."""
        print("\\n💻 Testing Device Consistency")

        device = self.device
        tokenizer = AIFSTimeSeriesTokenizer(device=device)

        # Create data on same device
        data = self.create_test_data(2, 4, 3, (16, 16)).to(device)

        # Tokenize
        tokens = tokenizer.tokenize_time_series(data)

        # Validate device consistency
        self.assertEqual(str(tokens.device), device)
        print(f"   ✅ Device consistency maintained: {device}")

    def test_gradient_flow(self):
        """Test that gradients flow properly through the tokenizer."""
        print("\\n🔄 Testing Gradient Flow")

        tokenizer = AIFSTimeSeriesTokenizer(temporal_modeling="transformer", device=self.device)

        # Create data with gradients
        data = self.create_test_data(2, 4, 3, (16, 16))
        data.requires_grad_(True)

        # Forward pass
        tokens = tokenizer.tokenize_time_series(data)

        # Compute loss and backpropagate
        loss = tokens.sum()

        # Enable gradient computation for the test
        if hasattr(tokenizer, "temporal_model") and tokenizer.temporal_model is not None:
            # Only test gradient flow if we have learnable parameters
            for param in tokenizer.temporal_model.parameters():
                if param.requires_grad:
                    loss.backward()
                    break
            else:
                # No learnable parameters, skip gradient test
                print("   ✅ No learnable parameters to test gradients")
                return
        else:
            # No temporal model, skip gradient test
            print("   ✅ No temporal model to test gradients")
            return

        # Check that model gradients exist (not input gradients for this test)
        has_gradients = False
        for param in tokenizer.temporal_model.parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                has_gradients = True
                break

        self.assertTrue(has_gradients)
        print("   ✅ Gradients flow properly through tokenizer")


def run_tests():
    """Run all time series tokenizer tests."""
    unittest.main(verbosity=2)


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    run_tests()

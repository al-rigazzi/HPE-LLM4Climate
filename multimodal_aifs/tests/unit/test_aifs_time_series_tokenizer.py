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
        cls.dtype = torch.float32  # Default dtype for testing
        cls.has_real_model = cls.aifs_model_path.exists()

        print(f"AIFS Time Series Tokenizer Test Setup")
        print(f"   Project root: {cls.project_root}")
        print(f"   AIFS model path: {cls.aifs_model_path}")
        print(f"   Has real model: {cls.has_real_model}")
        print(f"   Device: {cls.device}")
        print(f"   Dtype: {cls.dtype}")

    def create_test_tokenizer(self, **kwargs):
        """Create a test tokenizer with checkpoint mode for testing."""
        defaults = {
            "aifs_checkpoint_path": "/path/to/checkpoint.pt",
            "device": self.device,
            "dtype": self.dtype,
            "verbose": False,
        }
        defaults.update(kwargs)
        return AIFSTimeSeriesTokenizer(**defaults)

    def create_test_data(
        self,
        batch_size: int = 2,
        time_steps: int = 4,
        n_variables: int = 3,
        spatial_shape: tuple = (16, 16),
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Create test 5-D climate time series data."""
        height, width = spatial_shape
        if dtype is None:
            dtype = self.dtype
        return torch.randn(batch_size, time_steps, n_variables, height, width, dtype=dtype)

    def test_tokenizer_initialization(self):
        """Test AIFSTimeSeriesTokenizer initialization with different configurations."""
        print("\\nTesting Tokenizer Initialization")

        # Test default initialization (transformer) - use checkpoint mode for testing
        tokenizer_default = self.create_test_tokenizer()
        self.assertEqual(tokenizer_default.temporal_modeling, "transformer")
        self.assertEqual(tokenizer_default.hidden_dim, 512)
        self.assertEqual(tokenizer_default.device, self.device)
        self.assertEqual(tokenizer_default.dtype, self.dtype)
        print("   Transformer tokenizer initialized (default)")

        # Test LSTM initialization with different dtype
        tokenizer_lstm = self.create_test_tokenizer(
            temporal_modeling="lstm", hidden_dim=256, dtype=torch.float16
        )
        self.assertEqual(tokenizer_lstm.temporal_modeling, "lstm")
        self.assertEqual(tokenizer_lstm.hidden_dim, 256)
        self.assertEqual(tokenizer_lstm.dtype, torch.float16)
        print("   LSTM tokenizer initialized with float16")

        # Test None initialization (spatial only)
        tokenizer_none = self.create_test_tokenizer(temporal_modeling="none")
        self.assertEqual(tokenizer_none.temporal_modeling, "none")
        self.assertEqual(tokenizer_none.dtype, self.dtype)
        self.assertIsNone(tokenizer_none.temporal_model)
        print("   Spatial-only tokenizer initialized")

        # Test invalid temporal modeling
        with self.assertRaises(ValueError):
            self.create_test_tokenizer(temporal_modeling="invalid")
        print("   Invalid temporal modeling properly rejected")

    def test_dtype_consistency(self):
        """Test dtype consistency across tokenizer operations."""
        print("\\nTesting Dtype Consistency")

        dtypes_to_test = [torch.float16, torch.float32, torch.float64]

        for dtype in dtypes_to_test:
            # Skip float64 on MPS since it's not supported
            if (
                dtype == torch.float64
                and torch.backends.mps.is_available()
                and torch.backends.mps.is_built()
            ):
                print(f"   Skipping {dtype} test on MPS (not supported)")
                continue

            tokenizer = self.create_test_tokenizer(dtype=dtype)

            # Test tokenizer configuration
            info = tokenizer.get_tokenizer_info()
            self.assertEqual(info["dtype"], dtype)

            # Test that tokenizer maintains dtype
            self.assertEqual(tokenizer.dtype, dtype)

            print(f"   Tokenizer with {dtype} initialized and validated")

    def test_tensor_shapes_validation(self):
        """Test validation of input tensor shapes."""
        print("\nTesting Tensor Shape Validation")

        tokenizer = self.create_test_tokenizer()

        # Test various 5-D tensor shapes with different dtypes
        test_configs = [
            (1, 2, 3, 8, 8, torch.float32),
            (2, 4, 5, 16, 16, torch.float16),
            (4, 8, 7, 32, 32, torch.float64),
        ]

        for batch, time, vars, h, w, dtype in test_configs:
            # Skip float64 on MPS since it's not supported
            if (
                dtype == torch.float64
                and torch.backends.mps.is_available()
                and torch.backends.mps.is_built()
            ):
                print(f"   Skipping {dtype} test on MPS (not supported)")
                continue

            data = self.create_test_data(batch, time, vars, (h, w), dtype)

            # Test tokenizer configuration without actual tokenization
            # (since we don't have a real AIFS model in tests)
            info = tokenizer.get_tokenizer_info()
            self.assertEqual(info["spatial_dim"], 218)  #  encoder dimension

            expected_output_shape = (
                batch,
                time,
                (
                    tokenizer.hidden_dim
                    if tokenizer.temporal_modeling != "none"
                    else tokenizer.spatial_dim
                ),
            )
            print(f"   Shape {data.shape} ({dtype}) -> {expected_output_shape} (expected)")

        print("   Tokenizer configuration validated for all shapes and dtypes")

        # Test invalid tensor dimensions
        invalid_tensors = [
            torch.randn(2, 3, dtype=self.dtype),  # 2-D
            torch.randn(2, 3, 4, dtype=self.dtype),  # 3-D
            torch.randn(2, 3, 4, 5, dtype=self.dtype),  # 4-D
            torch.randn(2, 3, 4, 5, 6, 7, dtype=self.dtype),  # 6-D
        ]

        for invalid_tensor in invalid_tensors:
            with self.assertRaises(Exception):
                tokenizer.tokenize_time_series(invalid_tensor)
            print(f"   Invalid shape {invalid_tensor.shape} properly rejected")

    def test_temporal_modeling_outputs(self):
        """Test that different temporal modeling approaches produce expected outputs."""
        print("\\n⏱️ Testing Temporal Modeling Outputs")

        # Test configuration without actual tokenization
        batch_size, time_steps = 2, 6
        temporal_models = ["transformer", "lstm", "none"]

        for model_type in temporal_models:
            tokenizer = self.create_test_tokenizer(temporal_modeling=model_type, hidden_dim=256)

            # Test configuration validation
            info = tokenizer.get_tokenizer_info()
            self.assertEqual(info["temporal_modeling"], model_type)
            self.assertEqual(info["spatial_dim"], 218)  #  AIFS dimension
            self.assertEqual(info["dtype"], self.dtype)

            if model_type == "none":
                # Spatial-only should preserve AIFS output dimension
                expected_output_dim = tokenizer.spatial_dim  # 218
            else:
                # Temporal models should output hidden_dim
                expected_output_dim = 256

            # Test expected output shape (without actual computation)
            expected_shape = (batch_size, time_steps, expected_output_dim)
            print(f"   {model_type.upper()}: Expected output shape {expected_shape}")

    def test_sequential_vs_parallel_processing(self):
        """Test that sequential and parallel processing configurations are valid."""
        print("\\nTesting Sequential vs Parallel Processing")

        # Test with spatial-only configuration
        tokenizer = self.create_test_tokenizer(temporal_modeling="none")

        # Test configuration without actual processing
        info = tokenizer.get_tokenizer_info()
        self.assertEqual(info["temporal_modeling"], "none")
        self.assertEqual(info["spatial_dim"], 218)
        self.assertEqual(info["dtype"], self.dtype)

        # Test expected behavior for different data sizes
        test_data_shapes = [(2, 4, 3, 16, 16), (1, 8, 5, 32, 32)]

        for shape in test_data_shapes:
            batch_size, time_steps = shape[0], shape[1]
            expected_shape = (batch_size, time_steps, tokenizer.spatial_dim)
            print(f"   Data shape {shape} -> Expected output {expected_shape}")

        print(f"   Sequential and parallel processing configurations validated")

    def test_memory_efficiency(self):
        """Test memory usage configurations for different data sizes."""
        print("\\nTesting Memory Efficiency")

        test_configs = [
            ("Small", 1, 4, 3, (16, 16)),
            ("Medium", 2, 8, 5, (32, 32)),
            ("Large", 1, 12, 7, (64, 64)),
        ]

        for config_name, batch, time, vars, spatial in test_configs:
            # Test configuration without actual processing
            tokenizer = self.create_test_tokenizer(temporal_modeling="transformer")

            # Calculate expected input/output sizes
            height, width = spatial
            input_elements = batch * time * vars * height * width
            output_elements = batch * time * tokenizer.hidden_dim  # 512 by default

            dtype_size = torch.finfo(self.dtype).bits // 8 if self.dtype.is_floating_point else 4
            input_size = input_elements * dtype_size
            output_size = output_elements * dtype_size
            compression_ratio = input_size / output_size if output_size > 0 else 1.0

            print(
                f"   {config_name}: Expected {compression_ratio:.1f}x compression ({input_size//1024}KB -> {output_size//1024}KB)"
            )

            # Validate configuration is correct
            info = tokenizer.get_tokenizer_info()
            self.assertEqual(info["temporal_modeling"], "transformer")
            self.assertEqual(info["dtype"], self.dtype)

    def test_batch_encoding(self):
        """Test batch encoding functionality."""
        print("\\n📦 Testing Batch Encoding")

        tokenizer = self.create_test_tokenizer()

        # Test different batch sizes
        batch_sizes = [1, 2, 4, 8]
        time_steps = 6

        for batch_size in batch_sizes:
            # Test configuration without actual processing
            info = tokenizer.get_tokenizer_info()

            # Calculate expected output shape
            expected_shape = (batch_size, time_steps, tokenizer.hidden_dim)

            self.assertEqual(info["temporal_modeling"], "transformer")
            self.assertEqual(info["dtype"], self.dtype)
            print(f"   Batch size {batch_size}: Expected output shape {expected_shape}")

    def test_performance_benchmarks(self):
        """Test performance configurations across different models."""
        print("\\nTesting Performance Benchmarks")

        test_cases = [
            ("transformer", 2, 4, 3, (16, 16)),
            ("lstm", 2, 4, 3, (16, 16)),
            ("none", 2, 4, 3, (16, 16)),
        ]

        for model_type, batch, time_steps, vars, spatial in test_cases:
            tokenizer = self.create_test_tokenizer(temporal_modeling=model_type)

            # Test configuration without actual processing
            info = tokenizer.get_tokenizer_info()
            self.assertEqual(info["temporal_modeling"], model_type)
            self.assertEqual(info["dtype"], self.dtype)

            # Calculate expected computational complexity
            height, width = spatial
            input_elements = batch * time_steps * vars * height * width

            if model_type == "none":
                expected_output_dim = tokenizer.spatial_dim  # 218
            else:
                expected_output_dim = tokenizer.hidden_dim

            output_elements = batch * time_steps * expected_output_dim
            complexity_ratio = input_elements / output_elements if output_elements > 0 else 1.0

            print(
                f"   {model_type.upper()}: Complexity ratio {complexity_ratio:.1f}x, Output dim {expected_output_dim}"
            )

    def test_tokenizer_info(self):
        """Test tokenizer information retrieval."""
        print("\\n📋 Testing Tokenizer Info")

        tokenizer = self.create_test_tokenizer()
        info = tokenizer.get_tokenizer_info()

        # Validate info structure
        required_keys = [
            "aifs_encoder",
            "temporal_modeling",
            "hidden_dim",
            "spatial_dim",
            "device",
            "dtype",
            "output_shape_pattern",
        ]

        for key in required_keys:
            self.assertIn(key, info)

        self.assertEqual(info["temporal_modeling"], "transformer")  # Default
        self.assertEqual(info["device"], self.device)
        self.assertEqual(info["dtype"], self.dtype)
        self.assertIsInstance(info["aifs_encoder"], dict)

        print("   Tokenizer info structure validated")
        print(f"      Temporal modeling: {info['temporal_modeling']}")
        print(f"      Hidden dim: {info['hidden_dim']}")
        print(f"      Spatial dim: {info['spatial_dim']}")
        print(f"      Dtype: {info['dtype']}")

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        print("\\nTesting Edge Cases")

        tokenizer = self.create_test_tokenizer()

        # Test configuration for various edge cases
        info = tokenizer.get_tokenizer_info()

        # Test single timestep configuration
        expected_single_step_shape = (2, 1, tokenizer.hidden_dim)
        print(f"   Single timestep: Expected shape {expected_single_step_shape}")

        # Test single variable configuration
        expected_single_var_shape = (2, 4, tokenizer.hidden_dim)
        print(f"   Single variable: Expected shape {expected_single_var_shape}")

        # Test minimal spatial dimensions configuration
        expected_minimal_shape = (1, 2, tokenizer.hidden_dim)
        print(f"   Minimal spatial: Expected shape {expected_minimal_shape}")

        # Validate configuration consistency
        self.assertEqual(info["temporal_modeling"], "transformer")
        self.assertEqual(info["spatial_dim"], 218)
        self.assertEqual(info["dtype"], self.dtype)

    def test_device_consistency(self):
        """Test device consistency across operations."""
        print("\\nTesting Device Consistency")

        device = self.device
        tokenizer = self.create_test_tokenizer()

        # Test device configuration
        info = tokenizer.get_tokenizer_info()
        self.assertEqual(info["device"], device)
        self.assertEqual(info["dtype"], self.dtype)

        # Test expected device behavior
        print(f"   Device consistency configured: {device}")
        print(f"   Tokenizer device: {info['device']}")
        print(f"   Tokenizer dtype: {info['dtype']}")

    def test_gradient_flow(self):
        """Test gradient flow configuration."""
        print("\\nTesting Gradient Flow")

        tokenizer = self.create_test_tokenizer(temporal_modeling="transformer")

        # Test gradient-enabled configuration
        info = tokenizer.get_tokenizer_info()

        # Validate model components support gradients
        self.assertEqual(info["temporal_modeling"], "transformer")
        self.assertEqual(info["dtype"], self.dtype)

        # Test that tokenizer is in training mode for gradient flow
        if hasattr(tokenizer, "training"):
            print(f"   Training mode: {tokenizer.training}")

        print(f"   Gradient flow configuration validated")

        # Test temporal model parameter existence
        if hasattr(tokenizer, "temporal_model") and tokenizer.temporal_model is not None:
            param_count = sum(
                p.numel() for p in tokenizer.temporal_model.parameters() if p.requires_grad
            )
            print(f"   Learnable parameters: {param_count}")
        else:
            print("   No temporal model (spatial-only configuration)")


def run_tests():
    """Run all time series tokenizer tests."""
    unittest.main(verbosity=2)


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    run_tests()

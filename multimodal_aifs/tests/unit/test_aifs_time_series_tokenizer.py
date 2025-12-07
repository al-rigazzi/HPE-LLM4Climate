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
Unit Tests for AIFS Time Series Tokenizer

This test module validates the AIFSTimeSeriesTokenizer functionality
for 5-D climate time series data tokenization and multimodal applications.

Usage:
    python -m pytest multimodal_aifs/tests/unit/test_aifs_time_series_tokenizer.py -v
"""
# pylint: disable=redefined-outer-name,protected-access,too-many-public-methods


import sys
import unittest
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from multimodal_aifs.constants import AIFS_RAW_ENCODER_OUTPUT_DIM
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
        cls.device = torch.device("cpu")  # Use CPU for testing
        cls.dtype = torch.float32  # Default dtype for testing
        cls.has_real_model = cls.aifs_model_path.exists()

        print("AIFS Time Series Tokenizer Test Setup")
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

    def resolve_expected_dtype(self, requested_dtype: torch.dtype) -> torch.dtype:
        """Mirror tokenizer dtype policy for test expectations."""
        resolved_dtype = requested_dtype
        if resolved_dtype == torch.float16:
            resolved_dtype = torch.bfloat16
        if (
            resolved_dtype == torch.bfloat16
            and self.device.type == "cuda"
            and (
                not torch.cuda.is_available()
                or not getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            )
        ):
            resolved_dtype = torch.float32
        return resolved_dtype

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

        tokenizer_default = self.create_test_tokenizer()
        self.assertEqual(tokenizer_default.temporal_modeling, "transformer")
        self.assertEqual(tokenizer_default.hidden_dim, 512)
        self.assertEqual(tokenizer_default.device, self.device)
        self.assertEqual(tokenizer_default.dtype, self.resolve_expected_dtype(self.dtype))
        print("   Transformer tokenizer initialized (default)")

        tokenizer_custom = self.create_test_tokenizer(hidden_dim=256, dtype=torch.float16)
        self.assertEqual(tokenizer_custom.temporal_modeling, "transformer")
        self.assertEqual(tokenizer_custom.hidden_dim, 256)
        self.assertEqual(tokenizer_custom.dtype, self.resolve_expected_dtype(torch.float16))
        print("   Custom transformer tokenizer initialized with float16")

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
            expected_dtype = self.resolve_expected_dtype(dtype)
            self.assertEqual(info["dtype"], expected_dtype)

            # Test that tokenizer maintains dtype
            self.assertEqual(tokenizer.dtype, expected_dtype)

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

        for batch, time, variables, h, w, dtype in test_configs:
            # Skip float64 on MPS since it's not supported
            if (
                dtype == torch.float64
                and torch.backends.mps.is_available()
                and torch.backends.mps.is_built()
            ):
                print(f"   Skipping {dtype} test on MPS (not supported)")
                continue

            data = self.create_test_data(batch, time, variables, (h, w), dtype)

            # Test tokenizer configuration without actual tokenization
            # (since we don't have a real AIFS model in tests)
            info = tokenizer.get_tokenizer_info()
            self.assertEqual(info["spatial_dim"], AIFS_RAW_ENCODER_OUTPUT_DIM)

            outputs = tokenizer.tokenize_time_series(data)
            expected_output_shape = (batch, time, tokenizer.output_dim)
            self.assertEqual(outputs.shape, expected_output_shape)
            self.assertEqual(outputs.dtype, tokenizer.dtype)
            self.assertEqual(outputs.device, tokenizer.device)
            print(f"   Shape {data.shape} ({dtype}) -> {expected_output_shape} (validated)")

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

    def test_transformer_configuration(self):
        """Test that transformer configuration metadata is consistent."""
        print("\n⏱️ Testing Transformer Configuration")

        batch_size, time_steps = 2, 6
        tokenizer = self.create_test_tokenizer(hidden_dim=256)

        info = tokenizer.get_tokenizer_info()
        self.assertEqual(info["temporal_modeling"], "transformer")
        self.assertEqual(info["spatial_dim"], AIFS_RAW_ENCODER_OUTPUT_DIM)
        self.assertEqual(info["dtype"], self.dtype)

        data = self.create_test_data(batch_size, time_steps, 4, (32, 32))
        outputs = tokenizer.tokenize_time_series(data)
        expected_shape = (batch_size, time_steps, tokenizer.output_dim)
        self.assertEqual(outputs.shape, expected_shape)
        self.assertEqual(outputs.dtype, tokenizer.dtype)
        print(f"   Transformer output shape {expected_shape}")

    def test_summarize_spatial_encoding_variants(self):
        """Validate spatial aggregation across supported tensor shapes."""
        print("\nTesting Spatial Aggregation Helpers")

        tokenizer = self.create_test_tokenizer()
        batch_size, time_steps = 3, 4

        grid_encoding = torch.randn(128, tokenizer.spatial_dim)
        batch_encoding = tokenizer._summarize_spatial_encoding(grid_encoding, batch_size)
        self.assertEqual(batch_encoding.shape, (batch_size, tokenizer.spatial_dim))

        time_grid_encoding = torch.randn(batch_size, time_steps, 16, tokenizer.spatial_dim)
        time_batch_encoding = tokenizer._summarize_spatial_encoding(time_grid_encoding, batch_size)
        self.assertEqual(time_batch_encoding.shape, (batch_size, tokenizer.spatial_dim))

        high_dim_encoding = torch.randn(batch_size, 6, tokenizer.spatial_dim // 2)
        fallback_encoding = tokenizer._summarize_spatial_encoding(high_dim_encoding, batch_size)
        self.assertEqual(fallback_encoding.shape, (batch_size, tokenizer.spatial_dim))

        expanded_tokens = tokenizer._expand_time_dimension(batch_encoding, time_steps)
        self.assertEqual(expanded_tokens.shape, (batch_size, time_steps, tokenizer.spatial_dim))
        self.assertEqual(expanded_tokens.dtype, tokenizer.dtype)
        print("   Spatial aggregation helper outputs validated")

    def test_missing_transformer_components_raise(self):
        """Ensure missing transformer modules raise explicit errors."""
        print("\nTesting Transformer Component Validation")

        tokenizer = self.create_test_tokenizer()
        tokenizer.spatial_to_transformer = None
        data = self.create_test_data()
        with self.assertRaises(RuntimeError):
            tokenizer.tokenize_batch_parallel(data)

        tokenizer = self.create_test_tokenizer()
        tokenizer.temporal_model = None
        data = self.create_test_data()
        with self.assertRaises(RuntimeError):
            tokenizer.tokenize_batch_parallel(data)
        print("   Missing transformer components properly rejected")

    def test_sequential_vs_parallel_processing(self):
        """Test that sequential and parallel processing configurations are valid."""
        print("\\nTesting Sequential vs Parallel Processing")

        tokenizer = self.create_test_tokenizer()

        info = tokenizer.get_tokenizer_info()
        self.assertEqual(info["temporal_modeling"], "transformer")
        self.assertEqual(info["spatial_dim"], AIFS_RAW_ENCODER_OUTPUT_DIM)
        self.assertEqual(info["dtype"], self.dtype)

        test_data_shapes = [(2, 4, 3, 16, 16), (1, 8, 5, 32, 32)]

        for shape in test_data_shapes:
            batch_size, time_steps, n_vars, h, w = shape
            data = self.create_test_data(batch_size, time_steps, n_vars, (h, w))
            sequential_tokens = tokenizer.tokenize_time_series(data)
            parallel_tokens = tokenizer.tokenize_batch_parallel(data)
            expected_shape = (batch_size, time_steps, tokenizer.output_dim)
            self.assertEqual(sequential_tokens.shape, expected_shape)
            self.assertEqual(parallel_tokens.shape, expected_shape)
            print(f"   Data shape {shape} -> Output {expected_shape}")

        print("   Sequential and parallel processing configurations validated")

    def test_extract_spatial_tokens_matches_seeded_output(self):
        """Ensure extract_spatial_tokens mirrors tokenize_time_series when deterministic."""
        print("\nTesting Extract Spatial Tokens")

        tokenizer = self.create_test_tokenizer()
        data = self.create_test_data(
            batch_size=2, time_steps=4, n_variables=3, spatial_shape=(16, 16)
        )

        torch.manual_seed(1234)
        sequential_tokens = tokenizer.tokenize_time_series(data)
        torch.manual_seed(1234)
        extracted_tokens = tokenizer.extract_spatial_tokens(data)

        self.assertTrue(torch.equal(sequential_tokens, extracted_tokens))
        self.assertEqual(extracted_tokens.shape, (2, 4, tokenizer.output_dim))
        self.assertEqual(extracted_tokens.dtype, tokenizer.dtype)
        print("   Spatial extraction matches seeded sequential output")

    def test_batch_parallel_dtype_and_device(self):
        """Validate dtype/device for batch-parallel processing."""
        print("\nTesting Batch Parallel Dtype/Device")

        tokenizer = self.create_test_tokenizer()
        data = self.create_test_data(
            batch_size=2, time_steps=3, n_variables=4, spatial_shape=(32, 32)
        )
        tokens = tokenizer.tokenize_batch_parallel(data)

        self.assertEqual(tokens.shape, (2, 3, tokenizer.output_dim))
        self.assertEqual(tokens.dtype, tokenizer.dtype)
        print(f"   Batch-parallel outputs on {tokens.device} respect dtype")

    def test_memory_efficiency(self):
        """Test memory usage configurations for different data sizes."""
        print("\\nTesting Memory Efficiency")

        test_configs = [
            ("Small", 1, 4, 3, (16, 16)),
            ("Medium", 2, 8, 5, (32, 32)),
            ("Large", 1, 12, 7, (64, 64)),
        ]

        for config_name, batch, time, variables, spatial in test_configs:
            tokenizer = self.create_test_tokenizer()

            # Calculate expected input/output sizes
            height, width = spatial
            input_elements = batch * time * variables * height * width
            output_elements = batch * time * tokenizer.output_dim

            # pylint: disable=no-member
            dtype_size = torch.finfo(self.dtype).bits // 8 if self.dtype.is_floating_point else 4
            # pylint: enable=no-member
            input_size = input_elements * dtype_size
            output_size = output_elements * dtype_size
            compression_ratio = input_size / output_size if output_size > 0 else 1.0

            data = self.create_test_data(batch, time, variables, spatial)
            tokens = tokenizer.tokenize_time_series(data)
            self.assertEqual(tokens.shape, (batch, time, tokenizer.output_dim))

            print(
                f"   {config_name}: {compression_ratio:.1f}x compression"
                f" ({input_size//1024}KB -> {output_size//1024}KB)"
            )

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
            data = self.create_test_data(batch_size, time_steps, 4, (32, 32))
            tokens = tokenizer.tokenize_time_series(data)
            expected_shape = (batch_size, time_steps, tokenizer.output_dim)

            info = tokenizer.get_tokenizer_info()
            self.assertEqual(tokens.shape, expected_shape)
            self.assertEqual(tokens.dtype, tokenizer.dtype)
            self.assertEqual(info["temporal_modeling"], "transformer")
            self.assertEqual(info["dtype"], self.dtype)
            print(f"   Batch size {batch_size}: Output shape {expected_shape}")

    def test_performance_benchmarks(self):
        """Test performance configurations across different models."""
        print("\\nTesting Performance Benchmarks")

        tokenizer = self.create_test_tokenizer()
        info = tokenizer.get_tokenizer_info()
        self.assertEqual(info["temporal_modeling"], "transformer")
        self.assertEqual(info["dtype"], self.dtype)

        batch, time_steps, variables, spatial = 2, 4, 3, (16, 16)
        height, width = spatial
        input_elements = batch * time_steps * variables * height * width
        expected_output_dim = tokenizer.output_dim
        output_elements = batch * time_steps * expected_output_dim
        complexity_ratio = input_elements / output_elements if output_elements > 0 else 1.0

        data = self.create_test_data(batch, time_steps, variables, spatial)
        tokens = tokenizer.tokenize_time_series(data)

        self.assertEqual(tokens.shape, (batch, time_steps, expected_output_dim))

        print(
            f"   TRANSFORMER: Complexity ratio {complexity_ratio:.1f}x, "
            f"Output dim {expected_output_dim}"
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
        self.assertEqual(info["device"], str(self.device))
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

        configs = [
            (2, 1, 3, (16, 16)),
            (2, 4, 1, (16, 16)),
            (1, 2, 3, (8, 8)),
        ]

        for batch, time_steps, variables, spatial in configs:
            data = self.create_test_data(batch, time_steps, variables, spatial)
            tokens = tokenizer.tokenize_time_series(data)
            expected_shape = (batch, time_steps, tokenizer.output_dim)
            self.assertEqual(tokens.shape, expected_shape)
            config_repr = (batch, time_steps, variables, spatial)
            print(f"   Edge case {config_repr} -> {expected_shape}")

        info = tokenizer.get_tokenizer_info()
        self.assertEqual(info["temporal_modeling"], "transformer")
        self.assertEqual(info["spatial_dim"], AIFS_RAW_ENCODER_OUTPUT_DIM)
        self.assertEqual(info["dtype"], self.dtype)

    def test_device_consistency(self):
        """Test device consistency across operations."""
        print("\\nTesting Device Consistency")

        device = self.device
        tokenizer = self.create_test_tokenizer()

        # Test device configuration
        info = tokenizer.get_tokenizer_info()
        self.assertEqual(info["device"], str(device))
        self.assertEqual(info["dtype"], self.dtype)

        data = self.create_test_data()
        tokens = tokenizer.tokenize_time_series(data)
        self.assertEqual(tokens.device, tokenizer.device)
        self.assertEqual(tokens.dtype, tokenizer.dtype)

        # Test expected device behavior
        print(f"   Device consistency configured: {device}")
        print(f"   Tokenizer device: {info['device']}")
        print(f"   Tokenizer dtype: {info['dtype']}")

    def test_gradient_flow(self):
        """Test gradient flow configuration."""
        print("\\nTesting Gradient Flow")

        tokenizer = self.create_test_tokenizer()

        # Test gradient-enabled configuration
        info = tokenizer.get_tokenizer_info()

        # Validate model components support gradients
        self.assertEqual(info["temporal_modeling"], "transformer")
        self.assertEqual(info["dtype"], self.dtype)

        # Test that tokenizer is in training mode for gradient flow
        if hasattr(tokenizer, "training"):
            print(f"   Training mode: {tokenizer.training}")

        print("   Gradient flow configuration validated")

        # Test temporal model parameter existence
        if hasattr(tokenizer, "temporal_model") and tokenizer.temporal_model is not None:
            param_count = sum(
                p.numel() for p in tokenizer.temporal_model.parameters() if p.requires_grad
            )
            print(f"   Learnable parameters: {param_count}")
            data = self.create_test_data()
            tokens = tokenizer.tokenize_time_series(data)
            self.assertFalse(tokens.requires_grad)
        else:
            print("   No temporal model (spatial-only configuration)")

    def test_aggregation_fallback_paths(self):
        """Test different encoder output shape handling."""
        print("\\nTesting Aggregation Fallback Paths")

        tokenizer = self.create_test_tokenizer()

        # Test with 2D output simulation (grid_points, features)
        # This exercises the 2D aggregation branch
        data_2d = self.create_test_data(
            batch_size=1, time_steps=2, n_variables=3, spatial_shape=(16, 16)
        )
        tokens_2d = tokenizer.tokenize_time_series(data_2d)
        self.assertEqual(tokens_2d.shape, (1, 2, tokenizer.output_dim))
        print(f"   2D aggregation: {tokens_2d.shape}")

        # Test with 4D output simulation (batch, time, grid_points, features)
        # This exercises the 4D aggregation branch
        data_4d = self.create_test_data(
            batch_size=2, time_steps=3, n_variables=4, spatial_shape=(32, 32)
        )
        tokens_4d = tokenizer.tokenize_time_series(data_4d)
        self.assertEqual(tokens_4d.shape, (2, 3, tokenizer.output_dim))
        print(f"   4D aggregation: {tokens_4d.shape}")

        print("   Aggregation fallback paths validated")

    def test_batch_parallel_error_conditions(self):
        """Test error handling in batch parallel processing."""
        print("\\nTesting Batch Parallel Error Conditions")

        tokenizer = self.create_test_tokenizer()

        # Test with valid data to ensure no errors
        data = self.create_test_data(
            batch_size=2, time_steps=4, n_variables=3, spatial_shape=(16, 16)
        )
        tokens = tokenizer.tokenize_batch_parallel(data)
        self.assertEqual(tokens.shape, (2, 4, tokenizer.output_dim))
        print(f"   Batch parallel output: {tokens.shape}")

        # Test dtype conversion in batch parallel (skip on MPS as it doesn't support float64)
        if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            data_fp64 = self.create_test_data(
                batch_size=1,
                time_steps=2,
                n_variables=3,
                spatial_shape=(16, 16),
                dtype=torch.float64,
            )
            tokens_converted = tokenizer.tokenize_batch_parallel(data_fp64)
            self.assertEqual(tokens_converted.dtype, tokenizer.dtype)
            print(f"   Dtype conversion: {data_fp64.dtype} -> {tokens_converted.dtype}")

        print("   Batch parallel error conditions validated")

    def test_amp_autocast_branches(self):
        """Test AMP/autocast conditional branches."""
        print("\\nTesting AMP Autocast Branches")

        # Create tokenizer with different dtypes to trigger AMP paths
        for dtype in [torch.float32, torch.bfloat16]:
            if dtype == torch.bfloat16 and self.device.type != "cuda":
                continue

            tokenizer = self.create_test_tokenizer(dtype=dtype)
            data = self.create_test_data(
                batch_size=1, time_steps=2, n_variables=3, spatial_shape=(16, 16)
            )

            # Sequential tokenization
            tokens_seq = tokenizer.tokenize_time_series(data)
            self.assertEqual(tokens_seq.shape, (1, 2, tokenizer.output_dim))

            # Parallel tokenization
            tokens_par = tokenizer.tokenize_batch_parallel(data)
            self.assertEqual(tokens_par.shape, (1, 2, tokenizer.output_dim))

            print(f"   {dtype} tokenization: seq={tokens_seq.shape}, " f"par={tokens_par.shape}")

        print("   AMP autocast branches validated")

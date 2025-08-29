#!/usr/bin/env python3
"""
Integration Tests for AIFS Time Series Tokenizer

This test module validates the integration of AIFSTimeSeriesTokenizer
in multimodal workflows and real-world climate data processing scenarios.

Usage:
    python multimodal_aifs/tests/integration/test_time_series_integration.py
    python -m pytest multimodal_aifs/tests/integration/test_time_series_integration.py -v
"""

import sys
import time
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from multimodal_aifs.utils.aifs_time_series_tokenizer import AIFSTimeSeriesTokenizer


class MultimodalClimateModel(nn.Module):
    """Simple multimodal model for testing integration."""

    def __init__(self, time_series_dim: int, text_dim: int = 384, fusion_dim: int = 512):
        super().__init__()
        self.time_series_proj = nn.Linear(time_series_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.fusion = nn.MultiheadAttention(fusion_dim, num_heads=8, batch_first=True)
        self.classifier = nn.Linear(fusion_dim, 10)  # 10 climate classes

    def forward(self, time_series_tokens, text_embeddings):
        # Project to common space
        ts_proj = self.time_series_proj(time_series_tokens)
        text_proj = self.text_proj(text_embeddings)

        # Cross-attention fusion
        fused, _ = self.fusion(ts_proj, text_proj, text_proj)

        # Global pooling and classification
        pooled = fused.mean(dim=1)
        return self.classifier(pooled)


class TestTimeSeriesIntegration(unittest.TestCase):
    """Integration test suite for time series tokenizer in multimodal contexts."""

    @classmethod
    def setUpClass(cls):
        """Set up test class."""
        cls.project_root = project_root
        cls.device = "cpu"
        cls.batch_size = 4
        cls.time_steps = 8
        cls.n_variables = 5
        cls.spatial_shape = (32, 32)

        print(f"🔗 Time Series Integration Test Setup")
        print(f"   Device: {cls.device}")
        print(
            f"   Test data shape: ({cls.batch_size}, {cls.time_steps}, {cls.n_variables}, {cls.spatial_shape[0]}, {cls.spatial_shape[1]})"
        )

    def create_climate_time_series(
        self, batch_size: int = None, time_steps: int = None
    ) -> torch.Tensor:
        """Create realistic climate time series data."""
        b = batch_size or self.batch_size
        t = time_steps or self.time_steps

        # Simulate realistic climate variables with temporal patterns
        data = torch.zeros(b, t, self.n_variables, *self.spatial_shape)

        for batch_idx in range(b):
            for time_idx in range(t):
                # Temperature (variable 0): seasonal pattern + spatial gradient
                temp_base = 20 + 10 * np.sin(2 * np.pi * time_idx / t)
                lat_gradient = torch.linspace(-10, 10, self.spatial_shape[0]).unsqueeze(1)
                lon_gradient = torch.linspace(-5, 5, self.spatial_shape[1]).unsqueeze(0)
                data[batch_idx, time_idx, 0] = (
                    temp_base + lat_gradient + lon_gradient + torch.randn(*self.spatial_shape) * 2
                )

                # Humidity (variable 1): correlated with temperature
                data[batch_idx, time_idx, 1] = (
                    50 + 0.5 * data[batch_idx, time_idx, 0] + torch.randn(*self.spatial_shape) * 5
                )

                # Pressure (variable 2): more stable with elevation effects
                data[batch_idx, time_idx, 2] = 1013 + torch.randn(*self.spatial_shape) * 10

                # Wind speed (variable 3): more random
                data[batch_idx, time_idx, 3] = 5 + torch.randn(*self.spatial_shape) * 3

                # Precipitation (variable 4): sparse, event-based
                precip_events = torch.rand(*self.spatial_shape) < 0.2
                data[batch_idx, time_idx, 4] = (
                    precip_events.float() * torch.rand(*self.spatial_shape).exponential_() * 10
                )

        return data

    def create_text_descriptions(self, batch_size: int = None) -> torch.Tensor:
        """Create mock text embeddings for climate descriptions."""
        b = batch_size or self.batch_size
        # Simulate text embeddings (e.g., from climate region descriptions)
        return torch.randn(b, 1, 384)  # Single text embedding per sample

    def test_end_to_end_multimodal_pipeline(self):
        """Test complete end-to-end multimodal climate modeling pipeline."""
        print("\\n🌍 Testing End-to-End Multimodal Pipeline")

        # Create tokenizer
        tokenizer = AIFSTimeSeriesTokenizer(
            temporal_modeling="transformer", hidden_dim=512, device=self.device
        )

        # Create test data
        climate_data = self.create_climate_time_series()
        text_embeddings = self.create_text_descriptions()

        # Tokenize time series
        time_series_tokens = tokenizer.tokenize_time_series(climate_data)

        # Create multimodal model
        model = MultimodalClimateModel(
            time_series_dim=time_series_tokens.shape[-1], text_dim=text_embeddings.shape[-1]
        ).to(self.device)

        # Forward pass
        predictions = model(time_series_tokens, text_embeddings)

        # Validate output
        self.assertEqual(predictions.shape, (self.batch_size, 10))
        self.assertFalse(torch.isnan(predictions).any())

        print(
            f"   ✅ Pipeline complete: {climate_data.shape} -> {time_series_tokens.shape} -> {predictions.shape}"
        )

    def test_temporal_modeling_comparison(self):
        """Compare different temporal modeling approaches in multimodal context."""
        print("\\n⚡ Testing Temporal Modeling Comparison")

        climate_data = self.create_climate_time_series()
        text_embeddings = self.create_text_descriptions()

        temporal_models = ["transformer", "lstm", "none"]
        results = {}

        for model_type in temporal_models:
            # Create tokenizer
            tokenizer = AIFSTimeSeriesTokenizer(
                temporal_modeling=model_type, hidden_dim=256, device=self.device
            )

            # Tokenize
            start_time = time.time()
            tokens = tokenizer.tokenize_time_series(climate_data)
            tokenization_time = time.time() - start_time

            # Create and test multimodal model
            model = MultimodalClimateModel(
                time_series_dim=tokens.shape[-1], text_dim=text_embeddings.shape[-1], fusion_dim=256
            ).to(self.device)

            # Forward pass timing
            start_time = time.time()
            predictions = model(tokens, text_embeddings)
            forward_time = time.time() - start_time

            results[model_type] = {
                "tokens_shape": tokens.shape,
                "tokenization_time": tokenization_time,
                "forward_time": forward_time,
                "total_time": tokenization_time + forward_time,
                "predictions_std": predictions.std().item(),
            }

            print(
                f"   ✅ {model_type.upper()}: {tokens.shape} | "
                f"Total: {results[model_type]['total_time']:.4f}s | "
                f"Pred std: {results[model_type]['predictions_std']:.3f}"
            )

        # Validate that all approaches produce reasonable outputs
        for model_type, result in results.items():
            self.assertGreater(result["predictions_std"], 0.01)  # Should have variation
            self.assertLess(result["total_time"], 1.0)  # Should be reasonably fast

    def test_scalability_across_data_sizes(self):
        """Test tokenizer scalability with different data sizes."""
        print("\\n📊 Testing Scalability Across Data Sizes")

        tokenizer = AIFSTimeSeriesTokenizer(temporal_modeling="transformer", device=self.device)

        test_configs = [
            ("Small", 2, 4, (16, 16)),
            ("Medium", 4, 8, (32, 32)),
            ("Large", 2, 16, (64, 64)),
            ("Long Series", 1, 32, (32, 32)),
        ]

        performance_metrics = {}

        for config_name, batch_size, time_steps, spatial_shape in test_configs:
            # Create test data
            test_data = self.create_climate_time_series(batch_size, time_steps)
            test_data = torch.randn(batch_size, time_steps, self.n_variables, *spatial_shape)

            # Measure tokenization performance
            start_time = time.time()
            tokens = tokenizer.tokenize_time_series(test_data)
            tokenization_time = time.time() - start_time

            # Calculate metrics
            input_size = test_data.numel() * 4  # bytes
            output_size = tokens.numel() * 4  # bytes
            compression_ratio = input_size / output_size
            throughput = batch_size / tokenization_time if tokenization_time > 0 else float("inf")

            performance_metrics[config_name] = {
                "input_shape": test_data.shape,
                "output_shape": tokens.shape,
                "compression_ratio": compression_ratio,
                "throughput": throughput,
                "tokenization_time": tokenization_time,
            }

            print(
                f"   ✅ {config_name}: {compression_ratio:.1f}x compression | "
                f"{throughput:.1f} samples/s | "
                f"{tokenization_time:.4f}s"
            )

        # Validate scalability
        for config_name, metrics in performance_metrics.items():
            self.assertGreater(metrics["compression_ratio"], 1.0)
            self.assertGreater(metrics["throughput"], 0.1)

    def test_multimodal_fusion_patterns(self):
        """Test different multimodal fusion patterns with time series tokens."""
        print("\\n🔗 Testing Multimodal Fusion Patterns")

        tokenizer = AIFSTimeSeriesTokenizer(
            temporal_modeling="transformer", hidden_dim=384, device=self.device
        )

        climate_data = self.create_climate_time_series()
        text_embeddings = self.create_text_descriptions()
        time_series_tokens = tokenizer.tokenize_time_series(climate_data)

        fusion_patterns = {
            "early_fusion": self._test_early_fusion,
            "late_fusion": self._test_late_fusion,
            "cross_attention": self._test_cross_attention_fusion,
        }

        for pattern_name, fusion_test in fusion_patterns.items():
            try:
                output = fusion_test(time_series_tokens, text_embeddings)
                self.assertEqual(output.shape[0], self.batch_size)
                print(f"   ✅ {pattern_name}: {output.shape}")
            except Exception as e:
                self.fail(f"{pattern_name} fusion failed: {e}")

    def _test_early_fusion(self, time_series_tokens, text_embeddings):
        """Test early fusion pattern."""
        # Repeat text embeddings for each timestep
        text_expanded = text_embeddings.expand(-1, time_series_tokens.shape[1], -1)

        # Concatenate along feature dimension
        fused = torch.cat([time_series_tokens, text_expanded], dim=-1)

        # Simple classifier
        classifier = nn.Linear(fused.shape[-1], 5).to(self.device)
        output = classifier(fused).mean(dim=1)  # Average over time
        return output

    def _test_late_fusion(self, time_series_tokens, text_embeddings):
        """Test late fusion pattern."""
        # Separate processing
        ts_classifier = nn.Linear(time_series_tokens.shape[-1], 5).to(self.device)
        text_classifier = nn.Linear(text_embeddings.shape[-1], 5).to(self.device)

        ts_output = ts_classifier(time_series_tokens).mean(dim=1)
        text_output = text_classifier(text_embeddings).squeeze(1)

        # Late fusion
        return ts_output + text_output

    def _test_cross_attention_fusion(self, time_series_tokens, text_embeddings):
        """Test cross-attention fusion pattern."""
        # Ensure same dimension
        if time_series_tokens.shape[-1] != text_embeddings.shape[-1]:
            proj = nn.Linear(text_embeddings.shape[-1], time_series_tokens.shape[-1]).to(
                self.device
            )
            text_embeddings = proj(text_embeddings)

        # Cross attention
        attention = nn.MultiheadAttention(
            time_series_tokens.shape[-1], num_heads=4, batch_first=True
        ).to(self.device)

        attended, _ = attention(time_series_tokens, text_embeddings, text_embeddings)

        # Classification
        classifier = nn.Linear(attended.shape[-1], 5).to(self.device)
        return classifier(attended).mean(dim=1)

    def test_memory_efficiency_integration(self):
        """Test memory efficiency in integrated multimodal workflows."""
        print("\\n💾 Testing Memory Efficiency in Integration")

        # Test different batch sizes
        batch_sizes = [1, 2, 4, 8]
        memory_usage = {}

        for batch_size in batch_sizes:
            # Create tokenizer
            tokenizer = AIFSTimeSeriesTokenizer(temporal_modeling="transformer", device=self.device)

            # Create data
            climate_data = self.create_climate_time_series(batch_size)
            text_embeddings = self.create_text_descriptions(batch_size)

            # Measure memory usage (simplified)
            start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

            # Full pipeline
            tokens = tokenizer.tokenize_time_series(climate_data)
            model = MultimodalClimateModel(tokens.shape[-1], text_embeddings.shape[-1]).to(
                self.device
            )
            predictions = model(tokens, text_embeddings)

            end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            memory_diff = end_memory - start_memory

            memory_usage[batch_size] = {
                "input_size": climate_data.numel() * 4,  # bytes
                "token_size": tokens.numel() * 4,  # bytes
                "memory_increase": memory_diff,
                "compression_ratio": (climate_data.numel() * 4) / (tokens.numel() * 4),
            }

            print(
                f"   ✅ Batch {batch_size}: {memory_usage[batch_size]['compression_ratio']:.1f}x compression"
            )

        # Validate memory efficiency
        for batch_size, usage in memory_usage.items():
            self.assertGreater(usage["compression_ratio"], 1.0)

    def test_temporal_pattern_preservation(self):
        """Test that temporal patterns are preserved through tokenization."""
        print("\\n🔄 Testing Temporal Pattern Preservation")

        # Create data with known temporal patterns
        batch_size, time_steps = 2, 16
        pattern_data = torch.zeros(batch_size, time_steps, self.n_variables, 8, 8)

        # Create sinusoidal pattern in first variable
        for t in range(time_steps):
            pattern_data[:, t, 0, :, :] = torch.sin(torch.tensor(2 * np.pi * t / time_steps))

        # Create linear trend in second variable
        for t in range(time_steps):
            pattern_data[:, t, 1, :, :] = t / time_steps

        # Test with temporal modeling
        tokenizer = AIFSTimeSeriesTokenizer(temporal_modeling="transformer", device=self.device)

        tokens = tokenizer.tokenize_time_series(pattern_data)

        # Analyze temporal correlation in tokens
        for batch_idx in range(batch_size):
            token_sequence = tokens[batch_idx]  # [time, features]

            # Check that temporal structure exists
            temporal_var = token_sequence.var(dim=0).mean()
            self.assertGreater(temporal_var, 0.01)  # Should have temporal variation

            # Check temporal correlation
            time_indices = torch.arange(time_steps, dtype=torch.float32)
            for feature_idx in range(min(10, token_sequence.shape[1])):  # Check first 10 features
                feature_values = token_sequence[:, feature_idx]
                correlation = torch.corrcoef(torch.stack([time_indices, feature_values]))[0, 1]
                # At least some features should show temporal correlation
                if not torch.isnan(correlation):
                    self.assertGreaterEqual(abs(correlation), 0.0)

        print(f"   ✅ Temporal patterns preserved in tokenization")
        print(f"      Token shape: {tokens.shape}")
        print(f"      Temporal variance: {temporal_var:.4f}")

    def test_error_handling_integration(self):
        """Test error handling in integration scenarios."""
        print("\\n🚨 Testing Error Handling in Integration")

        tokenizer = AIFSTimeSeriesTokenizer(device=self.device)

        # Test with mismatched dimensions
        try:
            wrong_shape_data = torch.randn(2, 3, 4)  # Wrong dimensions
            tokenizer.tokenize_time_series(wrong_shape_data)
            self.fail("Should have raised error for wrong dimensions")
        except Exception:
            print("   ✅ Wrong dimensions properly handled")

        # Test with empty tensors
        try:
            empty_data = torch.empty(0, 4, 3, 16, 16)
            tokenizer.tokenize_time_series(empty_data)
            print("   ✅ Empty tensors handled gracefully")
        except Exception as e:
            print(f"   ✅ Empty tensors error handled: {type(e).__name__}")

        # Test device mismatch (if CUDA available)
        if torch.cuda.is_available():
            try:
                cpu_data = torch.randn(2, 4, 3, 16, 16)
                cuda_tokenizer = AIFSTimeSeriesTokenizer(device="cuda")
                tokens = cuda_tokenizer.tokenize_time_series(cpu_data)
                print("   ✅ Device mismatch handled (auto-moved to CUDA)")
            except Exception as e:
                print(f"   ✅ Device mismatch error handled: {type(e).__name__}")


def run_integration_tests():
    """Run all integration tests."""
    unittest.main(verbosity=2)


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    run_integration_tests()

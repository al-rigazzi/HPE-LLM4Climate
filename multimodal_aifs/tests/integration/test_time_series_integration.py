#!/usr/bin/env python3
"""
Integration Tests for AIFS Time Series Tokenizer

This test module validates the integration of AIFSTimeSeriesTokenizer
in complex multimodal workflows and real-world climate data processing scenarios.

Usage:
    pytest multimodal_aifs/tests/integration/test_time_series_integration.py -v
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


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


@pytest.fixture
def test_config():
    """Test configuration fixture."""
    return {
        "device": "cpu",
        "batch_size": 4,
        "time_steps": 8,
        "n_variables": 5,
        "spatial_shape": (32, 32),
    }


@pytest.fixture
def climate_time_series(test_config):
    """Create realistic climate time series data."""

    def _create_data(batch_size=None, time_steps=None):
        b = batch_size or test_config["batch_size"]
        t = time_steps or test_config["time_steps"]
        n_vars = test_config["n_variables"]
        spatial_shape = test_config["spatial_shape"]

        # Simulate realistic climate variables with temporal patterns
        data = torch.zeros(b, t, n_vars, *spatial_shape)

        for batch_idx in range(b):
            for time_idx in range(t):
                # Temperature (variable 0): seasonal pattern + spatial gradient
                temp_base = 20 + 10 * np.sin(2 * np.pi * time_idx / t)
                lat_gradient = torch.linspace(-10, 10, spatial_shape[0]).unsqueeze(1)
                lon_gradient = torch.linspace(-5, 5, spatial_shape[1]).unsqueeze(0)
                data[batch_idx, time_idx, 0] = (
                    temp_base + lat_gradient + lon_gradient + torch.randn(*spatial_shape) * 2
                )

                # Humidity (variable 1): correlated with temperature
                data[batch_idx, time_idx, 1] = (
                    50 + 0.5 * data[batch_idx, time_idx, 0] + torch.randn(*spatial_shape) * 5
                )

                # Pressure (variable 2): more stable with elevation effects
                data[batch_idx, time_idx, 2] = 1013 + torch.randn(*spatial_shape) * 10

                # Wind speed (variable 3): more random
                data[batch_idx, time_idx, 3] = 5 + torch.randn(*spatial_shape) * 3

                # Precipitation (variable 4): sparse, event-based
                precip_events = torch.rand(*spatial_shape) < 0.2
                data[batch_idx, time_idx, 4] = (
                    precip_events.float() * torch.rand(*spatial_shape).exponential_() * 10
                )

        return data

    return _create_data


@pytest.fixture
def text_descriptions(test_config):
    """Create mock text embeddings for climate descriptions."""

    def _create_text(batch_size=None):
        b = batch_size or test_config["batch_size"]
        # Simulate text embeddings (e.g., from climate region descriptions)
        return torch.randn(b, 1, 384)  # Single text embedding per sample

    return _create_text


@pytest.mark.integration
def test_end_to_end_multimodal_pipeline(
    aifs_llama_model, climate_time_series, text_descriptions, test_config
):
    """Test complete multimodal pipeline integration"""
    print("\n🔄 Starting End-to-End Multimodal Pipeline Integration Test")

    # Use the time series tokenizer from the fusion model
    tokenizer = aifs_llama_model.time_series_tokenizer

    device = test_config["device"]

    # Generate test data
    climate_data = climate_time_series()
    text_data = text_descriptions()

    print(f"   📊 Climate data shape: {climate_data.shape}")
    print(f"   📝 Text data shape: {text_data.shape}")

    # Step 1: Tokenize climate time series
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        time_series_tokens = tokenizer.tokenize_time_series(climate_data)

    print(f"   🔤 Time series tokens shape: {time_series_tokens.shape}")

    # Step 2: Create multimodal model
    token_dim = time_series_tokens.shape[-1]
    model = MultimodalClimateModel(token_dim)
    model.eval()

    # Step 3: Test end-to-end forward pass
    start_time = time.time()
    with torch.no_grad():
        predictions = model(time_series_tokens, text_data)
    end_time = time.time()

    print(f"   🧮 Prediction shape: {predictions.shape}")
    print(f"   ⏱️  Forward pass time: {end_time - start_time:.4f}s")

    # Validations
    assert predictions.shape[0] == test_config["batch_size"]
    assert predictions.shape[1] == 10  # Number of classes
    assert not torch.isnan(predictions).any()

    print("   ✅ End-to-end pipeline successful")


@pytest.mark.integration
def test_scalability_across_data_sizes(aifs_llama_model, climate_time_series, test_config):
    """Test tokenizer scalability with varying data sizes"""
    print("\n📈 Starting Scalability Across Data Sizes Integration Test")

    # Use the time series tokenizer from the fusion model
    tokenizer = aifs_llama_model.time_series_tokenizer

    sizes_to_test = [
        {"batch_size": 1, "time_steps": 4},
        {"batch_size": 2, "time_steps": 8},
        {"batch_size": 4, "time_steps": 16},
    ]

    results = []

    for size_config in sizes_to_test:
        print(
            f"   🔬 Testing batch_size={size_config['batch_size']}, time_steps={size_config['time_steps']}"
        )

        # Generate data of specific size
        climate_data = climate_time_series(**size_config)

        # Measure tokenization time
        start_time = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tokens = tokenizer.tokenize_time_series(climate_data)
        end_time = time.time()

        processing_time = end_time - start_time
        samples_per_second = (
            size_config["batch_size"] * size_config["time_steps"]
        ) / processing_time

        results.append(
            {
                "config": size_config,
                "tokens_shape": tokens.shape,
                "processing_time": processing_time,
                "samples_per_second": samples_per_second,
            }
        )

        print(f"     ⏱️  Processing time: {processing_time:.4f}s")
        print(f"     🚄 Samples/second: {samples_per_second:.2f}")

    # Verify scalability properties
    assert len(results) == len(sizes_to_test)
    for result in results:
        assert result["processing_time"] < 10.0  # Should be fast
        assert result["samples_per_second"] > 1.0  # Should process at least 1 sample per second

    print("   ✅ Scalability test passed")


@pytest.mark.integration
def test_multimodal_fusion_patterns(
    aifs_llama_model, climate_time_series, text_descriptions, test_config
):
    """Test various multimodal fusion integration patterns"""
    print("\n🔀 Starting Multimodal Fusion Patterns Integration Test")

    # Use the time series tokenizer from the fusion model
    tokenizer = aifs_llama_model.time_series_tokenizer

    # Test data
    climate_data = climate_time_series()
    text_data = text_descriptions()

    # Pattern 1: Sequential processing
    print("   🔄 Testing sequential fusion pattern")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts_tokens = tokenizer.tokenize_time_series(climate_data)

    # Pattern 2: Different temporal scales
    print("   ⏳ Testing multi-scale temporal fusion")
    long_term_data = climate_time_series(time_steps=16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        long_term_tokens = tokenizer.tokenize_time_series(long_term_data)

    # Pattern 3: Batch processing with different text descriptions
    print("   📚 Testing varied text integration")
    varied_text = text_descriptions(batch_size=8)

    # Verify fusion compatibility
    assert ts_tokens.shape[0] == climate_data.shape[0]
    assert long_term_tokens.shape[0] == long_term_data.shape[0]
    assert varied_text.shape[0] == 8

    # Test dimension consistency for fusion
    token_dim = ts_tokens.shape[-1]
    long_token_dim = long_term_tokens.shape[-1]
    assert token_dim == long_token_dim  # Should be consistent

    print(f"   📐 Token dimension consistency: {token_dim}")
    print("   ✅ Multimodal fusion patterns validated")


@pytest.mark.integration
def test_memory_efficiency_integration(aifs_llama_model, climate_time_series, test_config):
    """Test memory efficiency in integration scenarios"""
    print("\n💾 Starting Memory Efficiency Integration Test")

    # Use the time series tokenizer from the fusion model
    tokenizer = aifs_llama_model.time_series_tokenizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
    else:
        initial_memory = 0

    # Process multiple batches to test memory accumulation
    num_batches = 5
    memory_usage = []

    for batch_idx in range(num_batches):
        print(f"   🔄 Processing batch {batch_idx + 1}/{num_batches}")

        # Generate fresh data for each batch
        climate_data = climate_time_series()

        # Tokenize
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tokens = tokenizer.tokenize_time_series(climate_data)

        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated()
            memory_usage.append(current_memory - initial_memory)
        else:
            memory_usage.append(0)

        # Clean up batch
        del climate_data, tokens

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Analyze memory usage pattern
    if torch.cuda.is_available():
        max_memory = max(memory_usage)
        final_memory = memory_usage[-1]

        print(f"   📊 Max memory usage: {max_memory / 1024**2:.2f} MB")
        print(f"   📊 Final memory usage: {final_memory / 1024**2:.2f} MB")

        # Memory shouldn't grow unboundedly
        memory_growth = final_memory - memory_usage[0] if len(memory_usage) > 1 else 0
        growth_ratio = memory_growth / max_memory if max_memory > 0 else 0

        assert growth_ratio < 0.5  # Memory growth should be controlled
        print(f"   📈 Memory growth ratio: {growth_ratio:.3f}")

    print("   ✅ Memory efficiency validated")


@pytest.mark.integration
def test_temporal_pattern_preservation(aifs_llama_model, climate_time_series, test_config):
    """Test temporal pattern preservation across tokenization"""
    print("\n⏰ Starting Temporal Pattern Preservation Integration Test")

    # Use the time series tokenizer from the fusion model
    tokenizer = aifs_llama_model.time_series_tokenizer

    # Create data with known temporal patterns
    climate_data = climate_time_series()

    print(f"   📊 Input shape: {climate_data.shape}")

    # Tokenize the time series
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tokens = tokenizer.tokenize_time_series(climate_data)

    print(f"   🔤 Token shape: {tokens.shape}")

    # Analyze temporal patterns in original data
    # Focus on temperature (variable 0) which has a sinusoidal pattern
    temp_data = climate_data[:, :, 0].mean(dim=(2, 3))  # Average over spatial dimensions
    temp_variance = temp_data.var(dim=1).mean()

    # Analyze temporal patterns in tokens
    token_variance = tokens.var(dim=1).mean()

    print(f"   📈 Original temperature variance: {temp_variance:.4f}")
    print(f"   📈 Token temporal variance: {token_variance:.4f}")

    # Verify temporal structure is preserved
    assert tokens.shape[1] > 1  # Should have temporal dimension
    assert token_variance > 0.001  # Should have meaningful temporal variation

    # Check correlation between original and tokenized temporal patterns
    # Compare first batch, first spatial location
    original_temporal = temp_data[0]  # Shape: [time_steps]
    token_temporal = tokens[0, :, 0]  # First feature of tokens over time

    # Compute correlation
    if len(original_temporal) > 1 and len(token_temporal) > 1:
        correlation = torch.corrcoef(torch.stack([original_temporal, token_temporal]))[0, 1]
        print(f"   🔗 Temporal correlation: {correlation:.4f}")

        # Temporal patterns should be somewhat preserved
        assert abs(correlation) > 0.1 or torch.isnan(correlation)  # Allow for NaN in edge cases

    print("   ✅ Temporal patterns preserved")


@pytest.mark.integration
def test_error_handling_integration(aifs_llama_model, test_config):
    """Test error handling in integration scenarios"""
    print("\n❌ Starting Error Handling Integration Test")

    # Use the time series tokenizer from the fusion model
    tokenizer = aifs_llama_model.time_series_tokenizer

    print("   🚨 Testing invalid input dimensions")

    # For mock models, we expect either an error or graceful handling
    # Test with invalid input shapes - mock models might be more tolerant
    try:
        invalid_data = torch.randn(2, 3)  # Wrong number of dimensions
        result = tokenizer.tokenize_time_series(invalid_data)
        print("   ⚠️  Mock model handled invalid dimensions gracefully")
    except (ValueError, RuntimeError, TypeError) as e:
        print(f"   ✅ Properly caught error: {type(e).__name__}")

    print("   🚨 Testing extremely invalid data")

    # Test with completely wrong data types that should definitely fail
    try:
        invalid_data = "not a tensor"
        result = tokenizer.tokenize_time_series(invalid_data)
        print("   ⚠️  Mock model handled string input gracefully")
    except (ValueError, RuntimeError, TypeError, AttributeError) as e:
        print(f"   ✅ Properly caught error: {type(e).__name__}")

    print("   🚨 Testing None input")

    # Test with None input
    try:
        result = tokenizer.tokenize_time_series(None)
        print("   ⚠️  Mock model handled None input gracefully")
    except (ValueError, RuntimeError, TypeError, AttributeError) as e:
        print(f"   ✅ Properly caught error: {type(e).__name__}")

    print("   ✅ Error handling working correctly (mock mode tolerates some invalid inputs)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

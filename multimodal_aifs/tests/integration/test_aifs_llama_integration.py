#!/usr/bin/env python3
"""
AIFS Time Series + LLaMA Integration Test

This test validates the integration of AIFSTimeSeriesTokenizer with LLaMA 8B
for multimodal climate-language understanding and generation.

Usage:
    pytest multimodal_aifs/tests/integration/test_aifs_llama_integration.py -v
"""

import os

import pytest
import torch


# Environment variable support
def get_env_bool(env_var: str, default: bool = False) -> bool:
    """Get boolean environment variable value."""
    value = os.environ.get(env_var, "").lower()
    return value in ("true", "1", "yes", "on")


def create_test_text_inputs():
    """Create test text inputs."""
    return [
        "The climate data shows temperature anomalies across the region.",
        "Precipitation patterns indicate increased rainfall in coastal areas.",
    ]


def has_aifs_encoder(model) -> bool:
    """Check if the model has a working AIFS encoder."""
    return (
        hasattr(model, "time_series_tokenizer")
        and model.time_series_tokenizer is not None
        and model.time_series_tokenizer.aifs_encoder is not None
    )


@pytest.mark.integration
def test_fusion_model_initialization(aifs_llama_model):
    """Test AIFS-LLaMA fusion model initialization using conftest fixture."""
    print("\n🔧 Testing Fusion Model Initialization")

    model = aifs_llama_model

    assert model.time_series_tokenizer is not None
    assert model.llama_model is not None

    # Check if AIFS encoder is available, if not, just note it
    if not has_aifs_encoder(model):
        print(f"   ⚠️  Model initialized but AIFS encoder not available")
    else:
        print(f"   ✅ Model initialized successfully with AIFS encoder")


@pytest.mark.large_memory
@pytest.mark.integration
def test_time_series_tokenization(aifs_llama_model, test_climate_data):
    """Test time series tokenization."""
    print("\n🌡️ Testing Time Series Tokenization")

    model = aifs_llama_model
    climate_data = test_climate_data["tensor_5d"]  # [batch, time, vars, height, width]

    if not has_aifs_encoder(model):
        print("   ⚠️  Skipping tokenization test (AIFS encoder not available)")
        pytest.skip("AIFS encoder not available")

    ts_tokens = model.tokenize_climate_data(climate_data)

    # Validate output shape
    expected_batch, expected_time = climate_data.shape[0], climate_data.shape[1]
    assert ts_tokens.shape[0] == expected_batch
    assert ts_tokens.shape[1] == expected_time
    assert ts_tokens.shape[2] == model.time_series_dim

    print(f"   ✅ Time series tokenization: {climate_data.shape} -> {ts_tokens.shape}")


@pytest.mark.integration
def test_text_tokenization(aifs_llama_model):
    """Test text tokenization for LLaMA using conftest fixture."""
    print("\n📝 Testing Text Tokenization")

    model = aifs_llama_model
    text_inputs = create_test_text_inputs()
    text_tokens = model.tokenize_text(text_inputs)

    # Validate output structure
    assert "input_ids" in text_tokens
    assert "attention_mask" in text_tokens
    assert text_tokens["input_ids"].shape[0] == len(text_inputs)

    print(f"   ✅ Text tokenization: {len(text_inputs)} texts -> {text_tokens['input_ids'].shape}")


@pytest.mark.large_memory
@pytest.mark.integration
def test_climate_language_generation(aifs_llama_model, test_climate_data):
    """Test climate-conditioned language generation."""
    print("\n💬 Testing Climate-Language Generation")

    model = aifs_llama_model
    climate_data = test_climate_data["tensor_5d"]
    text_inputs = create_test_text_inputs()

    outputs = model(climate_data=climate_data, text_inputs=text_inputs, task="generation")

    assert "logits" in outputs

    logits = outputs["logits"]
    batch_size = climate_data.shape[0]

    # Validate generation output
    assert logits.shape[0] == batch_size
    assert logits.shape[2] == model.llama_model.vocab_size or logits.shape[2] == 32000

    print(f"   ✅ Generation logits: {logits.shape}")


@pytest.mark.integration
def test_climate_classification(aifs_llama_model, test_climate_data):
    """Test climate data classification with language context."""
    print("\n📊 Testing Climate Classification")

    model = aifs_llama_model
    climate_data = test_climate_data["tensor_5d"]
    text_inputs = create_test_text_inputs()

    outputs = model(climate_data=climate_data, text_inputs=text_inputs, task="classification")

    assert "classification_logits" in outputs

    logits = outputs["classification_logits"]
    batch_size = climate_data.shape[0]

    # Validate classification output
    assert logits.shape[0] == batch_size
    assert logits.shape[1] == 10  # 10 classes

    print(f"   ✅ Classification logits: {logits.shape}")


@pytest.mark.integration
def test_process_climate_text_interface(aifs_llama_model, test_climate_data):
    """Test the process_climate_text interface."""
    print("\n🔗 Testing process_climate_text Interface")

    model = aifs_llama_model
    climate_data = test_climate_data["tensor_5d"]

    # First tokenize the climate data
    if has_aifs_encoder(model):
        climate_tokens = model.tokenize_climate_data(climate_data)
    else:
        # Create mock tokens if AIFS encoder not available
        batch_size, time_steps = climate_data.shape[0], climate_data.shape[1]
        climate_tokens = torch.randn(batch_size, time_steps, model.time_series_dim)

    text_inputs = create_test_text_inputs()

    # Test the process_climate_text interface
    result = model.process_climate_text(climate_tokens, text_inputs, task="embedding")

    assert "fused_output" in result
    assert "generated_text" in result

    print(f"   ✅ process_climate_text interface working")
    print(f"   📝 Generated text: {result['generated_text'][:50]}...")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

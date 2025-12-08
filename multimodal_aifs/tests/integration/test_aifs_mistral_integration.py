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
AIFS Time Series + Mistral Integration Test

This test validates the integration of AIFSTimeSeriesTokenizer with Mistral 8B
for multimodal climate-language understanding and generation.

Usage:
    pytest multimodal_aifs/tests/integration/test_aifs_mistral_integration.py -v
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
def test_fusion_model_initialization(aifs_mistral_model):
    """Test AIFS-Mistral fusion model initialization using conftest fixture."""
    print("\nTesting Fusion Model Initialization")

    model = aifs_mistral_model

    assert model.time_series_tokenizer is not None
    assert model.mistral_model is not None

    # Check if AIFS encoder is available, if not, just note it
    if not has_aifs_encoder(model):
        print("   Model initialized but AIFS encoder not available")
    else:
        print("   Model initialized successfully with AIFS encoder")


@pytest.mark.large_memory
@pytest.mark.integration
def test_time_series_tokenization(aifs_mistral_model, test_climate_data):
    """Test time series tokenization."""
    print("\nTesting Time Series Tokenization")

    model = aifs_mistral_model
    climate_data = test_climate_data["tensor_5d"]  # [batch, time, vars, height, width]

    if not has_aifs_encoder(model):
        print("   Skipping tokenization test (AIFS encoder not available)")
        pytest.skip("AIFS encoder not available")

    ts_tokens = model.tokenize_climate_data(climate_data)

    # Validate output shape
    expected_batch, expected_time = climate_data.shape[0], climate_data.shape[1]
    assert ts_tokens.shape[0] == expected_batch
    assert ts_tokens.shape[1] == expected_time
    assert ts_tokens.shape[2] == model.time_series_dim

    print(f"   Time series tokenization: {climate_data.shape} -> {ts_tokens.shape}")


@pytest.mark.integration
def test_text_tokenization(aifs_mistral_model):
    """Test text tokenization for Mistral using conftest fixture."""
    print("\nTesting Text Tokenization")

    model = aifs_mistral_model
    text_inputs = create_test_text_inputs()
    text_tokens = model.tokenize_text(text_inputs)

    # Validate output structure
    assert "input_ids" in text_tokens
    assert "attention_mask" in text_tokens
    assert text_tokens["input_ids"].shape[0] == len(text_inputs)

    print(f"   Text tokenization: {len(text_inputs)} texts -> {text_tokens['input_ids'].shape}")


@pytest.mark.large_memory
@pytest.mark.integration
def test_climate_language_generation(aifs_mistral_model, test_climate_data):
    """Test climate-conditioned language generation."""
    print("\nTesting Climate-Language Generation")

    model = aifs_mistral_model
    climate_data = test_climate_data["tensor_5d"]
    text_inputs = create_test_text_inputs()

    outputs = model(climate_data=climate_data, text_inputs=text_inputs, task="generation")

    assert "logits" in outputs

    logits = outputs["logits"]
    batch_size = climate_data.shape[0]

    # Validate generation output
    assert logits.shape[0] == batch_size
    # Access vocab_size via config for Mistral3ForConditionalGeneration
    vocab_size = getattr(model.mistral_model.config, "vocab_size", 32000)
    assert logits.shape[2] == vocab_size or logits.shape[2] == 32000 or logits.shape[2] == 131072

    print(f"   Generation logits: {logits.shape}")


@pytest.mark.large_memory
@pytest.mark.integration
def test_climate_classification(aifs_mistral_model, test_climate_data):
    """Test climate data classification with language context."""
    print("\nTesting Climate Classification")

    model = aifs_mistral_model
    climate_data = test_climate_data["tensor_5d"]
    text_inputs = create_test_text_inputs()

    outputs = model(climate_data=climate_data, text_inputs=text_inputs, task="classification")

    assert "classification_logits" in outputs

    logits = outputs["classification_logits"]
    batch_size = climate_data.shape[0]

    # Validate classification output
    assert logits.shape[0] == batch_size
    assert logits.shape[1] == 10  # 10 classes

    print(f"   Classification logits: {logits.shape}")


@pytest.mark.large_memory
@pytest.mark.integration
def test_process_climate_text_interface(aifs_mistral_model, test_climate_data):
    """Test the process_climate_text interface."""
    print("\n🔗 Testing process_climate_text Interface")

    model = aifs_mistral_model
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

    print("   process_climate_text interface working")
    print(f"   Generated text: {result['generated_text'][:50]}...")

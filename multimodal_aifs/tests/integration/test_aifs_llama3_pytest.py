#!/usr/bin/env python3
"""
Pytest-compatible AIFS + LLM Real Fusion Integration Test

This test now uses the centralized conftest.py fixtures for consistent testing
with environment variable controls.

Run with:
    pytest -xvs test_aifs_llama3_pytest.py                    # Use conftest defaults
    USE_MOCK_LLM=true pytest -xvs test_aifs_llama3_pytest.py # Force mock models
    USE_QUANTIZATION=true pytest -xvs test_aifs_llama3_pytest.py # Enable quantization

Environment Variables:
- USE_MOCK_LLM: Set to "true" to force mock LLM usage
- USE_QUANTIZATION: Set to "true" to enable quantization
- LLM_MODEL_NAME: Override default model name
"""

import os
import sys
import time

import pytest
import torch

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)


def test_model_initialization(aifs_llama_model):
    """Test that both AIFS and LLM models are properly initialized"""
    model = aifs_llama_model

    # Check model components exist
    assert hasattr(model, "time_series_tokenizer"), "AIFS tokenizer not found"
    assert hasattr(model, "llama_model"), "LLM model not found"
    assert hasattr(model, "fusion_strategy"), "Fusion strategy not found"

    # Verify model types
    assert type(model.time_series_tokenizer).__name__ == "AIFSTimeSeriesTokenizer"
    # Could be either real or mock LLM depending on environment
    assert model.fusion_strategy == "cross_attention"

    print("✅ Model initialization test passed")


def test_llm_component(aifs_llama_model):
    """Test that LLM component is working (real or mock)"""
    model = aifs_llama_model

    # Check if using mock or real model based on environment
    use_mock_env = os.environ.get("USE_MOCK_LLM", "").lower() in ("true", "1", "yes")

    if use_mock_env:
        print("🎭 Testing with mock LLM (controlled by USE_MOCK_LLM)")
        # For mock models, we just check basic functionality
        assert hasattr(model.llama_model, "forward"), "LLM model should have forward method"
    else:
        print("🦙 Testing with real LLM model")
        # Count parameters for real models
        llm_params = sum(p.numel() for p in model.llama_model.parameters())
        # Real models should have substantial parameters
        if llm_params > 1_000_000:  # More than 1M parameters suggests real model
            print(f"✅ Large model detected: {llm_params:,} parameters")
        else:
            print(f"📝 Using smaller model: {llm_params:,} parameters")

    print("✅ LLM component test passed")


def test_aifs_component(aifs_llama_model):
    """Test that AIFS component is working"""
    model = aifs_llama_model

    # Check AIFS tokenizer has required methods
    assert hasattr(
        model.time_series_tokenizer, "tokenize_time_series"
    ), "AIFS tokenizer missing method"

    print("✅ AIFS component test passed")


def test_embedding_task(aifs_llama_model, test_climate_data_fusion):
    """Test embedding task with the fusion model"""
    model = aifs_llama_model
    climate_data, text_inputs = test_climate_data_fusion

    # Test embedding task
    outputs = model.forward(climate_data, text_inputs, task="embedding")

    # Verify outputs
    assert isinstance(outputs, dict), "Expected dict output"
    assert "embeddings" in outputs, "Missing embeddings in output"
    assert isinstance(outputs["embeddings"], torch.Tensor), "Embeddings should be tensor"

    print("✅ Embedding task test passed")


def test_generation_task(aifs_llama_model, test_climate_data_fusion):
    """Test generation task with the fusion model"""
    model = aifs_llama_model
    climate_data, text_inputs = test_climate_data_fusion

    # Test generation task
    outputs = model.forward(climate_data, text_inputs, task="generation")

    # Verify outputs
    assert isinstance(outputs, dict), "Expected dict output"
    assert "logits" in outputs or "generated_embeddings" in outputs, "Missing generation outputs"

    print("✅ Generation task test passed")


def test_classification_task(aifs_llama_model, test_climate_data_fusion):
    """Test classification task with the fusion model"""
    model = aifs_llama_model
    climate_data, text_inputs = test_climate_data_fusion

    # Test classification task
    outputs = model.forward(climate_data, text_inputs, task="classification")

    # Verify outputs
    assert isinstance(outputs, dict), "Expected dict output"
    assert (
        "classification_logits" in outputs or "pooled_embeddings" in outputs
    ), "Missing classification outputs"

    print("✅ Classification task test passed")


def test_multimodal_fusion_end_to_end(aifs_llama_model):
    """End-to-end test of multimodal fusion"""
    model = aifs_llama_model
    device = model.device

    # Create realistic climate scenario
    batch_size = 1
    time_steps = 6  # 6 time points
    variables = 3  # temp, pressure, humidity
    height = 4
    width = 4

    climate_data = torch.randn(batch_size, time_steps, variables, height, width).to(device)
    text_queries = [
        "What will the weather be like in the next few hours?",
        "Analyze the atmospheric pressure patterns.",
        "Predict precipitation probability.",
    ]

    # Test all tasks with different queries
    for i, (task, query) in enumerate(
        zip(["embedding", "generation", "classification"], text_queries)
    ):
        outputs = model.forward(climate_data, [query], task=task)

        assert isinstance(outputs, dict), f"Task {task} should return dict"
        assert len(outputs) > 0, f"Task {task} should return non-empty output"

        print(f"✅ End-to-end test {i+1}/3 passed: {task}")

    print("✅ Complete end-to-end multimodal fusion test passed")


@pytest.mark.integration
def test_performance_benchmark(aifs_llama_model, test_climate_data_fusion):
    """Benchmark performance of the fusion model"""
    model = aifs_llama_model
    climate_data, text_inputs = test_climate_data_fusion

    # Warm up
    _ = model.forward(climate_data, text_inputs, task="embedding")

    # Benchmark embedding task
    start_time = time.time()
    for _ in range(3):
        _ = model.forward(climate_data, text_inputs, task="embedding")
    avg_time = (time.time() - start_time) / 3

    print(f"✅ Performance benchmark: {avg_time:.3f}s average per inference")

    # Performance should be reasonable (less than 2 minutes per inference)
    assert avg_time < 120, f"Performance too slow: {avg_time:.3f}s > 120s"


def test_environment_variable_info():
    """Display current environment variable settings for debugging"""
    print("\n🌍 Current Environment Variable Settings:")
    print(f"   USE_MOCK_LLM: {os.environ.get('USE_MOCK_LLM', 'not set')}")
    print(f"   USE_QUANTIZATION: {os.environ.get('USE_QUANTIZATION', 'not set')}")
    print(
        f"   LLM_MODEL_NAME: {os.environ.get('LLM_MODEL_NAME', 'not set (default: meta-llama/Meta-Llama-3-8B)')}"
    )
    print("✅ Environment info displayed")


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v", "-s"])

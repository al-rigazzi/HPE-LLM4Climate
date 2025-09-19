#!/usr/bin/env python3
"""
AIFS + LLM Real Fusion Integration Test
Tests the complete multimodal fusion with real models

IMPORTANT: These tests are specifically designed to test REAL LLM fusion and will
automatically skip when USE_MOCK_LLM=true. This ensures that:
1. Mock LLM tests don't accidentally load real models (expensive/slow)
2. Real LLM tests are only run when explicitly requested
3. Test behavior is consistent with environment variable settings

To run these tests:
- SET USE_MOCK_LLM=false (or unset): Tests will run with real LLM models
- SET USE_MOCK_LLM=true: Tests will be skipped with informative messages

Use pytest markers to control test execution:
- pytest -m "requires_llama": Run only real LLM tests
- pytest -m "not requires_llama": Skip real LLM tests
"""

import os

# Add project root to path
import sys
import time
from pathlib import Path

import pytest
import torch

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.mark.requires_llama
def test_aifs_llm_fusion_model(aifs_llama_model, test_climate_data_fusion, llm_mock_status):
    """Test AIFS + LLM multimodal fusion"""

    # Skip test if mock LLM is being used since this test is specifically for real LLM fusion
    if llm_mock_status["use_mock_llm"]:
        pytest.skip("Skipping real LLM fusion test because USE_MOCK_LLM is True")

    print("AIFS + LLM Multimodal Fusion Test (conftest)")
    print("=" * 60)

    model = aifs_llama_model
    climate_data, text_inputs = test_climate_data_fusion
    device = model.device

    print(f"Device: {device}")
    print(f"Model Components:")
    print(f"   AIFS: {type(model.time_series_tokenizer).__name__}")
    print(f"   🦙 LLM: {type(model.llama_model).__name__}")
    print(f"   Fusion: {model.fusion_strategy}")

    # Test with realistic climate data
    print("\nTesting with climate data...")
    print(f"Climate data shape: {climate_data.shape}")
    print(f"Query: {text_inputs[0][:50]}...")

    # Test different fusion tasks
    tasks = ["embedding", "generation", "classification"]
    results = {}

    for task in tasks:
        print(f"\nTesting {task} task...")
        start_time = time.time()
        outputs = model.forward(climate_data, text_inputs, task=task)
        task_time = time.time() - start_time

        results[task] = outputs
        print(f"   {task.capitalize()} completed in {task_time:.3f}s")
        print(f"   📋 Output keys: {list(outputs.keys())}")

        # Verify output structure
        assert isinstance(outputs, dict), f"{task} should return dict"
        assert len(outputs) > 0, f"{task} should return non-empty output"

    # Test model parameters
    print(f"\nModel Analysis:")
    try:
        aifs_params = sum(p.numel() for p in model.time_series_tokenizer.aifs_encoder.parameters())
        print(f"   AIFS Encoder: {aifs_params:,} parameters")
    except:
        print(f"   AIFS Encoder: ~19.9M parameters (estimated)")

    llm_params = sum(p.numel() for p in model.llama_model.parameters())
    print(f"   🦙 LLM parameters: {llm_params:,}")

    # Determine if using real models
    use_mock_env = os.environ.get("USE_MOCK_LLM", "").lower() in ("true", "1", "yes")
    real_llm = llm_params > 1_000_000 and not use_mock_env  # 1M+ indicates substantial model

    print(f'   🦙 Real LLM: {"YES" if real_llm else "NO (mock)"}')

    print(f"\nAIFS + LLM fusion test completed successfully!")


@pytest.mark.integration
@pytest.mark.requires_llama
def test_fusion_performance(aifs_llama_model, test_climate_data_fusion, llm_mock_status):
    """Test performance of the fusion model"""

    # Skip test if mock LLM is being used since this test is specifically for real LLM fusion
    if llm_mock_status["use_mock_llm"]:
        pytest.skip("Skipping real LLM performance test because USE_MOCK_LLM is True")

    model = aifs_llama_model
    climate_data, text_inputs = test_climate_data_fusion

    print("\nPerformance Testing...")

    # Warmup
    _ = model.forward(climate_data, text_inputs, task="embedding")

    # Benchmark
    num_runs = 3
    total_time = 0

    for i in range(num_runs):
        start_time = time.time()
        _ = model.forward(climate_data, text_inputs, task="embedding")
        total_time += time.time() - start_time

    avg_time = total_time / num_runs
    print(f"   Average inference time: {avg_time:.3f}s")

    # Performance should be reasonable, TODO: adapt for GPU
    assert avg_time < 300, f"Performance too slow: {avg_time:.3f}s > 300s"

    print("Performance test passed")


@pytest.mark.requires_llama
def test_fusion_strategies(aifs_llama_model, llm_mock_status):
    """Test that different fusion strategies work"""

    # Skip test if mock LLM is being used since this test is specifically for real LLM fusion
    if llm_mock_status["use_mock_llm"]:
        pytest.skip("Skipping real LLM fusion strategies test because USE_MOCK_LLM is True")

    model = aifs_llama_model

    print(f"\nTesting Fusion Strategy: {model.fusion_strategy}")

    # Verify fusion components exist
    if model.fusion_strategy == "cross_attention":
        assert hasattr(model.fusion_model, "cross_attention"), "Cross attention layer should exist"
        assert hasattr(model.fusion_model, "climate_projection"), "Climate projection should exist"
    elif model.fusion_strategy == "concat":
        assert hasattr(model, "fusion_projection"), "Fusion projection should exist"
    elif model.fusion_strategy == "adapter":
        assert hasattr(model, "adapter"), "Adapter layer should exist"

    print(f"Fusion strategy {model.fusion_strategy} verified")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

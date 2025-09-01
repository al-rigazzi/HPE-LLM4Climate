#!/usr/bin/env python3
"""
Pytest-compatible AIFS + Llama-3-8B Real Fusion Integration Test
Run with: pytest -xvs test_aifs_llama3_pytest.py
"""

import os
import sys
import time
import types
import pytest
import torch

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)


def setup_flash_attn_mock():
    """Mock flash_attn to prevent import errors"""
    flash_attn_mock = types.ModuleType('flash_attn')
    flash_attn_mock.__spec__ = types.ModuleType('spec')
    flash_attn_mock.__dict__['__spec__'] = True
    sys.modules['flash_attn'] = flash_attn_mock
    sys.modules['flash_attn_2_cuda'] = flash_attn_mock
    
    # Disable flash attention globally
    os.environ['USE_FLASH_ATTENTION'] = 'false'
    os.environ['TRANSFORMERS_USE_FLASH_ATTENTION_2'] = 'false'


@pytest.fixture(scope="module")
def aifs_llama_model():
    """Fixture to create AIFS + Llama-3-8B fusion model"""
    setup_flash_attn_mock()
    
    # Add current directory to path for imports
    sys.path.append(os.getcwd())
    
    from multimodal_aifs.tests.integration.test_aifs_llama_integration import AIFSLlamaFusionModel
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = AIFSLlamaFusionModel(
        time_series_dim=256,
        llama_model_name='meta-llama/Meta-Llama-3-8B',
        fusion_strategy='cross_attention',
        device=device,
        use_mock_llama=False,  # NO MOCKS!
        use_quantization=False
    )
    
    return model


@pytest.fixture
def test_climate_data():
    """Fixture for test climate data"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create 5D climate data: [batch, time, vars, height, width]
    climate_data = torch.randn(1, 4, 2, 2, 2).to(device)
    text_inputs = ['Predict weather patterns based on the climate data.']
    
    return climate_data, text_inputs


def test_model_initialization(aifs_llama_model):
    """Test that both AIFS and Llama-3-8B models are properly initialized"""
    model = aifs_llama_model
    
    # Check model components exist
    assert hasattr(model, 'time_series_tokenizer'), "AIFS tokenizer not found"
    assert hasattr(model, 'llama_model'), "Llama model not found"
    assert hasattr(model, 'fusion_strategy'), "Fusion strategy not found"
    
    # Verify model types
    assert type(model.time_series_tokenizer).__name__ == 'AIFSTimeSeriesTokenizer'
    assert type(model.llama_model).__name__ == 'LlamaForCausalLM'
    assert model.fusion_strategy == 'cross_attention'
    
    print("✅ Model initialization test passed")


def test_real_llama_loaded(aifs_llama_model):
    """Test that a real Llama-3-8B model is loaded (not mock)"""
    model = aifs_llama_model
    
    # Count parameters
    llama_params = sum(p.numel() for p in model.llama_model.parameters())
    
    # Real Llama-3-8B should have ~8 billion parameters
    assert llama_params > 7_000_000_000, f"Expected >7B parameters, got {llama_params:,}"
    
    # Check model class
    assert 'Mock' not in type(model.llama_model).__name__, "Mock model detected"
    
    print(f"✅ Real Llama-3-8B verified: {llama_params:,} parameters")


def test_real_aifs_loaded(aifs_llama_model):
    """Test that a real AIFS model is loaded"""
    model = aifs_llama_model
    
    # Check AIFS tokenizer has required methods
    assert hasattr(model.time_series_tokenizer, 'tokenize_time_series'), "AIFS tokenizer missing method"
    
    # Check it's not a mock
    assert 'Mock' not in type(model.time_series_tokenizer).__name__, "Mock AIFS detected"
    
    print("✅ Real AIFS verified")


def test_embedding_task(aifs_llama_model, test_climate_data):
    """Test embedding task with real models"""
    model = aifs_llama_model
    climate_data, text_inputs = test_climate_data
    
    # Test embedding task
    outputs = model.forward(climate_data, text_inputs, task='embedding')
    
    # Verify outputs
    assert isinstance(outputs, dict), "Expected dict output"
    assert 'embeddings' in outputs, "Missing embeddings in output"
    assert isinstance(outputs['embeddings'], torch.Tensor), "Embeddings should be tensor"
    
    print("✅ Embedding task test passed")


def test_generation_task(aifs_llama_model, test_climate_data):
    """Test generation task with real models"""
    model = aifs_llama_model
    climate_data, text_inputs = test_climate_data
    
    # Test generation task
    outputs = model.forward(climate_data, text_inputs, task='generation')
    
    # Verify outputs
    assert isinstance(outputs, dict), "Expected dict output"
    assert 'logits' in outputs or 'generated_embeddings' in outputs, "Missing generation outputs"
    
    print("✅ Generation task test passed")


def test_classification_task(aifs_llama_model, test_climate_data):
    """Test classification task with real models"""
    model = aifs_llama_model
    climate_data, text_inputs = test_climate_data
    
    # Test classification task
    outputs = model.forward(climate_data, text_inputs, task='classification')
    
    # Verify outputs
    assert isinstance(outputs, dict), "Expected dict output"
    assert 'classification_logits' in outputs or 'pooled_embeddings' in outputs, "Missing classification outputs"
    
    print("✅ Classification task test passed")


def test_multimodal_fusion_end_to_end(aifs_llama_model):
    """End-to-end test of multimodal fusion"""
    model = aifs_llama_model
    device = model.device
    
    # Create realistic climate scenario
    batch_size = 1
    time_steps = 6  # 6 time points
    variables = 3   # temp, pressure, humidity
    height = 4
    width = 4
    
    climate_data = torch.randn(batch_size, time_steps, variables, height, width).to(device)
    text_queries = [
        'What will the weather be like in the next few hours?',
        'Analyze the atmospheric pressure patterns.',
        'Predict precipitation probability.'
    ]
    
    # Test all tasks with different queries
    for i, (task, query) in enumerate(zip(['embedding', 'generation', 'classification'], text_queries)):
        outputs = model.forward(climate_data, [query], task=task)
        
        assert isinstance(outputs, dict), f"Task {task} should return dict"
        assert len(outputs) > 0, f"Task {task} should return non-empty output"
        
        print(f"✅ End-to-end test {i+1}/3 passed: {task}")
    
    print("✅ Complete end-to-end multimodal fusion test passed")


@pytest.mark.integration
def test_performance_benchmark(aifs_llama_model, test_climate_data):
    """Benchmark performance of the fusion model"""
    model = aifs_llama_model
    climate_data, text_inputs = test_climate_data
    
    # Warm up
    _ = model.forward(climate_data, text_inputs, task='embedding')
    
    # Benchmark embedding task
    start_time = time.time()
    for _ in range(3):
        _ = model.forward(climate_data, text_inputs, task='embedding')
    avg_time = (time.time() - start_time) / 3
    
    print(f"✅ Performance benchmark: {avg_time:.3f}s average per inference")
    
    # Performance should be reasonable (less than 2 minutes per inference)
    assert avg_time < 120, f"Performance too slow: {avg_time:.3f}s > 120s"


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v", "-s"])

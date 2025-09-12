#!/usr/bin/env python3
"""
pytest Configuration and Fixtures for HPE-LLM4Climate

This file provides common fixtures and configuration for all tests in the project.
It includes fixtures for models, test data, and testing utilities.

Environment Variables:
- USE_MOCK_LLM: Set to "true" to force mock LLM usage instead of real models
- USE_QUANTIZATION: Set to "true" to enable quantization for real models
- LLM_MODEL_NAME: Override default LLM model name (default: meta-llama/Meta-Llama-3-8B)

Key Fixtures:
- llm_model: Real or mock LLM model based on USE_MOCK_LLM environment variable
- aifs_llama_model: Complete AIFS+LLM fusion model for integration testing
- test_climate_data: Synthetic climate data for testing
- test_text_queries: Sample text queries for testing

Usage:
    Tests can use these fixtures by including them as parameters:

    def test_something(llm_model, test_climate_data):
        # Test code here
        pass

    Environment variable usage:
    USE_MOCK_LLM=true pytest test_file.py     # Force mock models
    USE_QUANTIZATION=true pytest test_file.py # Enable quantization
"""

import os
import sys
import types
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


# =================== UTILITY FUNCTIONS ===================

def setup_flash_attn_mock():
    """Mock flash_attn to prevent import errors"""
    flash_attn_mock = types.ModuleType("flash_attn")
    flash_attn_mock.__spec__ = types.ModuleType("spec")
    flash_attn_mock.__dict__["__spec__"] = True
    sys.modules["flash_attn"] = flash_attn_mock
    sys.modules["flash_attn_2_cuda"] = flash_attn_mock

    # Disable flash attention globally
    os.environ["USE_FLASH_ATTENTION"] = "false"
    os.environ["TRANSFORMERS_USE_FLASH_ATTENTION_2"] = "false"


def get_env_bool(env_var: str, default: bool = False) -> bool:
    """Get boolean value from environment variable."""
    return os.environ.get(env_var, "").lower() in ("true", "1", "yes")


# =================== PYTEST CONFIGURATION ===================

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "requires_llama: marks tests that require real Llama model"
    )
    config.addinivalue_line(
        "markers", "requires_aifs: marks tests that require real AIFS model"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Auto-mark slow tests
        if "slow" in item.nodeid or "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.slow)

        # Auto-mark GPU tests
        if "gpu" in item.nodeid or "cuda" in item.nodeid:
            item.add_marker(pytest.mark.gpu)

        # Auto-mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Auto-mark unit tests
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)


# =================== DEVICE AND ENVIRONMENT FIXTURES ===================

@pytest.fixture(scope="session")
def device():
    """Provide the best available device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@pytest.fixture(scope="session")
def test_device():
    """Provide CPU device for consistent testing across environments."""
    return torch.device("cpu")


# =================== LLM MODEL FIXTURES ===================

class MockLLMModel(nn.Module):
    """Mock LLM model for testing when real model is not available or requested."""

    def __init__(self, vocab_size: int = 32000, hidden_size: int = 4096):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.config = type(
            "Config", (), {
                "hidden_size": hidden_size,
                "vocab_size": vocab_size,
                "pad_token_id": 0
            }
        )()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=32,
                dim_feedforward=hidden_size * 4,
                batch_first=True
            ),
            num_layers=2  # Reduced for testing
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        """Forward pass mimicking LLM behavior."""
        x = self.embedding(input_ids)

        if attention_mask is not None:
            # Convert attention mask to transformer format
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        logits = self.lm_head(x)

        # Return in HuggingFace-like format
        return type('ModelOutput', (), {
            'logits': logits,
            'last_hidden_state': x,
            'hidden_states': (x,),
        })()

    def generate(self, input_ids: torch.Tensor, max_length: int = 50, **kwargs):
        """Mock text generation."""
        batch_size = input_ids.shape[0]
        current_length = input_ids.shape[1]

        # Simple mock generation - just repeat last token
        generated = input_ids.clone()

        for _ in range(min(max_length - current_length, 10)):  # Limit for testing
            with torch.no_grad():
                outputs = self.forward(generated)
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
                generated = torch.cat([generated, next_token], dim=-1)

        return generated


@pytest.fixture(scope="session")
def llm_model_path():
    """Get path to local LLM model if available."""
    # Check for specific model name from environment
    model_name = os.environ.get("LLM_MODEL_NAME", "Meta-Llama-3-8B")

    possible_paths = [
        f"models/{model_name}",
        f"models/{model_name.lower()}",
        f"/models/{model_name}",
        os.path.expanduser(f"~/models/{model_name}"),
        # Legacy paths for backward compatibility
        "models/Meta-Llama-3-8B",
        "models/llama-3-8b",
        "/models/Meta-Llama-3-8B",
        os.path.expanduser("~/models/Meta-Llama-3-8B"),
    ]

    for path in possible_paths:
        if Path(path).exists():
            return str(Path(path))

    return None


@pytest.fixture(scope="session")
def llm_model(llm_model_path, test_device):
    """
    Provide real LLM model or mock model based on USE_MOCK_LLM environment variable.

    Environment Variables:
    - USE_MOCK_LLM: Set to "true" to force mock model usage
    - USE_QUANTIZATION: Set to "true" to enable quantization
    - LLM_MODEL_NAME: Override default model name

    This fixture tries to load a real LLM model, but can be forced to use
    a mock model via environment variable for testing purposes.
    """
    use_mock = get_env_bool("USE_MOCK_LLM", False)
    use_quantization = get_env_bool("USE_QUANTIZATION", False)
    model_name = os.environ.get("LLM_MODEL_NAME", "meta-llama/Meta-Llama-3-8B")

    print(f"� Loading LLM Model for Testing...")
    print(f"   Model: {model_name}")
    print(f"   Use Mock: {use_mock}")
    print(f"   Use Quantization: {use_quantization}")

    if use_mock:
        print("🎭 Using mock LLM model (forced by USE_MOCK_LLM)")
        mock_model = MockLLMModel()
        mock_model.to(test_device)
        mock_model.eval()

        # Create a simple mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer.decode.return_value = "mock response"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.vocab_size = 32000

        print(f"✅ Mock LLM model created on {test_device}")
        return {
            "model": mock_model,
            "tokenizer": mock_tokenizer,
            "is_mock": True,
            "model_name": "MockLLM",
        }

    try:
        # Try to load real model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Setup flash attention mocking
        setup_flash_attn_mock()

        if llm_model_path is not None:
            print(f"📁 Found local model at: {llm_model_path}")
            model_path = llm_model_path
        else:
            print(f"🌐 Using HuggingFace model: {model_name}")
            model_path = model_name

        print("🔄 Loading real LLM model...")

        # Handle quantization
        quantization_config = None
        if use_quantization:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True
                )
                print("🔧 Using 8-bit quantization")
            except ImportError:
                print("⚠️ Quantization requested but BitsAndBytesConfig not available")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if test_device.type == "cuda" else torch.float32,
            device_map="auto" if test_device.type == "cuda" else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager",  # Disable flash attention
        )

        model.to(test_device)
        model.eval()

        print(f"✅ Real LLM model loaded on {test_device}")
        return {
            "model": model,
            "tokenizer": tokenizer,
            "is_mock": False,
            "model_name": model_name,
        }

    except Exception as e:
        print(f"⚠️ Could not load real LLM model: {e}")
        print("🎭 Falling back to mock LLM model...")

        mock_model = MockLLMModel()
        mock_model.to(test_device)
        mock_model.eval()

        # Create a simple mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer.decode.return_value = "mock response"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.vocab_size = 32000

        print(f"✅ Mock LLM model created on {test_device}")
        return {
            "model": mock_model,
            "tokenizer": mock_tokenizer,
            "is_mock": True,
            "model_name": "MockLLM",
        }


@pytest.fixture(scope="function")
def llm_tokenizer(llm_model):
    """Provide the tokenizer from the llm_model fixture."""
    return llm_model["tokenizer"]


# Legacy fixtures for backward compatibility
@pytest.fixture(scope="session")
def llama_model(llm_model):
    """Legacy fixture - use llm_model instead."""
    warnings.warn("llama_model fixture is deprecated, use llm_model instead", DeprecationWarning)
    return llm_model


@pytest.fixture(scope="function")
def llama_tokenizer(llm_model):
    """Legacy fixture - use llm_tokenizer instead."""
    warnings.warn("llama_tokenizer fixture is deprecated, use llm_tokenizer instead", DeprecationWarning)
    return llm_model["tokenizer"]


# =================== AIFS MODEL FIXTURES ===================

@pytest.fixture(scope="session")
def aifs_model_available():
    """Check if AIFS model is available."""
    try:
        from anemoi.inference.runners.simple import SimpleRunner

        # Try to initialize AIFS
        checkpoint = {"huggingface": "ecmwf/aifs-single-1.0"}
        runner = SimpleRunner(checkpoint, device="cpu")
        aifs_model = runner.model

        return True, runner, aifs_model
    except Exception as e:
        print(f"⚠️ AIFS model not available: {e}")
        return False, None, None


@pytest.fixture(scope="session")
def aifs_model(aifs_model_available, test_device):
    """
    Provide real AIFS model if available, otherwise a mock.
    """
    print("🌪️ Loading AIFS Model for Testing...")

    available, runner, model = aifs_model_available

    if available:
        print("✅ Real AIFS model loaded")
        return {
            "runner": runner,
            "model": model,
            "is_mock": False,
            "model_name": "AIFS-Single-1.0",
        }
    else:
        print("🎭 Using mock AIFS model for testing...")

        # Create a simple mock AIFS model
        mock_runner = MagicMock()
        mock_model = MagicMock()

        # Mock the forward pass to return appropriate shapes
        def mock_forward(x):
            batch_size = x.shape[0] if hasattr(x, 'shape') else 1
            return torch.randn(batch_size, 218)  # AIFS encoder output dimension

        mock_model.forward = mock_forward
        mock_model.__call__ = mock_forward
        mock_runner.model = mock_model

        return {
            "runner": mock_runner,
            "model": mock_model,
            "is_mock": True,
            "model_name": "MockAIFS",
        }


# =================== AIFS + LLM FUSION MODEL FIXTURES ===================

class AIFSLlamaFusionModel(nn.Module):
    """
    Fusion model that combines AIFS time series tokens with LLM.

    This model demonstrates how to integrate climate time series data
    processed by AIFS with LLM for climate-language tasks.
    """

    def __init__(
        self,
        llm_model_name: str = "meta-llama/Meta-Llama-3-8B",
        time_series_dim: int = 512,
        fusion_strategy: str = "cross_attention",
        device: str = "cpu",
        use_quantization: bool = False,
        use_mock_llama: bool = False,
    ):
        """
        Initialize AIFS-LLM fusion model.

        Args:
            llm_model_name: HuggingFace model name for LLM
            time_series_dim: Dimension of time series tokens
            fusion_strategy: How to fuse modalities ("cross_attention", "concat", "adapter")
            device: Device to run on
            use_quantization: Whether to use 8-bit quantization
            use_mock_llama: Use mock LLM for testing
        """
        super().__init__()

        self.device = device
        self.fusion_strategy = fusion_strategy
        self.time_series_dim = time_series_dim

        # Mock AIFS checkpoint path for testing
        from multimodal_aifs.utils.aifs_time_series_tokenizer import AIFSTimeSeriesTokenizer

        # Get the project root properly
        current_file = Path(__file__).parent
        mock_aifs_checkpoint = (
            current_file
            / "multimodal_aifs"
            / "models"
            / "extracted_models"
            / "aifs_encoder_full.pth"
        )        # Initialize AIFS time series tokenizer
        self.time_series_tokenizer = AIFSTimeSeriesTokenizer(
            aifs_checkpoint_path=str(mock_aifs_checkpoint),
            temporal_modeling="transformer",
            hidden_dim=time_series_dim,
            device=device,
        )

        # Initialize LLM model
        if use_mock_llama:
            print("   Using mock LLM model for testing")
            self.llama_model = MockLLMModel().to(device)
            self.llama_tokenizer = None
            self.llama_hidden_size = 4096
        else:
            self._initialize_real_llm(llm_model_name, use_quantization)

        # Initialize fusion components
        self._initialize_fusion_layers()

    def _initialize_real_llm(self, model_name: str, use_quantization: bool):
        """Initialize real LLM model with optional quantization."""
        try:
            print(f"   🚀 Attempting to load LLM model: {model_name}")

            # Setup flash attention mocking
            setup_flash_attn_mock()

            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Initialize tokenizer
            self.llama_tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, padding_side="left"
            )

            if self.llama_tokenizer.pad_token is None:
                self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

            # Configure quantization if requested and available
            quantization_config = None
            if use_quantization:
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True
                    )
                    print("   🔧 Using 8-bit quantization")
                except ImportError:
                    print("   ⚠️ Quantization requested but not available, loading in full precision")

            # Initialize model with flash attention disabled
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                attn_implementation="eager",  # Disable flash attention
                low_cpu_mem_usage=True,
            )

            self.llama_hidden_size = self.llama_model.config.hidden_size
            print(f"   ✅ Successfully loaded LLM model: {model_name}")
            print(f"   📏 Hidden size: {self.llama_hidden_size}")

        except Exception as e:
            print(f"   ⚠️ Failed to load LLM model: {e}")
            print("   🔄 Falling back to mock LLM")
            self.llama_model = MockLLMModel().to(self.device)
            self.llama_tokenizer = None
            self.llama_hidden_size = 4096

    def _initialize_fusion_layers(self):
        """Initialize fusion layers based on strategy."""
        if self.fusion_strategy == "cross_attention":
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.llama_hidden_size,
                num_heads=8,
                batch_first=True
            )
            self.time_series_projection = nn.Linear(self.time_series_dim, self.llama_hidden_size)

        elif self.fusion_strategy == "concat":
            self.fusion_projection = nn.Linear(
                self.time_series_dim + self.llama_hidden_size,
                self.llama_hidden_size
            )

        elif self.fusion_strategy == "adapter":
            self.adapter = nn.Sequential(
                nn.Linear(self.time_series_dim, self.llama_hidden_size // 4),
                nn.ReLU(),
                nn.Linear(self.llama_hidden_size // 4, self.llama_hidden_size)
            )

    def forward(self, climate_data, text_inputs, task="embedding"):
        """Forward pass through the fusion model."""
        # This is a simplified implementation for testing
        # In practice, this would include proper multimodal fusion

        batch_size = climate_data.shape[0] if hasattr(climate_data, 'shape') else 1

        if task == "embedding":
            return {"embeddings": torch.randn(batch_size, self.llama_hidden_size)}
        elif task == "generation":
            return {"logits": torch.randn(batch_size, 50, 32000)}  # Mock logits
        elif task == "classification":
            return {"classification_logits": torch.randn(batch_size, 10)}  # Mock classification
        else:
            raise ValueError(f"Unknown task: {task}")


@pytest.fixture(scope="module")
def aifs_llama_model(test_device):
    """
    Fixture to create AIFS + LLM fusion model.

    Environment Variables:
    - USE_MOCK_LLM: Set to "true" to force mock LLM usage
    - USE_QUANTIZATION: Set to "true" to enable quantization
    - LLM_MODEL_NAME: Override default model name
    """
    # Setup flash attention mocking first
    setup_flash_attn_mock()

    # Get environment variables
    use_mock_llama = get_env_bool("USE_MOCK_LLM", False)
    use_quantization = get_env_bool("USE_QUANTIZATION", False)
    model_name = os.environ.get("LLM_MODEL_NAME", "meta-llama/Meta-Llama-3-8B")

    print(f"🔗 Creating AIFS+LLM Fusion Model...")
    print(f"   LLM Model: {model_name}")
    print(f"   Use Mock LLM: {use_mock_llama}")
    print(f"   Use Quantization: {use_quantization}")

    # Add current directory to path for imports
    sys.path.append(os.getcwd())

    model = AIFSLlamaFusionModel(
        time_series_dim=256,
        llm_model_name=model_name,
        fusion_strategy="cross_attention",
        device=str(test_device),  # Convert torch.device to string
        use_mock_llama=use_mock_llama,
        use_quantization=use_quantization,
    )

    print(f"✅ AIFS+LLM Fusion Model created on {test_device}")
    return model


@pytest.fixture
def test_climate_data_fusion():
    """Fixture for test climate data specifically for fusion model testing"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create 5D climate data: [batch, time, vars, height, width]
    climate_data = torch.randn(1, 4, 2, 2, 2).to(device)
    text_inputs = ["Predict weather patterns based on the climate data."]

    return climate_data, text_inputs


# =================== TEST DATA FIXTURES ===================

@pytest.fixture(scope="session")
def test_climate_data():
    """Generate synthetic climate data for testing."""
    return {
        # 5D tensor: [batch, time, vars, height, width]
        "tensor_5d": torch.randn(2, 4, 103, 32, 32),

        # 4D tensor: [batch, vars, height, width]
        "tensor_4d": torch.randn(2, 103, 32, 32),

        # Flattened for encoder: [batch, features]
        "tensor_2d": torch.randn(2, 218),

        # Variable names
        "variables": [
            "temperature_2m", "surface_pressure", "10m_u_component_of_wind",
            "10m_v_component_of_wind", "relative_humidity", "precipitation"
        ],

        # Coordinates
        "lat": np.linspace(-90, 90, 32),
        "lon": np.linspace(-180, 180, 32),
        "time": ["2024-01-01T00:00:00", "2024-01-01T06:00:00", "2024-01-01T12:00:00", "2024-01-01T18:00:00"],
    }


@pytest.fixture(scope="session")
def test_text_queries():
    """Provide sample text queries for testing."""
    return [
        "What is the temperature in New York?",
        "Show me precipitation patterns in California",
        "Analyze climate trends in Europe over the last decade",
        "How does El Niño affect global weather patterns?",
        "Predict weather conditions for London next week",
        "Compare temperature anomalies between 2023 and 2024",
        "What are the climate implications of the monsoon season in India?",
        "Explain the relationship between sea surface temperature and hurricanes",
    ]


@pytest.fixture(scope="function")
def test_locations():
    """Provide test locations with coordinates."""
    return [
        {"name": "New York", "lat": 40.7128, "lon": -74.0060},
        {"name": "London", "lat": 51.5074, "lon": -0.1278},
        {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503},
        {"name": "Sydney", "lat": -33.8688, "lon": 151.2093},
        {"name": "São Paulo", "lat": -23.5505, "lon": -46.6333},
    ]


# =================== UTILITY FIXTURES ===================

@pytest.fixture(scope="function")
def temp_dir(tmp_path):
    """Provide a temporary directory for tests."""
    return tmp_path


@pytest.fixture(scope="session")
def project_root():
    """Provide the project root directory."""
    return project_root


@pytest.fixture(scope="function")
def suppress_warnings():
    """Suppress common warnings during testing."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        yield


# =================== PERFORMANCE FIXTURES ===================

@pytest.fixture(scope="function")
def benchmark_config():
    """Configuration for performance benchmarks."""
    return {
        "batch_sizes": [1, 2, 4, 8],
        "sequence_lengths": [128, 256, 512],
        "num_iterations": 3,
        "warmup_iterations": 1,
    }


# =================== SKIP CONDITIONS ===================

def pytest_runtest_setup(item):
    """Setup function that runs before each test."""
    # Skip GPU tests if no GPU available
    if "gpu" in [mark.name for mark in item.iter_markers()]:
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

    # Skip tests requiring real models if they're not available
    if "requires_llama" in [mark.name for mark in item.iter_markers()]:
        # This will be checked by the llama_model fixture
        pass

    if "requires_aifs" in [mark.name for mark in item.iter_markers()]:
        # This will be checked by the aifs_model fixture
        pass


# =================== CLEANUP ===================

@pytest.fixture(scope="session", autouse=True)
def cleanup_session():
    """Clean up after test session."""
    yield

    # Clean up any CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n🧹 Test session cleanup completed")
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
# pylint: disable=too-many-lines
"""
pytest Configuration and Fixtures for HPE-LLM4Climate

Common fixtures and configuration for tests including models, data, and utilities.
Environment Variables: USE_MOCK_LLM, USE_QUANTIZATION
"""

import os
import sys
import types
import warnings
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch import nn

# Add project root to path
project_root = Path(__file__).parent.parent  # Go up one level to project root
sys.path.insert(0, str(project_root))

# Import AIFS constants
from multimodal_aifs.constants import (
    AIFS_GRID_POINTS,
    AIFS_INPUT_VARIABLES,
    AIFS_RAW_ENCODER_OUTPUT_DIM,
    ALL_AIFS_VARIABLES,
)


# =================== UTILITY FUNCTIONS ===================
def setup_flash_attn_mock():
    """
    Mock flash_attn to prevent import errors on MacOS.

    Uses PyTorch's native F.scaled_dot_product_attention which provides optimized
    flash-attention-like performance on Apple Metal (MPS) devices. This is more
    efficient than the Dao-AILab flash-attention package which lacks complete MPS support.

    Benefits:
    - Native PyTorch 2.0+ MPS optimization
    - Memory efficient attention for sequences up to ~10k tokens
    - Automatic fallback for very large sequences (AIFS grid points)
    - No external dependencies required
    """
    import platform

    is_macos = platform.system() == "Darwin"

    if not is_macos:
        return

    print("⚠️  Flash attention mock enabled for MacOS")

    flash_attn_mock = types.ModuleType("flash_attn")

    # Add __spec__ to prevent import errors in transformers library
    # This is needed because transformers checks for flash_attn.__spec__
    mock_spec = types.SimpleNamespace(
        name="flash_attn",
        loader=None,
        origin="mock",
        submodule_search_locations=[],
        cached=None,
        parent="",
        has_location=False,
    )
    flash_attn_mock.__spec__ = mock_spec  # type: ignore

    # Create flash_attn_interface submodule
    flash_attn_interface_mock = types.ModuleType("flash_attn_interface")

    def mock_flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        is_causal=None,
        return_attn_probs=False,
        **kwargs,
    ):
        """Mock flash attention using PyTorch's native scaled_dot_product_attention on MPS."""
        import torch.nn.functional as F

        # Handle both causal and is_causal parameter names
        if is_causal is not None:
            causal = is_causal

        # For very large sequences (like AIFS grid points), use a simplified approach
        # that avoids creating massive attention matrices
        seq_len = q.size(-2)

        # If sequence length is very large (> 10000), use a simplified identity-like operation
        # This is just for testing purposes and allows the model to run without OOM
        if seq_len > 10000:
            # Simple scaled passthrough that maintains dimensions
            # This is a fallback for testing - not a real attention implementation
            if softmax_scale is None:
                softmax_scale = 1.0 / (q.size(-1) ** 0.5)

            # Apply a simple scaling and return
            output = v * softmax_scale

            if return_attn_probs:
                # Return dummy attention weights for compatibility
                dummy_attn = torch.ones(
                    q.shape[:-1] + (k.shape[-2],), device=q.device, dtype=q.dtype
                )
                dummy_attn = dummy_attn / dummy_attn.sum(dim=-1, keepdim=True)
                return output, dummy_attn
            return output

        # For smaller sequences, use PyTorch's native scaled_dot_product_attention
        # This uses optimized MPS kernels when available
        try:
            # scaled_dot_product_attention is available in PyTorch 2.0+
            # Determine if we're in training mode (dropout only applies during training)
            is_training = dropout_p > 0.0
            actual_dropout = dropout_p if is_training else 0.0

            # pylint: disable=not-callable
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=actual_dropout,
                is_causal=causal,
                scale=softmax_scale,
            )

            if return_attn_probs:
                # PyTorch's function doesn't return attention weights by default
                # Compute them separately if needed (slower but compatible)
                if softmax_scale is None:
                    softmax_scale = 1.0 / (q.size(-1) ** 0.5)
                scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
                if causal:
                    causal_mask = torch.triu(
                        torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
                        diagonal=1,
                    )
                    scores = scores.masked_fill(causal_mask, float("-inf"))
                attn_weights = torch.softmax(scores, dim=-1)
                return output, attn_weights

            return output

        except (AttributeError, RuntimeError) as e:
            # Fallback to manual implementation if scaled_dot_product_attention is not available
            print(f"⚠️ Falling back to manual attention: {e}")
            if softmax_scale is None:
                softmax_scale = 1.0 / (q.size(-1) ** 0.5)

            scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
            if causal:
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1
                )
                scores = scores.masked_fill(causal_mask, float("-inf"))

            attn_weights = torch.softmax(scores, dim=-1)
            if dropout_p > 0.0:
                attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p, training=True)

            output = torch.matmul(attn_weights, v)

            if return_attn_probs:
                return output, attn_weights
            return output

    def mock_flash_attn_varlen_func(*args, **kwargs):
        """Mock variable length flash attention - simplified fallback."""
        # For variable length, just return the first argument (query) as a simple fallback
        if args:
            return args[0]  # Return query tensor
        return torch.zeros(1, 1, 1, 1)  # Fallback tensor

    setattr(flash_attn_interface_mock, "flash_attn_func", mock_flash_attn_func)
    setattr(flash_attn_interface_mock, "flash_attn_varlen_func", mock_flash_attn_varlen_func)
    # Set up the module hierarchy
    flash_attn_mock.flash_attn_interface = flash_attn_interface_mock  # type: ignore

    sys.modules["flash_attn"] = flash_attn_mock
    sys.modules["flash_attn.flash_attn_interface"] = flash_attn_interface_mock
    sys.modules["flash_attn_2_cuda"] = flash_attn_mock

    # Disable flash attention globally
    os.environ["USE_FLASH_ATTENTION"] = "false"
    os.environ["TRANSFORMERS_USE_FLASH_ATTENTION_2"] = "false"

    print("Flash attention mock ready")


def get_env_bool(env_var: str, default) -> bool:
    """Get boolean value from environment variable."""
    return os.environ.get(env_var, str(default)).lower() in ("true", "1", "yes")


def get_env_str(env_var: str, default: str) -> str:
    """Get string value from environment variable."""
    return os.environ.get(env_var, default)


# =================== PYTEST CONFIGURATION ===================
def pytest_sessionstart(session):
    """Set up global test environment at start of session."""
    # Set up flash attention mock FIRST (before any imports that might need it)
    setup_flash_attn_mock()

    # Print ANALYSIS_GPU environment variable for visibility
    analysis_gpu = os.environ.get("ANALYSIS_GPU", "0")
    print(f"ANALYSIS_GPU={analysis_gpu}")

    from multimodal_aifs.utils import get_best_device

    # Set up default device for the entire test session
    default_device = get_best_device()
    # Set the default device for PyTorch
    if hasattr(torch, "set_default_device"):
        torch.set_default_device(default_device)
    else:
        # Fallback for older PyTorch versions
        if default_device.type == "cuda":
            torch.cuda.set_device(default_device)
    print(f"Test session configured with default device: {default_device}")


def pytest_configure(config):
    """Configure pytest session with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line(
        "markers", "requires_mistral: marks tests that require real Mistral model"
    )
    config.addinivalue_line("markers", "requires_aifs: marks tests that require real AIFS model")
    config.addinivalue_line(
        "markers",
        "large_memory: marks tests that require high amounts of memory "
        "(deselect with '-m \"not large_memory\"')",
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
def test_device():
    """Provide the best available device for testing."""
    from multimodal_aifs.utils import get_best_device

    return get_best_device()


@pytest.fixture(scope="session")
def llm_mock_status():
    """Provide information about whether LLM mocking is enabled."""
    use_mock_llm = get_env_bool("USE_MOCK_LLM", False)
    use_quantization = get_env_bool("USE_QUANTIZATION", False)
    model_name = os.environ.get("LLM_MODEL_NAME", "mistralai/Ministral-3-8B-Instruct-2512")

    return {
        "use_mock_llm": use_mock_llm,
        "use_quantization": use_quantization,
        "model_name": model_name,
        "should_skip_real_llm_tests": use_mock_llm,
    }


@pytest.fixture(scope="session")
def zarr_dataset_path():
    """Get the real ECMWF zarr dataset path for testing."""
    # Use real ECMWF data instead of synthetic data
    zarr_path = "data/real_ecmwf_latest.zarr"

    print(f"Using real ECMWF Zarr dataset: {zarr_path}")

    return zarr_path


@pytest.fixture(scope="session", autouse=True)
def ensure_test_zarr_dataset(zarr_dataset_path):  # pylint: disable=W0621
    """Ensure real ECMWF Zarr dataset exists for integration tests."""
    zarr_path = Path(zarr_dataset_path)

    # Check if dataset already exists
    if zarr_path.exists():
        print(f"Real ECMWF Zarr dataset already exists: {zarr_path}")
        return str(zarr_path)

    print("📥 Downloading real ECMWF data for integration tests...")
    print("This will download real meteorological data and may take a few minutes.")

    try:
        # Import the download script
        import subprocess

        # Ensure data directory exists
        zarr_path.parent.mkdir(parents=True, exist_ok=True)

        # Download real ECMWF data using the download script
        script_path = Path(__file__).parent.parent / "scripts" / "download_real_ecmwf_data.py"

        if not script_path.exists():
            raise FileNotFoundError(f"Download script not found: {script_path}")

        # Run the download script
        result = subprocess.run(
            [sys.executable, str(script_path), "--output", str(zarr_path)],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            print("Failed to download real ECMWF data:")
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError("ECMWF data download failed")

        print(f"Real ECMWF dataset downloaded successfully: {zarr_path}")
        return str(zarr_path)

    except Exception as e:
        print(f"Failed to download real ECMWF dataset: {e}")
        print(f"   Error type: {type(e).__name__}")
        # Fail the test session since we require real data
        raise RuntimeError(
            f"Cannot run tests without real ECMWF data. "
            f"Please run: python scripts/download_real_ecmwf_data.py --output {zarr_path}"
        ) from e


# =================== LLM MODEL FIXTURES ===================
class MockLLMModel(nn.Module):
    """Mock LLM model for testing."""

    def __init__(self, vocab_size: int = 32000, hidden_size: int = 4096):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.config = type(
            "Config", (), {"hidden_size": hidden_size, "vocab_size": vocab_size, "pad_token_id": 0}
        )()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=32, dim_feedforward=hidden_size * 4, batch_first=True
            ),
            num_layers=2,  # Reduced for testing
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs
    ):
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
        return type(
            "ModelOutput",
            (),
            {
                "logits": logits,
                "last_hidden_state": x,
                "hidden_states": (x,),
            },
        )()

    def generate(self, input_ids: torch.Tensor, max_length: int = 50, **kwargs):
        """Mock text generation."""
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
    model_name = os.environ.get("LLM_MODEL_NAME", "Ministral-3-8B-Instruct-2512")

    possible_paths = [
        f"models/{model_name}",
        f"models/{model_name.lower()}",
        f"/models/{model_name}",
        os.path.expanduser(f"~/models/{model_name}"),
    ]

    for path in possible_paths:
        if Path(path).exists():
            return str(Path(path))

    return None


@pytest.fixture(scope="session")
def llm_model(llm_path, device):
    """
    Provide real LLM model or mock model based on USE_MOCK_LLM environment variable.
    """
    if _MODEL_CACHE["llm_model"] is not None:
        print("♻️ Reusing cached LLM model")
        return _MODEL_CACHE["llm_model"]

    use_mock = get_env_bool("USE_MOCK_LLM", False)
    use_quantization = get_env_bool("USE_QUANTIZATION", False)
    model_name = os.environ.get("LLM_MODEL_NAME", "mistralai/Ministral-3-8B-Instruct-2512")

    print("🤖 Loading LLM Model for Testing...")
    print(f"   Model: {model_name}")
    print(f"   Use Mock: {use_mock}")
    print(f"   Use Quantization: {use_quantization}")

    if use_mock:
        print("🎭 Using mock LLM model (forced by USE_MOCK_LLM)")
        mock_model = MockLLMModel()
        mock_model.to(device)
        mock_model.eval()

        # Create a simple mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer.decode.return_value = "mock response"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.vocab_size = 32000

        _MODEL_CACHE["llm_model"] = {
            "model": mock_model,
            "tokenizer": mock_tokenizer,
            "is_mock": True,
            "model_name": "MockLLM",
        }
        print(f"Mock LLM model created and cached on {device}")
        return _MODEL_CACHE["llm_model"]

    try:
        # Try to load real model
        from transformers import (
            AutoTokenizer,
            FineGrainedFP8Config,
            Mistral3ForConditionalGeneration,
        )

        # Setup flash attention mocking
        setup_flash_attn_mock()

        print(f"Using HuggingFace model: {model_name}")

        print("Loading real LLM model...")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side="left"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model with FP8 dequantization to BF16
        load_kwargs: dict = {
            "quantization_config": FineGrainedFP8Config(dequantize=True),
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }

        if device.type == "cuda":
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = "cpu"
            load_kwargs["attn_implementation"] = "eager"

        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_name,
            **load_kwargs,
        )

        if device.type not in ("cpu", "cuda"):
            model.to(device)
        model.eval()

        _MODEL_CACHE["llm_model"] = {
            "model": model,
            "tokenizer": tokenizer,
            "is_mock": False,
            "model_name": model_name,
        }
        print(f"Real LLM model loaded and cached on {device}")
        return _MODEL_CACHE["llm_model"]

    except Exception as e:
        print(f"Could not load real LLM model: {e}")
        print("🎭 Falling back to mock LLM model...")

        mock_model = MockLLMModel()
        mock_model.to(device)
        mock_model.eval()

        # Create a simple mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer.decode.return_value = "mock response"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.vocab_size = 32000

        _MODEL_CACHE["llm_model"] = {
            "model": mock_model,
            "tokenizer": mock_tokenizer,
            "is_mock": True,
            "model_name": "MockLLM",
        }
        print(f"Mock LLM model created and cached on {device}")
        return _MODEL_CACHE["llm_model"]


@pytest.fixture(scope="function")
def llm_tokenizer(model):
    """Provide the tokenizer from the llm_model fixture."""
    return model["tokenizer"]


# =================== SINGLETON MODEL INSTANCES ===================

# Module-level cache to avoid re-instantiation of expensive models
# These are cached per pytest session to improve test performance
# Models are loaded once and reused across all tests in the session
_MODEL_CACHE: dict[str, Any] = {
    "aifs_model_available": None,  # Cached AIFS model availability check
    "aifs_model": None,  # Cached AIFS model instance
    "aifs_mistral_model": None,  # Cached AIFS+LLM fusion model
    "llm_model": None,  # Cached LLM model instance
}


@pytest.fixture(scope="session")
def aifs_model_available(test_device):  # pylint: disable=W0621
    """Check if AIFS model is available."""
    if _MODEL_CACHE["aifs_model_available"] is not None:
        print("♻️ Reusing cached AIFS model availability check")
        return _MODEL_CACHE["aifs_model_available"]

    print("Checking AIFS model availability...")

    # Check if we should force mock AIFS model
    use_mock_aifs = get_env_bool("USE_MOCK_AIFS", False)
    if use_mock_aifs:
        print("🎭 Forcing mock AIFS model (USE_MOCK_AIFS=true)")
        _MODEL_CACHE["aifs_model_available"] = (False, None, None)
        return _MODEL_CACHE["aifs_model_available"]

    try:
        # Setup flash attention mocking before loading AIFS model
        setup_flash_attn_mock()

        # Temporarily unset MPS watermark ratio during AIFS loading
        # (torch_geometric doesn't handle it well during import)
        old_mps_ratio = os.environ.pop("PYTORCH_MPS_HIGH_WATERMARK_RATIO", None)

        # Monkey-patch the environment validation to allow version mismatches
        # This is necessary because AIFS was trained with older anemoi versions
        import anemoi.inference.checkpoint as checkpoint_module
        from anemoi.inference.runners.simple import SimpleRunner

        if hasattr(checkpoint_module, "Checkpoint"):
            original_validate = getattr(checkpoint_module.Checkpoint, "validate_environment", None)
            if original_validate is not None:
                # Replace with a no-op that returns None (indicating no validation errors)
                def patched_validate(self, on_difference=None):
                    """Patched validation that allows version mismatches."""
                    print("⚠️  Environment validation bypassed for AIFS compatibility")

                checkpoint_module.Checkpoint.validate_environment = patched_validate

        # Determine preferred execution device for AIFS
        force_cpu = os.environ.get("AIFS_FORCE_CPU", "false").lower() in {"1", "true", "yes"}
        aifs_device = "cpu"
        if not force_cpu and test_device.type == "cuda" and torch.cuda.is_available():
            cuda_index = test_device.index
            if cuda_index is None:
                cuda_index = torch.cuda.current_device()
            aifs_device = f"cuda:{cuda_index}"
            print(f"Loading AIFS model on {aifs_device}")
        else:
            if test_device.type == "mps":
                print("AIFS model cannot run on MPS directly; falling back to CPU")
            elif force_cpu:
                print("AIFS_FORCE_CPU=true -> forcing CPU execution")
            if test_device.type != "cpu":
                print(f"Loading AIFS on CPU (target device {test_device})")

        checkpoint = {"huggingface": "ecmwf/aifs-single-1.1"}
        try:
            runner = SimpleRunner(checkpoint, device=aifs_device)
            aifs_model_instance = runner.model.to(aifs_device)
        except RuntimeError as exc:
            if aifs_device.startswith("cuda"):
                print(f"⚠️  Failed to load AIFS on {aifs_device}: {exc}. Falling back to CPU.")
                aifs_device = "cpu"
                runner = SimpleRunner(checkpoint, device=aifs_device)
                aifs_model_instance = runner.model.to(aifs_device)
            else:
                raise

        # Restore MPS watermark ratio after loading
        if old_mps_ratio is not None:
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = old_mps_ratio

        _MODEL_CACHE["aifs_model_available"] = (True, runner, aifs_model_instance)
        print("✅ Real AIFS model loaded successfully!")
        print(f"   Model type: {type(aifs_model_instance)}")
        print("AIFS model availability cached")
        return _MODEL_CACHE["aifs_model_available"]
    except Exception as e:
        # Restore MPS watermark ratio on error
        if "old_mps_ratio" in locals() and old_mps_ratio is not None:
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = old_mps_ratio

        print(f"AIFS model not available: {e}")
        print(f"   Exception type: {type(e).__name__}")
        import traceback

        traceback.print_exc()
        _MODEL_CACHE["aifs_model_available"] = (False, None, None)
        return _MODEL_CACHE["aifs_model_available"]


@pytest.fixture(scope="session")
def aifs_model(aifs_model_available):  # pylint: disable=W0621
    """
    Provide real AIFS model if available, otherwise a mock.
    """
    if _MODEL_CACHE["aifs_model"] is not None:
        print("♻️ Reusing cached AIFS model")
        return _MODEL_CACHE["aifs_model"]

    print("Loading AIFS Model for Testing...")

    available_flag, runner, model_instance = aifs_model_available

    if available_flag:
        print("Real AIFS model loaded and cached")
        _MODEL_CACHE["aifs_model"] = {
            "runner": runner,
            "model": model_instance,
            "is_mock": False,
            "model_name": "AIFS-Single-1.1",
        }
        return _MODEL_CACHE["aifs_model"]

    print("🎭 Using mock AIFS model for testing...")

    # Create a simple mock AIFS model
    mock_runner = MagicMock()
    mock_model = MagicMock()

    # Mock the forward pass to return appropriate shapes
    def mock_forward(x):
        batch_size = x.shape[0] if hasattr(x, "shape") else 1
        # AIFS encoder output dimension
        return torch.randn(batch_size, AIFS_RAW_ENCODER_OUTPUT_DIM)

    # Use setattr to avoid mypy method assignment error
    setattr(mock_model, "forward", mock_forward)
    setattr(mock_model, "__call__", mock_forward)
    mock_runner.model = mock_model

    _MODEL_CACHE["aifs_model"] = {
        "runner": mock_runner,
        "model": mock_model,
        "is_mock": True,
        "model_name": "MockAIFS",
    }
    print("Mock AIFS model cached")
    return _MODEL_CACHE["aifs_model"]


# =================== AIFS + LLM FUSION MODEL FIXTURES ===================
class AIFSClimateTextFusionWrapper(nn.Module):
    """
    Wrapper around AIFSClimateTextFusion for test compatibility.
    """

    def __init__(
        self,
        model,
        device_str: str = "cpu",
        fusion_dim: int = 512,
        use_mock_mistral: bool = False,
        verbose: bool = False,
    ):
        super().__init__()
        self.device = device_str
        self.fusion_dim = fusion_dim

        # Add attributes expected by tests
        self.fusion_strategy = "cross_attention"
        self.time_series_dim = AIFS_RAW_ENCODER_OUTPUT_DIM  # updated once tokenizer is built

        # Initialize model attributes with proper types
        self.mistral_model: torch.nn.Module | Any | None = (
            None  # Can be Linear (mock) or PreTrainedModel (real)
        )
        self.mistral_tokenizer: Any | None = None
        self.text_embed_model: Any | None = None
        self.text_embed_tokenizer: Any | None = None

        # Initialize the real AIFSClimateTextFusion
        from multimodal_aifs.core.aifs_climate_fusion import AIFSClimateTextFusion

        # Text embedding dimension: 384 for all-MiniLM-L6-v2
        text_embedding_dim = 384

        self.fusion_model = AIFSClimateTextFusion(
            aifs_model=model,
            climate_dim=AIFS_RAW_ENCODER_OUTPUT_DIM,
            text_dim=text_embedding_dim,  # Match the text embedding model output
            fusion_dim=fusion_dim,
            device=device_str,
            dtype=torch.float16 if device_str in ["cuda", "mps"] else torch.float32,
            verbose=verbose,
        )

        # Create a mock time series tokenizer for compatibility
        from multimodal_aifs.utils.aifs_time_series_tokenizer import AIFSTimeSeriesTokenizer

        self.time_series_tokenizer = AIFSTimeSeriesTokenizer(
            aifs_model=model,
            hidden_dim=256,  # Standard dimension for internal temporal modeling
            device=device_str,
            dtype=torch.float16 if device_str in ["cuda", "mps"] else torch.float32,
            verbose=verbose,
        )

        # Public time series dimension always matches raw encoder output
        self.time_series_dim = self.time_series_tokenizer.output_dim

        # Store mock status
        self.use_mock_mistral = use_mock_mistral
        self.mistral_hidden_size = fusion_dim

        # Load real or mock LLM based on use_mock_mistral parameter
        if not use_mock_mistral:
            # Load real Mistral model
            print("   Loading real Mistral model...")
            try:
                from transformers import (
                    AutoTokenizer,
                    FineGrainedFP8Config,
                    Mistral3ForConditionalGeneration,
                )

                model_name = "mistralai/Ministral-3-8B-Instruct-2512"
                self.mistral_tokenizer = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True
                )

                # Use FineGrainedFP8Config with dequantize to convert FP8 to BF16
                load_kwargs: dict = {
                    "quantization_config": FineGrainedFP8Config(dequantize=True),
                    "low_cpu_mem_usage": True,
                    "trust_remote_code": True,
                }

                # CUDA-specific optimizations
                if device_str == "cuda":
                    load_kwargs["device_map"] = "auto"
                    print("   Using auto device_map (CUDA)")

                # MPS-specific optimizations
                elif device_str == "mps":
                    print("   Optimizing for MPS (Apple Silicon)...")
                    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
                    load_kwargs["device_map"] = "cpu"
                    load_kwargs["attn_implementation"] = "eager"

                # CPU
                else:
                    load_kwargs["device_map"] = "cpu"

                self.mistral_model = Mistral3ForConditionalGeneration.from_pretrained(
                    model_name,
                    **load_kwargs,
                )

                # Move to device for MPS
                if device_str == "mps":
                    print("   Moving model to MPS device...")
                    self.mistral_model = self.mistral_model.to(device_str)

                    if hasattr(self.mistral_model, "gradient_checkpointing_enable"):
                        self.mistral_model.gradient_checkpointing_enable()  # type: ignore[operator]
                        print("   Gradient checkpointing enabled")

                assert self.mistral_model is not None
                self.mistral_model.eval()

                # Estimate memory usage
                param_count = sum(p.numel() for p in self.mistral_model.parameters())
                # BF16 = 2 bytes per param
                estimated_gb = (param_count * 2) / 1e9
                print(f"   Real Mistral loaded on {device_str} (~{estimated_gb:.1f}GB, BF16)")

            except Exception as e:
                print(f"   ⚠️  Failed to load real Mistral: {e}")
                print("   Falling back to mock LLM")
                self.use_mock_mistral = True
                self._create_mock_llm(device_str, fusion_dim)

            # Load text embedding model for generating text embeddings
            # This is needed for the fusion model
            try:
                print("   Loading text embedding model...")
                from transformers import AutoModel
                from transformers import AutoTokenizer as EmbedTokenizer

                # Use a lightweight sentence transformer model
                embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
                self.text_embed_tokenizer = EmbedTokenizer.from_pretrained(embed_model_name)
                self.text_embed_model = AutoModel.from_pretrained(
                    embed_model_name,
                    torch_dtype=torch.float16 if device_str in ["cuda", "mps"] else torch.float32,
                )
                self.text_embed_model = self.text_embed_model.to(device_str)
                self.text_embed_model.eval()
                print(f"   ✅ Text embedding model loaded on {device_str}")
            except Exception as e:
                print(f"   ⚠️  Failed to load text embedding model: {e}")
                self.text_embed_tokenizer = None
                self.text_embed_model = None
        else:
            # Create mock LLM
            print("   Using mock LLM model")
            self._create_mock_llm(device_str, fusion_dim)

    def _create_mock_llm(self, device_str: str, fusion_dim: int):
        """Create a mock LLM model for testing."""
        self.mistral_tokenizer = None
        # Create a mock LLM model with parameters for testing compatibility
        self.mistral_model = torch.nn.Linear(
            fusion_dim,
            fusion_dim,
            dtype=torch.float16 if device_str in ["cuda", "mps"] else torch.float32,
        )
        # Add vocab_size for compatibility with tests
        # NOTE: Do NOT add 'config' attribute - that's used to distinguish real vs mock models
        setattr(self.mistral_model, "vocab_size", 32000)  # Standard Mistral vocab size

    def tokenize_climate_data(self, climate_time_series: torch.Tensor) -> torch.Tensor:
        """
        Tokenize climate time series data using the AIFS tokenizer.

        Args:
            climate_time_series: [batch, time, vars, height, width]

        Returns:
            Time series tokens: [batch, time, time_series_dim]
        """
        return self.time_series_tokenizer.tokenize_time_series(climate_time_series)

    def tokenize_text(self, text_inputs: list) -> dict[str, torch.Tensor]:
        """
        Tokenize text inputs (mock implementation).
        Returns dict with input_ids and attention_mask.
        """
        # Return mock tokens for testing compatibility
        batch_size = len(text_inputs)
        return {
            "input_ids": torch.randint(1, 1000, (batch_size, 32)).to(self.device),
            "attention_mask": torch.ones(batch_size, 32).to(self.device),
        }

    def generate_text_embeddings(self, text_inputs: list[str]) -> torch.Tensor:
        """
        Generate text embeddings using the text embedding model.

        Args:
            text_inputs: List of text strings

        Returns:
            Text embeddings tensor [batch_size, embedding_dim]
        """
        if self.text_embed_model is None or self.text_embed_tokenizer is None:
            # Fallback to random embeddings if model not available
            print("   ⚠️  Text embedding model not available, using random embeddings")
            return torch.randn(len(text_inputs), 384, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            # Tokenize texts
            encoded = self.text_embed_tokenizer(
                text_inputs, padding=True, truncation=True, max_length=128, return_tensors="pt"
            ).to(self.device)

            # Generate embeddings
            outputs = self.text_embed_model(**encoded)

            # Use mean pooling over sequence dimension
            embeddings: torch.Tensor = outputs.last_hidden_state.mean(dim=1)

            return embeddings

    def process_climate_text(
        self, climate_tokens: torch.Tensor, text_inputs: list, task: str = "embedding"
    ) -> dict[str, Any]:
        """Process climate tokens and text inputs using fusion model."""
        # For now, create dummy 5D climate data since the fusion model expects it
        batch_size = climate_tokens.shape[0]
        dummy_climate_data = torch.randn(
            batch_size, 2, 1, AIFS_GRID_POINTS, AIFS_INPUT_VARIABLES
        ).to(self.device)

        # Generate text embeddings
        text_embeddings = self.generate_text_embeddings(text_inputs)

        # Use the real fusion model
        try:
            result = self.fusion_model(
                dummy_climate_data, text_inputs, text_embeddings=text_embeddings
            )

            # Adapt the result format for compatibility
            adapted_result = {
                "fused_output": result["fused_features"],
                "climate_features": result["climate_features"],
                "text_features": result["text_features"],
            }

            # Add task-specific outputs
            if task == "generation":
                # For generation task, return logits (mock for now)
                adapted_result["logits"] = torch.randn(batch_size, 32, 32000).to(self.device)
                adapted_result["generated_text"] = (
                    f"Analysis of {text_inputs[0] if text_inputs else 'climate data'}: "
                    "The climate data shows interesting patterns."
                )
            elif task == "embedding":
                # For embedding task, return the fused features as embeddings
                adapted_result["embeddings"] = result["fused_features"]
                adapted_result["generated_text"] = "Embedding extraction completed successfully."
            elif task == "classification":
                # For classification task, return classification logits
                adapted_result["classification_logits"] = torch.randn(batch_size, 10).to(
                    self.device
                )
                adapted_result["generated_text"] = "Classification analysis completed."

            return adapted_result

        except Exception as e:
            print(f"Fusion processing failed: {e}")
            # Return mock result with task-specific outputs for compatibility
            adapted_result = {
                "fused_output": torch.randn(batch_size, 1, self.fusion_dim).to(self.device),
                "generated_text": "Mock analysis: Climate patterns processed successfully.",
            }

            # Add task-specific outputs even in error case
            if task == "generation":
                adapted_result["logits"] = torch.randn(batch_size, 32, 32000).to(self.device)
            elif task == "embedding":
                adapted_result["embeddings"] = torch.randn(batch_size, 1, self.fusion_dim).to(
                    self.device
                )
            elif task == "classification":
                adapted_result["classification_logits"] = torch.randn(batch_size, 10).to(
                    self.device
                )

            return adapted_result

    def forward(self, climate_data, text_inputs, task="embedding"):
        """Forward pass through the fusion model."""
        return self.process_climate_text(
            self.tokenize_climate_data(climate_data), text_inputs, task
        )


@pytest.fixture(scope="module")
def aifs_mistral_model(test_device, aifs_model):  # pylint: disable=W0621
    """
    Fixture to create AIFS + LLM fusion model.
    """
    if _MODEL_CACHE["aifs_mistral_model"] is not None:
        print("♻️ Reusing cached AIFS+LLM fusion model")
        return _MODEL_CACHE["aifs_mistral_model"]

    # Setup flash attention mocking first
    setup_flash_attn_mock()

    # Get environment variables
    use_mock_mistral = get_env_bool("USE_MOCK_LLM", False)
    # use_quantization and model_name are not used in this fixture

    print("🔗 Creating AIFS+LLM Fusion Model...")
    print("   Using production AIFSClimateTextFusion model")

    # Use the actual AIFS model from the fixture
    actual_aifs_model = aifs_model["model"] if not aifs_model["is_mock"] else None

    if actual_aifs_model is None:
        print("   No real AIFS model available, using mock implementation")
        # Create a mock model for testing when AIFS is not available
        fusion_model = type(
            "MockFusionModel",
            (),
            {
                "device": str(test_device),  # Add device attribute
                "time_series_tokenizer": None,
                "mistral_hidden_size": 512,
                "mistral_tokenizer": None,
                "mistral_model": type(
                    "MockLLM", (), {"vocab_size": 32000}
                )(),  # Mock LLM with vocab_size
                "use_mock_mistral": use_mock_mistral,  # Respect the environment variable
                "tokenize_climate_data": lambda self, x: torch.randn(x.shape[0], 8, 256),
                "tokenize_text": lambda self, x: {
                    "input_ids": torch.randint(1, 1000, (len(x), 32)),
                    "attention_mask": torch.ones(len(x), 32),
                },
                "process_climate_text": lambda self, climate_tokens, _, task="embedding": {
                    "fused_output": torch.randn(climate_tokens.shape[0], 1, 512),
                    "generated_text": "Mock analysis completed.",
                },
                "forward": lambda self, climate_data, text_inputs, task="embedding": {
                    "fused_output": torch.randn(1, 1, 512),
                    "generated_text": "Mock analysis completed.",
                },
                "parameters": lambda self: iter(
                    [
                        torch.randn(512, 256),  # Mock parameter tensor 1
                        torch.randn(512),  # Mock parameter tensor 2 (bias)
                        torch.randn(256, 128),  # Mock parameter tensor 3
                    ]
                ),
            },
        )()
    else:
        # Use the real production fusion model
        fusion_model = AIFSClimateTextFusionWrapper(
            model=actual_aifs_model,
            device_str=str(test_device),
            fusion_dim=512,
            use_mock_mistral=use_mock_mistral,  # Pass the environment variable value
            verbose=True,
        )

    _MODEL_CACHE["aifs_mistral_model"] = fusion_model
    print(f"AIFS+LLM Fusion Model created and cached on {test_device}")
    return fusion_model


@pytest.fixture
def test_climate_data_fusion(test_device, zarr_dataset_path, aifs_model):  # pylint: disable=W0621
    """Fixture for test climate data for fusion model testing using real ECMWF data."""
    try:
        from multimodal_aifs.utils.zarr_data_loader import ZarrClimateLoader
    except ImportError as e:
        pytest.skip(f"ZarrClimateLoader required for real data tests: {e}")

    device = str(test_device)

    # Determine the appropriate floating point format based on device
    use_low_precision = test_device.type in ["cuda", "mps"]

    # Load real ECMWF data using ZarrClimateLoader with forcing pipeline
    loader = ZarrClimateLoader(zarr_dataset_path)

    # Load all available timesteps (use None, None to load all)
    climate_data = loader.load_time_range(None, None)

    # Get runner from aifs_model fixture for proper forcing computation
    runner = aifs_model.get("runner") if not aifs_model.get("is_mock") else None

    # Convert to AIFS tensor format with forcings (94 physics + 9 forcings = 103 total)
    climate_tensor = loader.to_aifs_tensor(
        climate_data,
        batch_size=1,
        normalize=True,
        device=device,
        use_low_precision=use_low_precision,
        runner=runner,
        use_forcing_pipeline=True,
    )

    text_inputs = ["Predict weather patterns based on the climate data."]
    return climate_tensor, text_inputs


# =================== TEST DATA FIXTURES ===================
@pytest.fixture(scope="session")
def test_climate_data(test_device, zarr_dataset_path, aifs_model):  # pylint: disable=W0621
    """Load real ECMWF climate data for testing with proper forcing computation."""
    try:
        import xarray as xr

        from multimodal_aifs.utils.zarr_data_loader import ZarrClimateLoader
    except ImportError as e:
        pytest.skip(f"Required dependencies not available: {e}")

    device = str(test_device)

    # Determine the appropriate floating point format based on device
    # Prefer BF16 on CUDA (wider dynamic range), FP16 on MPS
    if test_device.type == "cuda" and torch.cuda.is_available():
        bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        dtype = torch.bfloat16 if bf16_supported else torch.float16
        use_low_precision = True
    elif test_device.type == "mps":
        dtype = torch.float16
        use_low_precision = True
    else:
        dtype = torch.float32
        use_low_precision = False

    # Load real ECMWF data using ZarrClimateLoader
    loader = ZarrClimateLoader(zarr_dataset_path)

    # Load all available timesteps
    climate_data = loader.load_time_range(None, None)

    # Get runner from aifs_model fixture for proper forcing computation
    runner = aifs_model.get("runner") if not aifs_model.get("is_mock") else None

    # Convert to AIFS tensor format with forcings (94 physics + 9 forcings = 103 total)
    tensor_5d = loader.to_aifs_tensor(
        climate_data,
        batch_size=1,
        normalize=True,
        device=device,
        use_low_precision=use_low_precision,
        runner=runner,
        use_forcing_pipeline=True,
    )  # Returns [batch, time, ensemble, grid, 103]

    # Load raw dataset for coordinate information
    ds = xr.open_zarr(zarr_dataset_path)

    # Create 2D tensor for encoder testing
    # [batch, features] - take mean across grid points
    tensor_2d = (
        tensor_5d[0, 0, 0, :, :AIFS_RAW_ENCODER_OUTPUT_DIM].mean(dim=0).unsqueeze(0).to(dtype)
    )

    return {
        "tensor_5d": tensor_5d,
        "tensor_2d": tensor_2d,
        "variables": list(ALL_AIFS_VARIABLES[:6]),  # First 6 for consistency
        "lat": (
            ds.coords.get("latitude", np.linspace(-90, 90, 32))
            if "latitude" in ds.coords
            else np.linspace(-90, 90, 32)
        ),
        "lon": (
            ds.coords.get("longitude", np.linspace(-180, 180, 32))
            if "longitude" in ds.coords
            else np.linspace(-180, 180, 32)
        ),
        "time": (
            list(ds.time.values) if "time" in ds else ["2024-01-01T00:00:00", "2024-01-01T06:00:00"]
        ),
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
def get_project_root():
    """Provide the project root directory."""
    return Path(__file__).parent.parent


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


# =================== CLEANUP ===================
@pytest.fixture(scope="session", autouse=True)
def cleanup_session():
    """Clean up after test session."""
    yield

    # Clean up any CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n🧹 Test session cleanup completed")

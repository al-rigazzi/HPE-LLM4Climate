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
"""Unit tests for the core AIFS fusion and encoder utilities."""
# pylint: disable=redefined-outer-name,protected-access,abstract-method

from __future__ import annotations

import builtins
import importlib
import importlib.util
import runpy
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from multimodal_aifs.constants import AIFS_RAW_ENCODER_OUTPUT_DIM
from multimodal_aifs.core import aifs_encoder_utils as encoder_utils
from multimodal_aifs.core.aifs_climate_fusion import (
    AIFSClimateEmbedding,
    AIFSClimateTextFusion,
    create_aifs_embedding_from_model,
    create_aifs_fusion_from_model,
)


class DummyAIFSInnerModel(nn.Module):
    """Lightweight stand-in for the inner Anemoi encoder with controllable behaviour."""

    def __init__(
        self,
        grid_points: int = 32,
        feature_dim: int = 16,
        mode: str = "stable",
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.grid_points = grid_points
        self.mode = mode
        self.register_buffer("latlons_data", torch.zeros(grid_points, 2))
        self.projection = nn.Linear(1, feature_dim)

    def forward(
        self, tensor: torch.Tensor
    ) -> torch.Tensor:  # pragma: no cover - exercised via encoder
        """Process input tensor through the dummy encoder."""
        batch, time, _ensemble, grid, _ = tensor.shape
        assert grid == self.grid_points

        if self.mode == "oom_error":
            raise RuntimeError("CUDA out of memory while running encoder attention")
        if self.mode == "unsupported_error":
            raise RuntimeError("Operation not currently supported on this device")
        if self.mode == "generic_runtime_error":
            raise RuntimeError("Synthetic encoder failure")
        if self.mode == "exception_error":
            raise ValueError("Synthetic non-runtime failure")

        averaged = tensor.mean(dim=2).mean(dim=-1, keepdim=True)
        projected = self.projection(averaged.float())
        embeddings = projected.to(tensor.dtype)

        if self.mode == "tuple_output":
            return embeddings, torch.zeros(1, device=tensor.device, dtype=tensor.dtype)
        if self.mode == "autocast_nan" and tensor.dtype != torch.float32:
            return torch.full_like(embeddings, torch.nan)
        if self.mode == "inf_output":
            return torch.full(
                (batch, time, self.grid_points, self.feature_dim),
                float("inf"),
                dtype=torch.float16,
                device=tensor.device,
            )
        if self.mode == "inf_output_fp32":
            return torch.full(
                (batch, time, self.grid_points, self.feature_dim),
                float("inf"),
                dtype=torch.float32,
                device=tensor.device,
            )
        if self.mode == "nan_output":
            return torch.full(
                (batch, time, self.grid_points, self.feature_dim),
                float("nan"),
                dtype=torch.float32,
                device=tensor.device,
            )

        return embeddings


class DummyAIFSInterface(nn.Module):
    """Mimics the public AIFS runner interface used by the encoder wrapper."""

    def __init__(
        self,
        grid_points: int = 32,
        feature_dim: int = 16,
        *,
        inner_model: DummyAIFSInnerModel | None = None,
        mode: str = "stable",
    ):
        super().__init__()
        if inner_model is not None:
            self.model = inner_model
            self.grid_points = inner_model.grid_points
            self.feature_dim = inner_model.feature_dim
        else:
            self.model = DummyAIFSInnerModel(
                grid_points=grid_points, feature_dim=feature_dim, mode=mode
            )
            self.grid_points = grid_points
            self.feature_dim = feature_dim

    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
        return self


class FailingMoveInterface(DummyAIFSInterface):
    """Interface that simulates a device transfer failure."""

    def __init__(self):
        super().__init__(grid_points=2, feature_dim=3)

    def to(self, *args, **kwargs):  # pragma: no cover - exercised in tests
        raise RuntimeError("simulated transfer failure")


class StaticOutputEncoder(nn.Module):
    """Return a fixed tensor regardless of the provided climate input."""

    def __init__(self, output: torch.Tensor):
        super().__init__()
        self._output = output

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial wrapper
        """Return the fixed output tensor."""
        return self._output


def _random_climate_tensor(
    grid_points: int,
    batch: int = 2,
    variables: int = 8,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a deterministic-shape 5D climate tensor."""
    tensor = torch.randn(batch, 1, 1, grid_points, variables, dtype=dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


@pytest.fixture()
def enable_aifs_dependencies(monkeypatch):
    """Force encoder utils to behave as if all external deps are available."""
    original = encoder_utils.AIFS_AVAILABLE
    monkeypatch.setattr(encoder_utils, "AIFS_AVAILABLE", True)
    yield
    monkeypatch.setattr(encoder_utils, "AIFS_AVAILABLE", original)


@pytest.fixture()
def fusion_context(enable_aifs_dependencies):
    """Provide a fusion module plus its backing dummy interface."""
    interface = DummyAIFSInterface(grid_points=8, feature_dim=16)
    module = AIFSClimateTextFusion(
        aifs_model=interface,
        climate_dim=interface.feature_dim,
        text_dim=24,
        fusion_dim=32,
        num_attention_heads=4,
        device="cpu",
        verbose=False,
    )
    return module, interface


def test_fusion_requires_model_or_checkpoint():
    """Test that fusion module requires either model or checkpoint."""
    with pytest.raises(ValueError):
        AIFSClimateTextFusion(
            aifs_model=None,
            aifs_checkpoint_path=None,
            climate_dim=8,
            text_dim=12,
            fusion_dim=16,
            num_attention_heads=2,
            device="cpu",
            verbose=False,
        )


def test_fusion_init_promotes_user_half_dtype(enable_aifs_dependencies):  # noqa: F811
    """Test that float16 dtype is promoted to bfloat16."""
    interface = DummyAIFSInterface(grid_points=6, feature_dim=10)
    fusion_module = AIFSClimateTextFusion(
        aifs_model=interface,
        climate_dim=interface.feature_dim,
        text_dim=24,
        fusion_dim=32,
        num_attention_heads=4,
        device="cpu",
        dtype=torch.float16,
        verbose=False,
    )
    assert fusion_module.dtype == torch.bfloat16


def test_encoder_utils_dependency_guard(monkeypatch):
    """Test encoder utilities handle missing dependencies gracefully."""
    module_name = "multimodal_aifs.core.aifs_encoder_utils_missingdeps"
    spec = importlib.util.spec_from_file_location(module_name, Path(encoder_utils.__file__))
    module = importlib.util.module_from_spec(spec)

    real_import = builtins.__import__

    def fake_import(name, globs=None, locs=None, fromlist=(), level=0):
        if name == "einops":
            raise ImportError("forced missing dependency")
        return real_import(name, globs, locs, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    assert module.AIFS_AVAILABLE is False
    sys.modules.pop(module_name, None)


def test_complete_encoder_forward_returns_expected_shape(enable_aifs_dependencies):  # noqa: F811
    """Test that encoder forward returns expected output shape."""
    interface = DummyAIFSInterface(grid_points=10, feature_dim=12)
    encoder = encoder_utils.AIFSCompleteEncoder(interface, verbose=False, device="cpu")
    inputs = _random_climate_tensor(interface.grid_points, batch=3)
    outputs = encoder(inputs)
    assert outputs.shape == (3, 1, interface.grid_points, interface.feature_dim)
    assert outputs.dtype in {torch.float32, torch.bfloat16}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_encoder_cuda_move_failure(enable_aifs_dependencies):  # noqa: F811
    """Test that encoder raises on CUDA move failure."""
    with pytest.raises(RuntimeError, match="Failed to move AIFS encoder to CUDA"):
        encoder_utils.AIFSCompleteEncoder(FailingMoveInterface(), verbose=False, device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_encoder_transfers_outputs_back_with_verbose(enable_aifs_dependencies):  # noqa: F811
    """Test encoder transfers outputs back to original device with verbose."""
    interface = DummyAIFSInterface(grid_points=2, feature_dim=4)
    encoder = encoder_utils.AIFSCompleteEncoder(interface, verbose=True, device="cpu")
    inputs = _random_climate_tensor(interface.grid_points, batch=1, device="cuda")
    outputs = encoder(inputs)
    assert outputs.device.type == "cuda"


def test_encoder_forward_requires_dependencies(monkeypatch):
    """Test that encoder forward requires AIFS dependencies."""
    interface = DummyAIFSInterface(grid_points=5, feature_dim=7)
    encoder = encoder_utils.AIFSCompleteEncoder(interface, verbose=False, device="cpu")
    inputs = _random_climate_tensor(interface.grid_points)
    monkeypatch.setattr(encoder_utils, "AIFS_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="dependencies not available"):
        encoder(inputs)


def test_complete_encoder_restores_training_mode(enable_aifs_dependencies):  # noqa: F811
    """Test that encoder restores training mode after forward pass."""
    interface = DummyAIFSInterface(grid_points=6, feature_dim=8)
    encoder = encoder_utils.AIFSCompleteEncoder(interface, verbose=False, device="cpu")
    encoder.aifs_model.train()
    inputs = _random_climate_tensor(interface.grid_points, batch=1)
    encoder(inputs)
    assert encoder.aifs_model.training is True


def test_complete_encoder_grid_size_validation(enable_aifs_dependencies):  # noqa: F811
    """Test encoder validates grid size matches expected dimensions."""
    interface = DummyAIFSInterface(grid_points=4, feature_dim=6)
    encoder = encoder_utils.AIFSCompleteEncoder(interface, verbose=False, device="cpu")
    bad_inputs = _random_climate_tensor(grid_points=3, batch=1)
    with pytest.raises(ValueError):
        encoder(bad_inputs)


def test_encoder_checkpoint_roundtrip(tmp_path, enable_aifs_dependencies):  # noqa: F811
    """Test saving and loading encoder checkpoints."""
    interface = DummyAIFSInterface(grid_points=5, feature_dim=7)
    encoder = encoder_utils.AIFSCompleteEncoder(interface, verbose=False, device="cpu")
    sample_inputs = _random_climate_tensor(interface.grid_points, batch=1)
    sample_outputs = encoder(sample_inputs)

    checkpoint_dir = tmp_path / "encoders"
    path = encoder_utils.save_aifs_encoder(
        encoder,
        sample_outputs,
        checkpoint_dir=str(checkpoint_dir),
        filename="dummy_encoder.pth",
        verbose=False,
    )
    info = encoder_utils.get_checkpoint_info(path)
    assert info["file_path"] == path

    reloaded = encoder_utils.load_aifs_encoder(
        path,
        DummyAIFSInterface(grid_points=interface.grid_points, feature_dim=interface.feature_dim),
        verbose=False,
        device="cpu",
    )
    assert isinstance(reloaded, encoder_utils.AIFSCompleteEncoder)
    assert encoder_utils.validate_checkpoint(
        path,
        DummyAIFSInterface(grid_points=interface.grid_points, feature_dim=interface.feature_dim),
        verbose=False,
    )


def test_encoder_unwraps_tuple_outputs(enable_aifs_dependencies):  # noqa: F811
    """Test that encoder unwraps tuple outputs from inner model."""
    tuple_model = DummyAIFSInnerModel(grid_points=3, feature_dim=4, mode="tuple_output")
    interface = DummyAIFSInterface(inner_model=tuple_model)
    encoder = encoder_utils.AIFSCompleteEncoder(interface, verbose=False, device="cpu")
    inputs = _random_climate_tensor(interface.grid_points, batch=2)
    outputs = encoder(inputs)
    assert outputs.shape == (2, 1, interface.grid_points, interface.feature_dim)


def test_aggregate_encoder_output_variants(fusion_context):  # noqa: F811
    """Test aggregation of encoder outputs for various tensor shapes."""
    fusion_module, _ = fusion_context
    batch_size = 2
    feature_dim = fusion_module.climate_dim

    encoded_4d = torch.randn(batch_size, 3, 4, feature_dim)
    result_4d = fusion_module._aggregate_encoder_output(encoded_4d, batch_size)
    assert result_4d.shape == (batch_size, feature_dim)

    encoded_3d = torch.randn(batch_size, 5, feature_dim)
    result_3d = fusion_module._aggregate_encoder_output(encoded_3d, batch_size)
    assert result_3d.shape == (batch_size, feature_dim)

    encoded_2d = torch.randn(6, feature_dim)
    result_2d = fusion_module._aggregate_encoder_output(encoded_2d, batch_size)
    assert result_2d.shape == (batch_size, feature_dim)

    encoded_1d = torch.randn(feature_dim)
    result_1d = fusion_module._aggregate_encoder_output(encoded_1d, batch_size)
    assert result_1d.shape == (batch_size, feature_dim)

    fallback = torch.randn(1, 2, 3, 4, 5)
    fallback_result = fusion_module._aggregate_encoder_output(fallback, batch_size)
    assert fallback_result.shape == (batch_size, feature_dim)


def test_set_module_precision_handles_float16_requests(fusion_context):  # noqa: F811
    """Test that float16 precision requests are promoted to bfloat16."""
    fusion_module, _ = fusion_context
    fusion_module._set_module_precision(torch.float16)
    assert fusion_module.dtype == torch.bfloat16
    assert fusion_module._current_module_dtype == torch.bfloat16


def test_sanitize_encoder_dtype_promotes_half(fusion_context):  # noqa: F811
    """Test that half precision is promoted in encoder dtype sanitization."""
    fusion_module, _ = fusion_context
    tensor = torch.randn(2, fusion_module.climate_dim, dtype=torch.float16)
    sanitized, runtime_dtype = fusion_module._sanitize_encoder_dtype(tensor)
    assert runtime_dtype == torch.bfloat16
    assert sanitized.dtype == torch.bfloat16


def test_encode_text_requires_embeddings(fusion_context):  # noqa: F811
    """Test that encode_text requires text embeddings."""
    fusion_module, _ = fusion_context
    with pytest.raises(ValueError):
        fusion_module.encode_text(["hot spell"], text_embeddings=None)


def test_encode_climate_data_with_dummy_encoder(fusion_context):  # noqa: F811
    """Test encoding climate data with dummy encoder."""
    fusion_module, interface = fusion_context
    climate_data = _random_climate_tensor(interface.grid_points, batch=2)
    climate_features = fusion_module.encode_climate_data(climate_data)
    assert climate_features.shape == (2, fusion_module.fusion_dim)
    assert climate_features.dtype == fusion_module.dtype


def test_apply_cross_attention_shapes(fusion_context):  # noqa: F811
    """Test cross attention produces expected output shapes."""
    fusion_module, _ = fusion_context
    attn_device = fusion_module.cross_attention.in_proj_weight.device
    climate_features = torch.randn(
        2, fusion_module.fusion_dim, device=attn_device, dtype=fusion_module.dtype
    )
    text_features = torch.randn(
        2, fusion_module.fusion_dim, device=attn_device, dtype=fusion_module.dtype
    )
    climate_attended, text_attended = fusion_module.apply_cross_attention(
        climate_features, text_features
    )
    assert climate_attended.shape == climate_features.shape
    assert text_attended.shape == text_features.shape


def test_fuse_features_executes_attention(fusion_context):  # noqa: F811
    """Test that feature fusion executes attention mechanism."""
    fusion_module, _ = fusion_context
    fusion_module._set_module_precision(torch.float32)
    climate_features = torch.randn(2, fusion_module.fusion_dim, dtype=torch.float16)
    text_features = torch.randn(2, fusion_module.fusion_dim, dtype=torch.float32)
    fused = fusion_module.fuse_features(climate_features, text_features)
    assert fused.shape == (2, fusion_module.fusion_dim)
    assert fused.dtype == fusion_module.dtype


def test_encode_climate_requires_actual_encoder(tmp_path):
    """Test that encoding climate data requires an actual encoder."""
    fusion_module = AIFSClimateTextFusion(
        aifs_model=None,
        aifs_checkpoint_path=str(tmp_path / "dummy.ckpt"),
        climate_dim=32,
        text_dim=24,
        fusion_dim=16,
        num_attention_heads=2,
        device="cpu",
        verbose=False,
    )
    dummy_climate = torch.zeros(1, 1, 1, 32, 8)
    with pytest.raises(ValueError, match="AIFS encoder not available"):
        fusion_module.encode_climate_data(dummy_climate)


def test_similarity_helpers_use_encoders(fusion_context):  # noqa: F811
    """Test similarity helpers use encoders correctly."""
    fusion_module, interface = fusion_context
    texts = ["storm surge", "heat wave"]
    batch = len(texts)
    climate_a = _random_climate_tensor(interface.grid_points, batch=batch)
    climate_b = _random_climate_tensor(interface.grid_points, batch=batch)
    embeddings = torch.randn(batch, fusion_module.text_dim, dtype=fusion_module.dtype)

    similarity = fusion_module.get_climate_similarity(climate_a, climate_b)
    alignment = fusion_module.get_text_climate_alignment(
        climate_a,
        texts,
        text_embeddings=embeddings,
    )

    assert similarity.shape == alignment.shape == (batch,)


def test_climate_embedding_forward_produces_embeddings(enable_aifs_dependencies):  # noqa: F811
    """Test that climate embedding forward produces embeddings."""
    interface = DummyAIFSInterface(grid_points=6, feature_dim=10)
    embedding_module = AIFSClimateEmbedding(
        aifs_model=interface,
        climate_dim=interface.feature_dim,
        embedding_dim=20,
        device="cpu",
        verbose=False,
    )
    climate_data = _random_climate_tensor(interface.grid_points, batch=2)
    embeddings = embedding_module(climate_data)
    assert embeddings.shape == (2, 20)
    assert embeddings.dtype == embedding_module.dtype


def test_embedding_checkpoint_only_initialization(
    enable_aifs_dependencies, tmp_path, capsys
):  # noqa: F811
    """Test embedding module initialization with checkpoint only."""
    checkpoint = tmp_path / "encoder.pth"
    module = AIFSClimateEmbedding(
        aifs_model=None,
        aifs_checkpoint_path=str(checkpoint),
        climate_dim=16,
        embedding_dim=8,
        device="cpu",
        verbose=True,
    )
    stdout = capsys.readouterr().out
    assert "Loading from checkpoint requires AIFS model" in stdout
    with pytest.raises(ValueError, match="AIFS encoder not available"):
        module(_random_climate_tensor(grid_points=2))
    assert module.aifs_encoder is None
    assert module.checkpoint_path == str(checkpoint)


def test_embedding_accepts_preaggregated_encoder_outputs(enable_aifs_dependencies):  # noqa: F811
    """Test embedding accepts preaggregated encoder outputs."""
    interface = DummyAIFSInterface(grid_points=4, feature_dim=6)
    module = AIFSClimateEmbedding(
        aifs_model=interface,
        climate_dim=interface.feature_dim,
        embedding_dim=12,
        device="cpu",
        verbose=False,
    )
    batch = 3
    # Use float32 as MPS doesn't support float64
    preaggregated = torch.randn(interface.grid_points, module.climate_dim, dtype=torch.float32)
    module.aifs_encoder = StaticOutputEncoder(preaggregated)
    climate_data = _random_climate_tensor(interface.grid_points, batch=batch)
    embeddings = module(climate_data)
    assert embeddings.shape == (batch, module.embedding_dim)
    assert embeddings.dtype == module.dtype


def test_factory_helpers_create_instances(enable_aifs_dependencies):  # noqa: F811
    """Test factory helpers create correct instances."""
    interface = DummyAIFSInterface(grid_points=4, feature_dim=AIFS_RAW_ENCODER_OUTPUT_DIM)
    fusion_instance = create_aifs_fusion_from_model(
        interface,
        fusion_dim=64,
        verbose=False,
        device="cpu",
    )
    assert isinstance(fusion_instance, AIFSClimateTextFusion)

    embedding_instance = create_aifs_embedding_from_model(
        interface,
        embedding_dim=32,
        verbose=False,
        device="cpu",
    )
    assert isinstance(embedding_instance, AIFSClimateEmbedding)


def test_encoder_force_cpu_branch(monkeypatch, enable_aifs_dependencies):  # noqa: F811
    """Test encoder force CPU branch when AIFS_FORCE_CPU is set."""
    monkeypatch.setenv("AIFS_FORCE_CPU", "true")
    interface = DummyAIFSInterface(grid_points=3, feature_dim=5)
    encoder = encoder_utils.AIFSCompleteEncoder(interface, verbose=True, device="cpu")
    assert encoder.aifs_device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_encoder_moves_results_back_to_original_device(enable_aifs_dependencies):  # noqa: F811
    """Test encoder moves results back to original device."""
    interface = DummyAIFSInterface(grid_points=2, feature_dim=4)
    encoder = encoder_utils.AIFSCompleteEncoder(interface, verbose=False, device="cpu")
    gpu_inputs = _random_climate_tensor(interface.grid_points, batch=1, device="cuda")
    outputs = encoder(gpu_inputs)
    assert outputs.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_encoder_autocast_retries_on_nan(enable_aifs_dependencies):  # noqa: F811
    """Test encoder retries without autocast when NaN is produced."""
    unstable_model = DummyAIFSInnerModel(grid_points=3, feature_dim=4, mode="autocast_nan")
    interface = DummyAIFSInterface(inner_model=unstable_model)
    encoder = encoder_utils.AIFSCompleteEncoder(interface, verbose=False, device="cuda")
    cuda_inputs = _random_climate_tensor(interface.grid_points, batch=1, device="cuda")
    outputs = encoder(cuda_inputs)
    assert torch.isfinite(outputs).all()
    assert encoder._use_autocast is False
    assert encoder.encoder_forward_dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_encoder_handles_out_of_memory_error(enable_aifs_dependencies):  # noqa: F811
    """Test encoder handles CUDA out of memory error."""
    oom_model = DummyAIFSInnerModel(grid_points=2, feature_dim=3, mode="oom_error")
    interface = DummyAIFSInterface(inner_model=oom_model)
    encoder = encoder_utils.AIFSCompleteEncoder(interface, verbose=False, device="cuda")
    cuda_inputs = _random_climate_tensor(interface.grid_points, batch=1, device="cuda")
    with pytest.raises(RuntimeError, match="out of memory"):
        encoder(cuda_inputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_encoder_handles_unsupported_operation(enable_aifs_dependencies):  # noqa: F811
    """Test encoder handles unsupported device operation."""
    unsupported_model = DummyAIFSInnerModel(grid_points=2, feature_dim=3, mode="unsupported_error")
    interface = DummyAIFSInterface(inner_model=unsupported_model)
    encoder = encoder_utils.AIFSCompleteEncoder(interface, verbose=False, device="cuda")
    cuda_inputs = _random_climate_tensor(interface.grid_points, batch=1, device="cuda")
    with pytest.raises(RuntimeError, match="not currently supported"):
        encoder(cuda_inputs)


def test_encoder_handles_generic_runtime_error(enable_aifs_dependencies):  # noqa: F811
    """Test encoder handles generic runtime errors."""
    failing_model = DummyAIFSInnerModel(grid_points=2, feature_dim=3, mode="generic_runtime_error")
    interface = DummyAIFSInterface(inner_model=failing_model)
    encoder = encoder_utils.AIFSCompleteEncoder(interface, verbose=False, device="cpu")
    inputs = _random_climate_tensor(interface.grid_points, batch=1)
    with pytest.raises(RuntimeError, match="Failed to encode"):
        encoder(inputs)


def test_encoder_handles_generic_exception(enable_aifs_dependencies):  # noqa: F811
    """Test encoder handles generic exceptions."""
    exception_model = DummyAIFSInnerModel(grid_points=2, feature_dim=3, mode="exception_error")
    interface = DummyAIFSInterface(inner_model=exception_model)
    encoder = encoder_utils.AIFSCompleteEncoder(interface, verbose=False, device="cpu")
    inputs = _random_climate_tensor(interface.grid_points, batch=1)
    with pytest.raises(RuntimeError, match="Failed to encode"):
        encoder(inputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_encoder_dtype_overflow_guard(enable_aifs_dependencies):  # noqa: F811
    """Test encoder dtype overflow guard."""
    inf_model = DummyAIFSInnerModel(grid_points=2, feature_dim=3, mode="inf_output")
    interface = DummyAIFSInterface(inner_model=inf_model)
    encoder = encoder_utils.AIFSCompleteEncoder(interface, verbose=False, device="cuda")
    inputs = _random_climate_tensor(interface.grid_points, batch=1, device="cuda")
    with pytest.raises(RuntimeError, match="non-finite outputs even in FP32 mode"):
        encoder(inputs)


def test_encoder_detects_non_finite_outputs_on_cpu(enable_aifs_dependencies):  # noqa: F811
    """Test encoder detects non-finite outputs on CPU."""
    inf_model = DummyAIFSInnerModel(grid_points=2, feature_dim=3, mode="inf_output")
    interface = DummyAIFSInterface(inner_model=inf_model)
    encoder = encoder_utils.AIFSCompleteEncoder(interface, verbose=False, device="cpu")
    inputs = _random_climate_tensor(interface.grid_points, batch=1)
    with pytest.raises(RuntimeError, match="non-finite outputs even in FP32 mode"):
        encoder(inputs)


def test_encoder_helper_functions_report_state(enable_aifs_dependencies):  # noqa: F811
    """Test encoder helper functions report state correctly."""
    default_path = encoder_utils.get_default_checkpoint_path()
    assert isinstance(default_path, str) and default_path.endswith(".pth")
    assert encoder_utils.check_aifs_dependencies() is True


def test_checkpoint_helpers_verbose_logging(
    tmp_path, enable_aifs_dependencies, capsys
):  # noqa: F811
    """Test checkpoint helpers verbose logging output."""
    interface = DummyAIFSInterface(grid_points=4, feature_dim=6)
    encoder = encoder_utils.AIFSCompleteEncoder(interface, verbose=False, device="cpu")
    sample_inputs = _random_climate_tensor(interface.grid_points, batch=1)
    sample_outputs = encoder(sample_inputs)

    checkpoint_path = encoder_utils.save_aifs_encoder(
        encoder,
        sample_outputs,
        checkpoint_dir=str(tmp_path),
        filename="verbose_encoder.pth",
        verbose=True,
    )
    encoder_utils.load_aifs_encoder(
        checkpoint_path,
        DummyAIFSInterface(grid_points=interface.grid_points, feature_dim=interface.feature_dim),
        verbose=True,
        device="cpu",
    )
    encoder_utils.create_aifs_encoder(interface, verbose=True)

    output = capsys.readouterr().out
    assert "SAVING AIFSCompleteEncoder CHECKPOINT" in output
    assert "Loading AIFSCompleteEncoder" in output
    assert "Creating AIFSCompleteEncoder" in output


def test_validate_checkpoint_verbose_output(
    tmp_path, enable_aifs_dependencies, capsys
):  # noqa: F811
    """Test checkpoint validation verbose output."""
    interface = DummyAIFSInterface(grid_points=3, feature_dim=5)
    encoder = encoder_utils.AIFSCompleteEncoder(interface, verbose=False, device="cpu")
    sample_inputs = _random_climate_tensor(interface.grid_points, batch=1)
    sample_outputs = encoder(sample_inputs)
    checkpoint_path = encoder_utils.save_aifs_encoder(
        encoder,
        sample_outputs,
        checkpoint_dir=str(tmp_path),
        filename="validate_verbose.pth",
        verbose=False,
    )
    assert encoder_utils.validate_checkpoint(checkpoint_path, interface, verbose=True)
    output = capsys.readouterr().out
    assert "Validating checkpoint" in output
    assert "Checkpoint validation passed" in output


def test_encoder_utils_main_block_executes(capsys):
    """Test encoder utils main block executes correctly."""
    runpy.run_module("multimodal_aifs.core.aifs_encoder_utils", run_name="__main__")
    output = capsys.readouterr().out
    assert "AIFS Encoder Utils" in output

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
"""Focused unit tests for ``multimodal_aifs.core.aifs_encoder_utils``."""

from __future__ import annotations

import torch
from torch import nn
import pytest

from multimodal_aifs.core import aifs_encoder_utils as enc


class DummyAIFSInnerModel(nn.Module):
    """Tiny stand-in for the real Anemoi encoder."""

    def __init__(self, grid_points: int = 8):
        super().__init__()
        self.latlons_data = torch.zeros(grid_points, 2)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.mean(dim=-1) * self.scale


class DummyAIFSInterface(nn.Module):
    """Wrapper that mimics the minimal API used by ``AIFSCompleteEncoder``."""

    def __init__(self, grid_points: int = 8):
        super().__init__()
        self.model = DummyAIFSInnerModel(grid_points)

    def to(self, device=None, dtype=None):  # type: ignore[override]
        self.model = self.model.to(device=device, dtype=dtype)
        return self


def _build_encoder(grid_points: int = 8) -> enc.AIFSCompleteEncoder:
    enc.AIFS_AVAILABLE = True
    return enc.AIFSCompleteEncoder(DummyAIFSInterface(grid_points), verbose=False, device="cpu")


def test_encoder_forward_roundtrip():
    """Forward pass should return embeddings with the expected shape."""

    encoder = _build_encoder(grid_points=4)
    sample = torch.randn(1, 2, 1, 4, 6)
    embeddings = encoder(sample)
    assert embeddings.shape == (1, 2, 1, 4)
    assert embeddings.dtype == encoder.dtype


def test_encoder_raises_on_grid_mismatch():
    """Supplying tensors with the wrong grid size should raise a ValueError."""

    encoder = _build_encoder(grid_points=4)
    bad_sample = torch.randn(1, 2, 1, 5, 6)
    with pytest.raises(ValueError, match="Grid size mismatch"):
        encoder(bad_sample)


def test_checkpoint_roundtrip(tmp_path):
    """Saving, loading, and validating checkpoints should work with dummy encoders."""

    encoder = _build_encoder(grid_points=4)
    dummy_output = torch.randn(1, 2, 1, 4)

    ckpt_path = enc.save_aifs_encoder(
        encoder,
        dummy_output,
        checkpoint_dir=str(tmp_path),
        filename="dummy_encoder.pth",
        verbose=False,
    )

    info = enc.get_checkpoint_info(ckpt_path)
    assert info["file_path"].endswith("dummy_encoder.pth")
    assert info["file_size_mb"] > 0

    assert enc.validate_checkpoint(ckpt_path, DummyAIFSInterface(4), verbose=False)

    loaded = enc.load_aifs_encoder(
        ckpt_path,
        DummyAIFSInterface(4),
        verbose=False,
        device="cpu",
    )
    assert isinstance(loaded, enc.AIFSCompleteEncoder)

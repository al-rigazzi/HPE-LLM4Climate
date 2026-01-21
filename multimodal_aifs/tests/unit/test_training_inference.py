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
"""Unit tests for ``multimodal_aifs.training.inference``."""

# pylint: disable=abstract-method,arguments-differ

from __future__ import annotations

import torch
import yaml
from torch import nn

from multimodal_aifs.training.inference import AIFSMultimodalInference


class DummyFusionModel(nn.Module):
    """Tiny stand-in for ``AIFSClimateTextFusion``."""

    def __init__(self, *_, **__):
        super().__init__()
        self._device = torch.device("cpu")

    def to(self, device=None, **_):  # type: ignore[override]
        if device is not None:
            self._device = torch.device(device)
        return self

    def eval(self):  # type: ignore[override]
        return self

    def __call__(self, climate_batch, text_inputs):
        batch = climate_batch["dynamic"].shape[0]
        fused = torch.arange(batch * 4, dtype=torch.float32, device=self._device).view(batch, 4)
        return {
            "fused_features": fused,
            "climate_features": fused + 1,
            "text_features": fused + 2,
            "attention_weights": torch.ones(batch, 1, 1, device=self._device),
        }


class DummyTokenizer:
    """Simple tokenizer stub returned by ``AutoTokenizer.from_pretrained``."""

    def __init__(self):
        self.pad_token = None
        self.pad_token_id = None
        self.eos_token = "<eos>"
        self.eos_token_id = 0

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        """Create a dummy tokenizer instance."""
        return cls()


def test_inference_predict_batch_and_location(tmp_path, monkeypatch):  # noqa: F811
    """Test AIFSMultimodalInference model/tokenizer orchestration."""

    monkeypatch.setattr(
        "multimodal_aifs.training.inference.AIFSClimateTextFusion",
        DummyFusionModel,
    )
    monkeypatch.setattr(
        "multimodal_aifs.training.inference.AutoTokenizer",
        DummyTokenizer,
    )

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "tokenizer").mkdir()

    config = {
        "model": {
            "aifs_encoder_path": "dummy_encoder.pth",
            "climate_dim": 4,
            "text_dim": 4,
            "fusion_dim": 4,
            "num_fusion_layers": 1,
            "dropout": 0.0,
            "llama_model_name": "dummy/model",
        }
    }
    (checkpoint_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    inference = AIFSMultimodalInference(str(checkpoint_dir), device="cpu")

    climate_sample = torch.randn(2, 200, 4, 4)
    result = inference.predict(climate_sample, "Describe temperature", return_attention=True)
    assert result["fused_features"].shape == (1, 4)
    assert "attention_weights" in result

    climate_list = [torch.randn(2, 200, 4, 4) for _ in range(3)]
    queries = ["q1", "q2", "q3"]
    batch_outputs = inference.batch_predict(climate_list, queries, batch_size=2)
    assert len(batch_outputs) == 3
    assert all("fused_features" in item for item in batch_outputs)

    summary = inference.analyze_location(climate_sample, "Berlin", analysis_type="trends")
    assert "Berlin" in summary

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
Unit tests for checkpoint management.

Tests the CheckpointManager module which handles saving and loading
checkpoints across different training stages.
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

from multimodal_aifs.training.checkpoint_manager import (
    CheckpointManager,
    CheckpointMetadata,
    TrainingStage,
)


class SimpleModel(nn.Module):
    """Simple model for testing checkpoints."""

    def __init__(self, input_dim: int = 10, output_dim: int = 5):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class TestTrainingStage:
    """Tests for TrainingStage enum."""

    def test_stage_values(self):
        """Test that training stages have correct values."""
        assert TrainingStage.RL_PRETRAINING.value == "rl_pretraining"
        assert TrainingStage.SUPERVISED_FINETUNING.value == "supervised_finetuning"
        assert TrainingStage.EVALUATION.value == "evaluation"

    def test_stage_from_string(self):
        """Test creating stage from string value."""
        stage = TrainingStage("rl_pretraining")
        assert stage == TrainingStage.RL_PRETRAINING


class TestCheckpointMetadata:
    """Tests for CheckpointMetadata dataclass."""

    def test_creation(self):
        """Test creating checkpoint metadata."""
        metadata = CheckpointMetadata(
            stage=TrainingStage.RL_PRETRAINING,
            step=1000,
            epoch=5,
            metrics={"reward": 0.85},
            config={"lr": 1e-5},
        )

        assert metadata.stage == TrainingStage.RL_PRETRAINING
        assert metadata.step == 1000
        assert metadata.epoch == 5
        assert metadata.metrics["reward"] == 0.85

    def test_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = CheckpointMetadata(
            stage=TrainingStage.RL_PRETRAINING,
            step=1000,
            epoch=5,
            metrics={"reward": 0.85},
        )

        data = metadata.to_dict()

        assert data["stage"] == "rl_pretraining"
        assert data["step"] == 1000
        assert data["epoch"] == 5
        assert data["metrics"]["reward"] == 0.85

    def test_from_dict(self):
        """Test creating metadata from dictionary."""
        data = {
            "stage": "supervised_finetuning",
            "step": 2000,
            "epoch": 10,
            "timestamp": "2025-01-28T10:00:00",
            "metrics": {"loss": 0.5},
            "config": {},
            "previous_stages": ["rl_pretraining"],
        }

        metadata = CheckpointMetadata.from_dict(data)

        assert metadata.stage == TrainingStage.SUPERVISED_FINETUNING
        assert metadata.step == 2000
        assert metadata.epoch == 10
        assert "rl_pretraining" in metadata.previous_stages

    def test_roundtrip(self):
        """Test metadata survives roundtrip serialization."""
        original = CheckpointMetadata(
            stage=TrainingStage.RL_PRETRAINING,
            step=1500,
            epoch=7,
            metrics={"reward": 0.9, "accuracy": 0.85},
            config={"lr": 2e-5, "batch_size": 8},
            previous_stages=[],
        )

        data = original.to_dict()
        restored = CheckpointMetadata.from_dict(data)

        assert restored.stage == original.stage
        assert restored.step == original.step
        assert restored.epoch == original.epoch
        assert restored.metrics == original.metrics
        assert restored.config == original.config


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create a temporary directory for checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def checkpoint_manager(self, temp_checkpoint_dir):
        """Create a checkpoint manager instance."""
        return CheckpointManager(
            checkpoint_dir=temp_checkpoint_dir,
            max_checkpoints_per_stage=3,
        )

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return SimpleModel()

    @pytest.fixture
    def optimizer(self, simple_model):
        """Create an optimizer for testing."""
        return torch.optim.Adam(simple_model.parameters(), lr=1e-4)

    def test_initialization(self, temp_checkpoint_dir):
        """Test checkpoint manager initialization."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

        assert manager.checkpoint_dir == temp_checkpoint_dir
        assert manager.checkpoint_dir.exists()

    def test_save_checkpoint(self, checkpoint_manager, simple_model, optimizer):
        """Test saving a checkpoint."""
        metadata = CheckpointMetadata(
            stage=TrainingStage.RL_PRETRAINING,
            step=100,
            epoch=1,
            metrics={"reward": 0.7},
        )

        path = checkpoint_manager.save_checkpoint(
            model=simple_model,
            optimizer=optimizer,
            scheduler=None,
            metadata=metadata,
        )

        assert path.exists()
        assert (path / "model.pt").exists()
        assert (path / "optimizer.pt").exists()
        assert (path / "metadata.json").exists()

    def test_load_checkpoint(self, checkpoint_manager, simple_model, optimizer):
        """Test loading a checkpoint."""
        original_weights = simple_model.linear.weight.clone()
        simple_model.linear.weight.data.fill_(1.0)

        metadata = CheckpointMetadata(
            stage=TrainingStage.RL_PRETRAINING,
            step=100,
            epoch=1,
        )

        checkpoint_manager.save_checkpoint(
            model=simple_model,
            optimizer=optimizer,
            scheduler=None,
            metadata=metadata,
        )

        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters())

        loaded_metadata = checkpoint_manager.load_checkpoint(
            model=new_model,
            optimizer=new_optimizer,
            stage=TrainingStage.RL_PRETRAINING,
            step=100,
        )

        assert loaded_metadata.step == 100
        assert loaded_metadata.epoch == 1
        assert torch.allclose(new_model.linear.weight, simple_model.linear.weight)

    def test_save_best_checkpoint(self, checkpoint_manager, simple_model, optimizer):
        """Test saving a best checkpoint."""
        metadata = CheckpointMetadata(
            stage=TrainingStage.RL_PRETRAINING,
            step=500,
            epoch=5,
            metrics={"reward": 0.95},
        )

        path = checkpoint_manager.save_checkpoint(
            model=simple_model,
            optimizer=optimizer,
            scheduler=None,
            metadata=metadata,
            is_best=True,
        )

        assert "best" in str(path)
        assert path.exists()

    def test_get_latest_checkpoint(self, checkpoint_manager, simple_model, optimizer):
        """Test getting the latest checkpoint."""
        for step in [100, 200, 300]:
            metadata = CheckpointMetadata(
                stage=TrainingStage.RL_PRETRAINING,
                step=step,
                epoch=step // 100,
            )
            checkpoint_manager.save_checkpoint(
                model=simple_model,
                optimizer=optimizer,
                scheduler=None,
                metadata=metadata,
            )

        path, metadata = checkpoint_manager.get_latest_checkpoint(TrainingStage.RL_PRETRAINING)

        assert path is not None
        assert metadata is not None
        assert metadata.step == 300

    def test_get_best_checkpoint(self, checkpoint_manager, simple_model, optimizer):
        """Test getting the best checkpoint."""
        metadata = CheckpointMetadata(
            stage=TrainingStage.RL_PRETRAINING,
            step=250,
            epoch=2,
            metrics={"reward": 0.9},
        )

        checkpoint_manager.save_checkpoint(
            model=simple_model,
            optimizer=optimizer,
            scheduler=None,
            metadata=metadata,
            is_best=True,
        )

        path, loaded_metadata = checkpoint_manager.get_best_checkpoint(TrainingStage.RL_PRETRAINING)

        assert path is not None
        assert loaded_metadata is not None
        assert loaded_metadata.step == 250

    def test_has_checkpoint(self, checkpoint_manager, simple_model, optimizer):
        """Test checking if checkpoint exists."""
        assert not checkpoint_manager.has_checkpoint(TrainingStage.RL_PRETRAINING)

        metadata = CheckpointMetadata(
            stage=TrainingStage.RL_PRETRAINING,
            step=100,
            epoch=1,
        )
        checkpoint_manager.save_checkpoint(
            model=simple_model,
            optimizer=optimizer,
            scheduler=None,
            metadata=metadata,
        )

        assert checkpoint_manager.has_checkpoint(TrainingStage.RL_PRETRAINING)
        assert checkpoint_manager.has_checkpoint(TrainingStage.RL_PRETRAINING, step=100)
        assert not checkpoint_manager.has_checkpoint(TrainingStage.RL_PRETRAINING, step=200)

    def test_checkpoint_cleanup(self, temp_checkpoint_dir, simple_model, optimizer):
        """Test that old checkpoints are cleaned up."""
        manager = CheckpointManager(
            checkpoint_dir=temp_checkpoint_dir,
            max_checkpoints_per_stage=2,
        )

        for step in [100, 200, 300, 400]:
            metadata = CheckpointMetadata(
                stage=TrainingStage.RL_PRETRAINING,
                step=step,
                epoch=step // 100,
            )
            manager.save_checkpoint(
                model=simple_model,
                optimizer=optimizer,
                scheduler=None,
                metadata=metadata,
            )

        stage_dir = temp_checkpoint_dir / "rl_pretraining"
        checkpoints = list(stage_dir.glob("step_*"))

        assert len(checkpoints) == 2
        checkpoint_steps = sorted(int(c.name.split("_")[1]) for c in checkpoints)
        assert checkpoint_steps == [300, 400]

    def test_training_history(self, checkpoint_manager, simple_model, optimizer):
        """Test training history tracking."""
        for step in [100, 200]:
            metadata = CheckpointMetadata(
                stage=TrainingStage.RL_PRETRAINING,
                step=step,
                epoch=step // 100,
                metrics={"reward": step / 1000},
            )
            checkpoint_manager.save_checkpoint(
                model=simple_model,
                optimizer=optimizer,
                scheduler=None,
                metadata=metadata,
            )

        history = checkpoint_manager.get_training_history()

        assert len(history) == 2
        assert history[0]["step"] == 100
        assert history[1]["step"] == 200

    def test_get_previous_stage_path(self, checkpoint_manager, simple_model, optimizer):
        """Test getting previous stage path for SFT."""
        metadata = CheckpointMetadata(
            stage=TrainingStage.RL_PRETRAINING,
            step=500,
            epoch=5,
        )
        checkpoint_manager.save_checkpoint(
            model=simple_model,
            optimizer=optimizer,
            scheduler=None,
            metadata=metadata,
            is_best=True,
        )

        rl_path = checkpoint_manager.get_previous_stage_path(TrainingStage.SUPERVISED_FINETUNING)

        assert rl_path is not None
        assert "rl_pretraining" in str(rl_path)

    def test_load_nonexistent_checkpoint(self, checkpoint_manager, simple_model):
        """Test loading a checkpoint that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            checkpoint_manager.load_checkpoint(
                model=simple_model,
                stage=TrainingStage.RL_PRETRAINING,
                step=9999,
            )

    def test_save_without_optimizer(self, checkpoint_manager, simple_model):
        """Test saving checkpoint without optimizer."""
        metadata = CheckpointMetadata(
            stage=TrainingStage.RL_PRETRAINING,
            step=100,
            epoch=1,
        )

        path = checkpoint_manager.save_checkpoint(
            model=simple_model,
            optimizer=None,
            scheduler=None,
            metadata=metadata,
        )

        assert path.exists()
        assert (path / "model.pt").exists()
        assert not (path / "optimizer.pt").exists()

    def test_cross_stage_checkpoint_loading(self, checkpoint_manager, simple_model, optimizer):
        """Test loading RL checkpoint for SFT initialization."""
        rl_metadata = CheckpointMetadata(
            stage=TrainingStage.RL_PRETRAINING,
            step=1000,
            epoch=10,
            metrics={"reward": 0.92},
        )

        simple_model.linear.weight.data.fill_(0.5)

        checkpoint_manager.save_checkpoint(
            model=simple_model,
            optimizer=optimizer,
            scheduler=None,
            metadata=rl_metadata,
            is_best=True,
        )

        new_model = SimpleModel()
        new_model.linear.weight.data.fill_(0.0)

        rl_checkpoint_path = checkpoint_manager.get_previous_stage_path(
            TrainingStage.SUPERVISED_FINETUNING
        )

        checkpoint_manager.load_checkpoint(
            model=new_model,
            checkpoint_path=rl_checkpoint_path,
        )

        assert torch.allclose(new_model.linear.weight, simple_model.linear.weight)


class TestCheckpointManagerPersistence:
    """Tests for checkpoint manager persistence across instances."""

    def test_training_log_persistence(self):
        """Test that training log persists across manager instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)

            model = SimpleModel()
            optimizer = torch.optim.Adam(model.parameters())

            manager1 = CheckpointManager(checkpoint_dir)
            metadata = CheckpointMetadata(
                stage=TrainingStage.RL_PRETRAINING,
                step=100,
                epoch=1,
            )
            manager1.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=None,
                metadata=metadata,
            )

            manager2 = CheckpointManager(checkpoint_dir)
            history = manager2.get_training_history()

            assert len(history) == 1
            assert history[0]["step"] == 100

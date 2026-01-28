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
Checkpoint management for multi-stage training.

This module provides a unified checkpoint structure that supports both
RL pre-training and supervised fine-tuning stages, ensuring consistency
across the training pipeline.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import torch
from torch import nn


class TrainingStage(Enum):
    """Enumeration of training stages."""

    RL_PRETRAINING = "rl_pretraining"
    SUPERVISED_FINETUNING = "supervised_finetuning"
    EVALUATION = "evaluation"


@dataclass
class CheckpointMetadata:
    """Metadata stored with each checkpoint."""

    stage: TrainingStage
    step: int
    epoch: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metrics: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    previous_stages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert metadata to dictionary for serialization."""
        return {
            "stage": self.stage.value,
            "step": self.step,
            "epoch": self.epoch,
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "config": self.config,
            "previous_stages": self.previous_stages,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointMetadata":
        """Create metadata from dictionary."""
        return cls(
            stage=TrainingStage(data["stage"]),
            step=data["step"],
            epoch=data["epoch"],
            timestamp=data.get("timestamp", ""),
            metrics=data.get("metrics", {}),
            config=data.get("config", {}),
            previous_stages=data.get("previous_stages", []),
        )


class CheckpointManager:
    """
    Manages model checkpoints across training stages.

    This class provides a unified interface for saving and loading checkpoints
    with consistent structure across RL pre-training and supervised fine-tuning.

    Checkpoint structure:
    ```
    checkpoint_dir/
      ├── rl_pretraining/
      │   ├── step_1000/
      │   │   ├── model.pt
      │   │   ├── optimizer.pt
      │   │   ├── scheduler.pt (optional)
      │   │   └── metadata.json
      │   └── best/
      │       └── ...
      ├── supervised_finetuning/
      │   └── ...
      └── training_log.json
    ```
    """

    MODEL_FILENAME = "model.pt"
    OPTIMIZER_FILENAME = "optimizer.pt"
    SCHEDULER_FILENAME = "scheduler.pt"
    METADATA_FILENAME = "metadata.json"
    TRAINING_LOG_FILENAME = "training_log.json"

    def __init__(
        self,
        checkpoint_dir: str | Path,
        max_checkpoints_per_stage: int = 5,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Base directory for checkpoints
            max_checkpoints_per_stage: Maximum checkpoints to keep per stage
            save_optimizer: Whether to save optimizer state
            save_scheduler: Whether to save scheduler state
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints_per_stage = max_checkpoints_per_stage
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._training_log: list[dict] = []
        self._load_training_log()

    def _load_training_log(self) -> None:
        """Load existing training log if available."""
        log_path = self.checkpoint_dir / self.TRAINING_LOG_FILENAME
        if log_path.exists():
            with open(log_path, encoding="utf-8") as f:
                self._training_log = json.load(f)

    def _save_training_log(self) -> None:
        """Save training log to disk."""
        log_path = self.checkpoint_dir / self.TRAINING_LOG_FILENAME
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self._training_log, f, indent=2)

    def _get_stage_dir(self, stage: TrainingStage) -> Path:
        """Get directory for a specific training stage."""
        stage_dir = self.checkpoint_dir / stage.value
        stage_dir.mkdir(parents=True, exist_ok=True)
        return stage_dir

    def _get_checkpoint_path(
        self, stage: TrainingStage, step: int | None = None, is_best: bool = False
    ) -> Path:
        """Get path for a specific checkpoint."""
        stage_dir = self._get_stage_dir(stage)
        if is_best:
            return stage_dir / "best"
        if step is not None:
            return stage_dir / f"step_{step}"
        raise ValueError("Either step or is_best must be specified")

    def _cleanup_old_checkpoints(self, stage: TrainingStage) -> None:
        """Remove old checkpoints exceeding the maximum count."""
        stage_dir = self._get_stage_dir(stage)
        checkpoints = sorted(
            [d for d in stage_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
            key=lambda x: int(x.name.split("_")[1]),
        )

        while len(checkpoints) > self.max_checkpoints_per_stage:
            oldest = checkpoints.pop(0)
            for file in oldest.iterdir():
                file.unlink()
            oldest.rmdir()

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        metadata: CheckpointMetadata,
        is_best: bool = False,
    ) -> Path:
        """
        Save a training checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state (optional)
            scheduler: Learning rate scheduler (optional)
            metadata: Checkpoint metadata
            is_best: Whether this is the best checkpoint so far

        Returns:
            Path to the saved checkpoint directory
        """
        checkpoint_path = self._get_checkpoint_path(metadata.stage, metadata.step, is_best=is_best)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        model_path = checkpoint_path / self.MODEL_FILENAME
        torch.save(model.state_dict(), model_path)

        if self.save_optimizer and optimizer is not None:
            optimizer_path = checkpoint_path / self.OPTIMIZER_FILENAME
            torch.save(optimizer.state_dict(), optimizer_path)

        if self.save_scheduler and scheduler is not None:
            scheduler_path = checkpoint_path / self.SCHEDULER_FILENAME
            torch.save(scheduler.state_dict(), scheduler_path)

        metadata_path = checkpoint_path / self.METADATA_FILENAME
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        log_entry = {
            "stage": metadata.stage.value,
            "step": metadata.step,
            "epoch": metadata.epoch,
            "timestamp": metadata.timestamp,
            "path": str(checkpoint_path),
            "is_best": is_best,
            "metrics": metadata.metrics,
        }
        self._training_log.append(log_entry)
        self._save_training_log()

        if not is_best:
            self._cleanup_old_checkpoints(metadata.stage)

        return checkpoint_path

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        stage: TrainingStage | None = None,
        step: int | None = None,
        load_best: bool = False,
        checkpoint_path: str | Path | None = None,
        device: str | torch.device = "cpu",
    ) -> CheckpointMetadata:
        """
        Load a training checkpoint.

        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            stage: Training stage to load from
            step: Specific step to load
            load_best: Whether to load the best checkpoint
            checkpoint_path: Direct path to checkpoint (overrides stage/step)
            device: Device to load tensors to

        Returns:
            Checkpoint metadata

        Raises:
            FileNotFoundError: If checkpoint does not exist
        """
        if checkpoint_path is not None:
            path = Path(checkpoint_path)
        elif stage is not None:
            path = self._get_checkpoint_path(stage, step, is_best=load_best)
        else:
            raise ValueError("Either checkpoint_path or stage must be specified")

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        model_path = path / self.MODEL_FILENAME
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

        if optimizer is not None:
            optimizer_path = path / self.OPTIMIZER_FILENAME
            if optimizer_path.exists():
                optimizer.load_state_dict(
                    torch.load(optimizer_path, map_location=device, weights_only=True)
                )

        if scheduler is not None:
            scheduler_path = path / self.SCHEDULER_FILENAME
            if scheduler_path.exists():
                scheduler.load_state_dict(
                    torch.load(scheduler_path, map_location=device, weights_only=True)
                )

        metadata_path = path / self.METADATA_FILENAME
        with open(metadata_path, encoding="utf-8") as f:
            metadata = CheckpointMetadata.from_dict(json.load(f))

        return metadata

    def get_latest_checkpoint(
        self, stage: TrainingStage
    ) -> tuple[Path | None, CheckpointMetadata | None]:
        """
        Get the latest checkpoint for a stage.

        Args:
            stage: Training stage

        Returns:
            Tuple of (checkpoint_path, metadata) or (None, None) if no checkpoints
        """
        stage_dir = self._get_stage_dir(stage)
        checkpoints = [d for d in stage_dir.iterdir() if d.is_dir() and d.name.startswith("step_")]

        if not checkpoints:
            return None, None

        latest = max(checkpoints, key=lambda x: int(x.name.split("_")[1]))
        metadata_path = latest / self.METADATA_FILENAME

        with open(metadata_path, encoding="utf-8") as f:
            metadata = CheckpointMetadata.from_dict(json.load(f))

        return latest, metadata

    def get_best_checkpoint(
        self, stage: TrainingStage
    ) -> tuple[Path | None, CheckpointMetadata | None]:
        """
        Get the best checkpoint for a stage.

        Args:
            stage: Training stage

        Returns:
            Tuple of (checkpoint_path, metadata) or (None, None) if no best checkpoint
        """
        best_path = self._get_checkpoint_path(stage, is_best=True)
        if not best_path.exists():
            return None, None

        metadata_path = best_path / self.METADATA_FILENAME
        with open(metadata_path, encoding="utf-8") as f:
            metadata = CheckpointMetadata.from_dict(json.load(f))

        return best_path, metadata

    def has_checkpoint(self, stage: TrainingStage, step: int | None = None) -> bool:
        """
        Check if a checkpoint exists.

        Args:
            stage: Training stage
            step: Specific step (None to check for any checkpoint)

        Returns:
            True if checkpoint exists
        """
        if step is not None:
            path = self._get_checkpoint_path(stage, step)
            return path.exists()

        stage_dir = self._get_stage_dir(stage)
        return any(d.is_dir() and d.name.startswith("step_") for d in stage_dir.iterdir())

    def get_training_history(self) -> list[dict]:
        """
        Get the complete training history.

        Returns:
            List of training log entries
        """
        return self._training_log.copy()

    def get_previous_stage_path(self, current_stage: TrainingStage) -> Path | None:
        """
        Get the best checkpoint from the previous training stage.

        This is used to initialize fine-tuning from the RL pre-training result.

        Args:
            current_stage: Current training stage

        Returns:
            Path to the best checkpoint from previous stage, or None
        """
        if current_stage == TrainingStage.SUPERVISED_FINETUNING:
            best_path, _ = self.get_best_checkpoint(TrainingStage.RL_PRETRAINING)
            if best_path is not None:
                return best_path

            latest_path, _ = self.get_latest_checkpoint(TrainingStage.RL_PRETRAINING)
            return latest_path

        return None

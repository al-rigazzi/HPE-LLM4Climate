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
Training pipeline for multimodal AIFS-Mistral fusion model.

This package contains data loaders, trainers, and utilities for training
the climate-text fusion model with on-the-fly sample generation.

Training Stages:
1. RL Pre-training: Uses verifiable rewards computed from numerical statistics
2. Supervised Fine-tuning: Refines model outputs using high-quality examples

Use TrainingPipeline for the complete workflow, or run stages independently
using RLTrainer and SupervisedFinetuningTrainer.
"""

from .checkpoint_manager import CheckpointManager, CheckpointMetadata, TrainingStage
from .climate_dataloader import ClimateTextDataLoader
from .location_masks import LocationMaskGenerator
from .prompt_generator import ClimatePromptGenerator
from .rl_trainer import RLTrainer, RLTrainingConfig
from .sft_trainer import SFTConfig, SupervisedFinetuningTrainer
from .statistics_computer import ClimateStatisticsComputer
from .train_pipeline import TrainingPipeline
from .verifiable_rewards import RewardBreakdown, VerifiableRewardComputer

__all__ = [
    # Data loading
    "ClimateTextDataLoader",
    "LocationMaskGenerator",
    "ClimateStatisticsComputer",
    "ClimatePromptGenerator",
    # Checkpoint management
    "CheckpointManager",
    "CheckpointMetadata",
    "TrainingStage",
    # RL Training
    "RLTrainer",
    "RLTrainingConfig",
    "VerifiableRewardComputer",
    "RewardBreakdown",
    # Supervised Fine-tuning
    "SupervisedFinetuningTrainer",
    "SFTConfig",
    # Pipeline orchestration
    "TrainingPipeline",
]

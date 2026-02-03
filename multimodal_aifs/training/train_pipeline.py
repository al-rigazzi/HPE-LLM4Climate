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
Training Pipeline Orchestrator.

This module provides a unified interface for running the multi-stage training
pipeline, which consists of:
1. RL pre-training with verifiable rewards
2. Supervised fine-tuning (optional, but recommended)

The stages can be run separately or together, and share a common checkpoint
structure for seamless transitions.
"""

import argparse
import os
from dataclasses import asdict
from enum import Enum
from pathlib import Path

import torch
import yaml

from ..utils.distributed_utils import (
    barrier,
    cleanup_distributed,
    is_distributed,
    is_main_process,
    print_rank0,
    setup_distributed,
    should_use_distributed,
)
from .checkpoint_manager import CheckpointManager, TrainingStage
from .climate_dataloader import ClimateTextDataLoader
from .rl_trainer import RLTrainer, RLTrainingConfig
from .sft_trainer import SFTConfig, SupervisedFinetuningTrainer


class PipelineStage(Enum):
    """Training pipeline stages."""

    RL_ONLY = "rl_only"
    SFT_ONLY = "sft_only"
    RL_THEN_SFT = "rl_then_sft"
    EVALUATE = "evaluate"


def load_config(config_path: str | Path) -> dict:
    """
    Load training configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: dict, config_path: str | Path) -> None:
    """
    Save training configuration to YAML file.

    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)


class TrainingPipeline:
    """
    Orchestrates the multi-stage training pipeline.

    This class manages the complete training workflow including:
    - Model and tokenizer initialization
    - Data loading
    - RL pre-training with verifiable rewards
    - Supervised fine-tuning
    - Checkpoint management
    - Evaluation

    The recommendation is to run RL pre-training first to teach the model
    to produce numerically accurate responses, then follow with supervised
    fine-tuning to improve response quality and coherence.
    """

    def __init__(
        self,
        model_name: str,
        zarr_paths: list[str] | str,
        checkpoint_dir: str | Path,
        rl_config: RLTrainingConfig | None = None,
        sft_config: SFTConfig | None = None,
        device: str = "auto",
    ):
        """
        Initialize the training pipeline.

        Args:
            model_name: HuggingFace model name or path
            zarr_paths: Path(s) to Zarr climate data files
            checkpoint_dir: Directory for saving checkpoints
            rl_config: Configuration for RL training (uses defaults if None)
            sft_config: Configuration for SFT training (uses defaults if None)
            device: Device to run on ('auto', 'cuda', 'cpu', 'mps')
        """
        self.model_name = model_name
        self.zarr_paths = zarr_paths if isinstance(zarr_paths, list) else [zarr_paths]
        self.checkpoint_dir = Path(checkpoint_dir)

        self.rl_config = rl_config or RLTrainingConfig()
        self.sft_config = sft_config or SFTConfig()

        if device != "auto":
            self.rl_config.device = device
            self.sft_config.device = device

        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.checkpoint_dir,
            max_checkpoints_per_stage=5,
        )

        self.model = None
        self.tokenizer = None
        self.dataloader = None

        self._initialized = False

    def initialize(self) -> None:
        """
        Initialize model, tokenizer, and dataloader.

        This is called lazily on first use to allow configuration
        modifications before loading heavy components.
        """
        if self._initialized:
            return

        print_rank0("Initializing training pipeline components...")

        print_rank0(f"Loading tokenizer: {self.model_name}")
        from transformers import AutoTokenizer, logging

        # Reduce transformers logging verbosity during model loading
        original_log_level = logging.get_verbosity()
        logging.set_verbosity_error()

        # Disable tqdm progress bars for non-main processes
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        is_local_main = local_rank == 0

        if is_distributed() and not is_main_process():
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            os.environ["TQDM_DISABLE"] = "1"

        # Multi-node tokenizer loading: local_rank 0 on each node loads first
        # to populate the node-local HF cache, then other ranks load.
        if is_distributed():
            if is_local_main:
                print_rank0("Local rank 0 loading tokenizer on this node...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                )
            barrier()  # Sync after local rank 0s finish
            if not is_local_main:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print_rank0("Tokenizer loaded successfully")
        print_rank0(f"Loading model: {self.model_name}")
        # Use Mistral3ForConditionalGeneration for Mistral 3 models
        if "Ministral-3" in self.model_name or "Mistral-3" in self.model_name:
            from transformers import Mistral3ForConditionalGeneration

            # For FP8 models in DDP, load directly to current GPU
            device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else "cpu"

            load_kwargs = {
                "trust_remote_code": True,
                "dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
                "device_map": device_map,
            }

            # With shared HF_HOME, all ranks can load in parallel from the same cache.
            # The tokenizer loading already ensured the cache is populated.
            print_rank0("Loading model (all ranks in parallel)...")
            self.model = Mistral3ForConditionalGeneration.from_pretrained(
                self.model_name, **load_kwargs
            )
        else:
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )

        # Note: We intentionally skip barrier after model loading because:
        # 1. Model loading for 8B params can take 15+ minutes
        # 2. NCCL default timeout is 10 minutes
        # 3. Each rank loads independently from shared cache
        # The DDP wrapper will synchronize model parameters when needed.

        # Restore original logging level
        logging.set_verbosity(original_log_level)
        print_rank0("Model loaded successfully")

        print_rank0("Initializing data loader")
        self.dataloader = ClimateTextDataLoader(
            zarr_paths=self.zarr_paths,
            mistral_model_name=self.model_name,
            samples_per_epoch=self.rl_config.batch_size * 100,
            cache_mistral_model=False,
            external_model=self.model,
            external_tokenizer=self.tokenizer,
            seed=self.rl_config.seed,
        )

        self._initialized = True
        print_rank0("Pipeline initialization complete")

    def run_rl_training(self) -> dict[str, list[float]]:
        """
        Run RL pre-training with verifiable rewards.

        Returns:
            Training history dictionary
        """
        self.initialize()

        print_rank0("\n" + "=" * 60)
        print_rank0("STAGE 1: RL Pre-training with Verifiable Rewards")
        print_rank0("=" * 60)

        trainer = RLTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            dataloader=self.dataloader,
            config=self.rl_config,
            checkpoint_manager=self.checkpoint_manager,
        )

        history = trainer.train()

        print_rank0("\nRL training complete")
        if history["mean_reward"]:
            print_rank0(f"Final reward: {history['mean_reward'][-1]:.4f}")

        return history

    def run_sft_training(self, from_rl_checkpoint: bool = True) -> dict[str, list[float]]:
        """
        Run supervised fine-tuning.

        Args:
            from_rl_checkpoint: If True, initialize from RL checkpoint

        Returns:
            Training history dictionary
        """
        self.initialize()

        print_rank0("\n" + "=" * 60)
        print_rank0("STAGE 2: Supervised Fine-tuning")
        print_rank0("=" * 60)

        if from_rl_checkpoint:
            if self.checkpoint_manager.has_checkpoint(TrainingStage.RL_PRETRAINING):
                trainer = SupervisedFinetuningTrainer.from_rl_checkpoint(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    dataloader=self.dataloader,
                    config=self.sft_config,
                    checkpoint_manager=self.checkpoint_manager,
                )
            else:
                print_rank0("No RL checkpoint found, starting SFT from scratch")
                trainer = SupervisedFinetuningTrainer(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    dataloader=self.dataloader,
                    config=self.sft_config,
                    checkpoint_manager=self.checkpoint_manager,
                )
        else:
            trainer = SupervisedFinetuningTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                dataloader=self.dataloader,
                config=self.sft_config,
                checkpoint_manager=self.checkpoint_manager,
            )

        history = trainer.train()

        print_rank0("\nSupervised fine-tuning complete")
        if history["loss"]:
            print_rank0(f"Final loss: {history['loss'][-1]:.4f}")

        return history

    def run_full_pipeline(self) -> dict[str, dict[str, list[float]]]:
        """
        Run the complete training pipeline (RL + SFT).

        This is the recommended approach:
        1. RL pre-training to learn numerical accuracy
        2. SFT to refine response quality

        Returns:
            Dictionary with 'rl' and 'sft' training histories
        """
        print_rank0("\n" + "#" * 60)
        print_rank0("FULL TRAINING PIPELINE: RL + SFT")
        print_rank0("#" * 60)

        rl_history = self.run_rl_training()

        sft_history = self.run_sft_training(from_rl_checkpoint=True)

        print_rank0("\n" + "#" * 60)
        print_rank0("TRAINING PIPELINE COMPLETE")
        print_rank0("#" * 60)
        if rl_history["mean_reward"]:
            print_rank0(f"RL final reward: {rl_history['mean_reward'][-1]:.4f}")
        if sft_history["loss"]:
            print_rank0(f"SFT final loss: {sft_history['loss'][-1]:.4f}")

        return {"rl": rl_history, "sft": sft_history}

    def evaluate(
        self, stage: TrainingStage = TrainingStage.SUPERVISED_FINETUNING
    ) -> dict[str, float]:
        """
        Evaluate the model from a specific training stage.

        Args:
            stage: Training stage to load checkpoint from

        Returns:
            Evaluation metrics dictionary
        """
        self.initialize()

        print_rank0(f"\nEvaluating model from {stage.value} checkpoint...")

        if self.checkpoint_manager.has_checkpoint(stage):
            self.checkpoint_manager.load_checkpoint(
                model=self.model,
                stage=stage,
                load_best=True,
            )
        else:
            print_rank0(f"No {stage.value} checkpoint found, using current model state")

        if stage == TrainingStage.RL_PRETRAINING:
            trainer = RLTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                dataloader=self.dataloader,
                config=self.rl_config,
            )
            metrics = trainer.evaluate()
        else:
            trainer = SupervisedFinetuningTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                dataloader=self.dataloader,
                config=self.sft_config,
            )
            metrics = trainer.evaluate()

        print_rank0("Evaluation results:")
        for key, value in metrics.items():
            print_rank0(f"  {key}: {value:.4f}")

        return metrics

    def save_final_model(self, output_dir: str | Path) -> None:
        """
        Save the final trained model.

        Args:
            output_dir: Directory to save the model
        """
        self.initialize()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print_rank0(f"Saving final model to {output_path}")

        if self.checkpoint_manager.has_checkpoint(TrainingStage.SUPERVISED_FINETUNING):
            self.checkpoint_manager.load_checkpoint(
                model=self.model,
                stage=TrainingStage.SUPERVISED_FINETUNING,
                load_best=True,
            )
        elif self.checkpoint_manager.has_checkpoint(TrainingStage.RL_PRETRAINING):
            self.checkpoint_manager.load_checkpoint(
                model=self.model,
                stage=TrainingStage.RL_PRETRAINING,
                load_best=True,
            )

        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        config = {
            "rl_config": asdict(self.rl_config),
            "sft_config": asdict(self.sft_config),
            "model_name": self.model_name,
            "training_history": self.checkpoint_manager.get_training_history(),
        }
        save_config(config, output_path / "training_config.yaml")

        print_rank0(f"Model saved to {output_path}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(description="Train climate LLM with verifiable RL rewards")

    parser.add_argument(
        "--stage",
        type=str,
        choices=["rl", "sft", "full", "evaluate"],
        default="full",
        help="Training stage to run",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="mistralai/Ministral-3-8B-Instruct-2512",
        help="HuggingFace model name or path",
    )

    parser.add_argument(
        "--zarr-paths",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to Zarr climate data files",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/climate_llm",
        help="Directory for checkpoints",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save final model",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--rl-epochs",
        type=int,
        default=3,
        help="Number of RL training epochs",
    )

    parser.add_argument(
        "--sft-epochs",
        type=int,
        default=3,
        help="Number of SFT training epochs",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate for training",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu, mps)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser


def main() -> None:
    """Main entry point for training pipeline."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup distributed training if environment variables indicate it
    # Must be done BEFORE any other operations to ensure print_rank0 works
    dist_config = None
    use_distributed = should_use_distributed()
    if use_distributed:
        dist_config = setup_distributed()
        print_rank0(f"Distributed training enabled: {dist_config.world_size} processes")

    try:
        if args.config:
            config = load_config(args.config)
            rl_config_dict = config.get("rl", {})
            sft_config_dict = config.get("sft", {})
        else:
            rl_config_dict = {}
            sft_config_dict = {}

        # Enable distributed training in configs if detected
        distributed = is_distributed()

        rl_config_dict.update(
            {
                "epochs": args.rl_epochs,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "device": args.device,
                "seed": args.seed,
                "distributed": distributed,
            }
        )

        sft_config_dict.update(
            {
                "epochs": args.sft_epochs,
                "learning_rate": args.learning_rate * 2,
                "batch_size": args.batch_size,
                "device": args.device,
                "seed": args.seed,
                "distributed": distributed,
            }
        )

        rl_config = RLTrainingConfig(**rl_config_dict)
        sft_config = SFTConfig(**sft_config_dict)

        pipeline = TrainingPipeline(
            model_name=args.model_name,
            zarr_paths=args.zarr_paths,
            checkpoint_dir=args.checkpoint_dir,
            rl_config=rl_config,
            sft_config=sft_config,
            device=args.device,
        )

        if args.stage == "rl":
            pipeline.run_rl_training()
        elif args.stage == "sft":
            pipeline.run_sft_training(from_rl_checkpoint=True)
        elif args.stage == "full":
            pipeline.run_full_pipeline()
        elif args.stage == "evaluate":
            pipeline.evaluate()

        # Only save on main process for distributed training
        if args.output_dir and is_main_process():
            pipeline.save_final_model(args.output_dir)

    finally:
        # Clean up distributed training
        if dist_config is not None:
            cleanup_distributed()


if __name__ == "__main__":
    main()

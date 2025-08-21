#!/usr/bin/env python3
"""
DeepSpeed-based training script for AIFS Climate-Text Fusion Model

This script trains the AIFS-based multimodal model using DeepSpeed for distributed training
with memory optimization and gradient accumulation.

Usage:
    # Single GPU
    python train_multimodal.py --config config.yaml

    # Multi-GPU with DeepSpeed
    deepspeed train_multimodal.py --config config.yaml --deepspeed deepspeed_config.json

    # Distributed training
    deepspeed --num_gpus=4 train_multimodal.py --config config.yaml
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from multimodal_aifs.core.aifs_climate_fusion import AIFSClimateTextFusion

try:
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam

    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("Warning: DeepSpeed not available. Install with: pip install deepspeed")

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: W&B not available. Install with: pip install wandb")

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ClimateTextDataset(Dataset):
    """
    Dataset for climate-text fusion training.

    This is a template dataset class that should be adapted to your specific data format.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        climate_data_shape: Tuple[int, int, int] = (2, 160, 64, 64),
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.climate_data_shape = climate_data_shape

        # Load your data index/metadata here
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict]:
        """Load dataset samples. Adapt this to your data format."""
        # Template implementation - replace with your data loading logic
        samples = []

        # Example: Load from a JSON index file
        if (self.data_path / "index.json").exists():
            with open(self.data_path / "index.json", "r") as f:
                samples = json.load(f)
        else:
            # Create dummy samples for testing
            logger.warning("No data found. Creating dummy samples for testing.")
            for i in range(100):
                samples.append(
                    {
                        "climate_file": f"climate_{i:03d}.pt",
                        "text": f"Climate analysis for sample {i}",
                        "target": f"This is the target text for sample {i}.",
                    }
                )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load climate data
        climate_data = self._load_climate_data(sample["climate_file"])

        # Tokenize text
        input_text = sample["text"]
        target_text = sample["target"]

        # Tokenize input and target
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "climate_data": climate_data,
            "input_ids": input_encoding["input_ids"].squeeze(0),
            "attention_mask": input_encoding["attention_mask"].squeeze(0),
            "labels": target_encoding["input_ids"].squeeze(0),
            "target_attention_mask": target_encoding["attention_mask"].squeeze(0),
        }

    def _load_climate_data(self, climate_file: str) -> torch.Tensor:
        """Load climate data from file."""
        climate_path = self.data_path / climate_file

        if climate_path.exists():
            return torch.load(climate_path)
        else:
            # Create dummy climate data for testing
            return torch.randn(*self.climate_data_shape)


class MultimodalTrainer:
    """DeepSpeed-based trainer for multimodal climate-text fusion."""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        self.model = AIFSClimateTextFusion(
            aifs_encoder_path=config["model"]["aifs_encoder_path"],
            climate_dim=config["model"].get("climate_dim", 1024),
            text_dim=config["model"].get("text_dim", 768),
            fusion_dim=config["model"].get("fusion_dim", 512),
            num_attention_heads=config["model"].get("num_fusion_layers", 4),
            dropout=config["model"].get("dropout", 0.1),
            device=str(self.device),
        )

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"]["llama_model_name"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize datasets
        self.train_dataset = ClimateTextDataset(
            data_path=config["data"]["train_path"],
            tokenizer=self.tokenizer,
            max_length=config["data"].get("max_length", 512),
        )

        self.val_dataset = ClimateTextDataset(
            data_path=config["data"]["val_path"],
            tokenizer=self.tokenizer,
            max_length=config["data"].get("max_length", 512),
        )

        # Training parameters
        self.epochs = config["training"]["epochs"]
        self.batch_size = config["training"]["batch_size"]
        self.learning_rate = config["training"]["learning_rate"]
        self.warmup_steps = config["training"].get("warmup_steps", 1000)
        self.gradient_accumulation_steps = config["training"].get("gradient_accumulation_steps", 1)
        self.max_grad_norm = config["training"].get("max_grad_norm", 1.0)

        # Initialize DeepSpeed
        self.model_engine = None
        self.optimizer = None
        self.lr_scheduler = None

    def setup_deepspeed(self, deepspeed_config: Optional[str] = None):
        """Initialize DeepSpeed engine."""
        if not DEEPSPEED_AVAILABLE:
            raise RuntimeError("DeepSpeed is not available. Please install deepspeed.")

        # Default DeepSpeed configuration
        if deepspeed_config is None:
            ds_config = {
                "train_batch_size": self.batch_size * self.gradient_accumulation_steps,
                "train_micro_batch_size_per_gpu": self.batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "optimizer": {
                    "type": "AdamW",
                    "params": {
                        "lr": self.learning_rate,
                        "betas": [0.9, 0.999],
                        "eps": 1e-8,
                        "weight_decay": 0.01,
                    },
                },
                "scheduler": {
                    "type": "WarmupLR",
                    "params": {
                        "warmup_min_lr": 0,
                        "warmup_max_lr": self.learning_rate,
                        "warmup_num_steps": self.warmup_steps,
                    },
                },
                "fp16": {
                    "enabled": True,
                    "loss_scale": 0,
                    "loss_scale_window": 1000,
                    "hysteresis": 2,
                    "min_loss_scale": 1,
                },
                "zero_optimization": {
                    "stage": 2,
                    "allgather_partitions": True,
                    "allgather_bucket_size": 2e8,
                    "overlap_comm": True,
                    "reduce_scatter": True,
                    "reduce_bucket_size": 2e8,
                    "contiguous_gradients": True,
                },
                "activation_checkpointing": {
                    "partition_activations": True,
                    "cpu_checkpointing": True,
                    "contiguous_memory_optimization": False,
                    "number_checkpoints": 4,
                    "synchronize_checkpoint_boundary": False,
                },
                "wall_clock_breakdown": False,
            }
        else:
            with open(deepspeed_config, "r") as f:
                ds_config = json.load(f)

        # Initialize DeepSpeed
        self.model_engine, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
            model=self.model,
            config=ds_config,
        )

        logger.info(f"DeepSpeed initialized with config: {ds_config}")

    def setup_dataloaders(self):
        """Setup data loaders."""
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        logger.info(f"Train dataset: {len(self.train_dataset)} samples")
        logger.info(f"Val dataset: {len(self.val_dataset)} samples")

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute training loss."""
        # Prepare climate batch
        climate_batch = {
            "dynamic": batch["climate_data"][:, :, :160, :, :],  # Dynamic variables
            "static": torch.zeros(batch["climate_data"].size(0), 11, 64, 64),  # Static data
        }

        # Prepare text inputs
        text_inputs = [
            self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["input_ids"]
        ]

        # Forward pass
        outputs = self.model_engine(climate_batch, text_inputs)
        fused_features = outputs["fused_features"]

        # Generate target embeddings for loss computation
        target_inputs = [
            self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["labels"]
        ]

        with torch.no_grad():
            target_outputs = self.model_engine.module.encode_text(target_inputs)
            target_embeddings = target_outputs[0]  # [batch, seq_len, hidden_size]

        # Compute contrastive loss or other appropriate loss
        # This is a simplified example - adapt based on your specific training objective
        loss = F.mse_loss(
            fused_features.mean(dim=1),  # [batch, hidden_size]
            target_embeddings.mean(dim=1),  # [batch, hidden_size]
        )

        return loss

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model_engine.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)

        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.model_engine.device) for k, v in batch.items()}

            # Forward pass
            loss = self.compute_loss(batch)

            # Backward pass
            self.model_engine.backward(loss)
            self.model_engine.step()

            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (step + 1)

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{avg_loss:.4f}",
                    "lr": (
                        f"{self.lr_scheduler.get_last_lr()[0]:.2e}" if self.lr_scheduler else "N/A"
                    ),
                }
            )

            # Log to wandb
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/avg_loss": avg_loss,
                        "train/learning_rate": (
                            self.lr_scheduler.get_last_lr()[0]
                            if self.lr_scheduler
                            else self.learning_rate
                        ),
                        "train/epoch": epoch,
                        "train/step": step + epoch * num_batches,
                    }
                )

        return total_loss / num_batches

    def validate(self) -> float:
        """Validate the model."""
        self.model_engine.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                batch = {k: v.to(self.model_engine.device) for k, v in batch.items()}

                loss = self.compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Log to wandb
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({"val/loss": avg_loss})

        return avg_loss

    def save_checkpoint(self, epoch: int, save_dir: str):
        """Save model checkpoint."""
        save_path = Path(save_dir) / f"checkpoint_epoch_{epoch}"
        save_path.mkdir(parents=True, exist_ok=True)

        # Save DeepSpeed checkpoint
        self.model_engine.save_checkpoint(str(save_path))

        # Save tokenizer
        self.tokenizer.save_pretrained(str(save_path / "tokenizer"))

        # Save training config
        with open(save_path / "config.yaml", "w") as f:
            yaml.dump(self.config, f)

        logger.info(f"Checkpoint saved to {save_path}")

    def train(self, deepspeed_config: Optional[str] = None):
        """Main training loop."""
        # Setup DeepSpeed
        self.setup_deepspeed(deepspeed_config)

        # Setup data loaders
        self.setup_dataloaders()

        # Initialize wandb
        if WANDB_AVAILABLE and self.config.get("wandb", {}).get("enabled", False):
            wandb.init(
                project=self.config["wandb"].get("project", "climate-text-fusion"),
                name=self.config["wandb"].get("run_name", f"train_{int(time.time())}"),
                config=self.config,
            )

        logger.info("Starting training...")

        best_val_loss = float("inf")

        for epoch in range(self.epochs):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss = self.validate()

            logger.info(
                f"Epoch {epoch+1}/{self.epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, self.config["training"]["save_dir"])
                logger.info(f"New best model saved (val_loss: {val_loss:.4f})")

            # Save periodic checkpoint
            if (epoch + 1) % self.config["training"].get("save_every", 5) == 0:
                self.save_checkpoint(epoch, self.config["training"]["save_dir"])

        logger.info("Training completed!")

        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()


def load_config(config_path: str) -> Dict:
    """Load training configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train multimodal climate-text fusion model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--deepspeed", type=str, help="Path to DeepSpeed config file")
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Local rank for distributed training"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Initialize trainer
    trainer = MultimodalTrainer(config)

    # Start training
    trainer.train(args.deepspeed)


if __name__ == "__main__":
    main()

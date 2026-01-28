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
Supervised Fine-Tuning Trainer.

This module implements supervised fine-tuning (SFT) that can be applied after
RL pre-training. The SFT stage helps refine the model's outputs using
high-quality examples or expert demonstrations.

Key features:
- Causal language modeling objective
- Support for instruction tuning format
- Can initialize from RL pre-training checkpoint
- Shared checkpoint structure with RL stage
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from ..utils.device_utils import autocast_if_available, resolve_device, supports_amp
from .checkpoint_manager import CheckpointManager, CheckpointMetadata, TrainingStage
from .climate_dataloader import ClimateTextDataLoader


@dataclass
class SFTConfig:
    """Configuration for supervised fine-tuning."""

    learning_rate: float = 2e-5
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    save_steps: int = 500
    eval_steps: int = 100
    log_steps: int = 10

    max_seq_length: int = 2048

    use_response_only_loss: bool = True

    device: str = "auto"
    mixed_precision: bool = True
    seed: int = 42


class SupervisedFinetuningTrainer:
    """
    Supervised fine-tuning trainer for climate LLM.

    This trainer performs standard causal language modeling on climate
    description tasks. It can be used:
    - As a standalone training approach
    - As a refinement stage after RL pre-training

    The training uses high-quality prompt-response pairs where the
    responses have been validated against ground truth statistics.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        dataloader: ClimateTextDataLoader,
        config: SFTConfig,
        checkpoint_manager: CheckpointManager | None = None,
    ):
        """
        Initialize supervised fine-tuning trainer.

        Args:
            model: The language model to fine-tune
            tokenizer: Tokenizer for the model
            dataloader: Climate data loader for generating samples
            config: Training configuration
            checkpoint_manager: Optional checkpoint manager
        """
        self.config = config
        self.device = resolve_device(config.device if config.device != "auto" else None)
        self._amp_supported = supports_amp(self.device)
        self._use_amp = config.mixed_precision and self._amp_supported

        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.checkpoint_manager = checkpoint_manager

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        total_steps = config.epochs * len(dataloader) // config.gradient_accumulation_steps
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=max(1, total_steps // 3),
            T_mult=2,
        )

        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")

        if self._use_amp and self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

    @classmethod
    def from_rl_checkpoint(
        cls,
        model: nn.Module,
        tokenizer,
        dataloader: ClimateTextDataLoader,
        config: SFTConfig,
        checkpoint_manager: CheckpointManager,
    ) -> "SupervisedFinetuningTrainer":
        """
        Create trainer initialized from RL pre-training checkpoint.

        Args:
            model: The language model (weights will be loaded from checkpoint)
            tokenizer: Tokenizer for the model
            dataloader: Climate data loader
            config: SFT configuration
            checkpoint_manager: Checkpoint manager with RL checkpoints

        Returns:
            Initialized trainer with RL weights loaded

        Raises:
            FileNotFoundError: If no RL checkpoint exists
        """
        rl_checkpoint = checkpoint_manager.get_previous_stage_path(
            TrainingStage.SUPERVISED_FINETUNING
        )

        if rl_checkpoint is None:
            raise FileNotFoundError(
                "No RL pre-training checkpoint found. "
                "Run RL training first or provide a checkpoint path."
            )

        device = resolve_device(config.device if config.device != "auto" else None)

        metadata = checkpoint_manager.load_checkpoint(
            model=model,
            stage=TrainingStage.RL_PRETRAINING,
            load_best=True,
            device=device,
        )

        print(f"Loaded RL checkpoint from step {metadata.step}, epoch {metadata.epoch}")
        print(f"RL metrics: {metadata.metrics}")

        trainer = cls(
            model=model,
            tokenizer=tokenizer,
            dataloader=dataloader,
            config=config,
            checkpoint_manager=checkpoint_manager,
        )

        return trainer

    def _prepare_batch(
        self, prompts: list[str], responses: list[str]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Prepare a batch for training.

        Args:
            prompts: List of prompts
            responses: List of responses

        Returns:
            Tuple of (inputs, labels)
        """
        full_texts = []
        prompt_lengths = []

        for prompt, response in zip(prompts, responses):
            full_text = f"{prompt}\n\n{response}"
            full_texts.append(full_text)

            prompt_tokens = self.tokenizer(
                prompt + "\n\n",
                add_special_tokens=False,
            )
            prompt_lengths.append(len(prompt_tokens["input_ids"]))

        inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs.pop("token_type_ids", None)

        labels = inputs["input_ids"].clone()

        if self.config.use_response_only_loss:
            for i, prompt_len in enumerate(prompt_lengths):
                labels[i, :prompt_len] = -100

        labels[inputs["attention_mask"] == 0] = -100

        return inputs, labels

    def train_step(self, prompts: list[str], responses: list[str]) -> dict[str, float]:
        """
        Perform a single training step.

        Args:
            prompts: Batch of prompts
            responses: Batch of responses

        Returns:
            Dictionary of step metrics
        """
        self.model.train()

        inputs, labels = self._prepare_batch(prompts, responses)

        with autocast_if_available(self.device):
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss / self.config.gradient_accumulation_steps

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return {
            "loss": loss.item() * self.config.gradient_accumulation_steps,
            "perplexity": torch.exp(outputs.loss).item(),
        }

    def train_epoch(self) -> dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()

        epoch_losses = []
        epoch_perplexities = []

        prompts_buffer = []
        responses_buffer = []
        accumulated_steps = 0

        for sample in self.dataloader:
            prompts_buffer.append(sample.prompt)
            responses_buffer.append(sample.llm_response)

            if len(prompts_buffer) >= self.config.batch_size:
                metrics = self.train_step(prompts_buffer, responses_buffer)
                epoch_losses.append(metrics["loss"])
                epoch_perplexities.append(metrics["perplexity"])

                accumulated_steps += 1

                if accumulated_steps >= self.config.gradient_accumulation_steps:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    accumulated_steps = 0
                    self.global_step += 1

                prompts_buffer = []
                responses_buffer = []

        if prompts_buffer:
            metrics = self.train_step(prompts_buffer, responses_buffer)
            epoch_losses.append(metrics["loss"])
            epoch_perplexities.append(metrics["perplexity"])

            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.global_step += 1

        return {
            "loss": sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0,
            "perplexity": (
                sum(epoch_perplexities) / len(epoch_perplexities) if epoch_perplexities else 0.0
            ),
        }

    def train(self) -> dict[str, list[float]]:
        """
        Run the full supervised fine-tuning loop.

        Returns:
            Dictionary of training history
        """
        history: dict[str, list[float]] = {
            "loss": [],
            "perplexity": [],
        }

        print(f"Starting supervised fine-tuning for {self.config.epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self._use_amp}")

        for epoch in range(self.config.epochs):
            self.epoch = epoch

            metrics = self.train_epoch()

            for key, values in history.items():
                if key in metrics:
                    values.append(metrics[key])

            print(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"Loss: {metrics['loss']:.4f}, "
                f"Perplexity: {metrics['perplexity']:.2f}"
            )

            is_best = metrics["loss"] < self.best_loss
            if is_best:
                self.best_loss = metrics["loss"]

            if self.checkpoint_manager is not None and (
                (epoch + 1) % (self.config.save_steps // len(self.dataloader) + 1) == 0 or is_best
            ):
                previous_stages = [TrainingStage.RL_PRETRAINING.value]

                metadata = CheckpointMetadata(
                    stage=TrainingStage.SUPERVISED_FINETUNING,
                    step=self.global_step,
                    epoch=epoch,
                    metrics=metrics,
                    config={
                        "learning_rate": self.config.learning_rate,
                        "batch_size": self.config.batch_size,
                    },
                    previous_stages=previous_stages,
                )

                self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    metadata=metadata,
                    is_best=is_best,
                )

                if is_best:
                    print(f"  New best model saved (loss: {self.best_loss:.4f})")

        print("Supervised fine-tuning complete")
        return history

    def evaluate(self, num_samples: int = 50) -> dict[str, float]:
        """
        Evaluate the model on a set of samples.

        Args:
            num_samples: Number of samples to evaluate on

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0
        sample_count = 0

        with torch.no_grad():
            for sample in self.dataloader:
                if sample_count >= num_samples:
                    break

                inputs, labels = self._prepare_batch([sample.prompt], [sample.llm_response])

                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss

                num_tokens = (labels != -100).sum().item()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
                sample_count += 1

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0

        return {
            "eval_loss": avg_loss,
            "eval_perplexity": torch.exp(torch.tensor(avg_loss)).item(),
            "num_samples": sample_count,
            "num_tokens": total_tokens,
        }

    def generate_sample(self, prompt: str, max_length: int = 512) -> str:
        """
        Generate a response for evaluation.

        Args:
            prompt: Input prompt
            max_length: Maximum generation length

        Returns:
            Generated response text
        """
        self.model.eval()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length // 2,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        input_length = inputs["input_ids"].shape[1]
        response_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

        return response

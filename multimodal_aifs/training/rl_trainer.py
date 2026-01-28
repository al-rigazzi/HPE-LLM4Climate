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
Reinforcement Learning Trainer with Verifiable Rewards.

This module implements a PPO-based RL training loop that uses verifiable rewards
computed from numerical climate statistics. The training approach follows the
principles of RLHF but uses mathematically verifiable rewards instead of human
preferences.

Key features:
- PPO algorithm with clipped objective
- Verifiable rewards from numerical statistics
- Value function baseline for variance reduction
- KL divergence penalty to prevent catastrophic forgetting
"""

from dataclasses import dataclass
from typing import Iterator

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..utils.device_utils import autocast_if_available, resolve_device, supports_amp
from .checkpoint_manager import CheckpointManager, CheckpointMetadata, TrainingStage
from .climate_dataloader import ClimateTextDataLoader, TrainingSample
from .statistics_computer import VariableStatistics
from .verifiable_rewards import RewardBreakdown, VerifiableRewardComputer


@dataclass
class RLTrainingConfig:
    """Configuration for RL training."""

    learning_rate: float = 1e-5
    epochs: int = 3
    ppo_epochs: int = 4
    batch_size: int = 4
    mini_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    kl_penalty_coef: float = 0.1
    target_kl: float = 0.02

    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 100
    log_steps: int = 10

    max_response_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    numerical_tolerance: float = 0.1
    penalize_hallucinations: bool = True

    device: str = "auto"
    mixed_precision: bool = True
    seed: int = 42


@dataclass
class RLBatch:
    """A batch of data for RL training."""

    climate_tensors: torch.Tensor
    prompts: list[str]
    responses: list[str]
    ground_truth_stats: list[list[VariableStatistics]]
    old_log_probs: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class ValueHead(nn.Module):
    """Value function head for PPO."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        """
        Initialize value head.

        Args:
            hidden_size: Input hidden size from base model
            dropout: Dropout probability
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute value estimates.

        Args:
            hidden_states: Hidden states from base model [batch, seq_len, hidden]

        Returns:
            Value estimates [batch]
        """
        pooled = hidden_states[:, -1, :]
        return self.layers(pooled).squeeze(-1)


class RLTrainer:
    """
    PPO-based RL trainer with verifiable rewards.

    This trainer uses PPO to optimize a language model based on rewards
    computed from numerical climate statistics verification.

    Training loop:
    1. Generate responses using current policy
    2. Compute verifiable rewards by comparing to ground truth
    3. Compute advantages using GAE
    4. Update policy using PPO clipped objective
    5. Update value function
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        dataloader: ClimateTextDataLoader,
        config: RLTrainingConfig,
        checkpoint_manager: CheckpointManager | None = None,
        reference_model: nn.Module | None = None,
    ):
        """
        Initialize RL trainer.

        Args:
            model: The language model to train
            tokenizer: Tokenizer for the model
            dataloader: Climate data loader for generating samples
            config: Training configuration
            checkpoint_manager: Optional checkpoint manager
            reference_model: Optional frozen reference model for KL penalty
        """
        self.config = config
        self.device = resolve_device(config.device if config.device != "auto" else None)
        self._amp_supported = supports_amp(self.device)
        self._use_amp = config.mixed_precision and self._amp_supported

        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.checkpoint_manager = checkpoint_manager

        if reference_model is not None:
            self.reference_model = reference_model.to(self.device)
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False
        else:
            self.reference_model = None

        hidden_size = self._get_hidden_size()
        self.value_head = ValueHead(hidden_size).to(self.device)

        self.reward_computer = VerifiableRewardComputer(
            numerical_tolerance=config.numerical_tolerance,
            penalize_hallucinations=config.penalize_hallucinations,
        )

        self.optimizer = AdamW(
            list(self.model.parameters()) + list(self.value_head.parameters()),
            lr=config.learning_rate,
            weight_decay=0.01,
        )

        total_steps = config.epochs * len(dataloader) // config.gradient_accumulation_steps
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=config.learning_rate * 0.1,
        )

        self.global_step = 0
        self.epoch = 0
        self.best_reward = float("-inf")

        if self._use_amp and self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

    def _get_hidden_size(self) -> int:
        """Get the hidden size of the model."""
        if hasattr(self.model, "config"):
            if hasattr(self.model.config, "hidden_size"):
                return self.model.config.hidden_size
            if hasattr(self.model.config, "d_model"):
                return self.model.config.d_model
        return 4096

    def generate_response(self, prompt: str) -> tuple[str, torch.Tensor]:
        """
        Generate a response for a prompt and compute log probabilities.

        Args:
            prompt: Input prompt

        Returns:
            Tuple of (generated_text, log_probs)
        """
        self.model.eval()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.config.max_response_length,
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_response_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                output_scores=True,
                return_dict_in_generate=True,
            )

        generated_ids = outputs.sequences[0]
        input_length = inputs["input_ids"].shape[1]
        response_ids = generated_ids[input_length:]

        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        if hasattr(outputs, "scores") and outputs.scores:
            log_probs = self._compute_log_probs_from_scores(
                outputs.scores, response_ids[: len(outputs.scores)]
            )
        else:
            log_probs = torch.zeros(1, device=self.device)

        return response, log_probs

    def _compute_log_probs_from_scores(
        self, scores: tuple, token_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probabilities from generation scores."""
        log_probs = []
        for i, score in enumerate(scores):
            if i < len(token_ids):
                probs = torch.softmax(score[0], dim=-1)
                token_prob = probs[token_ids[i]]
                log_probs.append(torch.log(token_prob + 1e-10))

        if log_probs:
            return torch.stack(log_probs)
        return torch.zeros(1, device=self.device)

    def compute_kl_divergence(
        self, current_log_probs: torch.Tensor, reference_log_probs: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence between current and reference policies."""
        return (torch.exp(current_log_probs) * (current_log_probs - reference_log_probs)).sum()

    def compute_advantages(
        self, rewards: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and returns.

        Args:
            rewards: Reward tensor [batch]
            values: Value estimates [batch]

        Returns:
            Tuple of (advantages, returns)
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        batch_size = rewards.shape[0]

        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                next_value = 0
                next_advantage = 0
            else:
                next_value = values[t + 1]
                next_advantage = advantages[t + 1]

            delta = rewards[t] + self.config.gamma * next_value - values[t]
            advantages[t] = delta + self.config.gamma * self.config.gae_lambda * next_advantage
            returns[t] = advantages[t] + values[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def collect_rollouts(
        self, num_samples: int
    ) -> Iterator[tuple[TrainingSample, str, torch.Tensor, float, RewardBreakdown]]:
        """
        Collect rollouts by generating responses and computing rewards.

        Args:
            num_samples: Number of samples to collect

        Yields:
            Tuples of (sample, response, log_probs, reward, breakdown)
        """
        sample_count = 0
        for sample in self.dataloader:
            if sample_count >= num_samples:
                break

            response, log_probs = self.generate_response(sample.prompt)

            ground_truth = self.dataloader.statistics_computer.compute_statistics(
                (sample.climate_tensor_t1 + sample.climate_tensor_t2) / 2,
                sample.mask,
            )

            result = self.reward_computer.compute_reward(
                response, ground_truth, return_breakdown=True
            )
            assert isinstance(result, tuple)
            reward, breakdown, _ = result

            yield sample, response, log_probs, reward, breakdown
            sample_count += 1

    def ppo_step(self, batch: RLBatch) -> dict[str, float]:
        """
        Perform a single PPO update step.

        Args:
            batch: Batch of RL data

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0

        batch_size = len(batch.prompts)

        for ppo_epoch in range(self.config.ppo_epochs):
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, self.config.mini_batch_size):
                end = min(start + self.config.mini_batch_size, batch_size)
                mb_indices = indices[start:end]

                mb_prompts = [batch.prompts[i] for i in mb_indices]
                mb_responses = [batch.responses[i] for i in mb_indices]
                mb_old_log_probs = batch.old_log_probs[mb_indices]
                mb_advantages = batch.advantages[mb_indices]
                mb_returns = batch.returns[mb_indices]

                combined_texts = [f"{p} {r}" for p, r in zip(mb_prompts, mb_responses)]
                inputs = self.tokenizer(
                    combined_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_response_length * 2,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                inputs.pop("token_type_ids", None)

                with autocast_if_available(self.device):
                    outputs = self.model(**inputs, output_hidden_states=True)
                    logits = outputs.logits

                    log_probs = torch.log_softmax(logits, dim=-1)
                    new_log_probs = log_probs[:, -1, :].mean(dim=-1)

                    hidden_states = outputs.hidden_states[-1]
                    values = self.value_head(hidden_states)

                    old_lp_sum = (
                        mb_old_log_probs.sum(dim=-1)
                        if mb_old_log_probs.dim() > 1
                        else mb_old_log_probs
                    )
                    ratio = torch.exp(new_log_probs - old_lp_sum)

                    surr1 = ratio * mb_advantages
                    surr2 = (
                        torch.clamp(
                            ratio,
                            1 - self.config.clip_epsilon,
                            1 + self.config.clip_epsilon,
                        )
                        * mb_advantages
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = 0.5 * ((values - mb_returns) ** 2).mean()

                    entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1).mean()

                    loss = (
                        policy_loss
                        + self.config.value_loss_coef * value_loss
                        - self.config.entropy_coef * entropy
                    )

                    if self.reference_model is not None:
                        with torch.no_grad():
                            ref_outputs = self.reference_model(**inputs)
                            ref_log_probs = torch.log_softmax(ref_outputs.logits, dim=-1)
                        kl = (torch.exp(log_probs) * (log_probs - ref_log_probs)).sum(dim=-1).mean()
                        loss = loss + self.config.kl_penalty_coef * kl
                        total_kl += kl.item()

                loss = loss / self.config.gradient_accumulation_steps

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()

            if (ppo_epoch + 1) % self.config.gradient_accumulation_steps == 0:
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

                self.optimizer.zero_grad()

        num_updates = self.config.ppo_epochs * (batch_size // self.config.mini_batch_size)
        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "kl_divergence": total_kl / num_updates if self.reference_model else 0.0,
        }

    def train_epoch(self) -> dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()

        epoch_rewards = []
        epoch_metrics: dict[str, list[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "kl_divergence": [],
            "numerical_accuracy": [],
            "trend_correctness": [],
            "coverage": [],
        }

        rollouts = list(self.collect_rollouts(len(self.dataloader)))

        prompts = [r[0].prompt for r in rollouts]
        responses = [r[1] for r in rollouts]
        log_probs_list = [r[2] for r in rollouts]
        rewards_list = [r[3] for r in rollouts]
        breakdowns = [r[4] for r in rollouts]
        samples = [r[0] for r in rollouts]

        max_len = max(lp.shape[0] for lp in log_probs_list)
        padded_log_probs = []
        for lp in log_probs_list:
            if lp.shape[0] < max_len:
                padding = torch.zeros(max_len - lp.shape[0], device=self.device)
                lp = torch.cat([lp, padding])
            padded_log_probs.append(lp)
        old_log_probs = torch.stack(padded_log_probs)

        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)
        epoch_rewards.extend(rewards_list)

        with torch.no_grad():
            dummy_inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_response_length,
            )
            dummy_inputs = {k: v.to(self.device) for k, v in dummy_inputs.items()}
            dummy_inputs.pop("token_type_ids", None)

            outputs = self.model(**dummy_inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            values = self.value_head(hidden_states)

        advantages, returns = self.compute_advantages(rewards, values)

        ground_truths = []
        for sample in samples:
            gt = self.dataloader.statistics_computer.compute_statistics(
                (sample.climate_tensor_t1 + sample.climate_tensor_t2) / 2,
                sample.mask,
            )
            ground_truths.append(gt)

        batch = RLBatch(
            climate_tensors=torch.stack([s.aifs_input_tensor for s in samples]),
            prompts=prompts,
            responses=responses,
            ground_truth_stats=ground_truths,
            old_log_probs=old_log_probs,
            rewards=rewards,
            advantages=advantages,
            returns=returns,
        )

        metrics = self.ppo_step(batch)

        for key in ["policy_loss", "value_loss", "entropy", "kl_divergence"]:
            epoch_metrics[key].append(metrics[key])

        for breakdown in breakdowns:
            epoch_metrics["numerical_accuracy"].append(breakdown.numerical_accuracy)
            epoch_metrics["trend_correctness"].append(breakdown.trend_correctness)
            epoch_metrics["coverage"].append(breakdown.coverage)

        self.scheduler.step()
        self.global_step += 1

        avg_metrics = {
            key: sum(values) / len(values) if values else 0.0
            for key, values in epoch_metrics.items()
        }
        avg_metrics["mean_reward"] = sum(epoch_rewards) / len(epoch_rewards)

        return avg_metrics

    def train(self) -> dict[str, list[float]]:
        """
        Run the full RL training loop.

        Returns:
            Dictionary of training history
        """
        history: dict[str, list[float]] = {
            "mean_reward": [],
            "policy_loss": [],
            "value_loss": [],
            "numerical_accuracy": [],
        }

        print(f"Starting RL training for {self.config.epochs} epochs")
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
                f"Reward: {metrics['mean_reward']:.4f}, "
                f"Policy Loss: {metrics['policy_loss']:.4f}, "
                f"Numerical Accuracy: {metrics['numerical_accuracy']:.4f}"
            )

            is_best = metrics["mean_reward"] > self.best_reward
            if is_best:
                self.best_reward = metrics["mean_reward"]

            if self.checkpoint_manager is not None and (
                (epoch + 1) % (self.config.save_steps // len(self.dataloader) + 1) == 0 or is_best
            ):
                metadata = CheckpointMetadata(
                    stage=TrainingStage.RL_PRETRAINING,
                    step=self.global_step,
                    epoch=epoch,
                    metrics=metrics,
                    config={
                        "learning_rate": self.config.learning_rate,
                        "ppo_epochs": self.config.ppo_epochs,
                        "clip_epsilon": self.config.clip_epsilon,
                    },
                )

                self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    metadata=metadata,
                    is_best=is_best,
                )

                if is_best:
                    print(f"  New best model saved (reward: {self.best_reward:.4f})")

        print("RL training complete")
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

        rewards = []
        accuracies = []
        coverages = []

        sample_count = 0
        for sample in self.dataloader:
            if sample_count >= num_samples:
                break

            response, _ = self.generate_response(sample.prompt)

            ground_truth = self.dataloader.statistics_computer.compute_statistics(
                (sample.climate_tensor_t1 + sample.climate_tensor_t2) / 2,
                sample.mask,
            )

            result = self.reward_computer.compute_reward(
                response, ground_truth, return_breakdown=True
            )
            assert isinstance(result, tuple)
            reward, breakdown, _ = result

            rewards.append(reward)
            accuracies.append(breakdown.numerical_accuracy)
            coverages.append(breakdown.coverage)

            sample_count += 1

        return {
            "mean_reward": sum(rewards) / len(rewards),
            "mean_numerical_accuracy": sum(accuracies) / len(accuracies),
            "mean_coverage": sum(coverages) / len(coverages),
            "num_samples": len(rewards),
        }

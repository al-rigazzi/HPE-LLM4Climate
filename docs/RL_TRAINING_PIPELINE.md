# RL Training Pipeline with Verifiable Rewards

This document describes the two-stage Reinforcement Learning (RL) training pipeline for the multimodal climate-text fusion model. The pipeline uses verifiable rewards computed from numerical statistics to train the model to produce accurate climate analysis.

## Overview

The training pipeline consists of two stages:

1. **RL Pre-training**: Uses Proximal Policy Optimization (PPO) with verifiable rewards based on numerical accuracy
2. **Supervised Fine-tuning (SFT)**: Refines the model using curated question-answer pairs

Both stages share a common checkpoint structure, enabling seamless transitions and separate triggering.

## Quick Start

```bash
# Run full pipeline (RL + SFT)
python -m multimodal_aifs.training.train_pipeline \
    --stage full \
    --model-path path/to/base/model \
    --data-path path/to/climate/data \
    --checkpoint-dir ./checkpoints

# Run only RL pre-training
python -m multimodal_aifs.training.train_pipeline \
    --stage rl \
    --model-path path/to/base/model \
    --data-path path/to/climate/data

# Run only SFT (requires existing RL checkpoint)
python -m multimodal_aifs.training.train_pipeline \
    --stage sft \
    --checkpoint-dir ./checkpoints

# Evaluate a trained model
python -m multimodal_aifs.training.train_pipeline \
    --stage evaluate \
    --checkpoint-dir ./checkpoints
```

## Architecture

### Training Stages

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Pipeline                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 1: RL Pre-training                                       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  • PPO with clipped objective                              │ │
│  │  • Verifiable rewards from numerical statistics            │ │
│  │  • GAE for advantage estimation                            │ │
│  │  • KL penalty for stability                                │ │
│  └────────────────────────────────────────────────────────────┘ │
│                           │                                      │
│                           ▼                                      │
│              [Checkpoint: RL_PRETRAINING]                       │
│                           │                                      │
│                           ▼                                      │
│  Stage 2: Supervised Fine-tuning                                │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  • Loads from RL checkpoint                                │ │
│  │  • Standard cross-entropy loss                             │ │
│  │  • Optional response-only loss                             │ │
│  │  • Learning rate warmup + decay                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                           │                                      │
│                           ▼                                      │
│           [Checkpoint: SUPERVISED_FINETUNING]                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Verifiable Rewards

The reward system computes scores by comparing LLM outputs against ground truth statistics from climate data:

| Component | Weight | Description |
|-----------|--------|-------------|
| `numerical_accuracy` | 40% | How close extracted numbers are to ground truth |
| `trend_correctness` | 25% | Whether trends (increasing/decreasing/stable) match |
| `coverage` | 20% | Fraction of expected values mentioned |
| `consistency` | 10% | Internal consistency of reported values |
| `format_quality` | 5% | Proper formatting of numerical outputs |

**Example reward computation:**
```python
from multimodal_aifs.training import VerifiableRewardComputer, StatisticsComputer

# Initialize components
stats_computer = StatisticsComputer(zarr_path="path/to/data.zarr")
reward_computer = VerifiableRewardComputer(statistics_computer=stats_computer)

# Compute reward for model response
reward, breakdown = reward_computer.compute_reward(
    response="The temperature at 500 hPa is 245.3 K with increasing trend",
    prompt="What is the temperature at 500 hPa?",
    climate_data=climate_tensor,
    variable="t_500"
)

print(f"Total reward: {reward:.4f}")
print(f"Breakdown: {breakdown}")
```

## Configuration

### RL Training Configuration

```python
from multimodal_aifs.training import RLTrainer, RLTrainingConfig

config = RLTrainingConfig(
    # PPO hyperparameters
    learning_rate=1e-5,
    ppo_epochs=4,
    clip_epsilon=0.2,
    value_loss_coef=0.5,
    entropy_coef=0.01,

    # GAE parameters
    gamma=0.99,
    gae_lambda=0.95,

    # Training settings
    batch_size=4,
    gradient_accumulation_steps=8,
    max_grad_norm=1.0,

    # KL penalty
    kl_penalty_coef=0.1,
    target_kl=0.01,

    # Mixed precision
    use_mixed_precision=True,
)

trainer = RLTrainer(
    model=model,
    tokenizer=tokenizer,
    reward_computer=reward_computer,
    config=config,
    checkpoint_manager=checkpoint_manager,
)
```

### SFT Configuration

```python
from multimodal_aifs.training import SupervisedFinetuningTrainer, SFTConfig

config = SFTConfig(
    learning_rate=2e-5,
    num_epochs=3,
    batch_size=8,
    gradient_accumulation_steps=4,

    # Warmup and decay
    warmup_ratio=0.1,
    weight_decay=0.01,

    # Response-only loss (only compute loss on response tokens)
    response_only_loss=True,

    # Evaluation
    eval_steps=500,
    save_steps=1000,
)

trainer = SupervisedFinetuningTrainer(
    model=model,
    tokenizer=tokenizer,
    config=config,
    checkpoint_manager=checkpoint_manager,
)
```

## Checkpoint Structure

All training stages use a unified checkpoint format:

```
checkpoints/
├── epoch_001_step_1000/
│   ├── model.pt              # Model state dict
│   ├── optimizer.pt          # Optimizer state
│   ├── scheduler.pt          # LR scheduler state (if applicable)
│   ├── metadata.json         # Training metadata
│   └── training_history.json # Metrics history
├── epoch_002_step_2000/
│   └── ...
└── best/
    └── ...
```

### Metadata Format

```json
{
    "epoch": 2,
    "global_step": 2000,
    "stage": "RL_PRETRAINING",
    "timestamp": "2026-01-28T10:30:00",
    "metrics": {
        "loss": 0.234,
        "reward_mean": 0.756,
        "kl_divergence": 0.008
    },
    "config": {
        "learning_rate": 1e-5,
        "batch_size": 4
    }
}
```

### Loading Checkpoints

```python
from multimodal_aifs.training import CheckpointManager, TrainingStage

manager = CheckpointManager(checkpoint_dir="./checkpoints")

# Load latest checkpoint
checkpoint = manager.load_latest()

# Load best checkpoint
checkpoint = manager.load_best()

# Load specific checkpoint
checkpoint = manager.load_checkpoint("epoch_002_step_2000")

# Load checkpoint from specific stage
checkpoint = manager.load_latest(stage=TrainingStage.RL_PRETRAINING)
```

## Command-Line Interface

### Full Pipeline

```bash
python -m multimodal_aifs.training.train_pipeline \
    --stage full \
    --model-path mistralai/Mistral-7B-Instruct-v0.2 \
    --data-path ./data/real_ecmwf_latest.zarr \
    --checkpoint-dir ./checkpoints \
    --rl-epochs 5 \
    --sft-epochs 3 \
    --batch-size 4 \
    --learning-rate 1e-5 \
    --device cuda
```

### RL-Only Training

```bash
python -m multimodal_aifs.training.train_pipeline \
    --stage rl \
    --model-path mistralai/Mistral-7B-Instruct-v0.2 \
    --data-path ./data/real_ecmwf_latest.zarr \
    --checkpoint-dir ./checkpoints \
    --rl-epochs 10 \
    --ppo-epochs 4 \
    --clip-epsilon 0.2 \
    --kl-penalty 0.1
```

### SFT-Only Training

```bash
python -m multimodal_aifs.training.train_pipeline \
    --stage sft \
    --checkpoint-dir ./checkpoints \
    --sft-data-path ./data/sft_samples.json \
    --sft-epochs 3 \
    --response-only-loss
```

### Evaluation

```bash
python -m multimodal_aifs.training.train_pipeline \
    --stage evaluate \
    --checkpoint-dir ./checkpoints \
    --eval-data-path ./data/eval_samples.json \
    --output-file ./results/evaluation.json
```

## Programmatic Usage

### Complete Example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from multimodal_aifs.training import (
    CheckpointManager,
    RLTrainer,
    RLTrainingConfig,
    SFTConfig,
    StatisticsComputer,
    SupervisedFinetuningTrainer,
    TrainingPipeline,
    VerifiableRewardComputer,
)

# Load base model and tokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# Initialize checkpoint manager
checkpoint_manager = CheckpointManager(
    checkpoint_dir="./checkpoints",
    max_checkpoints=5,  # Keep last 5 checkpoints
)

# Initialize reward computer
stats_computer = StatisticsComputer(zarr_path="./data/real_ecmwf_latest.zarr")
reward_computer = VerifiableRewardComputer(statistics_computer=stats_computer)

# Configure RL training
rl_config = RLTrainingConfig(
    learning_rate=1e-5,
    ppo_epochs=4,
    clip_epsilon=0.2,
    batch_size=4,
)

# Configure SFT
sft_config = SFTConfig(
    learning_rate=2e-5,
    num_epochs=3,
    response_only_loss=True,
)

# Create pipeline
pipeline = TrainingPipeline(
    model=model,
    tokenizer=tokenizer,
    checkpoint_manager=checkpoint_manager,
    rl_config=rl_config,
    sft_config=sft_config,
    reward_computer=reward_computer,
)

# Run full pipeline
pipeline.run(stage="full", train_dataloader=train_loader, eval_dataloader=eval_loader)

# Or run stages separately
pipeline.run(stage="rl", train_dataloader=train_loader)
pipeline.run(stage="sft", train_dataloader=sft_loader)
```

## Reward Breakdown Analysis

The `VerifiableRewardComputer` provides detailed breakdowns for debugging and analysis:

```python
reward, breakdown = reward_computer.compute_reward(
    response=model_response,
    prompt=prompt,
    climate_data=data,
    variable="t_500",
)

print(f"Total Reward: {breakdown.total_reward:.4f}")
print(f"  Numerical Accuracy: {breakdown.numerical_accuracy:.4f}")
print(f"  Trend Correctness:  {breakdown.trend_correctness:.4f}")
print(f"  Coverage:           {breakdown.coverage:.4f}")
print(f"  Consistency:        {breakdown.consistency:.4f}")
print(f"  Format Quality:     {breakdown.format_quality:.4f}")

# Access extracted values for debugging
for value in breakdown.extracted_values:
    print(f"  Found: {value.value} {value.unit} (confidence: {value.confidence:.2f})")
```

## Memory Optimization

For large models, enable memory optimizations:

```python
config = RLTrainingConfig(
    # Gradient accumulation to simulate larger batches
    batch_size=1,
    gradient_accumulation_steps=32,

    # Mixed precision
    use_mixed_precision=True,

    # Gradient checkpointing (set on model)
    # model.gradient_checkpointing_enable()
)
```

See [MEMORY_OPTIMIZATION.md](./MEMORY_OPTIMIZATION.md) for detailed memory optimization strategies.

## SLURM and Distributed Training

The training pipeline supports multi-node, multi-GPU training using PyTorch DistributedDataParallel (DDP). SLURM batch scripts are provided for running on HPC clusters.

### SLURM Job Scripts

Two SLURM scripts are provided in `multimodal_aifs/training/`:

- `slurm_rl_training.sh` - Multi-node RL training (default: 2 nodes × 4 GPUs)
- `slurm_sft_training.sh` - Single-node SFT training (default: 1 node × 4 GPUs)

### Quick SLURM Submission

```bash
# Submit RL training job
sbatch multimodal_aifs/training/slurm_rl_training.sh

# Submit SFT training after RL completes
sbatch --dependency=afterok:<rl_job_id> multimodal_aifs/training/slurm_sft_training.sh

# Override parameters at submission
sbatch --nodes=4 multimodal_aifs/training/slurm_rl_training.sh
```

### Environment Variables

Configure training via environment variables:

```bash
# Paths
export PROJECT_DIR=/path/to/HPE-LLM4Climate
export ZARR_DATA=/path/to/climate/data.zarr
export CHECKPOINT_DIR=/path/to/checkpoints

# Model
export MODEL_NAME=mistralai/Ministral-3-8B-Instruct-2512

# Training hyperparameters
export RL_EPOCHS=5
export SFT_EPOCHS=3
export BATCH_SIZE=4
export LEARNING_RATE=1e-5

# Submit with custom settings
sbatch multimodal_aifs/training/slurm_rl_training.sh
```

### Chain RL → SFT Jobs

To automatically run SFT after RL training:

```bash
# Submit RL job and capture job ID
RL_JOB=$(sbatch --parsable multimodal_aifs/training/slurm_rl_training.sh)
echo "RL Job ID: ${RL_JOB}"

# Submit SFT with dependency on RL completion
export RL_CHECKPOINT_DIR=/path/to/checkpoints/rl_training_${RL_JOB}
sbatch --dependency=afterok:${RL_JOB} multimodal_aifs/training/slurm_sft_training.sh
```

### Distributed Training Configuration

The training automatically detects SLURM environment variables:
- `SLURM_NTASKS` → World size
- `SLURM_PROCID` → Global rank
- `SLURM_LOCALID` → Local rank (GPU index)
- `SLURM_NODELIST` → Master address extraction

For manual distributed training without SLURM:

```bash
# Set environment variables manually
export MASTER_ADDR=node001
export MASTER_PORT=29500
export WORLD_SIZE=8
export RANK=0
export LOCAL_RANK=0

python -m multimodal_aifs.training.train_pipeline \
    --stage rl \
    --zarr-paths /path/to/data.zarr \
    --device cuda
```

### NCCL Settings

The SLURM scripts configure NCCL for optimal multi-node performance:

```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
```

Adjust these based on your cluster's network configuration.

## Troubleshooting

### Common Issues

**1. KL divergence exploding**
- Reduce `kl_penalty_coef` or increase `target_kl`
- Lower learning rate
- Increase PPO epochs for more stable updates

**2. Rewards not improving**
- Check that `StatisticsComputer` is loading data correctly
- Verify numerical extraction patterns in responses
- Increase reward weight for numerical accuracy

**3. Out of memory during RL training**
- Reduce batch size
- Increase gradient accumulation steps
- Enable mixed precision training
- Use gradient checkpointing

**4. SFT not loading RL checkpoint**
- Ensure RL training completed successfully
- Check checkpoint directory permissions
- Verify stage metadata in checkpoint

**5. NCCL timeout in distributed training**
- Increase `NCCL_TIMEOUT` environment variable
- Check network connectivity between nodes
- Verify firewall allows NCCL ports

**6. Uneven GPU memory usage**
- Ensure all processes start simultaneously
- Check that batch sizes are identical across ranks
- Verify data sharding is correct

### Logging

Enable verbose logging for debugging:

```python
import logging
logging.getLogger("multimodal_aifs.training").setLevel(logging.DEBUG)
```

## Testing

Run the training pipeline tests:

```bash
# All training tests
pytest multimodal_aifs/tests/unit/test_verifiable_rewards.py -v
pytest multimodal_aifs/tests/unit/test_checkpoint_manager.py -v

# Specific test classes
pytest multimodal_aifs/tests/unit/test_verifiable_rewards.py::TestRewardBreakdown -v
```

## Related Documentation

- [MEMORY_OPTIMIZATION.md](./MEMORY_OPTIMIZATION.md) - Memory optimization strategies
- [multimodal_aifs/training/README.md](../multimodal_aifs/training/README.md) - General training documentation
- [multimodal_aifs/docs/AIFS_Input_Preparation_Implementation.md](../multimodal_aifs/docs/AIFS_Input_Preparation_Implementation.md) - Data preparation

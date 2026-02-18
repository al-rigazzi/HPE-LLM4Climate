# RL/SFT Training Pipeline

This document describes the current training entrypoint in `multimodal_aifs/training/train_pipeline.py`.

## Pipeline Stages

The CLI supports these stages:

- `rl` — RL pre-training with verifiable rewards
- `sft` — supervised fine-tuning
- `full` — RL followed by SFT
- `evaluate` — evaluate from available checkpoints

## Main Entry Command

```bash
python -m multimodal_aifs.training.train_pipeline \
  --stage full \
  --model-name mistralai/Ministral-3-8B-Instruct-2512 \
  --zarr-paths data/real_ecmwf_latest.zarr \
  --checkpoint-dir checkpoints/climate_llm
```

## Common Workflows

### RL only

```bash
python -m multimodal_aifs.training.train_pipeline \
  --stage rl \
  --zarr-paths data/real_ecmwf_latest.zarr \
  --checkpoint-dir checkpoints/climate_llm \
  --rl-epochs 3 --batch-size 4
```

### SFT only

```bash
python -m multimodal_aifs.training.train_pipeline \
  --stage sft \
  --zarr-paths data/real_ecmwf_latest.zarr \
  --checkpoint-dir checkpoints/climate_llm \
  --sft-init rl --sft-epochs 3 --batch-size 4
```

### Resume RL from checkpoint

```bash
python -m multimodal_aifs.training.train_pipeline \
  --stage rl \
  --resume-from checkpoints/climate_llm/rl_pretraining/step_100 \
  --resume-stage rl \
  --zarr-paths data/real_ecmwf_latest.zarr \
  --checkpoint-dir checkpoints/climate_llm
```

### Resume SFT from checkpoint

```bash
python -m multimodal_aifs.training.train_pipeline \
  --stage sft \
  --sft-init resume \
  --resume-from checkpoints/climate_llm/supervised_finetuning/step_100 \
  --resume-stage sft \
  --zarr-paths data/real_ecmwf_latest.zarr \
  --checkpoint-dir checkpoints/climate_llm
```

## Key CLI Options

- `--zarr-paths` (required): one or more Zarr datasets
- `--checkpoint-dir`: where stage checkpoints are saved
- `--rl-epochs`, `--sft-epochs`, `--batch-size`, `--learning-rate`
- `--device auto|cuda|cpu|mps`
- `--reward-mode rl_only|full`

## SLURM Usage

Cluster scripts are provided in `multimodal_aifs/training/`:

- `slurm_rl_training.sh`
- `slurm_sft_training.sh`

Use `sbatch` directly with environment overrides (e.g. `RESUME_FROM`, `SFT_INIT`) as required.

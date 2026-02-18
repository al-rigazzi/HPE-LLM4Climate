# Training Module

This directory contains the active RL/SFT training implementation.

## Current Entry Point

- `train_pipeline.py` (module invocation: `python -m multimodal_aifs.training.train_pipeline`)

## Important Files

- `rl_trainer.py` - RL pre-training logic
- `sft_trainer.py` - supervised fine-tuning logic
- `checkpoint_manager.py` - stage-aware checkpoint management
- `climate_dataloader.py` - climate/text sample creation
- `statistics_computer.py` - climate statistics extraction
- `verifiable_rewards.py` - reward computation
- `slurm_rl_training.sh` - RL SLURM job script
- `slurm_sft_training.sh` - SFT SLURM job script
- `prompts/` - Jinja2 prompt templates

## Stage Commands

```bash
# Full pipeline (RL -> SFT)
python -m multimodal_aifs.training.train_pipeline \
  --stage full \
  --zarr-paths data/real_ecmwf_latest.zarr \
  --checkpoint-dir checkpoints/climate_llm

# RL only
python -m multimodal_aifs.training.train_pipeline \
  --stage rl \
  --zarr-paths data/real_ecmwf_latest.zarr \
  --checkpoint-dir checkpoints/climate_llm

# SFT only
python -m multimodal_aifs.training.train_pipeline \
  --stage sft \
  --sft-init rl \
  --zarr-paths data/real_ecmwf_latest.zarr \
  --checkpoint-dir checkpoints/climate_llm
```

## Resume Controls

- `--resume-from <checkpoint_path>`
- `--resume-stage rl|sft`
- `--sft-init rl|resume|fresh`

## Legacy Files

`config.yaml`, `launch.sh`, and `deepspeed_config.json` are kept in the repository, but the primary maintained path is `train_pipeline.py`.

# HPE-LLM4Climate

Multimodal climate analysis project combining an ECMWF AIFS encoder and a text LLM.

## Current Scope

- Real ECMWF/AIFS-compatible climate datasets (primary: Zarr)
- Multimodal climate + text fusion in `multimodal_aifs/core/`
- Training pipeline with RL pre-training and supervised fine-tuning in `multimodal_aifs/training/`
- Unit + integration tests in `multimodal_aifs/tests/`

## Repository Layout

```text
HPE-LLM4Climate/
├── multimodal_aifs/                    # Main package
│   ├── core/                           # AIFS encoder + fusion modules
│   ├── utils/                          # Data/device/distributed utilities
│   ├── training/                       # RL/SFT pipeline and scripts
│   ├── tests/                          # Unit and integration tests
│   ├── examples/                       # Runnable examples
│   ├── docs/                           # Package-specific technical docs
│   └── data/README.md                  # Data notes
├── data/real_ecmwf_latest.zarr/        # Real ECMWF dataset (local copy)
├── aifs-single-1.1/                    # AIFS model assets
├── checkpoints/                        # Training checkpoints
├── outputs/                            # Trained model outputs
├── docs/                               # Top-level operational docs
├── scripts/                            # Project utility scripts
├── .github/workflows/ci.yml            # GitHub CI workflow
└── .gitlab-ci.yml                      # GitLab CI pipeline
```

## Requirements

- Python 3.12+
- Git LFS (for large files)
- Optional accelerators:
  - CUDA GPU (recommended for real-model training)
  - MPS (Apple Silicon)
  - CPU (supported, slower)

## Setup

```bash
git clone --recurse-submodules https://github.com/al-rigazzi/HPE-LLM4Climate.git
cd HPE-LLM4Climate
git lfs install
git lfs pull

python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Quick Usage

### Run examples

```bash
python multimodal_aifs/examples/zarr_aifs_multimodal_example.py
python multimodal_aifs/examples/aifs_mistral_example.py
python multimodal_aifs/examples/multimodal_timeseries_demo.py
```

### Run tests

```bash
pytest multimodal_aifs/tests/unit/ -v --maxfail=5
pytest multimodal_aifs/tests/integration/ -v --maxfail=5 -m "not large_memory"
```

### Run training pipeline

```bash
python -m multimodal_aifs.training.train_pipeline \
  --stage full \
  --model-name mistralai/Ministral-3-8B-Instruct-2512 \
  --zarr-paths data/real_ecmwf_latest.zarr \
  --checkpoint-dir checkpoints/climate_llm
```

For SLURM usage, see `multimodal_aifs/training/slurm_rl_training.sh` and `multimodal_aifs/training/slurm_sft_training.sh`.

## Environment Variables

Common runtime/test toggles:

- `USE_MOCK_AIFS=true|false`
- `USE_MOCK_LLM=true|false`
- `USE_QUANTIZATION=true|false`
- `USE_REAL_ZARR=true|false`

## Documentation Index

- Top-level docs:
  - `docs/RL_TRAINING_PIPELINE.md`
  - `docs/MEMORY_OPTIMIZATION.md`
  - `docs/CI_CD_MIGRATION.md`
  - `docs/PRE_COMMIT_HOOK.md`
- Package docs:
  - `multimodal_aifs/README.md`
  - `multimodal_aifs/training/README.md`
  - `multimodal_aifs/tests/README.md`

## License

Apache 2.0. See `LICENSE`.

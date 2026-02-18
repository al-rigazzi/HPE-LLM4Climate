# multimodal_aifs

Main Python package for multimodal climate-text modeling.

## Package Layout

```text
multimodal_aifs/
├── core/
│   ├── aifs_encoder_utils.py
│   └── aifs_climate_fusion.py
├── utils/
│   ├── zarr_data_loader.py
│   ├── aifs_time_series_tokenizer.py
│   ├── climate_data_utils.py
│   ├── location_utils.py
│   ├── text_utils.py
│   ├── device_utils.py
│   └── distributed_utils.py
├── training/
│   ├── train_pipeline.py
│   ├── rl_trainer.py
│   ├── sft_trainer.py
│   ├── checkpoint_manager.py
│   ├── climate_dataloader.py
│   ├── statistics_computer.py
│   ├── verifiable_rewards.py
│   └── prompts/
├── tests/
├── examples/
├── docs/
└── constants.py
```

## Core Capabilities

- AIFS input preparation and encoder integration
- Climate/text fusion modules for multimodal processing
- RL + SFT training pipeline with checkpointing
- Zarr-based climate data loading

## Typical Commands

```bash
# Unit tests
pytest multimodal_aifs/tests/unit/ -v --maxfail=5

# Integration tests
pytest multimodal_aifs/tests/integration/ -v --maxfail=5 -m "not large_memory"

# Example scripts
python multimodal_aifs/examples/zarr_aifs_multimodal_example.py
python multimodal_aifs/examples/aifs_mistral_example.py
```

See subdirectory READMEs for training, examples, tests, and prompt templates.

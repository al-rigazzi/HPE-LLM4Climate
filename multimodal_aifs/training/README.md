# AIFS Multimodal Climate-Text Fusion Training

This directory contains the **production-ready training pipeline** for AIFS-based multimodal climate-text fusion using DeepSpeed optimization. All training examples and test scripts have been organized into the `examples/` subdirectory.

## 📁 Directory Structure

```
training/
├── train_multimodal.py          # Main AIFS training script
├── config.yaml                  # AIFS training configuration
├── deepspeed_config.json        # DeepSpeed optimization settings
├── launch.sh                    # Training launcher script
├── requirements.txt             # Dependencies
├── inference.py                 # AIFS model inference
├── prepare_data.py              # Data preparation utilities
├── examples/                    # Training examples and demos
│   ├── README.md               # Examples documentation
│   ├── mistral7b_final_success.py # Production AIFS+Mistral-7B-Instruct fusion
│   ├── train_mistral7b_8b.py      # Comprehensive training pipeline
│   └── spatial_comparative_analysis.py #  spatial analysis
└── *.pt                        # Saved model checkpoints (generated)
```

## 🚀 Quick Start

### 1. For Examples and Testing
```bash
# See all available examples
cd examples/
cat README.md

# Start with production-ready AIFS+Llama fusion
python examples/mistral7b_final_success.py

# Try comprehensive training pipeline
python examples/train_mistral7b_8b.py
```

### 2. For Production AIFS Training

```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment (optional)
./setup_env.sh

# Prepare your data
python prepare_data.py --output_dir multimodal/data/training --num_samples 1000

# Configure training (edit config.yaml)
# Then launch training
./launch.sh
```

## 🎯 AIFS Training Pipeline Features

The production AIFS training pipeline supports:
- **AIFS-based climate encoding** for  time-series processing
- **Cross-attention fusion** between AIFS climate features and text
- **DeepSpeed optimization** for memory efficiency and distributed training
- **Mixed precision training** (FP16) for faster training
- **Gradient checkpointing** to reduce memory usage
- **Flexible data loading** for various climate data formats
- **Proven scalability** up to 8B parameter models with AIFS

## ✅ Validated Performance

| Model Scale | Memory Usage | Status | Example Script |
|-------------|-------------|--------|----------------|
| **AIFS + Mistral-7B-Instruct (Simple)** | **8.5GB** | **✅ Production** | `examples/mistral7b_final_success.py` |
| **AIFS + Mistral-7B-Instruct (Full)** | **10.6GB** | **✅ Production** | `examples/train_mistral7b_8b.py` |
| **Spatial Analysis** | **Variable** | **✅ Production** | `examples/spatial_comparative_analysis.py` |

## 📊 System Requirements

- **Minimum**: 8GB RAM (for basic examples)
- **Recommended**: 16GB RAM (for large models)
- **Tested on**: 36GB RAM (all scales work)
- **CPU**: All examples use CPU training for maximum compatibility

training:
  epochs: 10
  batch_size: 4
  learning_rate: 5e-5
```

### 4. Start Training

```bash
# Single GPU training
python train_multimodal.py --config config.yaml

# Multi-GPU training with DeepSpeed
deepspeed train_multimodal.py --config config.yaml --deepspeed deepspeed_config.json

# Distributed training across multiple nodes
deepspeed --num_gpus=8 --num_nodes=2 train_multimodal.py --config config.yaml
```

## Training Features

### Cross-Attention Fusion with AIFS
The model uses cross-attention mechanisms to fuse AIFS climate features with text embeddings:
- AIFS encoder extracts  spatial-temporal features from weather data
- Text encoder processes natural language queries/descriptions
- Cross-attention layers enable bidirectional information flow
- Final fusion creates joint representations for downstream tasks

### Memory Optimization
- **DeepSpeed ZeRO**: Partitions optimizer states, gradients, and parameters
- **Gradient Checkpointing**: Trades compute for memory by recomputing activations
- **Mixed Precision**: Uses FP16 for faster training with lower memory usage
- **Activation Checkpointing**: Selectively saves/recomputes intermediate activations

### Monitoring
- **Progress bars** with real-time loss and learning rate tracking
- **Weights & Biases** integration for experiment tracking (optional)
- **Checkpointing** with best model selection based on validation loss

## Data Format

### Directory Structure
```
multimodal/data/training/
├── train/
│   ├── index.json
│   ├── climate_00001.pt
│   ├── climate_00002.pt
│   └── ...
└── val/
    ├── index.json
    ├── climate_00001.pt
    └── ...
```

### Index File Format
```json
[
  {
    "climate_file": "climate_00001.pt",
    "text": "Analyze climate patterns for the Pacific Northwest region.",
    "target": "The Pacific Northwest shows increased precipitation and moderate temperatures...",
    "location": "Pacific Northwest",
    "weather_condition": "increased precipitation"
  }
]
```

### Training Configuration
- `batch_size`: Per-GPU batch size
- `gradient_accumulation_steps`: Steps to accumulate gradients
- `learning_rate`: Peak learning rate for training
- `warmup_steps`: Learning rate warmup steps
- `max_grad_norm`: Gradient clipping threshold

### DeepSpeed Configuration
- `zero_stage`: ZeRO optimization level (1, 2, or 3)
- `fp16`: Enable mixed precision training
- `gradient_checkpointing`: Enable activation checkpointing
- `cpu_offload`: Offload optimizer states to CPU

##  Usage

### Custom Data Loading
Modify `ClimateTextDataset` class in `train_multimodal.py` to load your specific data format:

```python
def _load_climate_data(self, climate_file: str) -> torch.Tensor:
    # Your custom climate data loading logic
    return climate_tensor

def _load_samples(self) -> list[dict]:
    # Your custom sample loading logic
    return samples_list
```

### Custom Loss Functions
Modify `compute_loss` method for your specific training objective:

```python
def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    # Your custom loss computation
    return loss
```

### Hyperparameter Tuning
Use Weights & Biases sweeps for automated hyperparameter optimization:

```yaml
# Enable W&B in config.yaml
wandb:
  enabled: true
  project: "climate-text-fusion"
  entity: "your-username"
```

## Troubleshooting

### Memory Issues
- Reduce `batch_size` or increase `gradient_accumulation_steps`
- Enable `cpu_offload` in DeepSpeed config
- Use ZeRO stage 3 for very large models
- Enable gradient checkpointing

### Training Instability
- Lower learning rate
- Increase warmup steps
- Use gradient clipping
- Check data quality and preprocessing

### Performance Optimization
- Use multiple data loading workers
- Enable pin_memory for GPU training
- Use NVMe storage for faster data loading
- Profile with PyTorch profiler to identify bottlenecks

## Monitoring Training

### Real-time Monitoring
```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Monitor training progress
tail -f logs/training.log
```

## Contributing

When adding new features:
1. Update the configuration schema in `config.yaml`
2. Add corresponding command-line arguments
3. Update this README with usage examples
4. Test with both single-GPU and multi-GPU setups

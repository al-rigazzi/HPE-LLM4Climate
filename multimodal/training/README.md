# Multimodal Climate-Text Fusion Training

This directory contains scripts and configurations for training the multimodal climate-text fusion model using DeepSpeed for distributed training and memory optimization.

## Overview

The training pipeline supports:
- **Cross-attention fusion** between climate data and text
- **DeepSpeed optimization** for memory efficiency and distributed training
- **Mixed precision training** (FP16) for faster training
- **Gradient checkpointing** to reduce memory usage
- **Flexible data loading** for various climate data formats

## Quick Start

### 1. Install Dependencies

```bash
# Install training requirements
pip install -r requirements-training.txt

# For CUDA support (if not already installed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Prepare Data

```bash
# Create dummy data for testing (optional)
python prepare_data.py --output_dir data/training --num_samples 1000

# Or validate your existing data format
python prepare_data.py --output_dir /path/to/your/data --validate_only
```

### 3. Configure Training

Edit `config.yaml` to match your setup:

```yaml
model:
  prithvi_encoder_path: "path/to/prithvi.wxc.2300m.v1.pt"
  llama_model_name: "meta-llama/Meta-Llama-3-8B"

data:
  train_path: "data/training/train"
  val_path: "data/training/val"

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

### Cross-Attention Fusion
The model uses cross-attention mechanisms to fuse climate data features with text embeddings:
- Climate encoder extracts spatial-temporal features from weather data
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
data/training/
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

### Climate Data Format
Each `.pt` file should contain a PyTorch tensor with shape:
- `[time_steps, channels, height, width]`
- Example: `[2, 160, 64, 64]` for 2 time steps, 160 variables, 64x64 spatial grid

## Configuration Options

### Model Configuration
- `prithvi_encoder_path`: Path to pre-trained PrithviWxC weights
- `llama_model_name`: HuggingFace model name for text encoder
- `freeze_prithvi`: Whether to freeze climate encoder weights
- `fusion_mode`: Type of fusion ("cross_attention", "concatenate", "add")
- `num_fusion_layers`: Number of cross-attention fusion layers

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

## Advanced Usage

### Custom Data Loading
Modify `ClimateTextDataset` class in `train_multimodal.py` to load your specific data format:

```python
def _load_climate_data(self, climate_file: str) -> torch.Tensor:
    # Your custom climate data loading logic
    return climate_tensor

def _load_samples(self) -> List[Dict]:
    # Your custom sample loading logic
    return samples_list
```

### Custom Loss Functions
Modify `compute_loss` method for your specific training objective:

```python
def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
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

### Weights & Biases Dashboard
If enabled, view training metrics at: https://wandb.ai/your-username/climate-text-fusion

## Output

### Checkpoints
Saved in `checkpoints/multimodal/checkpoint_epoch_N/`:
- `mp_rank_00_model_states.pt`: Model weights
- `zero_pp_rank_0_mp_rank_00_optim_states.pt`: Optimizer states  
- `tokenizer/`: Saved tokenizer
- `config.yaml`: Training configuration

### Loading Trained Model
```python
from multimodal.core.climate_text_fusion import ClimateTextFusion

# Load trained model
model = ClimateTextFusion.from_pretrained("checkpoints/multimodal/checkpoint_epoch_5")
```

## Contributing

When adding new features:
1. Update the configuration schema in `config.yaml`
2. Add corresponding command-line arguments
3. Update this README with usage examples
4. Test with both single-GPU and multi-GPU setups

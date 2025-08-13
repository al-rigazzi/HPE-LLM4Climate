# PrithviWxC Multimodal Extensions

This directory contains utilities for extracting and working with components of the PrithviWxC model for multimodal applications.

## Overview

The PrithviWxC model is a powerful encoder-decoder architecture for weather and climate modeling. This module provides tools to extract the encoder portion for use in multimodal applications, transfer learning, and feature extraction tasks.

## Files

- `encoder_extractor.py` - Main script for extracting the encoder from a full PrithviWxC model
- `test_encoder_extractor.py` - Test script to verify the encoder extraction functionality
- `example_usage.py` - Examples of how to use the extracted encoder
- `__init__.py` - Package initialization

## Quick Start

### 1. Extract the Encoder

```bash
python multimodal/encoder_extractor.py \
    --config_path /data/config.yaml \
    --weights_path /data/weights/prithvi.wxc.2300m.v1.pt \
    --output_path /data/weights/prithvi_encoder.pt \
    --surf_scaler_path /data/climatology/musigma_surface.nc \
    --vert_scaler_path /data/climatology/musigma_vertical.nc
```

### 2. Use the Extracted Encoder

```python
from multimodal.encoder_extractor import PrithviWxC_Encoder
import torch

# Load the extracted encoder
checkpoint = torch.load('prithvi_encoder.pt')
encoder = PrithviWxC_Encoder(...)  # Initialize with saved config
encoder.load_state_dict(checkpoint['model_state_dict'])

# Use for feature extraction
batch = {
    'x': climate_data,          # [batch, time, channels, lat, lon]
    'static': static_data,      # [batch, channels_static, lat, lon]
    'climate': climate_baseline, # [batch, channels, lat, lon]
    'input_time': input_times,  # [batch]
    'lead_time': lead_times,    # [batch]
}

features = encoder(batch)  # [batch, n_tokens, local_seq, embed_dim]
```

## Architecture

The extracted encoder includes:

- **Input Preprocessing**: Normalization and scaling of climate variables
- **Patch Embedding**: Converts climate data into patch tokens
- **Position Encoding**: Spatial and temporal position encoding
- **Time Encoding**: Input and lead time embeddings
- **Masking**: Configurable masking for self-supervised learning
- **Transformer Encoder**: Multi-layer transformer with local-global attention

## Use Cases

### 1. Feature Extraction
Use the encoder to extract meaningful representations from climate data:
```python
climate_features = encoder(climate_batch)
# Use features for downstream tasks
```

### 2. Transfer Learning
Fine-tune the encoder for specific climate tasks:
```python
encoder = load_pretrained_encoder()
# Add task-specific head
classifier = nn.Linear(encoder.embed_dim, num_classes)
# Fine-tune on your data
```

### 3. Multimodal Fusion
Combine climate features with other modalities:
```python
climate_features = climate_encoder(climate_data)
satellite_features = satellite_encoder(satellite_data)
fused = fusion_layer([climate_features, satellite_features])
```

### 4. Climate Analysis
Analyze climate patterns and relationships:
```python
features = encoder(historical_data)
# Perform clustering, PCA, or other analysis
patterns = analyze_climate_patterns(features)
```

## Model Configuration

The encoder supports various configuration options:

- **Masking Mode**: `'global'`, `'local'`, or `'both'`
- **Position Encoding**: `'fourier'` or `'absolute'`
- **Residual Mode**: `'climate'`, `'temporal'`, or `'none'`
- **Architecture**: Configurable depth, width, and attention heads

## Testing

Run the test suite to verify functionality:

```bash
python multimodal/test_encoder_extractor.py
```

## Requirements

- PyTorch >= 2.0
- NumPy
- PyYAML
- h5py (for loading MERRA-2 data)
- xarray (for climate data handling)

## Data Format

The encoder expects input data in the following format:

- **Climate Data (`x`)**: `[batch, time, channels, lat, lon]`
  - Multi-timestep atmospheric and surface variables
  - Typically 2 timesteps for input

- **Static Data (`static`)**: `[batch, channels_static, lat, lon]`
  - Time-invariant surface properties (land fraction, topography, etc.)

- **Climate Baseline (`climate`)**: `[batch, channels, lat, lon]`
  - Climate normals or reference state (for residual mode)

- **Time Information**: 
  - `input_time`: Hours from reference time
  - `lead_time`: Forecast lead time in hours

## Output

The encoder outputs high-dimensional feature representations:
- Shape: `[batch, n_unmasked_tokens, local_sequence, embed_dim]`
- These features capture spatial-temporal climate patterns
- Can be used directly or further processed for downstream tasks

## Performance

The encoder significantly reduces the computational requirements compared to the full model:
- Encoder-only: ~1.2B parameters (vs 2.3B for full model)
- Faster inference for feature extraction tasks
- Suitable for real-time applications

## Examples

See `example_usage.py` for detailed examples of:
- Loading and using the encoder
- Feature extraction workflows
- Multimodal fusion patterns
- Downstream task integration

## Contributing

When adding new functionality:
1. Follow the existing code structure
2. Add appropriate tests
3. Update documentation
4. Ensure compatibility with the base PrithviWxC model

## Citation

If you use this encoder extraction in your research, please cite the original PrithviWxC paper and acknowledge this extension.

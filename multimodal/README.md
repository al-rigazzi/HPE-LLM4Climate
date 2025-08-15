# PrithviWxC Multimodal Extensions

This directory contains utilities for extracting and working with components of the PrithviWxC model for multimodal applications, including advanced location-aware climate analysis capabilities.

## Overview

The PrithviWxC model is a powerful encoder-decoder architecture for climate modeling and analysis. This module provides tools to extract the encoder portion for use in multimodal applications, transfer learning, and feature extraction tasks focused on long-term climate trends and projections. Additionally, it includes cutting-edge location-aware functionality that enables geographic-specific climate analysis.

## Key Features

### 🌍 Location-Aware Climate Analysis
- **Geographic Entity Resolution**: Automatically extract and resolve location references from natural language queries
- **Spatial Attention Masking**: Focus climate model attention on specific geographic regions
- **Multi-scale Analysis**: Support for coordinates, cities, countries, states, and large regions
- **Location-aware Fusion**: Combine climate data with text queries and geographic context

### 🤖 Multimodal Fusion
- **Climate-Text Integration**: Combine climate model outputs with natural language processing
- **Multiple Fusion Strategies**: Cross-attention, concatenation, and additive fusion modes
- **Risk Assessment**: Generate climate risk classifications with confidence estimates
- **Trend Analysis**: Project long-term climate trends for specific regions

## Files

### Core Components
- `encoder_extractor.py` - Extract encoder from full PrithviWxC model
- `climate_text_fusion.py` - Core multimodal fusion framework for climate and text
- `location_aware.py` - Geographic resolution and spatial attention infrastructure
- `location_aware_fusion.py` - Location-aware climate analysis system

### Examples and Demos
- `example_usage.py` - Basic encoder extraction and usage examples
- `practical_example.py` - Working demonstration of climate-text fusion
- `fusion_demo.py` - Comprehensive multimodal fusion capabilities
- `location_aware_example.py` - Complete location-aware analysis demonstration

### Testing
- `test_encoder_extractor.py` - Test encoder extraction functionality
- `test_fusion.py` - Test multimodal fusion components
- `test_location_aware.py` - Comprehensive location-aware system tests

### Utilities
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

### 2. Location-Aware Climate Analysis

```python
from multimodal.location_aware_fusion import LocationAwareClimateAnalysis
import torch

# Initialize location-aware system
model = LocationAwareClimateAnalysis()

# Analyze a geographic climate query
climate_features = torch.randn(1, 1024, 768)  # From Prithvi encoder
query = "What crops will be viable in Sweden by 2050?"

result = model.analyze_location_query(climate_features, query)

print(f"Location: {result['location']}")
print(f"Climate Risk: {result['climate_risk']}")
print(f"Confidence: {result['overall_confidence']:.1%}")
print(f"Analysis: {result['interpretation']}")
```

### 3. Basic Multimodal Fusion

```python
from multimodal.climate_text_fusion import ClimateTextFusion, FusionMode

# Initialize fusion system
fusion_model = ClimateTextFusion()

# Combine climate data with text query
result = fusion_model(
    climate_features=climate_features,
    text_query="Long-term drought patterns and agricultural sustainability",
    fusion_mode=FusionMode.CROSS_ATTENTION
)

features = result['fused_features']  # Combined climate-text features
assessment = result['climate_assessment']  # Generated analysis
```

## Geographic Resolution Backends

The location-aware system supports multiple geographic data sources:

### 🌐 **GeoPy/Nominatim (Recommended)**
```bash
pip install geopy
```
- **Pros**: Comprehensive global coverage, free, no API key required
- **Cons**: Requires internet connection
- **Data Source**: OpenStreetMap via Nominatim service
- **Coverage**: Worldwide with detailed city/region boundaries

### 🗺️ **GeoNames API**
```bash
pip install requests
# Register for free username at geonames.org
```
- **Pros**: Very comprehensive, official geographic data
- **Cons**: Requires API registration and internet
- **Coverage**: Worldwide with administrative boundaries

### 🏠 **Local Database (Fallback)**
- **Pros**: Fast, offline, no dependencies
- **Cons**: Limited coverage (major countries/regions only)
- **Coverage**: ~50 major countries, states, and regions

### Usage Examples

```python
from multimodal.location_aware_fusion import LocationAwareClimateAnalysis

# Auto-select best available backend
model = LocationAwareClimateAnalysis()

# Force specific backend
from multimodal.location_aware import GeographicResolver
model.geographic_resolver = GeographicResolver(backend='geopy')

# Test different queries
queries = [
    "Climate impact on Stockholm, Sweden",           # City-level precision
    "Drought risk in Central Valley, California",    # Regional analysis
    "What crops viable in 59.3°N, 18.1°E by 2050?", # Coordinate-based
    "Arctic ice melting trends"                      # Large region
]

for query in queries:
    result = model.analyze_location_query(climate_data, query)
    print(f"Location: {result['location']}")
    print(f"Risk: {result['climate_risk']} ({result['overall_confidence']:.1%})")
```

### Backend Comparison

| Backend | Coverage | Precision | Requirements | Speed | Best For |
|---------|----------|-----------|--------------|-------|----------|
| **GeoPy** | Global | High | Internet | Medium | Production use |
| **GeoNames** | Global | Very High | Internet + API Key | Medium | Research/Commercial |
| **Local** | Limited | Medium | None | Fast | Development/Demo |

### 4. Use the Extracted Encoder

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

### Core Encoder Architecture
The extracted encoder includes:

- **Input Preprocessing**: Normalization and scaling of climate variables
- **Patch Embedding**: Converts climate data into patch tokens
- **Position Encoding**: Spatial and temporal position encoding
- **Time Encoding**: Input and lead time embeddings
- **Masking**: Configurable masking for self-supervised learning
- **Transformer Encoder**: Multi-layer transformer with local-global attention

### Location-Aware Architecture
The location-aware system adds:

- **Geographic Resolver**: Extracts location entities from natural language
- **Spatial Cropper**: Creates attention masks for specific geographic regions
- **Location-Aware Attention**: Modifies transformer attention with spatial focus
- **Geographic Context Encoder**: Encodes location bounds as contextual features
- **Multimodal Fusion**: Combines climate data, text, and geographic context

## Use Cases

### 1. Geographic Climate Questions
Answer location-specific climate queries:
```python
model = LocationAwareClimateAnalysis()
result = model.analyze_location_query(
    climate_features,
    "What crops will be viable in Sweden by 2050?"
)
```

### 2. Regional Risk Assessment
Assess climate risks for specific regions:
```python
queries = [
    "Drought risk in California agriculture",
    "Arctic ice melting patterns",
    "Mediterranean climate resilience"
]
for query in queries:
    assessment = model.analyze_location_query(climate_features, query)
    print(f"{query}: {assessment['climate_risk']}")
```

### 3. Multi-scale Analysis
Support analysis from coordinates to continents:
```python
# Coordinate-level analysis
coordinate_query = "Climate at 40.7°N, 74.0°W"

# Country-level analysis
country_query = "Climate trends in Sweden"

# Regional analysis
regional_query = "Arctic climate changes"
```

### 4. Feature Extraction
Use the encoder to extract meaningful representations from climate data:
```python
climate_features = encoder(climate_batch)
# Use features for downstream tasks
```

### 5. Transfer Learning
Fine-tune the encoder for specific climate tasks:
```python
encoder = load_pretrained_encoder()
# Add task-specific head
classifier = nn.Linear(encoder.embed_dim, num_classes)
# Fine-tune on your data
```

### 6. Multimodal Fusion
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
  - `lead_time`: Climate projection time horizon in hours

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

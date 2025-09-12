# Multimodal AIFS Examples

This directory contains example scripts demonstrating the capabilities of the multimodal AIFS implementation for location-aware climate analysis.

## 📁 Example Structure

### `basic/`
Basic usage examples:
- **`aifs_encoder_demo.py`** - Simple AIFS encoder usage and climate data encoding
- **`location_aware_demo.py`** - Geographic operations and location encoding

### Root Examples
Complete integration examples:
- **`aifs_integration_example.py`** - AIFS integration with time series tokenization
- **`aifs_llama_example.py`** - AIFS + LLaMA multimodal fusion example
- **`multimodal_timeseries_demo.py`** - Time series multimodal demonstration
- **`zarr_aifs_multimodal_example.py`** - Zarr format integration with AIFS



## 🚀 Quick Start

### Basic AIFS Encoder Usage
```python
from multimodal_aifs.core.aifs_encoder_utils import AIFSCompleteEncoder, create_aifs_encoder
import torch

# Initialize AIFS encoder (use the new approach)
encoder = create_aifs_encoder(aifs_model)

# Encode climate data
climate_data = torch.randn(4, 218)  # Batch of climate data
encoded = encoder.encode_climate_data(climate_data)
print(f"Encoded: {climate_data.shape} -> {encoded.shape}")
```

### Climate-Text Fusion
```python
from multimodal_aifs.core import AIFSClimateTextFusion

# Initialize fusion module
fusion = AIFSClimateTextFusion(
    aifs_encoder_path="path/to/aifs/model.pth",
    fusion_dim=512
)

# Fuse climate and text
texts = ["High temperature anomaly detected"]
results = fusion(climate_data, texts)
print(f"Fused features: {results['fused_features'].shape}")
```

### Location-Aware Analysis
```python
from multimodal_aifs.core import AIFSGeographicResolver

# Initialize geographic resolver
resolver = AIFSGeographicResolver()

# Encode location
london_encoding = resolver.encode_location(51.5074, -0.1278)
print(f"Location encoding: {london_encoding.shape}")
```

## 📊 Example Datasets

The examples use various data sources:
- **Synthetic Data**: Generated climate patterns for testing
- **ECMWF Open Data**: Real weather data from ECMWF
- **Sample Locations**: Cities and regions worldwide
- **Climate Descriptions**: Text descriptions of weather conditions

## 🔧 Requirements

To run the examples, ensure you have:
- AIFS model weights (optional, examples work with synthetic data)
- Required dependencies: torch, numpy (for visualizations)
- ECMWF data access (for real data examples)

## 📈 Performance Examples

Examples include performance benchmarks:
- Encoding speed measurements
- Memory usage analysis
- Batch processing efficiency
- Scalability demonstrations

## 🎯 Use Cases Demonstrated

### Climate Analysis
- Temperature anomaly detection
- Precipitation pattern analysis
- Wind pattern characterization
- Pressure system tracking

### Multimodal Integration
- Climate-text alignment scoring
- Location-conditioned analysis
- Spatial attention visualization
- Cross-modal similarity computation

### Real-World Applications
- Weather station data analysis
- Satellite imagery integration
- Climate model validation
- Forecast verification

## 🏃‍♀️ Running Examples

### Individual Examples
```bash
# Run basic demo
python multimodal_aifs/examples/basic/aifs_encoder_demo.py

```

## 📝 Example Output

Examples produce various outputs:
- Performance metrics and timing
- Analysis results and statistics
- Model predictions and comparisons

## 🔍 Debugging Examples

For troubleshooting:
- Examples include verbose output options
- Error handling demonstrations
- Fallback modes for missing data/models
- Step-by-step execution logging

## 📚 Learning Path

Recommended order for exploring examples:
1. **Basic demos** - Understand individual components
2. **Integration examples** - See how components work together
3. ** examples** - Explore sophisticated use cases
4. **Application examples** - See real-world applications

## 🤝 Contributing Examples

When adding new examples:
1. Follow existing naming conventions
2. Include clear documentation and comments
3. Provide both real and synthetic data options
4. Add performance timing where relevant
5. Include error handling and validation
6. Test examples on different systems

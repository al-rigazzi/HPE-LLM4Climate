# Multimodal AIFS Examples

This directory contains example scripts demonstrating the capabilities of the multimodal AIFS implementation for location-aware climate analysis.

## 📁 Example Structure

### `basic/`
Basic usage examples:
- **`aifs_encoder_demo.py`** - Simple AIFS encoder usage and climate data encoding
- **`climate_text_fusion_demo.py`** - Basic climate-text fusion with AIFS
- **`location_aware_demo.py`** - Geographic operations and location encoding
- **`simple_pipeline_demo.py`** - End-to-end pipeline demonstration

### `advanced/`
Advanced usage examples:
- **`real_data_analysis.py`** - Real ECMWF data processing and analysis
- **`multimodal_attention_demo.py`** - Location-aware attention mechanisms
- **`spatial_analysis_demo.py`** - Spatial cropping and regional analysis
- **`climate_similarity_analysis.py`** - Climate pattern similarity analysis

### `applications/`
Application-specific examples:
- **`weather_forecast_analysis.py`** - Weather forecasting with multimodal analysis
- **`climate_change_detection.py`** - Climate change pattern detection
- **`regional_climate_comparison.py`** - Multi-region climate comparison
- **`extreme_weather_analysis.py`** - Extreme weather event analysis

## 🚀 Quick Start

### Basic AIFS Encoder Usage
```python
from multimodal_aifs.utils import AIFSEncoderWrapper
import torch

# Initialize AIFS encoder
encoder = AIFSEncoderWrapper("path/to/aifs/model.pth")

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
- Required dependencies: torch, numpy, matplotlib (for visualizations)
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

# Run advanced analysis
python multimodal_aifs/examples/advanced/real_data_analysis.py

# Run application example
python multimodal_aifs/examples/applications/weather_forecast_analysis.py
```

### With Jupyter Notebooks
Some examples are also available as Jupyter notebooks for interactive exploration:
```bash
jupyter notebook multimodal_aifs/examples/notebooks/
```

## 📝 Example Output

Examples produce various outputs:
- Performance metrics and timing
- Visualization plots (when matplotlib available)
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
3. **Advanced examples** - Explore sophisticated use cases
4. **Application examples** - See real-world applications

## 🤝 Contributing Examples

When adding new examples:
1. Follow existing naming conventions
2. Include clear documentation and comments
3. Provide both real and synthetic data options
4. Add performance timing where relevant
5. Include error handling and validation
6. Test examples on different systems

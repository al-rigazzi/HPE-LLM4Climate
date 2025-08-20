# ECMWF AIFS Integration

## Overview

This project now includes the **ECMWF Artificial Intelligence Forecasting System (AIFS) Single v1.0** as an alternative climate AI backend. AIFS is ECMWF's operational data-driven weather forecasting model that can be used alongside or instead of PrithviWxC for different use cases.

## Submodule Location

- **Path**: `aifs-single-1.0/`
- **Source**: https://huggingface.co/ecmwf/aifs-single-1.0
- **License**: CC-BY-4.0

## Contents

The AIFS submodule includes:

- **Model Checkpoint**: `aifs-single-mse-1.0.ckpt` - Pre-trained model weights
- **Configuration Files**:
  - `config_pretraining.yaml` - Pre-training configuration
  - `config_finetuning.yaml` - Fine-tuning configuration
- **Jupyter Notebook**: `run_AIFS_v1.ipynb` - Usage examples and demonstrations
- **Assets**: Visualization examples and documentation images

## Key Features

AIFS Single v1.0 provides:

- **Upper-level atmospheric variables** with improved performance at 50 and 100 hPa
- **Enhanced precipitation forecasting** with better total precipitation skill
- **Extended variable coverage**:
  - 100-meter winds
  - Snow-fall predictions
  - Surface solar radiation
  - Land variables (soil moisture, soil temperature)
- **Multi-day forecasts** up to 10+ days
- **Global coverage** with high spatial resolution

## Integration with HPE-LLM4Climate

### Use Cases

1. **Alternative Climate Backend**: Use AIFS instead of PrithviWxC for global weather forecasting
2. **Specialized Applications**: Leverage AIFS for operational weather prediction scenarios
3. **Extended Forecasting**: Use AIFS for longer-term (10+ day) forecasts
4. **Operational Deployment**: Deploy AIFS for production weather services
5. **Research Applications**: Study different AI approaches to climate modeling

### When to Use AIFS vs PrithviWxC

**Use AIFS for:**
- Global weather forecasting (10+ days)
- Operational weather prediction
- Standard meteorological variables
- Real-time forecasting applications
- Large-scale atmospheric dynamics

**Use PrithviWxC for:**
- Regional climate analysis
- Multimodal climate + text integration
- Custom climate applications
- Research and experimental scenarios
- Fine-grained spatial analysis

### Data Compatibility

AIFS works with standard meteorological variables that align with:
- **MERRA-2 datasets** (already integrated in this project)
- **ERA5 reanalysis data**
- **Real-time weather observations**

## Usage Instructions

### Initial Setup

When cloning this repository with submodules:

```bash
git clone --recurse-submodules https://github.com/al-rigazzi/HPE-LLM4Climate.git
```

### Updating Submodule

To update to the latest AIFS version:

```bash
cd aifs-single-1.0
git pull origin main
cd ..
git add aifs-single-1.0
git commit -m "Update AIFS submodule to latest version"
```

### Running AIFS

1. **Navigate to submodule**:
   ```bash
   cd aifs-single-1.0
   ```

2. **Install dependencies** (see AIFS documentation for requirements)

3. **Run the example notebook**:
   ```bash
   jupyter notebook run_AIFS_v1.ipynb
   ```

## Integration Examples

### Using AIFS as Alternative Backend

Replace PrithviWxC with AIFS for global forecasting:

```python
# Instead of PrithviWxC multimodal system
# from multimodal.location_aware_fusion import LocationAwareFusion

# Use AIFS for climate prediction
from aifs_wrapper import AIFSPredictor  # Custom wrapper

# Initialize AIFS model
aifs_model = AIFSPredictor('aifs-single-1.0/aifs-single-mse-1.0.ckpt')

# Run predictions
location = (40.7128, -74.0060)  # New York City
forecast = aifs_model.predict(location, days=10)
```

### Switching Between Models

Create a flexible climate backend system:

```python
class ClimateBackend:
    def __init__(self, model_type="aifs"):
        if model_type == "aifs":
            self.model = AIFSPredictor('aifs-single-1.0/aifs-single-mse-1.0.ckpt')
        elif model_type == "prithvi":
            self.model = LocationAwareFusion()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def predict(self, location, **kwargs):
        return self.model.predict(location, **kwargs)

# Use AIFS for this session
climate_ai = ClimateBackend("aifs")
results = climate_ai.predict((lat, lon), days=7)
```

### Data Pipeline Integration

Use with the MERRA-2 processing pipeline:

```python
from dataset.data_loader import MERRA2DataLoader
from aifs_integration import process_for_aifs

# Load processed MERRA-2 data
dataset = MERRA2DataLoader('multimodal/dataset/example_output/merra2_prithvi_*.npz')

# Convert to AIFS-compatible format
aifs_input = process_for_aifs(dataset)

# Run AIFS inference
aifs_predictions = aifs_model.forecast(aifs_input)
```

## Future Development

Planned integrations:

1. **Automated Benchmarking**: Regular comparison pipelines
2. **Hybrid Models**: Combine AIFS global forecasts with Prithvi regional details
3. **Real-time Processing**: Stream weather data through both models
4. **Uncertainty Quantification**: Ensemble predictions using multiple models

## References

- [AIFS Technical Documentation](https://huggingface.co/ecmwf/aifs-single-1.0)
- [ECMWF Official AIFS Page](https://www.ecmwf.int/en/research/projects/aifs)
- [Anemoi Library](https://github.com/ecmwf/anemoi) - AIFS implementation framework

## License

AIFS Single v1.0 is released under CC-BY-4.0 license. Please refer to the original repository for complete licensing terms.

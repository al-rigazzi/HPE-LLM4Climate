# Multimodal AIFS for Climate Analysis

This directory contains the complete implementation of multimodal AIFS (Artificial Intelligence Forecasting System) for climate analysis, integrating ECMWF's AIFS with text processing capabilities.

## Directory Structure

```
multimodal_aifs/
├── README.md                    # This file
├── requirements-aifs.txt        # AIFS-specific requirements
│
├── core/                       # Core multimodal fusion modules
│   ├── __init__.py
│   ├── aifs_encoder_utils.py   # AIFS encoder utilities
│   ├── aifs_climate_fusion.py  # Climate-text fusion implementation
│   ├── aifs_location_aware.py  # Location-aware geographic processing
│   └── aifs_location_aware_fusion.py # Complete AIFS multimodal system
│
├── utils/                      # Utility modules
│   ├── __init__.py
│   ├── aifs_encoder_utils.py   # AIFS encoder utilities (DEPRECATED - moved to core)
│   ├── aifs_time_series_tokenizer.py # Time series tokenization
│   ├── climate_data_utils.py   # Climate data processing
│   ├── location_utils.py       # Geographic and spatial utilities
│   └── text_utils.py          # Text processing utilities
│
├── models/                     # Model files and checkpoints
│   └── extracted_models/       # Extracted AIFS encoder models
│       ├── aifs_encoder_full.pth
│       └── aifs_encoder_state_dict.pth
│
├── scripts/                    # Utility and processing scripts
│   ├── extract_aifs_encoder.py # Extract encoder from AIFS model
│   └── check_encoder_signature.py # Check encoder compatibility
│
├── data/                       # ECMWF data processing and storage
│   ├── README.md              # Data processing documentation
│   ├── scripts/               # Data processing scripts
│   │   ├── test_ecmwf_download.py # ECMWF data download test
│   │   ├── real_data_pipeline.py # Real data processing pipeline
│   │   └── apply_encoder_to_data.py # Apply encoder to ECMWF data
│   ├── grib/                  # GRIB weather data files (cache files)
│   └── results/               # Data processing results
│
├── analysis/                   # Analysis and inspection tools
│   ├── aifs_detailed_analysis.py # Detailed AIFS model analysis
│   ├── aifs_model_summary.py   # Model summary and statistics
│   ├── analyze_aifs_processor.py # Processor analysis
│   ├── explain_aifs_processor.py # Processor explanations
│   ├── inspect_aifs_checkpoint.py # Checkpoint inspection
│   ├── aifs_checkpoint_analysis.md # Analysis documentation
│   └── processor_analysis.json # Processor analysis results
│
├── results/                    # Processing results and outputs
│   └── benchmarks/            # Performance benchmark results
│
├── examples/                   # Example usage and demonstrations
│   ├── README.md              # Example documentation
│   ├── aifs_integration_example.py # Integration example
│   ├── aifs_llama_example.py  # AIFS + LLaMA example
│   ├── multimodal_timeseries_demo.py # Time series demo
│   ├── zarr_aifs_multimodal_example.py # Zarr integration example
│   └── basic/                 # Basic examples
│       ├── location_aware_demo.py
│       └── aifs_encoder_demo.py
│
├── tests/                     # Test suite
│   ├── README.md              # Test documentation
│   ├── unit/                  # Unit tests
│   │   ├── test_aifs_encoder_utils.py
│   │   ├── test_aifs_time_series_tokenizer.py
│   │   ├── test_climate_data_utils.py
│   │   ├── test_location_utils.py
│   │   └── test_text_utils*.py
│   ├── integration/           # Integration tests
│   │   ├── test__integration.py # Moved from root
│   │   ├── test__tokenizer_integration.py # Moved from root
│   │   ├── test_aifs_encoder_integration.py # Moved from root
│   │   ├── test_aifs_climate_fusion.py
│   │   ├── test_time_series_integration.py
│   │   ├── test_aifs_llama_integration.py
│   │   ├── test_real_llama_integration.py
│   │   ├── test_aifs_llama3_real_fusion.py
│   │   ├── test_aifs_llama3_pytest.py
│   │   └── zarr/              # Zarr-specific tests
│   └── benchmarks/            # Performance benchmarks
│       └── test_time_series_performance.py
│
├── training/                   # Training examples and utilities
│   ├── README.md              # Training documentation
│   ├── config.yaml            # Training configuration
│   ├── deepspeed_config.json  # DeepSpeed configuration
│   ├── inference.py           # Inference utilities
│   ├── prepare_data.py        # Data preparation
│   ├── setup_env.sh          # Environment setup
│   ├── train_multimodal.py   # Main training script
│   └── examples/              # Training examples
│       ├── README.md
│       ├── llama3_final_success.py # Simple AIFS+LLaMA fusion
│       ├── spatial_comparative_analysis.py # Spatial analysis
│       └── train_llama3_8b.py # LLaMA-3-8B training
│
└── docs/                      # Documentation
    ├── README.md
    ├── zarr_integration_guide.md
    └── architecture/              # Architecture diagrams and tools
        ├── create_aifs_architecture_diagram.py
        ├── create_aifs_attention_detail.py
        └── README.md
```

## Architecture Diagrams

Generate professional architecture diagrams for the AIFS multimodal system:

```bash
# From project root (convenience script)
python generate_aifs_architecture_diagrams.py

# Or directly from architecture directory
cd multimodal_aifs/docs/architecture
python create_aifs_architecture_diagram.py     # Main AIFS system architecture
python create_aifs_attention_detail.py         # Cross-attention mechanism detail
```

Output files:
- `aifs_multimodal_architecture_diagram.{png,pdf}` - Complete AIFS system overview
- `aifs_cross_attention_detail.{png,pdf}` - Detailed attention mechanism


## Quick Start

### Installation
```bash
# Install AIFS-specific requirements (includes Zarr support)
pip install -r multimodal_aifs/requirements-aifs.txt

# Install base requirements
pip install -r requirements.txt
```

### Zarr Data Support

The AIFS multimodal system supports modern Zarr format climate datasets:

```bash
# Load Zarr climate data
python multimodal_aifs/examples/zarr_aifs_multimodal_example.py
    --zarr-path /path/to/climate.zarr
    --start-time "2024-01-01"
    --end-time "2024-01-07"

# Load from cloud storage
python multimodal_aifs/examples/zarr_aifs_multimodal_example.py
    --zarr-path s3://bucket/era5.zarr
    --variables temperature_2m surface_pressure
```

**Zarr Advantages:**
- **Cloud-optimized**: Efficient access from S3, GCS, Azure
- **Chunked access**: Load only needed time periods/regions
- **Compressed**: Reduced storage and bandwidth requirements
- **Parallel I/O**: Faster loading with distributed computing

### Data Format Support

**Supported Input Formats:**
- **GRIB files** (ECMWF, NOAA) - via cfgrib/xarray
- **NetCDF files** - via xarray
- **Zarr datasets** - cloud-optimized, chunked access ⭐ **NEW**
- **HDF5 files** - via h5py/xarray
- **Raw tensor data** - NumPy arrays, PyTorch tensors

**Output Format:**
- 5D tensors: `[batch, time, variables, height, width]`
- Compatible with AIFS TimeSeries tokenizer
- Ready for multimodal fusion with Llama 3-8B

### Basic Usage
```python
from multimodal_aifs.core.aifs_encoder_utils import AIFSCompleteEncoder, create_aifs_encoder
from multimodal_aifs.core.aifs_climate_fusion import AIFSClimateTextFusion

# Initialize AIFS encoder (new approach)
encoder = create_aifs_encoder(aifs_model)

# Initialize multimodal fusion
fusion = AIFSClimateTextFusion(
    aifs_model=aifs_model
    aifs_encoder_path="multimodal_aifs/models/extracted_models/aifs_encoder_full.pth"
)
```

### Running Examples
```bash
# Basic AIFS encoder demo
python multimodal_aifs/examples/basic/aifs_encoder_demo.py

# Location-aware climate analysis
python multimodal_aifs/examples/basic/location_aware_demo.py

# Integration example
python multimodal_aifs/examples/aifs_integration_example.py
```

### Running Tests
```bash
# Run all tests
python -m pytest multimodal_aifs/tests/ -v

# Run specific test categories
python -m pytest multimodal_aifs/tests/unit/ -v
python -m pytest multimodal_aifs/tests/integration/ -v
```

## Key Features

- **Real AIFS Integration**: Uses actual ECMWF AIFS encoder (19.8M parameters)
- **Multimodal Fusion**: Climate data + text processing with cross-attention
- **Location-Aware Processing**: Geographic and spatial analysis capabilities
- **Performance Optimized**: High-throughput batch processing (>500k samples/s)
- **Comprehensive Testing**: Unit, integration, and system tests
- **Robust Error Handling**: Graceful fallbacks and error recovery

## Model Information

The AIFS encoder (`extracted_models/aifs_encoder_full.pth`) contains:
- **Architecture**: GraphTransformerForwardMapper
- **Parameters**: 19,884,832 total parameters
- **Input**: 218-dimensional climate features
- **Output**: 1024-dimensional embeddings
- **Source**: ECMWF AIFS v1.0 (extracted encoder module)

## Results and Analysis

- **Test Coverage**: 27/27 tests passing (100% success rate)
- **Performance**: >500k samples/s throughput on CPU
- **Real Model Integration**: Successfully loads and uses actual AIFS encoder
- **Cross-platform**: Tested on macOS ARM64, supports CPU/GPU

## Contributing

See the main project README for contribution guidelines. All AIFS-related development should happen within this `multimodal_aifs/` directory to maintain organization.

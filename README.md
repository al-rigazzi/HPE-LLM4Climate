# 🌍 HPE-LLM4Climate: Multimodal Climate Analysis Large Language Model

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-orange.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-87.8%25-brightgreen.svg)](coverage.xml)

## Overview

This repository implements a multimodal large language model designed for climate data analysis, combining ECMWF's AI Forecasting System (AIFS) encoder with text-based language models. The system enables zero-shot analysis of numerical climate datasets and investigation of high-order climate features through natural language interfaces.

### Research Goals

- Investigate zero-shot capabilities of LLM models on numerical climate datasets
- Enable identification of high-order features (vortices, tornadoes, droughts) without supervised examples
- Develop production-ready multimodal climate analysis tools
- Bridge climate science and natural language processing

### Key Features

- **ECMWF AIFS Integration**: Uses AIFS encoder for climate data processing with 542,080 grid points and 103 variables
- **Multimodal Fusion**: Cross-attention mechanisms combining climate embeddings with text features
- **Zarr Data Support**: Cloud-optimized climate data loading and processing
- **Production Standards**: 10.00/10 pylint score, comprehensive test suite, modern Python 3.12+ type hints
- **Multi-Platform**: Native support for Apple Silicon (MPS), NVIDIA GPUs (CUDA), and CPU-only systems

## Repository Structure

```
HPE-LLM4Climate/
├── multimodal_aifs/                   # Main implementation
│   ├── core/                          # Core fusion modules
│   │   ├── aifs_climate_fusion.py     # AIFS-text fusion model
│   │   └── aifs_encoder_utils.py      # AIFS encoder utilities
│   ├── utils/                         # Data processing utilities
│   │   ├── zarr_data_loader.py        # Zarr climate data loader
│   │   ├── aifs_time_series_tokenizer.py # Time series tokenization
│   │   ├── climate_data_utils.py      # Climate data utilities
│   │   ├── location_utils.py          # Geographic utilities
│   │   └── text_utils.py              # Text processing utilities
│   ├── constants.py                   # Centralized constants and configurations
│   ├── examples/                      # Working examples
│   │   ├── zarr_aifs_multimodal_example.py # Zarr→AIFS→LLM pipeline
│   │   ├── aifs_mistral_example.py      # AIFS-Mistral integration
│   │   └── multimodal_timeseries_demo.py # Time series demonstration
│   ├── tests/                         # Comprehensive test suite
│   │   ├── integration/               # Integration tests
│   │   │   └── zarr/                  # Zarr integration tests
│   │   └── unit/                      # Unit tests
│   └── training/                      # Training examples
│       └── examples/                  # Training scripts
├── aifs-single-1.1/                   # ECMWF AIFS model (submodule)
│   ├── aifs-single-mse-1.1.ckpt      # AIFS model weights
│   └── config_pretraining.yaml       # AIFS configuration
├── data/                              # Data directory
│   └── real_ecmwf_latest.zarr/       # Real ECMWF climate data
├── scripts/                           # Utility scripts
└── docs/                              # Documentation
```

## Prerequisites

- **Python**: 3.12+ (exactly 3.12 required for current configuration)
- **Git LFS**: Required for downloading large data files (see installation below)
- **Hardware**:
  - **Apple Silicon (M1/M2/M3)**: Full native support with MPS acceleration ✅
  - **NVIDIA GPUs**: CUDA support for training/inference ✅
  - **CPU**: Intel/AMD x86_64 support ✅
- **Memory**: Minimum 8GB RAM (16GB+ recommended for full models)
- **Storage**: ~10GB for model weights and test data

### Device Compatibility

- **Apple Silicon**: Native ARM64 with MPS acceleration for PyTorch operations
- **NVIDIA GPUs**: CUDA 11.8+ or 12.x support
- **CPU-only**: Cross-platform support (Linux/Windows/macOS)

## Installation

### 1. Prerequisites Setup

#### Install Git LFS (Required)

The repository uses Git LFS to manage large data files (zarr datasets, model weights).

**macOS:**
```bash
brew install git-lfs
git lfs install
```

**Ubuntu/Debian:**
```bash
sudo apt-get install git-lfs
git lfs install
```

**Windows:**
Download from [git-lfs.github.com](https://git-lfs.github.com/) or use:
```bash
winget install git-lfs
git lfs install
```

#### Clone the Repository

After installing Git LFS, clone the repository with LFS support:

```bash
git clone https://github.com/al-rigazzi/HPE-LLM4Climate.git
cd HPE-LLM4Climate

# Pull LFS files (including real ECMWF data)
git lfs pull
```

### 2. Clone Repository

```bash
git clone --recurse-submodules https://github.com/al-rigazzi/HPE-LLM4Climate.git
cd HPE-LLM4Climate

# Pull LFS files (including real ECMWF data and model weights)
git lfs pull

# If already cloned, initialize submodules
git submodule update --init --recursive
```

### 2. Python Environment

```bash
# Create virtual environment
python3.12 -m venv llm4climate
source llm4climate/bin/activate  # Windows: llm4climate\Scripts\activate

# Or using conda
conda create -n llm4climate python=3.12
conda activate llm4climate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### Tested Configuration

The following versions are tested and verified working:

```
torch==2.4.0                    # PyTorch with MPS/CUDA support
transformers==4.55.2            # HuggingFace transformers
zarr==3.1.1                     # Cloud-optimized arrays
xarray==2025.8.0                # N-dimensional labeled arrays
numpy==2.3.2                    # Numerical computing
pytest==8.4.1                   # Testing framework
```

For AIFS-specific dependencies:
```bash
pip install anemoi-models        # AIFS model loading
pip install einops              # Tensor operations
```

## Quick Start

### Basic Usage

```python
from multimodal_aifs.core.aifs_climate_fusion import AIFSClimateTextFusion
from multimodal_aifs.utils.zarr_data_loader import ZarrClimateLoader
import torch

# Load real ECMWF climate data
loader = ZarrClimateLoader('data/real_ecmwf_latest.zarr')
climate_data = loader.load_time_range(None, None)  # Load all time steps
climate_tensor = loader.to_aifs_tensor(climate_data, batch_size=1, device='cpu')

# Initialize fusion model
fusion_model = AIFSClimateTextFusion(
    aifs_model=None,  # Uses mock for testing
  climate_dim=102,  # Raw AIFS encoder output
    text_dim=768,     # Text model hidden size
    fusion_dim=512,   # Output dimension
    device='cpu'
)

# Process climate and text
text_inputs = ["Analyze temperature patterns in this climate data."]
result = fusion_model(climate_tensor, text_inputs)
print(f"Fusion output shape: {result['fused_features'].shape}")
```

### Run Examples

```bash
# Zarr data integration example
python multimodal_aifs/examples/zarr_aifs_multimodal_example.py

# AIFS-Mistral integration
python multimodal_aifs/examples/aifs_mistral_example.py

# Time series demonstration
python multimodal_aifs/examples/multimodal_timeseries_demo.py
```

### Run Tests

```bash
# Full test suite
pytest multimodal_aifs/tests/ -v

# Integration tests only
pytest multimodal_aifs/tests/integration/ -v

# Zarr integration tests
pytest multimodal_aifs/tests/integration/zarr/ -v
```
## Configuration

### Environment Variables

The system supports several environment variables for configuration:

- `USE_MOCK_AIFS`: Use mock AIFS model instead of real model (useful for testing)
- `USE_MOCK_LLM`: Use mock LLM instead of real language model
- `USE_QUANTIZATION`: Enable model quantization for memory efficiency
- `LLM_MODEL_NAME`: Specify which language model to use (default: `mistralai/Mistral-7B-Instruct-v0.3`)

```bash
# Example: Run with mock models for fast testing
USE_MOCK_AIFS=true USE_MOCK_LLM=true python multimodal_aifs/examples/zarr_aifs_multimodal_example.py
```

### AIFS Constants

Key constants defined in `multimodal_aifs/constants.py`:

- `AIFS_GRID_POINTS = 542080`: Spatial grid points in AIFS model
- `AIFS_INPUT_VARIABLES = 103`: Total input variables (90 prognostic + 13 forcing)
- `AIFS_RAW_ENCODER_OUTPUT_DIM = 102`: Raw encoder output dimension
- `ALL_AIFS_VARIABLES`: Complete list of 103 climate variables

## Testing

### Test Structure

```
multimodal_aifs/tests/
├── integration/                    # Integration tests
│   ├── zarr/                      # Zarr data integration
│   │   ├── test_cpu_mistral_zarr.py # CPU-optimized Mistral + AIFS + Zarr
│   │   ├── test_real_mistral_zarr.py # Full model integration
│   │   └── test_zarr_integration.py # Zarr loader tests
│   ├── test_aifs_climate_fusion.py # Climate fusion tests
│   └── test_aifs_encoder_integration.py # AIFS encoder tests
└── unit/                          # Unit tests
    └── test_aifs_time_series_tokenizer.py # Tokenizer tests
```

### Running Tests

```bash
# All tests
pytest multimodal_aifs/tests/ -v

# Integration tests only
pytest multimodal_aifs/tests/integration/ -v

# Specific test with verbose output
pytest multimodal_aifs/tests/integration/zarr/test_cpu_mistral_zarr.py -v -s

# With mock models (faster)
USE_MOCK_AIFS=true pytest multimodal_aifs/tests/integration/ -v
```

### Code Quality

This repository maintains production-ready code standards:

- **Pylint Score**: 10.00/10 (perfect score across all core modules)
- **Type Hints**: Modern Python 3.12+ type annotations throughout
- **Test Coverage**: Comprehensive test suite with integration and unit tests
- **Documentation**: Extensive docstrings and API documentation

## Research Applications

This system enables research in:

- **Climate Data Analysis**: Zero-shot analysis of numerical climate datasets
- **Multimodal AI**: Combining numerical data with natural language processing
- **Climate Communication**: Natural language interfaces for climate data
- **Pattern Recognition**: Identification of climate patterns without supervised training

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes and add tests
4. Ensure all tests pass and pylint score remains 10.00/10
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run code quality checks
python -m pylint multimodal_aifs/core/ --fail-under=10.0
python -m mypy multimodal_aifs/
python -m isort multimodal_aifs/
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

This matches the license used by ECMWF AIFS for code and scripts.

## Acknowledgments

- **ECMWF AIFS**: AI Forecasting System integration
  - Paper: [AIFS - ECMWF's data-driven forecasting system](https://arxiv.org/abs/2406.01465)
  - Model: [ecmwf/aifs-single-1.1](https://huggingface.co/ecmwf/aifs-single-1.1)
- **Mistral AI**: Mistral model series and open-source AI contributions
- **HuggingFace**: Transformers library and model hub infrastructure

## Contact

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and community support

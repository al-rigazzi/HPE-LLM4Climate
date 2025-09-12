# 🌍 HPE-LLM4Climate: Multimodal Climate AI System
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-orange.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![Experimental](https://img.shields.io/badge/Status-Experimental-red.svg)](#)

## ⚠️ **EXPERIMENTAL REPOSITORY** ⚠️

This is an **experimental research repository** featuring a **multimodal climate AI system** that combines ECMWF's AI Forecasting System (AIFS) with large language models:

**🌪️ Primary Implementation: AIFS-based Multimodal System** (`/multimodal_aifs/`) ✅ **Fully Operational**
- **ECMWF AI Forecasting System** integration with 19M parameter encoder
- **Meta-Llama-3-8B integration** with cross-attention fusion
- **Zarr format support** for cloud-optimized climate data
- **Real-time testing** on CPU/GPU with comprehensive test suite

## 🚀 Overview

This **operational project** implements a multimodal fusion system that bridges climate science and natural language processing using ECMWF's AIFS as the primary climate encoder:

### 🌪️ **AIFS Multimodal Implementation** (`/multimodal_aifs/`) ✅ **Primary System**
- **ECMWF AI Forecasting System** integration with 19M parameter encoder
- **Zarr format support** for cloud-optimized climate data
- **Meta-Llama-3-8B integration** with multimodal fusion
- **Spatial region loading** with coordinate wrapping
- **Real-time testing** on CPU/GPU with comprehensive test suite

### 🎯 **Applications Enabled**
- **Climate Trend Analysis**: AI assistants that explain long-term climate patterns and projections
- **Location-Aware Climate Analysis**: Geographic-specific climate assessments for regions, countries, or coordinates
- **Climate Impact Assessment**: Automated analysis of climate change effects on various sectors
- **Agricultural Climate Planning**: Long-term farming recommendations based on climate projections
- **Climate Risk Assessment**: Analysis of future climate risks and adaptation strategies
- **Climate Education**: Interactive tools for learning about climate science and long-term trends

## ✨ Key Features

### 🏗️ Core Components
- **AIFS Encoder Integration**: Standalone ECMWF AIFS climate encoder with full model compatibility
- **Multimodal Fusion Framework**: Combines AIFS climate features and text data using cross-attention
- **Location-Aware Climate Analysis**: Geographic-specific analysis with spatial attention masking
- **Transformer Integration**: Support for Llama 3, BERT, and other HuggingFace models
- **Production-Ready Architecture**: Comprehensive testing and validation infrastructure

### 🌍 Geographic Intelligence
- **Multi-Backend Resolution**: GeoPy/Nominatim geographic coordinate resolution
- **Spatial Context Integration**: Location-aware processing and analysis
- **Multi-Scale Analysis**: From coordinate-level to global climate assessment
- **Real-World Integration**: OpenStreetMap and geographic database support

### 🔧 Fusion Strategies
- **Cross-Attention Fusion**: Deep interaction between climate and text features
- **Concatenation Fusion**: Simple feature combination for fast inference
- **Additive Fusion**: Element-wise feature integration
- **Location-Aware Fusion**: Geographic context integration with climate data

### 🧪 Production Ready
- Comprehensive test suite with encoder validation
- Multiple usage examples and demonstrations
- Complete documentation and API reference
- Real-world application templates

### 🌪️ ECMWF AIFS Integration
- **Primary Climate Backend**: AIFS provides state-of-the-art global weather forecasting
- **AIFS Single v1.0**: Operational AI forecasting system from ECMWF
- **Extended Variables**: Upper-air, precipitation, radiation, and land variables
- **Modular Architecture**: Clean integration with multimodal text processing

> **Current Status**: ECMWF's Artificial Intelligence Forecasting System (AIFS) serves as the primary climate AI backend, providing operational weather forecasting capabilities with state-of-the-art accuracy. See [`multimodal_aifs/README.md`](multimodal_aifs/README.md) for details.

## 📋 Prerequisites

- **Python**: 3.12 (exactly 3.12 required - only supported version)
- **Hardware**:
  - **Apple Silicon Macs (M1/M2/M3)**: Native support with MPS acceleration ✅
  - **NVIDIA GPUs**: CUDA support for accelerated training/inference ✅
  - **Intel/AMD CPUs**: Currently being tested.
- **Memory**: At least 16GB RAM (32GB+ recommended for full models)
- **Storage**: ~50GB free space for model weights and data

### 🍎 Apple Silicon Support

This project has **full native support** for Apple Silicon Macs (M1/M2/M3):
- ✅ **Native ARM64 compatibility** with all dependencies
- ✅ **MPS (Metal Performance Shaders)** acceleration for PyTorch operations
- ✅ **Optimized memory usage** for Apple Silicon architecture
- ✅ **Comprehensive testing** on macOS 15.6 with Apple Silicon

**Note**: For text generation with large language models on Apple Silicon, the system automatically uses CPU to avoid MPS compatibility issues, ensuring reliable operation.

## 🛠️ Installation

### 1. Install Git LFS (Required)

The AIFS model file (~948MB) is stored using Git LFS. Install Git LFS before cloning:

```bash
# On macOS
brew install git-lfs

# On Ubuntu/Debian
sudo apt-get install git-lfs

# On Windows (using Chocolatey)
choco install git-lfs

# Or download from: https://git-lfs.github.io/

# Initialize Git LFS for your user account (run once)
git lfs install
```

### 2. Clone the Repository

```bash
# Clone with submodules (includes ECMWF AIFS)
git clone --recurse-submodules https://github.com/al-rigazzi/HPE-LLM4Climate.git
cd HPE-LLM4Climate

# Or if already cloned, initialize submodules
git submodule update --init --recursive
```

### 3. Set Up Python Environment

```bash
# Create virtual environment
python -m venv llm4climate
source llm4climate/bin/activate  # On Windows: llm4climate\Scripts\activate

# Or using conda
conda create -n llm4climate python=3.12
conda activate llm4climate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 🏆 **Working AIFS Multimodal Configuration**

The following configuration is **tested and operational** for the AIFS multimodal implementation:

**Core Dependencies (Verified Working):**
```bash
# Core ML Framework
torch==2.4.0                  # PyTorch with MPS support
torch-geometric==2.4.0        # Graph neural networks
transformers==4.55.2          # HuggingFace transformers
accelerate==1.10.0            # Model acceleration
bitsandbytes==0.42.0          # Quantization support

# Climate Data Processing
zarr==3.1.1                   # Cloud-optimized arrays
xarray==2025.8.0             # N-dimensional labeled arrays
cfgrib==0.9.15.0             # GRIB file support
ecmwf-opendata==0.3.22       # ECMWF data access

# Scientific Computing
numpy==2.3.2                  # Numerical computing
matplotlib==3.10.5           # Plotting and visualization

# Testing and Development
pytest==8.4.1                # Testing framework
psutil==7.0.0                 # System monitoring
```

**Installation for AIFS Multimodal:**
```bash
# Create environment (Python 3.12 required)
python -m venv llm4climate-aifs
source llm4climate-aifs/bin/activate

# Install exact working versions
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
pip install transformers==4.55.2 accelerate==1.10.0 bitsandbytes==0.42.0
pip install zarr==3.1.1 xarray==2025.8.0 numpy==2.3.2
pip install pytest==8.4.1 matplotlib==3.10.5 psutil==7.0.0
pip install cfgrib==0.9.15.0 ecmwf-opendata==0.3.22

# Install remaining requirements
pip install -r requirements.txt
```

> **✅ Tested**: This exact configuration successfully runs the complete zarr → AIFS → Meta-Llama-3-8B pipeline on Apple Silicon (M-series) and Intel/AMD CPUs.

**Alternative Dependencies:**
**Alternative Dependencies:**
- `torch>=2.4.0` - Deep learning framework with MPS support for Apple Silicon
- `transformers>=4.55.0` - HuggingFace transformers for LLMs (Llama 3, BERT, etc.)
- `numpy`, `pandas`, `xarray` - Scientific computing and data manipulation
- `matplotlib` - Data visualization and plotting
- `huggingface_hub` - Model downloads and HuggingFace integration
- `accelerate`, `tokenizers`, `safetensors` - Optimized model handling
- `geopy>=2.3.0` - Geographic data processing for location-aware analysis
- `requests>=2.31.0` - HTTP requests for geographic API integration
- `tqdm`, `PyYAML`, `h5py`, `packaging` - Utilities and file handling

**Optional Geographic Extensions:**
```bash
# For enhanced geographic capabilities
pip install shapely geopandas folium pycountry
```

### 5. Verified System Configurations

#### 🍎 Apple Silicon (Tested Configuration)
**System:**
- **Hardware**: Apple Silicon M-series (M1/M2/M3)
- **OS**: macOS 15.6+ (Darwin 24.6.0)
- **Architecture**: ARM64 native
- **GPU**: MPS (Metal Performance Shaders) acceleration
- **Python**: 3.12.8 (CPython) - **ONLY SUPPORTED VERSION**

**Tested Package Versions:**
```
torch==2.4.0                  # MPS-optimized for Apple Silicon
transformers==4.55.2          # Full Llama 3 support
numpy==2.3.2                  # ARM64 optimized
pandas==2.3.1                 # Native Apple Silicon
accelerate==1.10.0            # MPS acceleration support
deepspeed==0.17.5             # Distributed training support
geopy==2.4.1                  # Geographic processing
huggingface_hub==0.34.4      # Model downloads
safetensors==0.6.2           # Efficient model storage
```

#### 🖥️ NVIDIA GPU Systems
**System:**
- **Hardware**: NVIDIA GPU with CUDA 11.8+ or 12.x
- **OS**: Linux/Windows
- **Python**: 3.12 (exactly 3.12 required - only supported version)

**Installation:**
```bash
# CUDA 12.1 (recommended)
pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121
pip install transformers>=4.55.2 accelerate>=1.10.0
pip install -r requirements.txt
```

#### 💻 CPU-Only Systems
**System:**
- **Hardware**: Intel/AMD x86_64 or ARM64
- **OS**: Linux/Windows/macOS
- **Python**: 3.12 (exactly 3.12 required - only supported version)

**Installation:**
```bash
pip install torch==2.4.0+cpu torchvision==0.19.0+cpu
pip install -r requirements.txt
```

> **Apple Silicon Users**: The system automatically handles MPS compatibility issues by using CPU for text generation while leveraging MPS for other operations. This ensures maximum reliability and performance.

### 6. Supported Language Models

The system supports a wide range of transformer models for climate-text fusion:

#### 🦙 **Meta Llama Models (Recommended)**
- `meta-llama/Meta-Llama-3-8B` - **Tested ✅** - Superior language understanding (**HF approval required**)
- `meta-llama/Llama-2-7b-hf` - Standard Llama 2 (**HF approval required**)
- `meta-llama/Llama-2-7b-chat-hf` - Chat-optimized version (**HF approval required**)

#### 🤖 **Alternative Models (No Access Required)**
- `microsoft/DialoGPT-medium` - **Tested ✅** - Conversational AI
- `bert-base-uncased` - **Tested ✅** - Encoder-only for embeddings
- `roberta-base` - **Tested ✅** - Robust language understanding
- `distilbert-base-uncased` - Lightweight BERT variant
- `google/flan-t5-small` - Text-to-text generation
- `facebook/opt-350m` - Lightweight GPT alternative

#### 📝 **Usage Examples**
```python
# Using Meta Llama 3 (best performance)
fusion_model = ClimateTextFusion(
    llama_model_name="meta-llama/Meta-Llama-3-8B"
)
```

**Model Access**: Meta Llama models require HuggingFace account approval. Alternative models work immediately without approval.

## 📥 Data Setup

### 1. Download AIFS Model Weights

The system will automatically download AIFS model weights on first use, or you can verify the setup:

```bash
# Verify AIFS model exists (should be ~948MB)
ls -la aifs-single-1.0/aifs-single-mse-1.0.ckpt

# If missing, the model will be downloaded automatically
# The AIFS model is included as a git submodule
```

### 2. Verify AIFS Installation

```bash
# Test AIFS encoder extraction
python multimodal_aifs/scripts/extract_aifs_encoder.py

# Verify the extracted encoder
python multimodal_aifs/scripts/check_encoder_signature.py
```

This extracts the AIFS encoder (~20MB) from the full model (948MB) and validates the installation.

## 🚀 Quick Start

## 🚀 Quick Start

### ✅ **Primary AIFS Multimodal System**

```bash
# Navigate to AIFS implementation
cd multimodal_aifs/

# Try complete zarr integration example
python examples/zarr_aifs_multimodal_example.py

# Test AIFS+Llama integration
python training/examples/llama3_final_success.py

# Run comprehensive tests
python -m pytest tests/integration/ -v
```

### Basic AIFS Multimodal Fusion

```python
from multimodal_aifs.core.aifs_climate_fusion import AIFSClimateTextFusion
import torch

# Initialize the AIFS-based fusion model
fusion_model = AIFSClimateTextFusion(
    aifs_encoder_path='aifs-single-1.0/aifs-single-mse-1.0.ckpt',
    climate_dim=1024,
    text_dim=4096,  # Llama-3-8B hidden size
    fusion_dim=512,
    num_attention_heads=8,
    device='cpu'  # or 'cuda' for GPU
)

# Prepare climate data (AIFS format)
climate_data = torch.randn(1, 20, 64, 64)  # [batch, variables, lat, lon]

# Prepare text data
text_inputs = [
    "What will the temperature patterns be like in the Northern Hemisphere?",
    "How might precipitation change in tropical regions?"
]

# Process through the fusion model
fusion_outputs = fusion_model(climate_data, text_inputs)
print(f"Fusion output shape: {fusion_outputs.shape}")
```

### Training Example (AIFS)

```python
# Run AIFS training
python multimodal_aifs/training/train_multimodal.py
```

### Simplified Example (Recommended for Beginners)

```python
# Run the practical AIFS example
python multimodal_aifs/examples/zarr_aifs_multimodal_example.py
```

This demonstrates:
- Loading a simplified fusion model
- Processing sample climate and text data
- Climate impact assessment classification
- Feature visualization

## 📁 **Experimental Repository Structure**

This repository contains **two different multimodal implementations** for research and experimentation:

```
HPE-LLM4Climate/                 # 🧪 EXPERIMENTAL REPOSITORY
├── README.md                    # Main project documentation
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Project configuration
├── example_aifs_encoder_usage.py # Basic AIFS encoder demo
│
├── 🟢 multimodal_aifs/         # ✅ WORKING IMPLEMENTATION
│   ├── README.md               # AIFS multimodal documentation
│   ├── core/                   # AIFS fusion modules
│   │   ├── aifs_climate_fusion.py      # AIFS-Llama fusion
│   │   ├── aifs_location_aware.py      # AIFS geographic processing
│   │   └── aifs_location_aware_fusion.py # Complete AIFS system
│   ├── utils/                  # AIFS utility modules
│   │   ├── zarr_data_loader.py          # Zarr climate data loader ✅
│   │   ├── aifs_encoder_utils.py        # AIFS encoder utilities ✅
│   │   └── aifs_time_series_tokenizer.py # Time series tokenization ✅
│   ├── examples/               # Working AIFS examples
│   │   ├── zarr_aifs_multimodal_example.py # Zarr→AIFS→Llama pipeline ✅
│   │   └── basic/              # Basic AIFS examples
│   ├── tests/                  # Comprehensive test suite ✅
│   │   ├── integration/zarr/   # Zarr integration tests ✅
│   │   ├── unit/               # Unit tests ✅
│   │   └── benchmarks/         # Performance tests ✅
│   └── docs/                   # AIFS documentation
│
├── 🌪️ aifs-single-1.0/         # ECMWF AIFS model
│   ├── aifs-single-mse-1.0.ckpt        # AIFS checkpoint ✅
│   ├── config_pretraining.yaml         # AIFS configuration ✅
│   └── run_AIFS_v1.ipynb               # AIFS example notebook ✅
```

### 🎯 **Choose Your Path**

- **🟢 Use AIFS Implementation** (`/multimodal_aifs/`) for:
  - ✅ Working zarr data integration
  - ✅ Real Meta-Llama-3-8B support
  - ✅ Comprehensive test suite
  - ✅ Production-ready examples

## 🧪 Testing & Validation

### Comprehensive Test Suite

The system includes extensive testing across multiple components.

## 🔧 Configuration Options

### AIFS Fusion Configuration

```python
from multimodal_aifs.core.aifs_climate_fusion import AIFSClimateTextFusion

# Cross-attention: Deep feature interaction (recommended)
fusion_model = AIFSClimateTextFusion(
    aifs_encoder_path='aifs-single-1.0/aifs-single-mse-1.0.ckpt',
    climate_dim=1024,
    text_dim=4096,  # Meta-Llama-3-8B hidden size
    fusion_dim=512,
    num_attention_heads=8
)

# Lightweight configuration for testing
fusion_model = AIFSClimateTextFusion(
    aifs_encoder_path='aifs-single-1.0/aifs-single-mse-1.0.ckpt',
    climate_dim=512,
    text_dim=768,   # Smaller text models
    fusion_dim=256,
    num_attention_heads=4
)
```

### Model Selection

```python
# Production model (high quality, Meta Llama)
fusion_model = AIFSClimateTextFusion(
    text_model_name='meta-llama/Meta-Llama-3-8B',
    text_dim=4096
)

# Testing model (fast inference, less memory)
fusion_model = AIFSClimateTextFusion(
    text_model_name='prajjwal1/bert-tiny',
    text_dim=128
)
```


## 🔬 Research Applications

This system enables novel research in:

- **Climate Science**: Automated analysis and explanation of climate patterns
- **Natural Language Processing**: Domain-specific language models for meteorology
- **Human-Computer Interaction**: Conversational interfaces for scientific data
- **Emergency Management**: Automated alert systems and risk communication
- **Education**: Interactive climate learning tools

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

See our [contribution guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ECMWF AIFS**: AI Forecasting System model
  - Paper: [Lang et al. (2024) - AIFS - ECMWF's data-driven forecasting system](https://arxiv.org/abs/2406.01465)
  - Model: [ecmwf/aifs-single-1.0](https://huggingface.co/ecmwf/aifs-single-1.0)
- **Meta**: Llama model series and open-source AI contributions

## 📬 Contact

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and community support

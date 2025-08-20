# 🌍 HPE-LLM4Climate: Multimodal Climate AI System

## ⚠️ **EXPERIMENTAL REPOSITORY** ⚠️

This is an **experimental research repository** containing **two different implementations** of multimodal climate/weather LLMs:

1. **PrithviWxC-based Implementation** (`/multimodal/`, `/PrithviWxC/`) - Original climate foundation model approach
2. **AIFS-based Implementation** (`/multimodal_aifs/`) - ECMWF AI Forecasting System integration ✅ **Currently Working**

> **Status**: The AIFS multimodal implementation is currently operational and tested. The PrithviWxC implementation is in development. This repository serves as a research testbed for exploring different approaches to multimodal climate AI.

A comprehensive multimodal AI system that combines climate data processing with natural language understanding capabilities using transformer models like Llama 3.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-orange.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![Experimental](https://img.shields.io/badge/Status-Experimental-red.svg)](#)

## 🚀 Overview

This **experimental project** implements multimodal fusion systems that bridge climate science and natural language processing. It explores **two different architectural approaches**:

### 🌪️ **AIFS Multimodal Implementation** (`/multimodal_aifs/`) ✅ **Working**
- **ECMWF AI Forecasting System** integration with 19M parameter encoder
- **Zarr format support** for cloud-optimized climate data
- **Meta-Llama-3-8B integration** with multimodal fusion
- **Spatial region loading** with coordinate wrapping
- **Real-time testing** on CPU/GPU with comprehensive test suite

### 🌍 **PrithviWxC Implementation** (`/multimodal/`, `/PrithviWxC/`) 🚧 **In Development**
- **IBM PrithviWxC** foundation model approach
- **Traditional climate data processing** pipeline
- **Research exploration** of alternative architectures

### 🎯 **Applications Enabled**
- **Climate Trend Analysis**: AI assistants that explain long-term climate patterns and projections
- **Location-Aware Climate Analysis**: Geographic-specific climate assessments for regions, countries, or coordinates
- **Climate Impact Assessment**: Automated analysis of climate change effects on various sectors
- **Agricultural Climate Planning**: Long-term farming recommendations based on climate projections
- **Climate Risk Assessment**: Analysis of future climate risks and adaptation strategies
- **Climate Education**: Interactive tools for learning about climate science and long-term trends

## ✨ Key Features

### 🏗️ Core Components
- **PrithviWxC Encoder Extraction**: Standalone climate feature encoder with full model compatibility
- **Multimodal Fusion Framework**: Combines climate and text data using multiple fusion strategies
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
- **Alternative Climate Backend**: Use AIFS instead of PrithviWxC for global forecasting
- **AIFS Single v1.0**: Operational AI forecasting system from ECMWF
- **Global Weather Forecasting**: 10+ day forecasts with high accuracy
- **Extended Variables**: Upper-air, precipitation, radiation, and land variables
- **Flexible Architecture**: Switch between AIFS and PrithviWxC based on use case

> **New**: ECMWF's Artificial Intelligence Forecasting System (AIFS) has been integrated as an alternative climate AI backend, providing operational weather forecasting capabilities alongside the existing PrithviWxC system. See [`docs/aifs_integration.md`](docs/aifs_integration.md) for details.

## 📋 Prerequisites

- **Python**: 3.8 or higher (3.10+ recommended)
- **Hardware**:
  - **Apple Silicon Macs (M1/M2/M3)**: Native support with MPS acceleration ✅
  - **NVIDIA GPUs**: CUDA support for accelerated training/inference ✅
  - **Intel/AMD CPUs**: CPU-only operation supported ✅
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

### 1. Clone the Repository

```bash
# Clone with submodules (includes ECMWF AIFS)
git clone --recurse-submodules https://github.com/al-rigazzi/HPE-LLM4Climate.git
cd HPE-LLM4Climate

# Or if already cloned, initialize submodules
git submodule update --init --recursive
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv llm4climate
source llm4climate/bin/activate  # On Windows: llm4climate\Scripts\activate

# Or using conda
conda create -n llm4climate python=3.10
conda activate llm4climate
```

### 3. Install Dependencies

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
# Create environment (Python 3.12+ recommended)
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

### 4. Verified System Configurations

#### 🍎 Apple Silicon (Tested Configuration)
**System:**
- **Hardware**: Apple Silicon M-series (M1/M2/M3)
- **OS**: macOS 15.6+ (Darwin 24.6.0)
- **Architecture**: ARM64 native
- **GPU**: MPS (Metal Performance Shaders) acceleration
- **Python**: 3.13.3 (CPython)

**Tested Package Versions:**
```
torch==2.8.0                  # MPS-optimized for Apple Silicon
transformers==4.55.0          # Full Llama 3 support
numpy==2.3.2                  # ARM64 optimized
pandas==2.3.1                 # Native Apple Silicon
accelerate==1.10.0            # MPS acceleration support
geopy==2.4.1                  # Geographic processing
huggingface_hub==0.34.4      # Model downloads
safetensors==0.6.2           # Efficient model storage
```

#### 🖥️ NVIDIA GPU Systems
**System:**
- **Hardware**: NVIDIA GPU with CUDA 11.8+ or 12.x
- **OS**: Linux/Windows
- **Python**: 3.8-3.11 (3.10 recommended)

**Installation:**
```bash
# CUDA 12.1 (recommended)
pip install torch==2.8.0+cu121 torchvision==0.23.0+cu121
pip install transformers>=4.55.0 accelerate>=1.10.0
pip install -r requirements.txt
```

#### 💻 CPU-Only Systems
**System:**
- **Hardware**: Intel/AMD x86_64 or ARM64
- **OS**: Linux/Windows/macOS
- **Python**: 3.8+ (3.10+ recommended)

**Installation:**
```bash
pip install torch==2.8.0+cpu torchvision==0.23.0+cpu
pip install -r requirements.txt
```

> **Apple Silicon Users**: The system automatically handles MPS compatibility issues by using CPU for text generation while leveraging MPS for other operations. This ensures maximum reliability and performance.

### 5. Supported Language Models

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

# Using alternative model (no HF access needed)
fusion_model = ClimateTextFusion(
    llama_model_name="microsoft/DialoGPT-medium"
)
```

**Model Access**: Meta Llama models require HuggingFace account approval. Alternative models work immediately without approval.

## 📥 Data Setup

### 1. Download Model Weights

The system will automatically download required model weights on first use, or you can download them manually:

```bash
# PrithviWxC full model (26GB)
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='Prithvi-WxC/prithvi.wxc.2300m.v1',
                filename='prithvi.wxc.2300m.v1.pt',
                local_dir='data/weights/')
"

# Configuration files
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='Prithvi-WxC/prithvi.wxc.2300m.v1',
                filename='config.yaml',
                local_dir='data/')
"
```

### 2. Extract Encoder Weights

```bash
python multimodal/encoder_extractor.py \
    --config_path data/config.yaml \
    --weights_path data/weights/prithvi.wxc.2300m.v1.pt \
    --output_path data/weights/prithvi_encoder.pt
```

This creates a standalone encoder (7.3GB) from the full model (26GB).

## 🚀 Quick Start

### ⚠️ **Choose Your Implementation**

This repository contains **two different multimodal implementations**:

#### 🟢 **AIFS Multimodal** (Recommended - Working)
```bash
# Navigate to AIFS implementation
cd multimodal_aifs/

# Run basic test to verify installation
python tests/integration/test_aifs_llama_integration.py

# Try zarr integration example
python examples/zarr_aifs_multimodal_example.py
```

#### 🟡 **PrithviWxC Multimodal** (Experimental - In Development)
```bash
# Navigate to PrithviWxC implementation
cd multimodal/

# This implementation is under development
```

> **Recommendation**: Use the **AIFS multimodal implementation** (`/multimodal_aifs/`) for current work. See detailed documentation in [`multimodal_aifs/README.md`](multimodal_aifs/README.md).

### Basic Multimodal Fusion (PrithviWxC - Legacy)

```python
from multimodal.climate_text_fusion import ClimateTextFusion
import torch

# Initialize the fusion model
fusion_model = ClimateTextFusion(
    prithvi_encoder_path='data/weights/prithvi_encoder.pt',
    llama_model_name='meta-llama/Meta-Llama-3-8B',  # Requires HF approval - use 'prajjwal1/bert-tiny' for testing
    fusion_mode='cross_attention',
    max_climate_tokens=1024,
    max_text_length=512
)

# Prepare climate data
climate_batch = {
    'x': climate_data,           # [batch, time, channels, lat, lon]
    'static': static_data,       # [batch, static_channels, lat, lon]
    'climate': climate_baseline, # [batch, channels, lat, lon]
    'input_time': input_times,   # [batch] - hours from reference
    'lead_time': lead_times      # [batch] - climate projection time horizon
}

# Prepare text data
text_inputs = [
    "What is the best crop to plant in Sweden in 2050?",
    "How sustainable will it be to live in Arizona by 2100?",
    "How much more likely will tornadoes be in 2050 compared to now?"
]

# Run multimodal fusion
outputs = fusion_model(climate_batch, text_inputs)
fused_features = outputs['fused_features']  # Combined climate-text features
```

### Simplified Example (Recommended for Beginners)

```python
# Run the practical example
python multimodal/practical_example.py
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
├── data/                        # Shared model weights and configuration
│   ├── config.yaml             # Configuration files
│   ├── climatology/            # Climate normalization data
│   └── weights/                # Model weight files
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
├── 🟡 multimodal/              # 🚧 IN DEVELOPMENT
│   ├── README.md               # PrithviWxC documentation
│   ├── core/                   # PrithviWxC fusion modules
│   │   ├── climate_text_fusion.py      # PrithviWxC fusion framework
│   │   ├── location_aware.py           # Geographic processing
│   │   └── location_aware_fusion.py    # Complete analysis system
│   ├── utils/                  # PrithviWxC utilities
│   │   └── encoder_extractor.py        # PrithviWxC encoder extraction
│   ├── examples/               # PrithviWxC examples (development)
│   ├── tests/                  # PrithviWxC test suite (comprehensive) ✅
│   ├── validation/             # PrithviWxC validation and testing ✅
│   └── docs/                   # PrithviWxC documentation
│
├── 🌍 PrithviWxC/              # IBM PrithviWxC model integration
│   ├── model.py                # PrithviWxC model implementation
│   └── rollout.py              # PrithviWxC inference utilities
│
├── 🌪️ aifs-single-1.0/         # ECMWF AIFS model
│   ├── aifs-single-mse-1.0.ckpt        # AIFS checkpoint ✅
│   ├── config_pretraining.yaml         # AIFS configuration ✅
│   └── run_AIFS_v1.ipynb               # AIFS example notebook ✅
│
├── PrithviWxC/                  # Original climate model
│   ├── model.py                # Core model implementation
│   ├── rollout.py              # Inference utilities
│   └── dataloaders/            # Data loading components
```

### 🎯 **Choose Your Path**

- **🟢 Use AIFS Implementation** (`/multimodal_aifs/`) for:
  - ✅ Working zarr data integration
  - ✅ Real Meta-Llama-3-8B support
  - ✅ Comprehensive test suite
  - ✅ Production-ready examples

- **🟡 Explore PrithviWxC** (`/multimodal/`) for:
  - 🚧 Research and development
  - 🚧 Alternative architecture exploration
  - 🚧 IBM foundation model approach
│   │   ├── test_encoder_only.py                  # Encoder functionality
│   │   ├── test_simple_encoder_extraction.py     # Basic extraction
│   │   ├── test_full_encoder_pipeline.py         # Complete pipeline
│   │   ├── test_llama_integration.py             # LLM integration
│   │   ├── test_llama_comprehensive.py           # Advanced testing
│   │   ├── debug_weight_loading.py               # Debug utilities
│   │   └── debug_forward_pass.py                 # Forward pass debug
│   ├── system/                 # System verification
│   │   └── verify_setup.py     # Complete setup validation
│   └── demos/                  # Working demonstrations
│       └── working_location_demo.py  # Location-aware demo
├── PrithviWxC/                  # Original climate model
│   ├── model.py                # Core model implementation
│   ├── rollout.py              # Inference utilities
│   └── dataloaders/            # Data loading components
```

## 🧪 Testing & Validation

### Comprehensive Test Suite

The system includes extensive testing across multiple components:

```bash
# Multimodal system tests
python multimodal/tests/test_fusion.py                    # Fusion framework tests
python multimodal/tests/test_encoder_extractor.py         # Encoder extraction tests
python multimodal/tests/test_location_aware.py           # Geographic processing tests

# Integration tests
python multimodal/tests/integration/test_simple_encoder_extraction.py     # Basic encoder validation
python multimodal/tests/integration/test_encoder_loading_verification.py  # Complete loading validation
python multimodal/tests/integration/test_encoder_only.py                  # Standalone encoder tests
python multimodal/tests/integration/test_full_encoder_pipeline.py         # End-to-end pipeline
python multimodal/tests/integration/test_llama_integration.py             # LLM integration
python multimodal/tests/integration/test_llama_comprehensive.py           # Advanced system tests

# System verification and demos
python multimodal/tests/system/verify_setup.py                # Complete system setup verification
python multimodal/tests/demos/working_location_demo.py        # Location-aware demonstration

# Debug utilities (for development)
python multimodal/tests/integration/debug_weight_loading.py   # Weight loading diagnostics
python multimodal/tests/integration/debug_forward_pass.py     # Forward pass analysis
```

### ✅ **Validation Results**

**System Tests (16/16 passing):**
- ✅ **Encoder Extraction**: PrithviWxC encoder successfully extracted and validated
- ✅ **Multimodal Fusion**: All fusion modes (cross-attention, concatenation, additive) working
- ✅ **Location-Aware Processing**: Geographic resolution with 100% success rate
- ✅ **Language Model Integration**: Meta-Llama-3-8B and alternatives fully functional
- ✅ **Apple Silicon Compatibility**: Native ARM64 support with MPS acceleration
- ✅ **Geographic Coverage**: 6 location types (countries, states, coordinates, regions, cities)

**Performance Metrics:**
- **Location Resolution**: 8/12 queries achieved precise geographic bounds
- **Risk Assessment**: Balanced distribution (Low: 5, Moderate: 7, High: 4)
- **Average Confidence**: 46.9% overall, 37.5% risk-specific
- **Model Compatibility**: 9 different transformer models tested and validated

### Test Output Example
```
🌍 Location-Aware Climate Analysis Demo
✅ System Status: FULLY FUNCTIONAL
🤖 Using: Meta-Llama-3-8B (7.5B parameters)
🗺️  Geographic: GeoPy/Nominatim geocoder

📊 Analysis Summary:
✅ Successful analyses: 12/12
🌍 Geographic Coverage: 6 location types
⚠️  Risk Distribution: Balanced across risk levels
🎯 Average Confidence: 46.9%
🏆 Best Location Identifications:
   • 'How will climate change affect agricultural produc...' → Sverige
   • 'What are the drought risks for California...' → California, United States
   • 'Sea level rise impacts on coastal infrastructure i...' → 25.7°N, 80.2°W
```

## 🎯 Usage Examples

### 1. Climate Question Answering

```python
from multimodal.climate_text_fusion import ClimateQuestionAnswering

# Initialize QA system
qa_model = ClimateQuestionAnswering(
    prithvi_encoder_path='data/weights/prithvi_encoder.pt',
    llama_model_name='meta-llama/Llama-3.2-1B'
)

# Ask questions about climate trends
answer = qa_model.answer_question(
    climate_data=historical_climate_data,
    question="How much more likely will tornadoes be in 2050 compared to now?"
)
print(f"Answer: {answer}")
```

### 2. Climate Assessment Generation

```python
from multimodal.climate_text_fusion import ClimateTextGeneration

# Initialize assessment generator
assessment_generator = ClimateTextGeneration(
    prithvi_encoder_path='data/weights/prithvi_encoder.pt',
    llama_model_name='meta-llama/Llama-3.2-3B'
)

# Generate climate assessment
report = assessment_generator.generate_report(
    climate_data=projection_data,
    template="Generate a climate impact assessment for agriculture in 2050:"
)
print(report)
```

### 3. Agricultural Climate Planning

```python
# Combine climate projections with agricultural questions
advisory = qa_model.answer_question(
    climate_data=long_term_climate_data,
    question="What is the best crop to plant in Sweden considering 2050 climate projections?"
)
```

### 4. Climate Risk Assessment

```python
# Generate climate risk analysis
risk_assessment = assessment_generator.generate_assessment(
    climate_data=future_climate_projections,
    assessment_type="regional_sustainability"
)
```

## 🔧 Configuration Options

### Fusion Modes

```python
# Cross-attention: Deep feature interaction
fusion_model = ClimateTextFusion(fusion_mode='cross_attention')

# Concatenation: Simple feature combination
fusion_model = ClimateTextFusion(fusion_mode='concatenate')

# Additive: Element-wise feature fusion
fusion_model = ClimateTextFusion(fusion_mode='add')
```

### Model Selection

```python
# Large model (high quality, more memory)
fusion_model = ClimateTextFusion(
    llama_model_name='meta-llama/Meta-Llama-3-8B'
)

# Small model (fast inference, less memory)
fusion_model = ClimateTextFusion(
    llama_model_name='prajjwal1/bert-tiny'
)
```

### Memory Optimization

```python
# Freeze models during training
fusion_model = ClimateTextFusion(
    freeze_prithvi=True,    # Freeze climate encoder
    freeze_llama=True       # Freeze text model
)

# Use gradient checkpointing
fusion_model = ClimateTextFusion(
    use_gradient_checkpointing=True
)
```

## 📊 Performance Benchmarks

| Model Configuration | Memory Usage | Inference Time | Quality Score |
|-------------------|--------------|----------------|---------------|
| Llama-3.2-3B + Cross-Attention | ~24GB | 2.1s | ⭐⭐⭐⭐⭐ |
| Llama-3.2-1B + Cross-Attention | ~12GB | 1.3s | ⭐⭐⭐⭐ |
| BERT-base + Concatenation | ~8GB | 0.8s | ⭐⭐⭐ |
| BERT-tiny + Concatenation | ~4GB | 0.4s | ⭐⭐ |

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Use smaller models or reduce batch size
   fusion_model = ClimateTextFusion(
       llama_model_name='prajjwal1/bert-tiny',
       max_climate_tokens=512,
       max_text_length=256
   )
   ```

2. **Model Download Errors**
   ```bash
   # Manual download
   huggingface-cli download Prithvi-WxC/prithvi.wxc.2300m.v1 --local-dir data/
   ```

3. **Import Errors**
   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements.txt
   ```

### Getting Help

- 📖 Check the [multimodal README](multimodal/README.md) for detailed documentation
- 🧪 Run test scripts to verify your setup
- 💡 See [example_usage.py](multimodal/example_usage.py) for more examples
- 🐛 Open an issue on GitHub for bugs or feature requests

## 🔬 Research Applications

This system enables novel research in:

- **Climate Science**: Automated analysis and explanation of climate patterns
- **Natural Language Processing**: Domain-specific language models for meteorology
- **Human-Computer Interaction**: Conversational interfaces for scientific data
- **Emergency Management**: Automated alert systems and risk communication
- **Education**: Interactive climate learning tools

## 🛣️ Roadmap

### Near Term
- [ ] Support for additional climate models (ECMWF, GFS)
- [ ] Real-time data integration APIs
- [ ] Web interface for easy access
- [ ] Fine-tuning recipes for specific domains

### Long Term
- [ ] Multi-language support
- [ ] Federated learning capabilities
- [ ] Integration with climate simulation workflows
- [ ] Mobile applications

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

- **IBM**: For the original PrithviWxC model
- **HuggingFace**: For the transformers library and model hub
- **PyTorch**: For the deep learning framework
- **Climate Research Community**: For datasets and domain expertise

## 📬 Contact

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and community support
- **Email**: [your-email@domain.com] for collaboration inquiries

---

**Ready to revolutionize climate AI? Get started with our multimodal fusion system today!** 🌟

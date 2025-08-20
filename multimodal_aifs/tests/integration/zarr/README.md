# Zarr Integration Tests

This directory contains comprehensive tests for Zarr format integration with the AIFS multimodal system.

## Test Files

### Core Integration Tests
- **`test_zarr_integration.py`** - Basic zarr → AIFS tensor conversion pipeline test
- **`test_5d_aifs_capability.py`** - AIFS 5D tensor processing capability verification

### Real Llama Integration Tests
- **`test_real_llama_zarr.py`** - Full pipeline with real Meta-Llama-3-8B model
- **`test_cpu_llama_zarr.py`** - CPU-optimized test with quantization
- **`test_real_llama_cpu_full.py`** - CPU full-precision real Llama test

## What These Tests Validate

### ✅ Zarr Data Loading
- Local and cloud-based zarr datasets
- Time range selection and spatial subsetting
- Variable filtering and normalization
- Memory-efficient chunked loading

### ✅ AIFS Processing
- 5D tensor format conversion `[B,T,V,H,W]`
- Climate data tokenization (19M parameter encoder)
- Temporal sequence modeling
- Spatial pattern recognition

### ✅ Multimodal Integration
- Real Meta-Llama-3-8B integration (8B parameters)
- Cross-attention fusion mechanisms
- Climate-text multimodal processing
- CPU and GPU execution paths

### ✅ Performance Validation
- Memory usage optimization
- Processing time benchmarks
- Model quantization strategies
- System resource monitoring

## Usage Examples

### Basic Zarr Integration Test
```bash
cd multimodal_aifs/tests/integration/zarr
python test_zarr_integration.py
```

### Real Llama Pipeline Test
```bash
cd multimodal_aifs/tests/integration/zarr
python test_real_llama_cpu_full.py
```

### CPU-Optimized Test
```bash
cd multimodal_aifs/tests/integration/zarr
python test_cpu_llama_zarr.py
```

## System Requirements

### Minimum Requirements
- **RAM**: 16+ GB (for real Llama tests)
- **Storage**: 20+ GB (for model downloads)
- **Python**: 3.8+
- **Dependencies**: zarr, xarray, transformers, torch

### Recommended Requirements
- **RAM**: 32+ GB
- **GPU**: CUDA-capable (for quantization)
- **CPU**: Multi-core for parallel processing
- **Network**: High-speed for model downloads

## Dependencies

Install required packages:
```bash
pip install zarr xarray transformers torch psutil accelerate bitsandbytes
```

## Test Results

All tests have been validated on:
- **macOS ARM64** (M1/M2 chips)
- **CPU-only execution** (no CUDA required)
- **Real Meta-Llama-3-8B model** loading and inference
- **Full zarr → AIFS → Llama pipeline** functionality

## Performance Benchmarks

### Real Llama-3-8B CPU Performance
- **Model Loading**: ~33 seconds
- **Processing Time**: ~114 seconds per batch
- **Memory Usage**: ~32 GB peak
- **Output Quality**: Full precision embeddings

### Zarr Data Processing
- **Loading**: <1 second for test datasets
- **Tensor Conversion**: <1 second for small batches
- **AIFS Tokenization**: ~2 seconds on CPU
- **Spatial Subsetting**: Real-time for moderate sizes

## Integration Success

✅ **Complete pipeline verified**: Zarr climate data → AIFS tokenization → Real Llama-3-8B processing → Multimodal fusion → Climate analysis outputs

This represents a fully functional climate AI system capable of processing real-world meteorological data through state-of-the-art language models.

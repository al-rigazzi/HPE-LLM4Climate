# Multimodal Test Suite

This directory contains the test suite for the multimodal climate analysis system components.

## Test Files

- **`test_encoder_extractor.py`** - Tests for PrithviWxC encoder extraction and validation
- **`test_fusion.py`** - Tests for multimodal climate-text fusion capabilities
- **`test_location_aware.py`** - Tests for location-aware processing and geographic functionality

## Running Tests

All tests can be run using pytest or directly with Python:

```bash
cd multimodal/tests/
python test_location_aware.py
python test_fusion.py
python test_encoder_extractor.py

# Or using pytest
python -m pytest . -v
```

## Test Coverage

The test suite validates:
- **Encoder Extraction**: PrithviWxC encoder extraction and weight validation
- **Multimodal Fusion**: Climate-text fusion across different strategies
- **Location-Aware Processing**: Geographic resolution and spatial attention
- **Cross-Platform Compatibility**: Apple Silicon MPS and CUDA device support

## System Requirements

- Python 3.8+
- PyTorch with MPS support (Apple Silicon) or CUDA (NVIDIA)
- All dependencies from `requirements.txt`
- Optional: HuggingFace authentication for gated models

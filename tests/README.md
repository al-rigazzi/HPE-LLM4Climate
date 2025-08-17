# Test Suite Organization

This directory contains the comprehensive test suite for the HPE-LLM4Climate project, organized by test type and purpose.

## 📁 Directory Structure

### `integration/`
Integration tests for complex system interactions:
- **`test_simple_encoder_extraction.py`** - Basic encoder extraction and validation
- **`test_encoder_loading_verification.py`** - Complete encoder loading verification
- **`test_encoder_only.py`** - Standalone encoder functionality tests
- **`test_full_encoder_pipeline.py`** - End-to-end encoder pipeline testing
- **`test_llama_comprehensive.py`** - Comprehensive Meta-Llama-3-8B integration testing
- **`test_llama_integration.py`** - Multiple Llama model configuration testing
- **`test_llama_with_prithvi.py`** - Prithvi encoder + Llama model fusion testing
- **`debug_weight_loading.py`** - Weight loading diagnostics and debugging
- **`debug_forward_pass.py`** - Forward pass analysis and debugging

### `system/`
System-level validation and setup tests:
- **`verify_setup.py`** - Complete system setup verification

### `demos/`
Demonstration scripts and examples:
- **`working_location_demo.py`** - Location-aware climate analysis demo

## 🧪 Core Unit Tests

The core unit tests are located in `multimodal/tests/`:
- **`test_location_aware.py`** - Geographic processing and location-aware functionality
- **`test_fusion.py`** - Multimodal climate-text fusion testing
- **`test_encoder_extractor.py`** - Encoder extraction and validation

## 🚀 Running Tests

### Unit Tests (Multimodal Components)
```bash
# Core functionality tests
cd multimodal
python -m pytest tests/ -v

# Or run individually
python tests/test_location_aware.py
python tests/test_fusion.py
python tests/test_encoder_extractor.py
```

### Integration Tests (Encoder Pipeline)
```bash
# Encoder validation and extraction
python tests/integration/test_simple_encoder_extraction.py
python tests/integration/test_encoder_loading_verification.py
python tests/integration/test_encoder_only.py
python tests/integration/test_full_encoder_pipeline.py

# LLM integration
python tests/integration/test_llama_comprehensive.py
python tests/integration/test_llama_integration.py
python tests/integration/test_llama_with_prithvi.py
```
python tests/integration/test_mps_fix.py
```

### System Verification
```bash
# Setup and environment validation
python tests/system/verify_setup.py
```

### Demos
```bash
# Working system demonstration
python tests/demos/working_location_demo.py
```

## 📊 Test Coverage

- **Unit Tests**: 21 tests covering core functionality
- **Integration Tests**: 4 comprehensive integration scenarios
- **System Tests**: 1 complete environment validation
- **Demos**: 1 working system demonstration

All tests are designed to work across:
- **Platforms**: Ubuntu 24.04, macOS-latest
- **Python Versions**: 3.12, 3.13
- **Devices**: CPU, CUDA, MPS (Apple Silicon)

# Test Suite Organization

This directory contains the comprehensive test suite for the HPE-LLM4Climate project, organized by test type and purpose.

## 📁 Directory Structure

### `integration/`
Integration tests for complex system interactions:
- **`test_llama_comprehensive.py`** - Comprehensive Meta-Llama-3-8B integration testing
- **`test_llama_integration.py`** - Multiple Llama model configuration testing
- **`test_llama_with_prithvi.py`** - Prithvi encoder + Llama model fusion testing
- **`test_mps_fix.py`** - Apple Silicon MPS compatibility testing

### `system/`
System-level validation and setup tests:
- **`verify_setup.py`** - Complete system setup verification

### `demos/`
Demonstration scripts and examples:
- **`working_location_demo.py`** - Location-aware climate analysis demo

## 🧪 Core Unit Tests

The core unit tests are located in `multimodal/tests/`:
- **`test_location_aware.py`** - Geographic processing and location-aware functionality (16 tests)
- **`test_fusion.py`** - Multimodal climate-text fusion (5 tests)
- **`test_encoder_extractor.py`** - Encoder extraction and validation

## 🚀 Running Tests

### Unit Tests (CI/CD Pipeline)
```bash
# Core functionality tests (run by CI/CD)
cd multimodal
python -m pytest tests/ -v

# Or run individually
python tests/test_location_aware.py
python tests/test_fusion.py
python tests/test_encoder_extractor.py
```

### Integration Tests
```bash
# Llama model integration
python tests/integration/test_llama_comprehensive.py
python tests/integration/test_llama_integration.py
python tests/integration/test_llama_with_prithvi.py

# Platform compatibility
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

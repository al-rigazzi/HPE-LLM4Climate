# Multimodal AIFS Tests

This directory contains comprehensive tests for the multimodal AIFS implementation, organized by test type and purpose.

## 📁 Test Structure

### `unit/`
Unit tests for individual components:
- **`test_aifs_encoder_utils.py`** - AIFS encoder wrapper and utilities testing
- **`test_aifs_time_series_tokenizer.py`** - Time series tokenizer functionality and validation
- **`test_climate_data_utils.py`** - Climate data processing and normalization testing
- **`test_location_utils.py`** - Geographic and spatial operations testing
- **`test_text_utils.py`** - Climate text processing and embedding testing

### `integration/`
Integration tests for complex interactions:
- **`test_aifs_climate_fusion.py`** - Climate-text fusion with AIFS encoder
- **`test_time_series_integration.py`** - Time series tokenizer multimodal integration
- **`test_aifs_location_aware.py`** - Location-aware components integration
- **`test_aifs_full_pipeline.py`** - End-to-end multimodal pipeline testing
- **`test_real_data_pipeline.py`** - Real ECMWF data processing pipeline
- **`test_aifs_llama_integration.py`** - AIFS + LLaMA fusion model testing
- **`test_real_llama_integration.py`** - Real LLaMA-3-8B integration tests
- **`test_aifs_llama3_real_fusion.py`** - ⭐ **AIFS + Real Llama-3-8B fusion (NO MOCKS)**
- **`test_aifs_llama3_pytest.py`** - Pytest-compatible real fusion tests

### `benchmarks/`
Performance benchmarking tests:
- **`test_time_series_performance.py`** - Comprehensive time series tokenizer benchmarks

### `system/`
System-level validation:
- **`test_aifs_system_setup.py`** - Complete AIFS system setup verification
- **`test_multimodal_aifs_integration.py`** - Full multimodal AIFS integration

## 🧪 Running Tests

### Individual Test Files
```bash
# Run specific test
python -m pytest multimodal_aifs/tests/unit/test_aifs_encoder_utils.py -v

# Run time series tests
python -m pytest multimodal_aifs/tests/unit/test_aifs_time_series_tokenizer.py -v

# Run integration tests
python -m pytest multimodal_aifs/tests/integration/ -v

# Run performance benchmarks
python -m pytest multimodal_aifs/tests/benchmarks/ -v

# Run all tests
python -m pytest multimodal_aifs/tests/ -v
```

### Real Model Integration Tests (⭐ NEW)
```bash
# Run AIFS + Real Llama-3-8B fusion test (standalone)
cd "/path/to/HPE-LLM4Climate"
PYTHONPATH="$PWD:$PYTHONPATH" python multimodal_aifs/tests/integration/test_aifs_llama3_real_fusion.py

# Run with pytest
pytest -xvs multimodal_aifs/tests/integration/test_aifs_llama3_pytest.py

# Run all real LLaMA integration tests
python multimodal_aifs/tests/integration/test_real_llama_integration.py
```

**Note**: Real model tests require:
- ~8GB+ RAM for Llama-3-8B model
- HuggingFace transformers library
- Internet connection for model download
- Time: ~30s model loading + ~2min per test

### Time Series Test Runner
```bash
# Run comprehensive time series tests
python multimodal_aifs/tests/run_time_series_tests.py

# Quick time series tests (no benchmarks)
python multimodal_aifs/tests/run_time_series_tests.py --quick

# Only unit tests
python multimodal_aifs/tests/run_time_series_tests.py --unit-only

# Only benchmarks
python multimodal_aifs/tests/run_time_series_tests.py --benchmarks-only
```

### Direct Execution
```bash
# Run individual test modules
python multimodal_aifs/tests/unit/test_aifs_encoder_utils.py
python multimodal_aifs/tests/integration/test_aifs_climate_fusion.py
```

## ✅ Test Coverage

The test suite covers:
- **AIFS Encoder Integration**: Model loading, encoding, batch processing
- **Time Series Tokenization**: 5-D tensor processing, temporal modeling, compression
- **Climate Data Processing**: Normalization, feature adjustment, synthetic data
- **Location Operations**: Distance calculations, spatial cropping, coordinate encoding
- **Text Processing**: Climate keyword extraction, location parsing, embeddings
- **Multimodal Fusion**: Climate-text attention, location-aware fusion, time series integration
- **Real Data**: ECMWF data processing, model compatibility
- **Performance**: Benchmarking across scales, configurations, and temporal patterns

## 🎯 Test Categories

### Unit Tests
- Component isolation
- Input/output validation
- Error handling
- Performance benchmarks

### Integration Tests
- Multi-component workflows
- Real model integration
- Data pipeline validation
- End-to-end scenarios

### System Tests
- Environment setup
- Dependency verification
- Performance validation
- Resource management

## 🚀 Continuous Integration

Tests are designed to:
- Work with/without AIFS model files
- Gracefully handle missing dependencies
- Provide informative failure messages
- Support both CPU and GPU execution
- Include synthetic data fallbacks

## 📊 Performance Benchmarks

Key performance metrics tracked:
- AIFS encoder loading time
- Time series tokenization throughput
- Climate data encoding speed
- Text processing throughput
- Compression ratios and memory efficiency
- Memory usage patterns
- Batch processing efficiency
- Temporal modeling performance

## 🔧 Development Guidelines

When adding new tests:
1. Follow existing naming conventions
2. Include both positive and negative test cases
3. Add docstrings with clear descriptions
4. Use synthetic data when real data unavailable
5. Include performance assertions where relevant
6. Test error conditions and edge cases

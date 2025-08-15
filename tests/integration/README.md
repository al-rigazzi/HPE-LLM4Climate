# Integration Tests

This directory contains integration tests that verify the interaction between different system components, particularly focusing on language model integration and platform compatibility.

## 🧪 Test Files

### Language Model Integration
- **`test_llama_comprehensive.py`** - Comprehensive testing of Meta-Llama-3-8B with location-aware climate analysis
- **`test_llama_integration.py`** - Tests multiple Llama model variants and configurations
- **`test_llama_with_prithvi.py`** - Tests Meta-Llama-3-8B integration with real Prithvi encoder weights

### Platform Compatibility  
- **`test_mps_fix.py`** - Apple Silicon MPS device compatibility testing and fixes

## 🚀 Running Integration Tests

```bash
# Run all integration tests
cd tests/integration
python test_llama_comprehensive.py
python test_llama_integration.py  
python test_llama_with_prithvi.py
python test_mps_fix.py

# Or run specific tests
python test_llama_comprehensive.py  # Full Llama-3-8B system test
python test_mps_fix.py             # Apple Silicon compatibility
```

## 📋 Test Requirements

### Authentication
Some tests require HuggingFace authentication for gated models:
```bash
huggingface-cli login
```

### Models Tested
- Meta-Llama-3-8B (gated - requires authentication)
- Meta-Llama-2-7B variants (gated)
- DialoGPT-medium (public)
- DistilBERT, BERT, RoBERTa (public fallbacks)

### System Requirements
- **Memory**: Minimum 16GB RAM for Llama models
- **Storage**: ~15GB for model weights
- **Platform**: macOS (MPS), Linux (CUDA/CPU), Windows (CPU)

## ⚠️ Known Issues

### Apple Silicon (MPS)
- MPS has known compatibility issues with some text generation models
- Tests include CPU fallback strategies
- See `test_mps_fix.py` for detailed workarounds

### Memory Management
- Large language models require significant memory
- Tests include memory monitoring and cleanup
- Consider using smaller models for development testing

## 📊 Test Coverage

These integration tests cover:
- ✅ Multi-model language processing
- ✅ Climate-text fusion workflows  
- ✅ Cross-platform device management
- ✅ Memory optimization strategies
- ✅ Authentication and model loading
- ✅ Error handling and fallbacks

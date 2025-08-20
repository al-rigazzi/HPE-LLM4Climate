# Integration Tests

This directory contains integration tests that verify the interaction between different system components, focusing on encoder extraction, weight loading, and language model integration.

## 🧪 Test Files

### Encoder Pipeline Tests
- **`test_simple_encoder_extraction.py`** - Basic encoder extraction and forward pass validation
- **`test_encoder_loading_verification.py`** - Complete encoder loading verification with zero missing keys
- **`test_encoder_only.py`** - Standalone encoder functionality and weight validation
- **`test_full_encoder_pipeline.py`** - End-to-end encoder extraction, saving, loading, and inference

### Language Model Integration
- **`test_llama_comprehensive.py`** - Comprehensive testing of Meta-Llama-3-8B with location-aware climate analysis
- **`test_llama_integration.py`** - Tests multiple Llama model variants and configurations
- **`test_llama_with_prithvi.py`** - Tests Meta-Llama-3-8B integration with real Prithvi encoder weights

### Debug and Development Tools
- **`debug_weight_loading.py`** - Detailed diagnostics for weight loading and architecture analysis
- **`debug_forward_pass.py`** - Forward pass debugging and tensor analysis

## 🚀 Running Integration Tests

```bash
# Run encoder pipeline tests
cd tests/integration
python test_simple_encoder_extraction.py      # Basic validation
python test_encoder_loading_verification.py   # Complete loading test
python test_encoder_only.py                   # Standalone encoder test
python test_full_encoder_pipeline.py          # Full pipeline test

# Run language model integration tests
python test_llama_comprehensive.py            # Full Llama-3-8B system test
python test_llama_integration.py              # Multiple model variants
python test_llama_with_prithvi.py            # Llama + Prithvi integration

# Debug utilities (for development)
python debug_weight_loading.py                # Weight loading diagnostics
python debug_forward_pass.py                  # Forward pass analysis
```

## 📋 Test Requirements

### Authentication
Some tests require HuggingFace authentication for gated models:
```bash
huggingface-cli login
```

### Models Tested
- **PrithviWxC Encoder**: Extracted from full model checkpoints
- **Meta-Llama-3-8B**: Gated model requiring authentication
- **Meta-Llama-2-7B variants**: Gated models
- **DialoGPT-medium**: Public alternative model
- **DistilBERT, BERT, RoBERTa**: Public fallback models

### System Requirements
- **Memory**: Minimum 16GB RAM for full models
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

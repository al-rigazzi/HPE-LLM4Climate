# System Tests

This directory contains system-level tests that verify the overall environment setup, dependencies, and system configuration.

## 🧪 Test Files

### Setup Verification
- **`verify_setup.py`** - Comprehensive system setup verification script

## 🚀 Running System Tests

```bash
# Verify complete system setup
python tests/system/verify_setup.py
```

## 📋 What Gets Verified

### Python Environment
- ✅ Python version compatibility (3.12+)
- ✅ Virtual environment activation
- ✅ Package installation verification

### Core Dependencies
- ✅ PyTorch installation and device compatibility
- ✅ Transformers library availability
- ✅ GeoPy geographic processing
- ✅ Scientific computing libraries (NumPy, pandas)

### System Components
- ✅ Multimodal module imports
- ✅ Core functionality initialization
- ✅ Geographic resolution services
- ✅ Model loading capabilities

### Platform Specific
- ✅ **macOS**: MPS availability and compatibility
- ✅ **Linux**: CUDA detection (if available)
- ✅ **Windows**: CPU fallback verification

### Data Requirements
- ✅ Required directories exist (`data/weights`, `data/climatology`)
- ✅ Mock weight files for testing
- ✅ Configuration file validation

## 🔧 Troubleshooting

If system verification fails, check:

### Common Issues
1. **Missing Dependencies**: Run `pip install -r requirements.txt`
2. **Python Version**: Ensure Python 3.12+ is installed
3. **GeoPy Issues**: Install with `pip install geopy`
4. **MPS Problems**: See integration tests for workarounds

### Environment Setup
```bash
# Create virtual environment
python -m venv llm4climate
source llm4climate/bin/activate  # Linux/macOS
# llm4climate\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
pip install geopy nominatim

# Verify setup
python tests/system/verify_setup.py
```

## 📊 Success Criteria

System verification passes when:
- ✅ All core imports successful
- ✅ Geographic resolution working
- ✅ Model loading functional
- ✅ Platform compatibility confirmed
- ✅ No critical dependency issues

The verification script provides detailed output showing exactly what passed or failed, with specific guidance for resolving any issues.

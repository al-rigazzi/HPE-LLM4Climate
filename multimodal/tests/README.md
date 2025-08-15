# Test Suite

This directory contains the comprehensive test suite for the multimodal climate analysis system.

## Test Files

- `test_corrected_encoder*.py` - Tests for corrected encoder functionality
- `test_encoder_extractor*.py` - Tests for encoder extraction utilities
- `test_fusion.py` - Tests for multimodal fusion capabilities
- `test_location_aware.py` - Tests for location-aware processing

## Running Tests

All tests can be run using pytest or directly with Python:

```bash
cd tests/
python test_location_aware.py
python test_fusion.py
```

## Test Status

All tests are currently passing with 100% success rate, validating:
- Location-aware processing
- Multimodal fusion
- Encoder extraction
- Geographic resolution

The test suite ensures reliability across different Apple Silicon configurations and MPS device compatibility.

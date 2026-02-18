# Test Suite

Test code is organized into unit and integration coverage.

## Structure

- `unit/` - component-level tests
- `integration/` - multi-component and end-to-end style tests
- `integration/zarr/` - Zarr-specific integration tests
- `test_input_preparation.py` - AIFS input preparation checks
- `test_training_pipeline.py` - high-level training pipeline checks
- `run_time_series_tests.py` - helper runner for tokenizer/time-series coverage

## Run Commands

```bash
# Unit tests
pytest multimodal_aifs/tests/unit/ -v --maxfail=5

# Integration tests
pytest multimodal_aifs/tests/integration/ -v --maxfail=5 -m "not large_memory"

# Zarr integration subset
pytest multimodal_aifs/tests/integration/zarr/ -v --maxfail=5

# Package-level tests
pytest multimodal_aifs/tests/test_input_preparation.py -v
pytest multimodal_aifs/tests/test_training_pipeline.py -v
```

## Runtime Toggles

- `USE_MOCK_AIFS`
- `USE_MOCK_LLM`
- `USE_QUANTIZATION`
- `USE_REAL_ZARR`

These are read by fixtures in `multimodal_aifs/conftest.py`.

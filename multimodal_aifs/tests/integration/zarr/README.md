# Zarr Integration Tests

This directory validates the Zarr -> AIFS -> multimodal flow.

## Current Test Files

- `test_zarr_integration.py`
- `test_cpu_mistral_zarr.py`
- `test_real_mistral_zarr.py`
- `test_real_mistral_cpu_full.py`

## Run Commands

```bash
# All zarr integration tests
pytest multimodal_aifs/tests/integration/zarr/ -v --maxfail=5

# Core pipeline smoke test
pytest multimodal_aifs/tests/integration/zarr/test_zarr_integration.py::test_zarr_to_aifs_pipeline -v
```

## Execution Modes

- Mock-friendly mode:
  - `USE_MOCK_LLM=true USE_REAL_ZARR=false`
- Real-model mode:
  - `USE_MOCK_LLM=false USE_REAL_ZARR=true`

## Expected Coverage

- Zarr dataset loading
- Conversion to AIFS-compatible tensor layout
- Fusion path integration with text model components

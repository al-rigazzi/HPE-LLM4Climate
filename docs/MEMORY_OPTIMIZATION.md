# Memory Optimization Guide

This guide summarizes practical memory controls for the current codebase.

## Primary Controls

- Use mock models for fast development: `USE_MOCK_LLM=true`
- Disable mock models for real-model tests: `USE_MOCK_LLM=false`
- Use quantization where supported (primarily CUDA): `USE_QUANTIZATION=true`
- Use mock Zarr data in constrained CI/dev: `USE_REAL_ZARR=false`

## Platform Guidance

### CUDA

- Best option for real-model training/inference
- Recommended for quantized execution and large-memory workloads

### MPS (Apple Silicon)

- Supported for development and inference
- For heavy tests, prefer excluding large-memory markers when needed:
  - `pytest ... -m "not large_memory"`

### CPU

- Fully supported
- Best paired with mock-mode flows for faster iteration

## Practical Test Profiles

### Fast development profile

```bash
USE_MOCK_LLM=true USE_REAL_ZARR=false pytest multimodal_aifs/tests/unit/ -v --maxfail=5
```

### Fuller integration profile

```bash
USE_MOCK_LLM=false USE_REAL_ZARR=true pytest multimodal_aifs/tests/integration/ -v --maxfail=5
```

### Resource-constrained integration profile

```bash
USE_MOCK_LLM=true pytest multimodal_aifs/tests/integration/ -v -m "not large_memory" --maxfail=5
```

## Troubleshooting

- OOM during integration: switch to mock LLM and exclude `large_memory`
- Slow first run: expected when models are downloaded/cached
- Unexpected skips: check `USE_MOCK_LLM` and marker selection

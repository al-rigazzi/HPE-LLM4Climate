# Memory Optimization Guide for HPE-LLM4Climate

## Overview

Running the full AIFS + Mistral-7B multimodal fusion model requires significant memory. This guide explains the memory requirements, optimization strategies, and platform-specific considerations.

## Memory Requirements

### Model Sizes

- **AIFS Model**: ~250M parameters (~500MB in float16)
- **Mistral-7B-Instruct**: ~7B parameters
  - **Float32**: ~28GB
  - **Float16**: ~14GB
  - **8-bit quantization** (CUDA only): ~7GB
  - **4-bit quantization** (CUDA only): ~3.5GB

### Total System Requirements

| Configuration | AIFS | Mistral | Total | Peak Memory | Platforms |
|--------------|------|---------|-------|-------------|-----------|
| **Mock models** | 500MB | 50MB | ~1GB | ~1GB | All |
| **Full precision (float32)** | 500MB | 28GB | ~29GB | ~29GB | CPU |
| **Half precision (float16)** | 500MB | 14GB | ~15GB | ~15GB | CUDA, MPS |
| **8-bit quantization** | 500MB | 7GB | ~8GB | ~8GB | CUDA only |
| **4-bit quantization** | 500MB | 3.5GB | ~4GB | ~4GB | CUDA only |
| **MPS optimized (float16 + checkpointing)** | 500MB | 14GB | ~15GB | **~9.5GB** ✨ | MPS only |

## Platform-Specific Optimizations

### CUDA (NVIDIA GPUs) - Best Performance ✅

**Recommended**: Use 4-bit quantization for optimal memory/performance balance.

```bash
# Install bitsandbytes for quantization
pip install bitsandbytes>=0.44.0

# Run tests with real models (uses 4-bit quantization automatically)
USE_MOCK_LLM=false pytest multimodal_aifs/tests/integration/
```

**Automatic optimizations applied**:
- 4-bit NF4 quantization with double quantization
- Automatic device mapping across GPUs
- Float16 compute dtype for attention operations
- **Memory reduction: 14GB → 3.5GB (75% savings)**

### MPS (Apple Silicon) - Optimized with Gradient Checkpointing ✨

**NEW**: Gradient checkpointing reduces peak memory by ~35% without quantization.

**Automatic MPS optimizations**:
- **Gradient checkpointing**: Trades compute for memory (reduces activations storage)
- **Aggressive memory management**: `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`
- **Float16 precision**: 50% reduction vs float32
- **Eager attention**: Memory-efficient implementation
- **Peak memory: ~9.5GB** (vs ~15GB without optimizations)

**Usage**:
```bash
# Run fusion tests on MPS with optimizations (works on 16GB+ Mac)
USE_MOCK_LLM=false pytest multimodal_aifs/tests/integration/test_aifs_climate_fusion.py -v

# If still experiencing issues, use mock models
USE_MOCK_LLM=true pytest multimodal_aifs/tests/
```

**Hardware requirements**:
- **Minimum**: 16GB unified memory (M1/M2/M3 with 16GB)
- **Recommended**: 24GB+ unified memory for comfortable headroom
- **Ideal**: 32GB+ for running multiple tests or development

**Limitations**:
- No quantization support (bitsandbytes incompatible with MPS)
- Gradient checkpointing adds ~15-20% compute overhead
- Some operations still fall back to CPU

### CPU

**Not recommended** for real models due to slow performance (~5-10 minutes per inference).

Use mock models for development:
```bash
USE_MOCK_LLM=true pytest multimodal_aifs/tests/
```

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_MOCK_LLM` | `false` | Use lightweight mock models instead of real Mistral |
| `USE_QUANTIZATION` | `auto` | Enable quantization (CUDA only, automatic in fixtures) |
| `LLM_MODEL_NAME` | `mistralai/Mistral-7B-Instruct-v0.3` | Mistral model variant |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | `0.7` | MPS memory allocation limit (0.0-1.0) |

### Test Execution Examples

```bash
# Development: Fast tests with mock models (recommended)
USE_MOCK_LLM=true pytest multimodal_aifs/tests/

# Integration: Real models on CUDA with 4-bit quantization
USE_MOCK_LLM=false pytest multimodal_aifs/tests/integration/

# Specific test with real models
USE_MOCK_LLM=false pytest multimodal_aifs/tests/integration/test_aifs_climate_fusion.py::test_multimodal_fusion -v

# Skip memory-intensive tests
pytest multimodal_aifs/tests/ -m "not large_memory"
```

## Troubleshooting

### Out of Memory (OOM) Errors

**Symptom**: `MPS backend out of memory` or `CUDA out of memory`

**Solutions**:
1. Use mock models: `USE_MOCK_LLM=true`
2. Close other applications to free memory
3. For MPS, increase watermark: `export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.9`
4. For CUDA, quantization should be automatic (verify bitsandbytes is installed)

### Slow Loading Times

**Symptom**: Model loading takes 2-5 minutes

**Explanation**: Normal for first run. Models are downloaded and cached by HuggingFace.

**Solutions**:
- Subsequent runs use cached models (much faster)
- Pre-download models: `huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3`

### Tests Skip Due to Real Models

**Symptom**: Tests marked as "skipped" with message about mock LLM

**Explanation**: Tests marked with `@pytest.mark.requires_mistral` skip when `USE_MOCK_LLM=true`

**Solution**: Run with real models: `USE_MOCK_LLM=false pytest ...`

## Best Practices

### For Development
- Use mock models (`USE_MOCK_LLM=true`) for fast iteration
- Mock models have realistic interfaces but use minimal memory
- Run full integration tests periodically on CUDA systems

### For CI/CD
```yaml
# Fast tests with mocks
test-mock:
  script: USE_MOCK_LLM=true pytest multimodal_aifs/tests/

# Integration tests on GPU runners
test-integration:
  script: USE_MOCK_LLM=false pytest multimodal_aifs/tests/integration/
  tags: [gpu]
```

### For Production
- Use CUDA systems with 4-bit quantization
- Monitor memory usage with `nvidia-smi` or `watch nvidia-smi`
- Consider using smaller models or model distillation for deployment

## Future Optimizations

Potential improvements under consideration:
- [ ] Gradient checkpointing for training
- [ ] Flash attention 2 support (when MPS compatible)
- [ ] Model pruning and distillation
- [ ] Quantization-aware training
- [ ] Mixed precision training (amp)
- [ ] Smaller Mistral variants (e.g., Mistral-3B)

## Hardware Recommendations

| Use Case | Minimum | Recommended |
|----------|---------|-------------|
| **Development (mock)** | 8GB RAM | 16GB RAM |
| **Testing (MPS)** | 32GB RAM | 64GB RAM |
| **Testing (CUDA)** | 16GB VRAM | 24GB VRAM |
| **Production** | 24GB VRAM | 40GB VRAM |

## References

- [bitsandbytes Documentation](https://github.com/TimDettmers/bitsandbytes)
- [Hugging Face Quantization Guide](https://huggingface.co/docs/transformers/main/en/quantization)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
- [AIFS Model Card](https://huggingface.co/ecmwf/aifs-single-1.1)

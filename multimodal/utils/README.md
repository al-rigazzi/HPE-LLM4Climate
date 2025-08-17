# Multimodal Utilities

This directory contains utility modules for encoder extraction, model processing, and geographic functionality.

## Files

### `encoder_extractor.py`
Core utility for extracting encoder components from PrithviWxC models.

**Key Classes:**
- `PrithviWxC_Encoder`: Standalone encoder class with full feature parity
- Functions for weight extraction, validation, and loading

**Features:**
- Climate residual mode support (160 input + 8 static channels)
- Automatic architecture detection from checkpoints
- Zero missing keys validation
- Support for various residual modes: climate, temporal, none

### `requirements-geo.txt`
Additional geographic processing dependencies for enhanced location-aware functionality.

## Usage Examples

```python
from multimodal.utils.encoder_extractor import PrithviWxC_Encoder, extract_encoder_weights

# Extract encoder from full model checkpoint
encoder = PrithviWxC_Encoder.from_checkpoint("path/to/prithvi.pt")

# Save extracted encoder
encoder.save_pretrained("extracted_encoder.pt")

# Load pre-extracted encoder
encoder = PrithviWxC_Encoder.from_pretrained("extracted_encoder.pt")

# Generate climate embeddings
climate_embeddings = encoder(climate_data)
```

## Dependencies

Core utilities require:
- PyTorch 2.0+
- NumPy, pandas
- Standard multimodal dependencies

Geographic utilities additionally require:
- GeoPy for coordinate resolution
- Additional packages listed in `requirements-geo.txt`

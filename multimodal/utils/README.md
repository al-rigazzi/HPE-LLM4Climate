# Utilities

This directory contains utility modules for encoder extraction and processing.

## Modules

### `encoder_extractor.py`
Core utility for extracting encoder components from PrithviWxC models.

### `corrected_encoder.py`
Corrected encoder implementations with fixes and improvements.

## Usage

```python
from multimodal.utils import extract_encoder, PrithviWxC_Encoder

# Extract encoder from model
encoder = extract_encoder(model_path)
```

These utilities support the core multimodal functionality by providing clean interfaces for model component extraction and processing.

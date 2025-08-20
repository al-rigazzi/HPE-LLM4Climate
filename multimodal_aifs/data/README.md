# ECMWF Data Processing

This directory contains all ECMWF data processing scripts, GRIB files, and results for the multimodal AIFS climate analysis system.

## Directory Structure

```
multimodal_aifs/data/
├── README.md                          # This file
├── scripts/                           # Data processing scripts
│   ├── test_ecmwf_download.py        # ECMWF data download test
│   ├── real_data_pipeline.py         # Real data processing pipeline
│   └── apply_encoder_to_data.py      # Apply AIFS encoder to ECMWF data
├── grib/                             # GRIB files and indices
│   ├── ecmwf_real_data_2025-08-18.grib
│   ├── ecmwf_real_data_2025-08-18.grib.5b7b6.idx
│   ├── test_2025-08-18.grib
│   └── test_download_2025-08-18.grib
└── results/                          # Processing results
    ├── real_data_encoding_20250819_171339.json
    └── real_encoded_data_20250819_171339.pt
```

## Scripts Overview

### `test_ecmwf_download.py`
- **Purpose**: Test script for downloading ECMWF data
- **Features**: Downloads sample weather data from ECMWF Open Data
- **Usage**: `python multimodal_aifs/data/scripts/test_ecmwf_download.py`

### `real_data_pipeline.py`
- **Purpose**: Complete pipeline for processing real ECMWF data
- **Features**: Downloads, processes, and analyzes real weather data
- **Usage**: `python multimodal_aifs/data/scripts/real_data_pipeline.py`

### `apply_encoder_to_data.py`
- **Purpose**: Apply AIFS encoder to ECMWF climate data
- **Features**:
  - Downloads real weather data from ECMWF Open Data
  - Applies extracted AIFS encoder to the data
  - Saves encoded results for analysis
- **Usage**: `python multimodal_aifs/data/scripts/apply_encoder_to_data.py`

## Data Files

### GRIB Files (`grib/`)
- **Format**: GRIB (GRIdded Binary) meteorological data format
- **Source**: ECMWF Open Data service
- **Content**: Real weather data including temperature, pressure, wind, etc.
- **Index Files**: `.idx` files provide metadata and indexing for GRIB files

### Results (`results/`)
- **Encoded Data**: `.pt` files containing PyTorch tensors with encoded climate data
- **Metadata**: `.json` files with processing metadata, timestamps, and statistics

## Usage Examples

### Download and Process New Data
```bash
# Download new ECMWF data
cd /path/to/project
python multimodal_aifs/data/scripts/test_ecmwf_download.py

# Process with real data pipeline
python multimodal_aifs/data/scripts/real_data_pipeline.py

# Apply AIFS encoder
python multimodal_aifs/data/scripts/apply_encoder_to_data.py
```

### Working with GRIB Files
```python
import xarray as xr

# Load GRIB data
data = xr.open_dataset('multimodal_aifs/data/grib/ecmwf_real_data_2025-08-18.grib',
                      engine='cfgrib')
print(data)
```

### Loading Encoded Results
```python
import torch
import json

# Load encoded data
encoded_data = torch.load('multimodal_aifs/data/results/real_encoded_data_20250819_171339.pt')

# Load metadata
with open('multimodal_aifs/data/results/real_data_encoding_20250819_171339.json', 'r') as f:
    metadata = json.load(f)

print(f"Encoded shape: {encoded_data.shape}")
print(f"Processing time: {metadata['processing_time']}")
```

## Dependencies

For running the data processing scripts, you need:

```bash
# ECMWF data access
pip install ecmwf-opendata

# GRIB file processing
pip install cfgrib xarray

# Scientific computing
pip install numpy torch

# Optional: for advanced GRIB processing
pip install eccodes
```

## Data Sources

- **ECMWF Open Data**: Free access to ECMWF weather data
  - URL: https://www.ecmwf.int/en/forecasts/datasets/open-data
  - License: Creative Commons Attribution 4.0 International
  - Update frequency: 4 times daily (00, 06, 12, 18 UTC)

## File Naming Conventions

- **GRIB files**: `{source}_{type}_{date}.grib`
- **Encoded results**: `{type}_encoded_data_{timestamp}.pt`
- **Metadata**: `{type}_encoding_{timestamp}.json`

## Notes

- GRIB files are large (typically 10-100 MB each)
- Encoded results are compressed PyTorch tensors
- All timestamps are in UTC
- Results include processing metadata for reproducibility

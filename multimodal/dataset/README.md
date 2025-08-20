# MERRA-2 Dataset Processor for PrithviWxC_Encoder

This directory contains tools for downloading, processing, and loading MERRA-2 reanalysis data specifically formatted for use with the PrithviWxC_Encoder model.

## Overview

The PrithviWxC_Encoder model requires specific atmospheric variables from MERRA-2 reanalysis:
- **20 surface variables** (temperature, pressure, humidity, radiation, etc.)
- **4 static variables** (land/ocean/ice fractions, surface geopotential)
- **10 vertical variables** (on 14 pressure levels)

This processing system downloads data from 6 different MERRA-2 collections, extracts only the required variables, aligns them to a common spatiotemporal grid, and saves them in an efficient numpy format.

## Features

- ✅ **Complete Variable Coverage**: All 34 variables required by PrithviWxC_Encoder
- ✅ **Multiple Temporal Resolutions**: 1-hourly, 3-hourly, or monthly averages
- ✅ **Flexible Date Ranges**: Process any time period from 1980-present
- ✅ **Efficient Storage**: Compressed numpy format for fast loading
- ✅ **Quality Control**: Built-in validation and outlier detection
- ✅ **PyTorch Integration**: Ready-to-use dataset and data loader classes
- ✅ **Automatic Alignment**: Handles different collection grids and temporal resolutions

## Quick Start

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set up NASA Earthdata credentials (required for downloads)
export EARTHDATA_USERNAME="your_username"
export EARTHDATA_PASSWORD="your_password"
```

### 2. Process Data

```bash
# Process 3-hourly data for January 2020
python merra2_dataset_processor.py \
    --start_date 2020-01-01 \
    --end_date 2020-01-31 \
    --temporal_resolution 3H \
    --output_dir ./processed_data

# Process monthly data for 2020
python merra2_dataset_processor.py \
    --start_date 2020-01-01 \
    --end_date 2020-12-31 \
    --temporal_resolution Monthly \
    --output_dir ./processed_data
```

### 3. Load and Use Data

```python
from data_loader import PrithviMERRA2Dataset, MERRA2DataLoader

# Create dataset
dataset = PrithviMERRA2Dataset(
    dataset_path="processed_data/merra2_prithvi_2020-01-01_2020-01-31_3H.npz",
    input_time_steps=2,
    time_step_hours=6,
    lead_time_hours=6
)

# Create PyTorch data loader
dataloader = MERRA2DataLoader.create_dataloader(
    dataset_path="processed_data/merra2_prithvi_2020-01-01_2020-01-31_3H.npz",
    batch_size=4,
    shuffle=True
)

# Use in training loop
for batch in dataloader:
    # batch contains:
    # - 'x': Input data [batch, time, vars, lat, lon]
    # - 'static': Static data [batch, vars, lat, lon]
    # - 'target': Target data [batch, vars, lat, lon]
    # - 'input_time': Input time offset [batch]
    # - 'lead_time': Lead time [batch]
    pass
```

## File Structure

```
dataset/
├── merra2_dataset_processor.py  # Main processing script
├── data_loader.py              # PyTorch dataset and data loader
├── config.py                   # Configuration and metadata
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── examples/                   # Usage examples
    ├── basic_usage.py          # Basic processing example
    ├── batch_processing.py     # Process multiple time periods
    └── integration_test.py     # Test with PrithviWxC_Encoder
```

## MERRA-2 Collections Used

| Collection | Description | Variables | Resolution |
|------------|-------------|-----------|------------|
| M2T1NXLND | Land surface | EFLUX, GWETROOT, HFLUX, LAI, TS, Z0M | 1-hourly |
| M2I1NXASM | Atmospheric surface | PS, QV2M, SLP, T2M, TQI, TQL, TQV, U10M, V10M, FRLAND, FROCEAN, PHIS | 1-hourly |
| M2I3NPASM | Atmospheric profiles | H, OMEGA, PL, QI, QL, QV, T, U, V | 3-hourly |
| M2I1NXRAD | Radiation | LWGAB, LWGEM, LWTUP, SWGNT, SWTNT | 1-hourly |
| M2TMNXGLC | Land ice/glaciers | FRACI | Monthly |
| M2I3NPCLD | Cloud diagnostics | CLOUD | 3-hourly |

## Command Line Options

### merra2_dataset_processor.py

```
Required Arguments:
  --start_date DATE        Start date (YYYY-MM-DD)
  --end_date DATE         End date (YYYY-MM-DD)

Optional Arguments:
  --temporal_resolution {1H,3H,Monthly}  Target temporal resolution (default: 3H)
  --output_dir DIR        Output directory (default: ./processed_data)
  --cache_dir DIR         Cache directory for downloads (default: ./merra2_cache)
  --output_filename FILE  Custom output filename
  --earthdata_username    NASA Earthdata username
  --earthdata_password    NASA Earthdata password
  --clean_cache          Remove cache after processing
```

## Output Format

Processed datasets are saved as compressed numpy archives (.npz) containing:

```python
{
    'surface': np.ndarray,      # Shape: (time, 20_vars, lat, lon)
    'static': np.ndarray,       # Shape: (4_vars, lat, lon)
    'vertical': np.ndarray,     # Shape: (time, 10_vars, 14_levels, lat, lon)
    'coordinates': dict,        # Time, lat, lon, level coordinates
    'surface_vars': list,       # Surface variable names
    'static_vars': list,        # Static variable names
    'vertical_vars': list,      # Vertical variable names
}
```

Additional metadata is saved in `.metadata.json` files.

## NASA Earthdata Setup

To download MERRA-2 data, you need a free NASA Earthdata account:

1. Register at: https://urs.earthdata.nasa.gov/
2. Set environment variables:
   ```bash
   export EARTHDATA_USERNAME="your_username"
   export EARTHDATA_PASSWORD="your_password"
   ```
3. Or pass credentials as command line arguments

## Performance and Storage

### Processing Performance
- **3-hourly data**: ~15 minutes per month (6 collections)
- **1-hourly data**: ~45 minutes per month (higher resolution)
- **Monthly data**: ~2 minutes per month (climatological)

### Storage Requirements
- **3-hourly**: ~2 GB per month (compressed)
- **1-hourly**: ~6 GB per month (compressed)
- **Monthly**: ~50 MB per month (compressed)

### Memory Requirements
- Minimum 8 GB RAM recommended
- 16+ GB RAM for processing multiple months simultaneously

## Quality Control

The processor includes automatic quality control:

- **Valid range checking**: Ensures variables are within physically reasonable bounds
- **Missing value detection**: Identifies and handles missing/invalid data
- **Outlier detection**: Uses interquartile range method to flag anomalous values
- **Spatial consistency**: Validates that land/ocean/ice fractions sum to 1

## Integration with PrithviWxC_Encoder

The processed datasets are directly compatible with the PrithviWxC_Encoder model:

```python
from multimodal.utils.encoder_extractor import PrithviWxC_Encoder
from multimodal.dataset.data_loader import PrithviMERRA2Dataset

# Load processed dataset
dataset = PrithviMERRA2Dataset("processed_data/merra2_prithvi_2020-01-01_2020-01-31_3H.npz")

# Create encoder model
encoder = PrithviWxC_Encoder(
    in_channels=160,  # 20 surface + 10*14 vertical vars
    in_channels_static=4,
    # ... other config parameters
)

# Process data through encoder
for batch in DataLoader(dataset):
    encoded_features = encoder(batch)
```

## Troubleshooting

### Common Issues

1. **Download failures**: Check NASA Earthdata credentials and network connection
2. **Memory errors**: Reduce date range or increase system RAM
3. **Missing variables**: Some variables may not be available for all time periods
4. **Disk space**: Ensure sufficient space for cache and output directories

### Error Messages

- `"NASA Earthdata credentials not provided"`: Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD
- `"No valid datasets were processed"`: Check date range and data availability
- `"Unexpected dimensions for variable X"`: May indicate corrupted or non-standard MERRA-2 files

## Examples

See the `examples/` directory for detailed usage examples:

- `basic_usage.py`: Simple processing of a single time period
- `batch_processing.py`: Processing multiple years efficiently
- `integration_test.py`: Full integration with PrithviWxC_Encoder

## Contributing

To contribute to this dataset processing system:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code passes linting (black, flake8)
5. Submit a pull request

## References

- Gelaro, R., et al. (2017). The modern-era retrospective analysis for research and applications, version 2 (MERRA-2). Journal of climate, 30(14), 5419-5454.
- MERRA-2 Documentation: https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/
- NASA Earthdata: https://earthdata.nasa.gov/

## License

This dataset processing system is released under the same license as the parent HPE-LLM4Climate project.

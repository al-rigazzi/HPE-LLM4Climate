# ECMWF Data Cache

This directory contains cached ECMWF climate data for the multimodal AIFS climate analysis system.

## Directory Structure

```
multimodal_aifs/data/
├── README.md                          # This file
└── grib/                             # ECMWF cached data files
    ├── ecmwf_20250910_12_10u_10v_2d_2t_msl_skt_sp_tcw_lsm_z_slor_sdor_sfc.cache.npy
    ├── ecmwf_20250910_12_gh_t_u_v_w_q_1000_925_850_700_600_500_400_300_250_200_150_100_50.cache.npy
    ├── ecmwf_20250910_12_vsw_sot_1_2.cache.npy
    ├── ecmwf_20250910_18_10u_10v_2d_2t_msl_skt_sp_tcw_lsm_z_slor_sdor_sfc.cache.npy
    ├── ecmwf_20250910_18_gh_t_u_v_w_q_1000_925_850_700_600_500_400_300_250_200_150_100_50.cache.npy
    └── ecmwf_20250910_18_vsw_sot_1_2.cache.npy
```

## Overview

The cached ECMWF data files contain weather forecast and atmospheric parameters processed for climate analysis. These are binary numpy arrays that have been extracted and cached from GRIB format for faster access during development and testing.

## Data Files

The GRIB cache contains weather data from ECMWF forecasts with the following variables:

**Surface-level variables (sfc):**
- 10u, 10v: 10m u/v wind components
- 2t: 2m temperature
- msl: Mean sea level pressure
- skt: Skin temperature
- sp: Surface pressure
- tcw: Total column water
- lsm: Land sea mask
- z: Geopotential
- slor, sdor: Slope parameters

**Pressure-level variables:**
- gh: Geopotential height
- t: Temperature
- u, v: Wind components
- w: Vertical velocity
- q: Specific humidity
- Levels: 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 hPa

**Additional variables:**
- vsw: Vertical integral of water vapour
- sot: Soil temperature

## Usage

### Working with Cache Files
```python
import numpy as np

# Load cached ECMWF data
data = np.load('multimodal_aifs/data/grib/ecmwf_20250910_12_10u_10v_2d_2t_msl_skt_sp_tcw_lsm_z_slor_sdor_sfc.cache.npy')
print(f"Cached data shape: {data.shape}")

# Load pressure-level data
pressure_data = np.load('multimodal_aifs/data/grib/ecmwf_20250910_12_gh_t_u_v_w_q_1000_925_850_700_600_500_400_300_250_200_150_100_50.cache.npy')
print(f"Pressure data shape: {pressure_data.shape}")
```

These cached files are automatically used by the AIFS processor for climate data analysis and can be used directly with numpy for debugging or analysis purposes.

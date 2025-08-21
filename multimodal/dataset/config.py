"""
Configuration file for MERRA-2 Dataset Processor

This file contains configuration settings for downloading and processing
MERRA-2 data for use with PrithviWxC_Encoder.
"""

import numpy as np

# NASA Earthdata configuration
EARTHDATA_CONFIG = {
    # You can set these environment variables or modify here
    "username": None,  # Set to your NASA Earthdata username or use EARTHDATA_USERNAME env var
    "password": None,  # Set to your NASA Earthdata password or use EARTHDATA_PASSWORD env var
    "base_url": "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2",
}

# Processing configuration
PROCESSING_CONFIG = {
    # Default temporal resolution
    "default_temporal_resolution": "3H",
    # Spatial configuration
    "target_spatial_resolution": {"lat": 0.5, "lon": 0.625},  # degrees  # degrees
    # Whether to apply quality control
    "apply_quality_control": True,
    # Whether to fill missing values
    "fill_missing_values": True,
    "fill_method": "linear",  # linear, nearest, or cubic
    # Compression settings
    "compression": {"enabled": True, "level": 6},  # 0-9, higher = more compression
}

# Variable mappings and units
VARIABLE_INFO = {
    "surface_vars": {
        "EFLUX": {
            "long_name": "Latent heat flux",
            "units": "W m-2",
            "standard_name": "surface_upward_latent_heat_flux",
        },
        "GWETROOT": {
            "long_name": "Root zone soil wetness",
            "units": "1",
            "standard_name": "volume_fraction_of_condensed_water_in_soil_at_root_depth",
        },
        "HFLUX": {
            "long_name": "Sensible heat flux",
            "units": "W m-2",
            "standard_name": "surface_upward_sensible_heat_flux",
        },
        "LAI": {"long_name": "Leaf area index", "units": "1", "standard_name": "leaf_area_index"},
        "LWGAB": {
            "long_name": "Surface absorbed longwave radiation",
            "units": "W m-2",
            "standard_name": "surface_net_downward_longwave_flux",
        },
        "LWGEM": {
            "long_name": "Surface emitted longwave radiation",
            "units": "W m-2",
            "standard_name": "surface_upward_longwave_flux",
        },
        "LWTUP": {
            "long_name": "Upwelling longwave radiation at TOA",
            "units": "W m-2",
            "standard_name": "toa_outgoing_longwave_flux",
        },
        "PS": {
            "long_name": "Surface pressure",
            "units": "Pa",
            "standard_name": "surface_air_pressure",
        },
        "QV2M": {
            "long_name": "2-meter specific humidity",
            "units": "kg kg-1",
            "standard_name": "specific_humidity",
        },
        "SLP": {
            "long_name": "Sea level pressure",
            "units": "Pa",
            "standard_name": "air_pressure_at_mean_sea_level",
        },
        "SWGNT": {
            "long_name": "Surface net downward shortwave radiation",
            "units": "W m-2",
            "standard_name": "surface_net_downward_shortwave_flux",
        },
        "SWTNT": {
            "long_name": "TOA net downward shortwave radiation",
            "units": "W m-2",
            "standard_name": "toa_net_downward_shortwave_flux",
        },
        "T2M": {
            "long_name": "2-meter temperature",
            "units": "K",
            "standard_name": "air_temperature",
        },
        "TQI": {
            "long_name": "Total precipitable ice water",
            "units": "kg m-2",
            "standard_name": "atmosphere_mass_content_of_cloud_ice",
        },
        "TQL": {
            "long_name": "Total precipitable liquid water",
            "units": "kg m-2",
            "standard_name": "atmosphere_mass_content_of_cloud_liquid_water",
        },
        "TQV": {
            "long_name": "Total precipitable water vapor",
            "units": "kg m-2",
            "standard_name": "atmosphere_mass_content_of_water_vapor",
        },
        "TS": {
            "long_name": "Surface skin temperature",
            "units": "K",
            "standard_name": "surface_temperature",
        },
        "U10M": {
            "long_name": "10-meter eastward wind",
            "units": "m s-1",
            "standard_name": "eastward_wind",
        },
        "V10M": {
            "long_name": "10-meter northward wind",
            "units": "m s-1",
            "standard_name": "northward_wind",
        },
        "Z0M": {
            "long_name": "Surface roughness",
            "units": "m",
            "standard_name": "surface_roughness_length_for_momentum_in_air",
        },
    },
    "static_vars": {
        "FRACI": {
            "long_name": "Ice fraction",
            "units": "1",
            "standard_name": "sea_ice_area_fraction",
        },
        "FRLAND": {
            "long_name": "Land fraction",
            "units": "1",
            "standard_name": "land_area_fraction",
        },
        "FROCEAN": {
            "long_name": "Ocean fraction",
            "units": "1",
            "standard_name": "sea_area_fraction",
        },
        "PHIS": {
            "long_name": "Surface geopotential height",
            "units": "m2 s-2",
            "standard_name": "surface_geopotential",
        },
    },
    "vertical_vars": {
        "CLOUD": {
            "long_name": "Cloud fraction",
            "units": "1",
            "standard_name": "cloud_area_fraction",
        },
        "H": {
            "long_name": "Geopotential height",
            "units": "m",
            "standard_name": "geopotential_height",
        },
        "OMEGA": {
            "long_name": "Vertical velocity (omega)",
            "units": "Pa s-1",
            "standard_name": "lagrangian_tendency_of_air_pressure",
        },
        "PL": {"long_name": "Pressure levels", "units": "Pa", "standard_name": "air_pressure"},
        "QI": {
            "long_name": "Cloud ice mixing ratio",
            "units": "kg kg-1",
            "standard_name": "mass_fraction_of_cloud_ice_in_air",
        },
        "QL": {
            "long_name": "Cloud liquid water mixing ratio",
            "units": "kg kg-1",
            "standard_name": "mass_fraction_of_cloud_liquid_water_in_air",
        },
        "QV": {
            "long_name": "Specific humidity",
            "units": "kg kg-1",
            "standard_name": "specific_humidity",
        },
        "T": {"long_name": "Temperature", "units": "K", "standard_name": "air_temperature"},
        "U": {"long_name": "Eastward wind", "units": "m s-1", "standard_name": "eastward_wind"},
        "V": {"long_name": "Northward wind", "units": "m s-1", "standard_name": "northward_wind"},
    },
}

# Quality control configuration
QUALITY_CONTROL = {
    "valid_ranges": {
        # Surface variables
        "T2M": [200, 350],  # 2m temperature (K)
        "TS": [200, 350],  # Surface temperature (K)
        "PS": [40000, 110000],  # Surface pressure (Pa)
        "SLP": [90000, 110000],  # Sea level pressure (Pa)
        "QV2M": [0, 0.05],  # 2m specific humidity (kg/kg)
        "U10M": [-50, 50],  # 10m wind components (m/s)
        "V10M": [-50, 50],
        "LAI": [0, 10],  # Leaf area index
        "GWETROOT": [0, 1],  # Soil wetness fraction
        "Z0M": [0, 10],  # Surface roughness (m)
        # Radiation variables
        "LWGAB": [-500, 500],  # Longwave radiation (W/m2)
        "LWGEM": [0, 800],
        "LWTUP": [0, 500],
        "SWGNT": [-100, 1400],  # Shortwave radiation (W/m2)
        "SWTNT": [-100, 1400],
        "EFLUX": [-200, 800],  # Heat fluxes (W/m2)
        "HFLUX": [-200, 800],
        # Precipitable water
        "TQV": [0, 100],  # Total precipitable water (kg/m2)
        "TQL": [0, 50],  # Precipitable liquid water (kg/m2)
        "TQI": [0, 50],  # Precipitable ice water (kg/m2)
        # Static variables
        "FRLAND": [0, 1],  # Fractions
        "FROCEAN": [0, 1],
        "FRACI": [0, 1],
        "PHIS": [-500, 100000],  # Surface geopotential (m2/s2)
        # Vertical variables
        "T": [150, 350],  # Temperature (K)
        "QV": [0, 0.05],  # Specific humidity (kg/kg)
        "QI": [0, 0.01],  # Cloud ice (kg/kg)
        "QL": [0, 0.01],  # Cloud liquid water (kg/kg)
        "U": [-200, 200],  # Wind components (m/s)
        "V": [-200, 200],
        "OMEGA": [-10, 10],  # Vertical velocity (Pa/s)
        "H": [-1000, 35000],  # Geopotential height (m)
        "PL": [1, 110000],  # Pressure (Pa)
        "CLOUD": [0, 1],  # Cloud fraction
    },
    "missing_value_flags": [-999, -9999, 1e20, np.nan],
    "outlier_detection": {
        "method": "iqr",  # interquartile range
        "threshold": 3.0,  # number of IQRs beyond Q1/Q3
    },
}

# Dataset metadata
DATASET_METADATA = {
    "title": "MERRA-2 Dataset for PrithviWxC_Encoder",
    "description": "Processed MERRA-2 reanalysis data containing variables required by PrithviWxC_Encoder model",
    "source": "NASA Global Modeling and Assimilation Office (GMAO) MERRA-2",
    "references": [
        "Gelaro, R., et al. (2017). The modern-era retrospective analysis for research and applications, version 2 (MERRA-2). Journal of climate, 30(14), 5419-5454."
    ],
    "contact": "Generated by merra2_dataset_processor.py",
    "conventions": "PrithviWxC_Encoder v1.0",
}

#!/usr/bin/env python3
"""
Create Test Zarr Dataset for Integration Tests

This script creates a small synthetic climate dataset in Zarr format
that can be used for testing without requiring large downloads.

Usage:
    python scripts/create_test_zarr.py
    python scripts/create_test_zarr.py --output test_climate_small.zarr --size small
    python scripts/create_test_zarr.py --output test_climate_large.zarr --size large --real-data
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import zarr

    print(f"✅ Using zarr version: {zarr.__version__}")
except ImportError:
    print("❌ zarr not available. Install with: pip install zarr")
    sys.exit(1)

try:
    from multimodal_aifs.utils.climate_data_utils import create_synthetic_climate_data

    print("✅ Climate data utilities available")
except ImportError:
    print("❌ Could not import climate data utilities")
    sys.exit(1)


def create_synthetic_zarr_dataset(
    output_path: str, size: str = "small", real_patterns: bool = False
) -> None:
    """
    Create a synthetic climate dataset in Zarr format.

    Args:
        output_path: Path where to save the Zarr dataset
        size: Dataset size ('tiny', 'small', 'medium', 'large')
        real_patterns: Whether to include realistic climate patterns
    """

    # Define size configurations
    size_configs = {
        "tiny": {
            "time_steps": 4,
            "lat_size": 8,
            "lon_size": 8,
            "n_variables": 3,
            "description": "Tiny dataset for quick tests",
        },
        "small": {
            "time_steps": 24,  # 1 day hourly
            "lat_size": 32,
            "lon_size": 32,
            "n_variables": 5,
            "description": "Small dataset for regular testing",
        },
        "medium": {
            "time_steps": 168,  # 1 week hourly
            "lat_size": 64,
            "lon_size": 64,
            "n_variables": 10,
            "description": "Medium dataset for comprehensive testing",
        },
        "large": {
            "time_steps": 720,  # 1 month hourly
            "lat_size": 128,
            "lon_size": 128,
            "n_variables": 15,
            "description": "Large dataset for performance testing",
        },
    }

    if size not in size_configs:
        raise ValueError(f"Size must be one of: {list(size_configs.keys())}")

    config = size_configs[size]
    print(f"\n🌍 Creating {size} climate dataset")
    print(f"   📊 Configuration: {config['description']}")
    print(f"   ⏰ Time steps: {config['time_steps']}")
    print(f"   🌐 Spatial: {config['lat_size']} x {config['lon_size']}")
    print(f"   📈 Variables: {config['n_variables']}")

    # Create coordinate arrays
    times = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(config["time_steps"])]

    # Create realistic coordinate grids
    lats = np.linspace(-90, 90, config["lat_size"])
    lons = np.linspace(-180, 180, config["lon_size"])

    # Variable names (standard climate variables)
    var_names = [
        "temperature_2m",  # 2m temperature (K)
        "relative_humidity",  # Relative humidity (%)
        "surface_pressure",  # Surface pressure (Pa)
        "wind_speed_10m",  # 10m wind speed (m/s)
        "total_precipitation",  # Total precipitation (mm)
        "cloud_cover",  # Cloud cover (%)
        "solar_radiation",  # Solar radiation (W/m²)
        "geopotential_500",  # 500 hPa geopotential (m²/s²)
        "specific_humidity",  # Specific humidity (kg/kg)
        "sea_level_pressure",  # Sea level pressure (Pa)
        "wind_u_10m",  # U-component of wind (m/s)
        "wind_v_10m",  # V-component of wind (m/s)
        "soil_temperature",  # Soil temperature (K)
        "snow_depth",  # Snow depth (m)
        "boundary_layer_height",  # Boundary layer height (m)
    ]

    # Use only the required number of variables
    selected_vars = var_names[: config["n_variables"]]

    print(f"   🔬 Variables: {', '.join(selected_vars)}")

    # Create data arrays for each variable
    data_vars = {}

    for i, var_name in enumerate(selected_vars):
        print(f"   📊 Generating {var_name}...")

        if real_patterns:
            # Create realistic patterns
            data = create_realistic_variable_data(var_name, times, lats, lons, config)
        else:
            # Use synthetic data utility
            synthetic_data = create_synthetic_climate_data(
                batch_size=1,
                time_steps=config["time_steps"],
                n_variables=1,
                spatial_shape=(config["lat_size"], config["lon_size"]),
            )
            # Extract single variable: [1, T, 1, H, W] -> [T, H, W]
            data = synthetic_data[0, :, 0, :, :].numpy()

        # Create DataArray
        data_vars[var_name] = xr.DataArray(
            data,
            dims=["time", "latitude", "longitude"],
            coords={"time": times, "latitude": lats, "longitude": lons},
            attrs={
                "long_name": f'{var_name.replace("_", " ").title()}',
                "units": get_variable_units(var_name),
                "description": f"Synthetic {var_name} data for testing",
            },
        )

    # Create the complete dataset
    ds = xr.Dataset(
        data_vars,
        coords={"time": times, "latitude": lats, "longitude": lons},
        attrs={
            "title": f"Synthetic Climate Test Dataset ({size})",
            "description": config["description"],
            "source": "Synthetic data generated for AIFS testing",
            "created": datetime.now().isoformat(),
            "conventions": "CF-1.8",
            "spatial_resolution": f'{180/config["lat_size"]:.2f} degrees',
            "temporal_resolution": "1 hour",
            "variables": len(selected_vars),
            "time_steps": config["time_steps"],
            "grid_size": f'{config["lat_size"]}x{config["lon_size"]}',
        },
    )

    print(f"\n💾 Saving to Zarr format: {output_path}")

    # Save to Zarr with simple chunking (no compression for compatibility)
    encoding = {}
    for var in selected_vars:
        encoding[var] = {"chunks": (min(24, config["time_steps"]), 16, 16)}

    print("   � Using simple chunking for compatibility")

    ds.to_zarr(output_path, mode="w", encoding=encoding)  # Verify the saved dataset
    print(f"✅ Dataset saved successfully!")

    # Load and verify
    ds_verify = xr.open_zarr(output_path)

    print(f"\n📊 Dataset Summary:")
    print(f"   📁 Path: {output_path}")
    print(f"   📏 Shape: {dict(ds_verify.dims)}")
    print(f"   🔢 Variables: {list(ds_verify.data_vars.keys())}")
    print(f"   📅 Time range: {ds_verify.time.values[0]} to {ds_verify.time.values[-1]}")
    print(
        f"   🌐 Spatial extent: {ds_verify.latitude.min().values:.1f}° to {ds_verify.latitude.max().values:.1f}° lat"
    )
    print(
        f"   🌐                   {ds_verify.longitude.min().values:.1f}° to {ds_verify.longitude.max().values:.1f}° lon"
    )

    # Calculate file size
    zarr_path = Path(output_path)
    if zarr_path.exists():
        total_size = sum(f.stat().st_size for f in zarr_path.rglob("*") if f.is_file())
        print(f"   💽 File size: {total_size / 1024**2:.2f} MB")

    print(f"\n🎉 Test dataset ready for use!")
    return output_path


def create_realistic_variable_data(var_name: str, times, lats, lons, config):
    """Create realistic patterns for specific climate variables."""

    time_steps = len(times)
    lat_size = len(lats)
    lon_size = len(lons)

    # Create base grids
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    time_grid = np.arange(time_steps)

    if var_name == "temperature_2m":
        # Temperature with latitude gradient and diurnal cycle
        base_temp = 288.15 - 0.5 * np.abs(lat_grid)  # Base temperature with lat gradient
        diurnal = 10 * np.sin(2 * np.pi * time_grid / 24)[:, None, None]  # Daily cycle
        seasonal = 5 * np.cos(2 * np.pi * time_grid / (24 * 365))[:, None, None]  # Seasonal
        noise = np.random.normal(0, 2, (time_steps, lat_size, lon_size))
        return base_temp[None, :, :] + diurnal + seasonal + noise

    elif var_name == "relative_humidity":
        # Humidity patterns (higher near equator and coasts)
        base_humidity = 50 + 30 * np.exp(-np.abs(lat_grid) / 30)
        daily_variation = 20 * np.sin(2 * np.pi * time_grid / 24 + np.pi)[:, None, None]
        noise = np.random.normal(0, 5, (time_steps, lat_size, lon_size))
        humidity = base_humidity[None, :, :] + daily_variation + noise
        return np.clip(humidity, 0, 100)

    elif var_name == "surface_pressure":
        # Pressure with realistic patterns
        base_pressure = 101325 - 12 * np.abs(lat_grid)  # Lower pressure at higher latitudes
        daily_cycle = 100 * np.sin(2 * np.pi * time_grid / 24)[:, None, None]
        noise = np.random.normal(0, 500, (time_steps, lat_size, lon_size))
        return base_pressure[None, :, :] + daily_cycle + noise

    elif var_name == "wind_speed_10m":
        # Wind patterns (stronger at higher latitudes)
        base_wind = 5 + 5 * np.abs(lat_grid) / 90
        daily_variation = 3 * np.sin(2 * np.pi * time_grid / 24)[:, None, None]
        noise = np.random.normal(0, 2, (time_steps, lat_size, lon_size))
        wind = base_wind[None, :, :] + daily_variation + noise
        return np.clip(wind, 0, None)

    elif var_name == "total_precipitation":
        # Precipitation (more near equator, sparse events)
        precipitation_prob = 0.1 + 0.2 * np.exp(-np.abs(lat_grid) / 20)
        events = np.random.random((time_steps, lat_size, lon_size)) < precipitation_prob[None, :, :]
        intensity = np.random.exponential(2, (time_steps, lat_size, lon_size))
        return events * intensity

    else:
        # Default: use synthetic data
        synthetic_data = create_synthetic_climate_data(
            batch_size=1, time_steps=time_steps, n_variables=1, spatial_shape=(lat_size, lon_size)
        )
        return synthetic_data[0, :, 0, :, :].numpy()


def get_variable_units(var_name: str) -> str:
    """Get appropriate units for climate variables."""
    units_map = {
        "temperature_2m": "K",
        "relative_humidity": "%",
        "surface_pressure": "Pa",
        "wind_speed_10m": "m/s",
        "total_precipitation": "mm",
        "cloud_cover": "%",
        "solar_radiation": "W/m²",
        "geopotential_500": "m²/s²",
        "specific_humidity": "kg/kg",
        "sea_level_pressure": "Pa",
        "wind_u_10m": "m/s",
        "wind_v_10m": "m/s",
        "soil_temperature": "K",
        "snow_depth": "m",
        "boundary_layer_height": "m",
    }
    return units_map.get(var_name, "unknown")


def download_real_climate_data(output_path: str):
    """
    Download a small real climate dataset for testing.
    Uses publicly available ERA5 sample data.
    """
    print("🌍 Downloading real climate data...")
    print("📡 This requires internet connection and may take a few minutes")

    try:
        from urllib.parse import urlparse

        import requests

        # Use a small ERA5 sample dataset (this is a placeholder URL)
        # In practice, you'd use actual data sources like:
        # - ECMWF ERA5 samples
        # - NOAA datasets
        # - Climate model outputs

        print("⚠️  Real data download not implemented yet.")
        print("💡 For now, creating realistic synthetic data...")

        return create_synthetic_zarr_dataset(output_path, size="small", real_patterns=True)

    except ImportError:
        print("❌ requests not available for downloading")
        print("💡 Creating synthetic data instead...")
        return create_synthetic_zarr_dataset(output_path, size="small", real_patterns=True)


def main():
    parser = argparse.ArgumentParser(
        description="Create test Zarr climate dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/create_test_zarr.py
  python scripts/create_test_zarr.py --size tiny --output test_tiny.zarr
  python scripts/create_test_zarr.py --size large --real-data
  python scripts/create_test_zarr.py --download-real --output era5_sample.zarr
        """,
    )

    parser.add_argument(
        "--output",
        "-o",
        default="test_climate.zarr",
        help="Output Zarr file path (default: test_climate.zarr)",
    )

    parser.add_argument(
        "--size",
        "-s",
        choices=["tiny", "small", "medium", "large"],
        default="small",
        help="Dataset size (default: small)",
    )

    parser.add_argument(
        "--real-data", "-r", action="store_true", help="Use realistic climate patterns"
    )

    parser.add_argument(
        "--download-real",
        "-d",
        action="store_true",
        help="Download real climate data instead of synthetic",
    )

    parser.add_argument("--force", "-f", action="store_true", help="Overwrite existing output file")

    args = parser.parse_args()

    # Check if output already exists
    if Path(args.output).exists() and not args.force:
        print(f"❌ Output file already exists: {args.output}")
        print("💡 Use --force to overwrite or choose a different output path")
        return 1

    try:
        if args.download_real:
            result_path = download_real_climate_data(args.output)
        else:
            result_path = create_synthetic_zarr_dataset(args.output, args.size, args.real_data)

        print(f"\n✅ Success! Test dataset created at: {result_path}")
        print(f"\n🧪 To use in tests:")
        print(
            f"   ZARR_PATH={result_path} python -m pytest multimodal_aifs/tests/integration/zarr/ -v"
        )

        return 0

    except Exception as e:
        print(f"❌ Error creating test dataset: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

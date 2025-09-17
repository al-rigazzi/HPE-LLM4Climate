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


def create_synthetic_zarr_dataset(
    output_path: str, size: str = "small", real_patterns: bool = False
) -> None:
    """
    Create a synthetic climate dataset in Zarr format compatible with AIFS dimensions.

    Args:
        output_path: Path where to save the Zarr dataset
        size: Dataset size ('tiny', 'small', 'medium', 'large')
        real_patterns: Whether to include realistic climate patterns
    """

    # AIFS-compatible dimensions
    aifs_grid_points = 542080  # Real AIFS grid size
    aifs_variables = 103  # Number of AIFS input variables (90 prognostic + 13 forcing)
    aifs_timesteps = 2  # AIFS expects exactly 2 timesteps (t-6h and t0)

    # Define size configurations (but override with AIFS dimensions)
    size_configs = {
        "tiny": {
            "time_steps": aifs_timesteps,
            "n_variables": min(10, aifs_variables),
            "grid_points": min(10000, aifs_grid_points),
            "description": "Tiny dataset with AIFS-compatible dimensions",
        },
        "small": {
            "time_steps": aifs_timesteps,
            "n_variables": min(25, aifs_variables),
            "grid_points": min(50000, aifs_grid_points),
            "description": "Small dataset with AIFS-compatible dimensions",
        },
        "medium": {
            "time_steps": aifs_timesteps,
            "n_variables": min(103, aifs_variables),  # Use all 103 variables for testing
            "grid_points": min(100000, aifs_grid_points),
            "description": "Medium dataset with AIFS-compatible dimensions",
        },
        "large": {
            "time_steps": aifs_timesteps,
            "n_variables": aifs_variables,
            "grid_points": aifs_grid_points,
            "description": "Full AIFS-compatible dataset dimensions",
        },
    }

    if size not in size_configs:
        raise ValueError(f"Size must be one of: {list(size_configs.keys())}")

    config = size_configs[size]
    print(f"\n🌍 Creating {size} AIFS-compatible climate dataset")
    print(f"   📊 Configuration: {config['description']}")
    print(f"   ⏰ Time steps: {config['time_steps']} (AIFS standard)")
    print(f"   🌐 Grid points: {config['grid_points']:,} (AIFS grid)")
    print(f"   📈 Variables: {config['n_variables']} (AIFS variables)")

    # Create time coordinates (AIFS expects exactly 2 timesteps)
    times = [
        datetime(2024, 1, 1, 0, 0, 0),  # t-6h (reference time)
        datetime(2024, 1, 1, 6, 0, 0),  # t0 (forecast time)
    ]

    # AIFS variable names - exact 103 input features (90 prognostic + 13 forcing)
    aifs_var_names = [
        # Prognostic variables (90 total)
        # Atmospheric variables at 13 pressure levels (6 vars × 13 levels = 78)
        "q_50",
        "q_100",
        "q_150",
        "q_200",
        "q_250",
        "q_300",
        "q_400",
        "q_500",
        "q_600",
        "q_700",
        "q_850",
        "q_925",
        "q_1000",
        "t_50",
        "t_100",
        "t_150",
        "t_200",
        "t_250",
        "t_300",
        "t_400",
        "t_500",
        "t_600",
        "t_700",
        "t_850",
        "t_925",
        "t_1000",
        "u_50",
        "u_100",
        "u_150",
        "u_200",
        "u_250",
        "u_300",
        "u_400",
        "u_500",
        "u_600",
        "u_700",
        "u_850",
        "u_925",
        "u_1000",
        "v_50",
        "v_100",
        "v_150",
        "v_200",
        "v_250",
        "v_300",
        "v_400",
        "v_500",
        "v_600",
        "v_700",
        "v_850",
        "v_925",
        "v_1000",
        "w_50",
        "w_100",
        "w_150",
        "w_200",
        "w_250",
        "w_300",
        "w_400",
        "w_500",
        "w_600",
        "w_700",
        "w_850",
        "w_925",
        "w_1000",
        "z_50",
        "z_100",
        "z_150",
        "z_200",
        "z_250",
        "z_300",
        "z_400",
        "z_500",
        "z_600",
        "z_700",
        "z_850",
        "z_925",
        "z_1000",
        # Surface prognostic variables (12 total)
        "10u",
        "10v",
        "2d",
        "2t",
        "msl",
        "skt",
        "sp",
        "tcw",
        "stl1",
        "stl2",
        "swvl1",
        "swvl2",
        # Forcing variables (13 total)
        "cos_latitude",
        "cos_longitude",
        "sin_latitude",
        "sin_longitude",
        "cos_julian_day",
        "cos_local_time",
        "sin_julian_day",
        "sin_local_time",
        "insolation",
        "lsm",
        "sdor",
        "slor",
        "z",
    ]

    assert config["n_variables"] is not None, "Must indicate number of variables in config"

    # Use only the required number of variables
    selected_vars = aifs_var_names[: config["n_variables"]]

    print(
        f"   🔬 Variables: {', '.join(selected_vars[:5])}{'...' if len(selected_vars) > 5 else ''}"
    )

    # Create data arrays for each variable
    data_vars = {}

    for var_name in selected_vars:
        print(f"   📊 Generating {var_name}...")

        if real_patterns:
            # Create realistic patterns for AIFS grid
            data = create_realistic_aifs_variable_data(
                var_name, config["time_steps"], config["grid_points"]
            )
        else:
            # Use synthetic data
            data = create_synthetic_aifs_data(config["time_steps"], config["grid_points"])

        # Create DataArray with AIFS-compatible dimensions
        # AIFS format: [time, variables, grid_points]
        data_vars[var_name] = xr.DataArray(
            data,
            dims=["time", "grid_point"],
            coords={"time": times, "grid_point": np.arange(config["grid_points"])},
            attrs={
                "long_name": f'{var_name.replace("_", " ").title()}',
                "units": get_variable_units(var_name),
                "description": f"Synthetic {var_name} data for AIFS testing",
                "aifs_standard": True,
                "grid_points": config["grid_points"],
            },
        )

    num_vars = len(selected_vars)
    # Create the complete dataset
    ds = xr.Dataset(
        data_vars,
        coords={"time": times, "grid_point": np.arange(config["grid_points"])},
        attrs={
            "title": f"Synthetic AIFS-Compatible Climate Test Dataset ({size})",
            "description": config["description"],
            "source": "Synthetic data generated for AIFS testing",
            "created": datetime.now().isoformat(),
            "conventions": "AIFS-1.0",
            "aifs_grid_points": config["grid_points"],
            "aifs_variables": num_vars,
            "aifs_timesteps": config["time_steps"],
            "standard_aifs_dims": f"{config['time_steps']}x{num_vars}x{config['grid_points']}",
            "note": "Data follows AIFS input format: [time, variables, grid_points]",
        },
    )

    print(f"\n💾 Saving to Zarr format: {output_path}")

    # Save to Zarr with simple chunking for AIFS compatibility
    encoding = {}
    for var in selected_vars:
        # Chunk by time (keep all grid points together for AIFS processing)
        encoding[var] = {"chunks": (config["time_steps"], config["grid_points"])}

    print("   � Using AIFS-compatible chunking")

    ds.to_zarr(output_path, mode="w", encoding=encoding)

    # Verify the saved dataset
    print("✅ Dataset saved successfully!")

    # Load and verify
    ds_verify = xr.open_zarr(output_path)

    print("\n📊 Dataset Summary:")
    print(f"   📁 Path: {output_path}")
    print(f"   📏 Shape: {dict(ds_verify.dims)}")
    print(f"   🔢 Variables: {list(ds_verify.data_vars.keys())}")
    print(f"   📅 Time range: {ds_verify.time.values[0]} to {ds_verify.time.values[-1]}")
    print(f"   🌐 Grid points: {ds_verify.grid_point.size:,}")
    print(f"   📊 AIFS dimensions: [{config['time_steps']}, {num_vars}, {config['grid_points']}]")

    # Calculate file size
    zarr_path = Path(output_path)
    if zarr_path.exists():
        total_size = sum(f.stat().st_size for f in zarr_path.rglob("*") if f.is_file())
        print(f"   💽 File size: {total_size / 1024**2:.2f} MB")

    print("\n🎉 AIFS-compatible test dataset ready for use!")
    return output_path


def create_realistic_aifs_variable_data(var_name: str, time_steps: int, grid_points: int):
    """Create realistic patterns for specific AIFS climate variables."""

    # Create base patterns for the flattened AIFS grid
    grid_indices = np.arange(grid_points)

    if var_name == "temperature_2m":
        # Temperature with latitude gradient (simulated via grid position)
        base_temp = (
            288.15 - 0.5 * np.sin(grid_indices * 2 * np.pi / grid_points) * 30
        )  # Lat gradient
        diurnal = 10 * np.sin(2 * np.pi * np.arange(time_steps) / 24)[:, None]  # Daily cycle
        seasonal = 5 * np.cos(2 * np.pi * np.arange(time_steps) / (24 * 365))[:, None]  # Seasonal
        noise = np.random.normal(0, 2, (time_steps, grid_points))
        return base_temp[None, :] + diurnal + seasonal + noise

    if var_name == "relative_humidity":
        # Humidity patterns
        base_humidity = 50 + 30 * np.sin(grid_indices * 4 * np.pi / grid_points)
        daily_variation = 20 * np.sin(2 * np.pi * np.arange(time_steps) / 24 + np.pi)[:, None]
        noise = np.random.normal(0, 5, (time_steps, grid_points))
        humidity = base_humidity[None, :] + daily_variation + noise
        return np.clip(humidity, 0, 100)

    if var_name == "surface_pressure":
        # Pressure with realistic patterns
        base_pressure = 101325 - 12 * np.abs(np.sin(grid_indices * 2 * np.pi / grid_points)) * 1000
        daily_cycle = 100 * np.sin(2 * np.pi * np.arange(time_steps) / 24)[:, None]
        noise = np.random.normal(0, 500, (time_steps, grid_points))
        return base_pressure[None, :] + daily_cycle + noise

    if var_name == "total_precipitation":
        # Precipitation (sparse events)
        precipitation_prob = 0.1 + 0.2 * np.sin(grid_indices * 6 * np.pi / grid_points)
        events = np.random.random((time_steps, grid_points)) < precipitation_prob[None, :]
        intensity = np.random.exponential(2, (time_steps, grid_points))
        return events * intensity

    return create_synthetic_aifs_data(time_steps, grid_points)


def create_synthetic_aifs_data(time_steps: int, grid_points: int):
    """Create synthetic AIFS-compatible data."""
    # Create data with realistic statistical properties
    data = np.random.normal(0, 1, (time_steps, grid_points))

    # Add some spatial correlation (simplified)
    for t in range(time_steps):
        # Add smooth spatial patterns
        spatial_pattern = np.sin(np.arange(grid_points) * 0.01) * 0.5
        data[t, :] += spatial_pattern

        # Add temporal correlation
        if t > 0:
            data[t, :] += 0.3 * data[t - 1, :]

    return data


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
    # print("🌍 Downloading real climate data...")
    # print("📡 This requires internet connection and may take a few minutes")

    try:
        print("⚠️  Real data download not implemented yet.")
        print("💡 For now, creating realistic synthetic data...")

        return create_synthetic_zarr_dataset(output_path, size="small", real_patterns=True)

    except ImportError:
        print("❌ requests not available for downloading")
        print("💡 Creating synthetic data instead...")
        return create_synthetic_zarr_dataset(output_path, size="small", real_patterns=True)


def main():
    """Create a zarr file complying to AIFS standards."""
    parser = argparse.ArgumentParser(
        description="Create AIFS-compatible test Zarr climate dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/create_test_zarr.py
  python scripts/create_test_zarr.py --size tiny --output test_tiny.zarr
  python scripts/create_test_zarr.py --size large --real-data
  python scripts/create_test_zarr.py --download-real --output era5_sample.zarr

AIFS Dimensions:
  - Time steps: Always 2 (t-6h and t0)
  - Variables: Up to 103 (AIFS standard)
  - Grid points: Up to 542,080 (AIFS grid)
  - Format: [time, variables, grid_points]
        """,
    )

    parser.add_argument(
        "--output",
        "-o",
        default="test_aifs_small.zarr",
        help="Output Zarr file path (default: test_aifs_small.zarr)",
    )

    parser.add_argument(
        "--size",
        "-s",
        choices=["tiny", "small", "medium", "large"],
        default="small",
        help="Dataset size with AIFS-compatible dimensions (default: small)",
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

        print(f"\n✅ Success! AIFS-compatible test dataset created at: {result_path}")
        print(f"\n🧪 To use in tests:")
        print(
            f"   ZARR_PATH={result_path} python -m pytest multimodal_aifs/tests/integration/zarr/"
        )
        print(f"\n📋 AIFS tensor format: [time, variables, grid_points]")
        print(f"   Example: python scripts/process_aifs_data.py {result_path}")

        return 0

    except Exception as e:
        print("❌ Error creating test dataset: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

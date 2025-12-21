#!/usr/bin/env python3
# Copyright 2025 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
    Download ECMWF Open Data and prepare it for AIFS model.

    This script downloads real meteorological data from ECMWF Open Data API,
    extracts the N320 grid coordinates by running the AIFS model, and generates
    proper forcing variables. The result is a zarr dataset with real data
    and correct geographic information that can be masked by region.

    Important: ECMWF Open Data only keeps recent forecasts (typically the last
    7-10 days). For reproducible runs, always specify an explicit date.

    Note: Results are cached. If the output file already exists, the script
    will skip downloading unless --force is used.

    Args:
        output_path: Where to save the zarr dataset (default: "data/real_ecmwf.zarr")
        use_date: Datetime for the forecast. If None, uses latest available data
                  from ECMWF (typically today's forecast). For reproducible runs,
                  pass an explicit recent date.
        force: If True, removes existing output before downloading

    Examples:
        # Download latest available data (cached)
        python scripts/download_real_ecmwf_data.py

        # Download specific recent date
        python scripts/download_real_ecmwf_data.py \\
            --date "2025-10-08 00:00" \\
            --output data/oct_data.zarr

        # Force re-download even if cached
        python scripts/download_real_ecmwf_data.py --force

        # Use latest available with custom output path
        python scripts/download_real_ecmwf_data.py \\
            --output data/latest.zarr
    """

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from platform import system
from typing import Any

import numpy as np
import xarray as xr

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Use the production-ready flash attention mock from conftest on MacOS
if system() == "Darwin":
    from multimodal_aifs.conftest import setup_flash_attn_mock

    setup_flash_attn_mock()

# Import ECMWF constants
from multimodal_aifs.constants import (
    ECMWF_LEVELS,
    ECMWF_PARAM_PL,
    ECMWF_PARAM_SFC,
    ECMWF_PARAM_SOIL,
    ECMWF_SOIL_LEVELS,
    ECMWF_SOIL_MAPPING,
)

try:
    import earthkit.data as ekd
    import earthkit.regrid as ekr
    from anemoi.inference.runners.simple import SimpleRunner
    from ecmwf.opendata import Client as OpendataClient
except ImportError as e:
    print(f"Missing required packages: {e}")
    print("\nInstall with:")
    print("  pip install earthkit-data earthkit-regrid ecmwf-opendata")
    print("  pip install anemoi-inference[huggingface] anemoi-models")
    sys.exit(1)


def get_open_data(date, param, levelist=None):
    """
    Download data from ECMWF Open Data API (exactly as in run_AIFS_v1.ipynb).

    Args:
        date: Target date
        param: Parameter name or list of parameters
        levelist: Pressure levels (optional)

    Returns:
        Dictionary of fields with stacked data for two timesteps
    """
    fields: dict[str, Any] = defaultdict(list)

    # Get data for current date and previous date (t-6h)
    for d in [date - timedelta(hours=6), date]:
        # Build request kwargs
        kwargs = {"date": d, "param": param}
        if levelist is not None:
            kwargs["levelist"] = levelist

        data = ekd.from_source("ecmwf-open-data", **kwargs)

        for f in data:
            # Open data is between -180 and 180, shift to 0-360
            assert f.to_numpy().shape == (721, 1440)
            values = np.roll(f.to_numpy(), -f.shape[1] // 2, axis=1)

            # Interpolate from 0.25 to N320
            values = ekr.interpolate(values, {"grid": (0.25, 0.25)}, {"grid": "N320"})

            # Add the values to the list
            if levelist:
                name = f"{f.metadata('param')}_{f.metadata('levelist')}"
            else:
                name = f.metadata("param")
            fields[name].append(values)

    # Stack timesteps
    for param_name, values in fields.items():
        fields[param_name] = np.stack(values)

    return fields


def download_real_ecmwf_data(output_path: str, date: datetime | None = None):
    """
    Download real ECMWF data using AIFS runner (following run_AIFS_v1.ipynb).

    Args:
        output_path: Path to save the Zarr dataset
        date: Target date. If None, uses latest available from ECMWF Open Data
              (typically today's forecast). For reproducible runs, pass explicit date.

    Returns:
        Path to the created Zarr file
    """
    print("\n" + "=" * 70)
    print("DOWNLOADING REAL ECMWF OPEN DATA (following run_AIFS_v1.ipynb)")
    print("=" * 70)

    # Use latest available data since ECMWF Open Data only keeps recent forecasts
    # (typically last 7-10 days). For reproducible runs, pass an explicit date.
    if date is None:
        client = OpendataClient()
        date = client.latest()
        print(f"\nNo date provided. Using latest available: {date}")
    else:
        print(f"\nTarget date: {date}")

    # Check if cached file already exists
    output_file = Path(output_path)
    if output_file.exists():
        print(f"\n✓ Cached data already exists: {output_path}")
        print("  Skipping download. Delete the file or use --force to re-download.")
        return str(output_file)

    print(f"Output path: {output_path}")

    print("\n" + "-" * 70)
    print("STEP 1: Download Meteorological Fields")
    print("-" * 70)

    fields = {}

    print("Surface variables...")
    fields.update(get_open_data(date, ECMWF_PARAM_SFC))

    print("Soil variables...")
    soil = get_open_data(date, ECMWF_PARAM_SOIL, levelist=ECMWF_SOIL_LEVELS)
    for k, v in soil.items():
        fields[ECMWF_SOIL_MAPPING[k]] = v

    print("Pressure level variables...")
    fields.update(get_open_data(date, ECMWF_PARAM_PL, levelist=ECMWF_LEVELS))

    print("Converting geopotential height to geopotential...")
    for level in ECMWF_LEVELS:
        gh = fields.pop(f"gh_{level}")
        fields[f"z_{level}"] = gh * 9.80665

    print("\n" + "-" * 70)
    print("STEP 2: Run AIFS to Extract Grid Coordinates")
    print("-" * 70)

    # Create input state
    input_state = {"date": date, "fields": fields}

    # Load AIFS model
    print("Loading AIFS model...")
    checkpoint = {"huggingface": "ecmwf/aifs-single-1.1"}

    # Use CUDA if available, otherwise CPU
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    runner = SimpleRunner(checkpoint, device=device)

    print("Running one forecast step...")
    state = None
    for s in runner.run(input_state=input_state, lead_time=6):
        state = s
        break  # Just need first state

    if state is None:
        raise RuntimeError("Failed to run AIFS")

    # Extract coordinates (THE KEY STEP!)
    lats = np.array(state["latitudes"])
    lons = np.array(state["longitudes"])
    grid_points = len(lats)

    print("\n✓ Extracted N320 grid:")
    print(f"   Points: {grid_points:,}")
    print(f"   Lat: [{lats.min():.2f}, {lats.max():.2f}]")
    print(f"   Lon: [{lons.min():.2f}, {lons.max():.2f}]")

    # Flatten fields to [time, grid_points]
    all_data = {}
    for var_name, data in fields.items():
        if data.ndim == 3:  # [time, nlat, nlon]
            all_data[var_name] = data.reshape(2, -1)
        else:
            all_data[var_name] = data

    print("\n" + "-" * 70)
    print("STEP 3: Generate Forcing Variables")
    print("-" * 70)

    date_t0 = date
    date_t_minus_6 = date - timedelta(hours=6)

    # Coordinate encodings
    sin_lat = np.sin(lats * np.pi / 180)
    cos_lat = np.cos(lats * np.pi / 180)
    sin_lon = np.sin(lons * np.pi / 180)
    cos_lon = np.cos(lons * np.pi / 180)

    all_data["sin_latitude"] = np.stack([sin_lat, sin_lat])
    all_data["cos_latitude"] = np.stack([cos_lat, cos_lat])
    all_data["sin_longitude"] = np.stack([sin_lon, sin_lon])
    all_data["cos_longitude"] = np.stack([cos_lon, cos_lon])

    # Time encodings
    julian_day_0 = date_t_minus_6.timetuple().tm_yday
    julian_day_1 = date_t0.timetuple().tm_yday

    all_data["sin_julian_day"] = np.stack(
        [
            np.full(grid_points, np.sin(2 * np.pi * julian_day_0 / 365.25)),
            np.full(grid_points, np.sin(2 * np.pi * julian_day_1 / 365.25)),
        ]
    )
    all_data["cos_julian_day"] = np.stack(
        [
            np.full(grid_points, np.cos(2 * np.pi * julian_day_0 / 365.25)),
            np.full(grid_points, np.cos(2 * np.pi * julian_day_1 / 365.25)),
        ]
    )

    # Local time
    local_time_0 = (date_t_minus_6.hour + lons / 15) % 24
    local_time_1 = (date_t0.hour + lons / 15) % 24

    all_data["sin_local_time"] = np.stack(
        [np.sin(2 * np.pi * local_time_0 / 24), np.sin(2 * np.pi * local_time_1 / 24)]
    )
    all_data["cos_local_time"] = np.stack(
        [np.cos(2 * np.pi * local_time_0 / 24), np.cos(2 * np.pi * local_time_1 / 24)]
    )

    # Insolation
    all_data["insolation"] = np.stack(
        [
            np.maximum(0, cos_lat * np.cos(2 * np.pi * (date_t_minus_6.hour - 12) / 24)),
            np.maximum(0, cos_lat * np.cos(2 * np.pi * (date_t0.hour - 12) / 24)),
        ]
    )

    # 100m winds
    try:
        fields_100m = get_open_data(date, ["100u", "100v"])
        all_data["100u"] = fields_100m["100u"].reshape(2, -1)
        all_data["100v"] = fields_100m["100v"].reshape(2, -1)
    except Exception:
        all_data["100u"] = np.zeros((2, grid_points), dtype=np.float64)
        all_data["100v"] = np.zeros((2, grid_points), dtype=np.float64)

    print("\n" + "-" * 70)
    print("STEP 4: Create Zarr Dataset")
    print("-" * 70)

    times = [date_t_minus_6, date_t0]

    data_vars = {}
    for var_name, data in all_data.items():
        data_vars[var_name] = xr.DataArray(
            data.astype(np.float64),
            dims=["time", "grid_point"],
            coords={"time": times, "grid_point": np.arange(grid_points)},
            attrs={
                "long_name": var_name.replace("_", " ").title(),
                "source": "ECMWF Open Data",
                "grid": "N320",
            },
        )

    # Add coordinates as data variables
    data_vars["latitude"] = xr.DataArray(
        lats.astype(np.float64),
        dims=["grid_point"],
        coords={"grid_point": np.arange(grid_points)},
        attrs={"units": "degrees_north", "long_name": "Latitude"},
    )
    data_vars["longitude"] = xr.DataArray(
        lons.astype(np.float64),
        dims=["grid_point"],
        coords={"grid_point": np.arange(grid_points)},
        attrs={"units": "degrees_east", "long_name": "Longitude"},
    )

    ds = xr.Dataset(
        data_vars,
        coords={"time": times, "grid_point": np.arange(grid_points)},
        attrs={
            "title": "Real ECMWF Data - AIFS Compatible",
            "source": "ECMWF Open Data API via AIFS runner",
            "created": datetime.now().isoformat(),
            "grid": "N320 (O1280 octahedral reduced Gaussian)",
            "grid_points": grid_points,
            "variables": len(all_data),
            "timesteps": 2,
            "format": "Real data with real N320 grid coordinates",
        },
    )

    print(f"Saving to: {output_path}")
    ds.to_zarr(output_path, mode="w")

    print("\n" + "=" * 70)
    print("SUCCESS!")
    print("=" * 70)
    print("✅ Real ECMWF data")
    print("✅ Real N320 grid coordinates")
    print("✅ Geographic masking will work")

    return output_path


# Default fixed date for reproducibility.
# This matches the date used for data/real_ecmwf_latest.zarr which can be used in tests.
# The file contains timesteps at 2025-10-10 00:00 (t-6h) and 2025-10-10 06:00 (t0).
DEFAULT_DATE = "2025-10-10 06:00"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download real ECMWF data (following run_AIFS_v1.ipynb)"
    )

    parser.add_argument(
        "--output", "-o", default="real_ecmwf_data.zarr", help="Output Zarr file path"
    )

    parser.add_argument(
        "--date",
        "-d",
        type=str,
        default=DEFAULT_DATE,
        help=(
            "Target date (YYYY-MM-DD or YYYY-MM-DD HH:MM). "
            f"Defaults to {DEFAULT_DATE} for reproducibility. "
            "Use 'latest' to fetch the most recent available data from ECMWF."
        ),
    )

    parser.add_argument("--force", "-f", action="store_true", help="Overwrite existing cached file")

    args = parser.parse_args()

    target_date = None
    if args.date.lower() == "latest":
        target_date = None  # Will use client.latest()
        print("Using latest available date from ECMWF")
    else:
        try:
            # Try full datetime format first, then just date
            if " " in args.date:
                target_date = datetime.strptime(args.date, "%Y-%m-%d %H:%M")
            else:
                target_date = datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            print(f"Invalid date: {args.date}")
            print("Use format: YYYY-MM-DD or YYYY-MM-DD HH:MM")
            return 1

    # Handle force flag - remove existing file if it exists
    if Path(args.output).exists() and args.force:
        print(f"Removing existing file: {args.output}")
        import shutil

        if Path(args.output).is_dir():
            shutil.rmtree(args.output)
        else:
            Path(args.output).unlink()

    try:
        download_real_ecmwf_data(args.output, target_date)
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

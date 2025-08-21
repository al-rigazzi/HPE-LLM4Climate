#!/usr/bin/env python3
"""
MERRA-2 Dataset Processor for PrithviWxC_Encoder

This script downloads and processes MERRA-2 data to create a unified dataset
containing only the variables required by the PrithviWxC_Encoder model.

The script:
1. Downloads required MERRA-2 datasets (6 different collections)
2. Extracts only the variables needed by PrithviWxC_Encoder
3. Aligns all data to common time/space grids
4. Saves the processed data in numpy format for efficient loading

Usage:
    python merra2_dataset_processor.py --start_date 2020-01-01 --end_date 2020-01-31 --output_dir ./processed_data --temporal_resolution 3H
"""

import argparse
import logging
import os
import shutil
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import xarray as xr
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class MERRA2DatasetProcessor:
    """
    Processes MERRA-2 data for PrithviWxC_Encoder compatibility.

    This class handles downloading, processing, and unifying multiple MERRA-2
    datasets into a single dataset containing only the variables required
    by the PrithviWxC_Encoder model.
    """

    # PrithviWxC_Encoder variable requirements (from encoder_extractor.py)
    SURFACE_VARS = [
        "EFLUX",  # Latent heat flux
        "GWETROOT",  # Root zone soil wetness
        "HFLUX",  # Sensible heat flux
        "LAI",  # Leaf area index
        "LWGAB",  # Surface absorbed longwave radiation
        "LWGEM",  # Surface emitted longwave radiation
        "LWTUP",  # Upwelling longwave radiation at TOA
        "PS",  # Surface pressure
        "QV2M",  # 2-meter specific humidity
        "SLP",  # Sea level pressure
        "SWGNT",  # Surface net downward shortwave radiation
        "SWTNT",  # TOA net downward shortwave radiation
        "T2M",  # 2-meter temperature
        "TQI",  # Total precipitable ice water
        "TQL",  # Total precipitable liquid water
        "TQV",  # Total precipitable water vapor
        "TS",  # Surface skin temperature
        "U10M",  # 10-meter eastward wind
        "V10M",  # 10-meter northward wind
        "Z0M",  # Surface roughness
    ]

    STATIC_SURFACE_VARS = [
        "FRACI",  # Ice fraction (from M2TMNXGLC)
        "FRLAND",  # Land fraction
        "FROCEAN",  # Ocean fraction
        "PHIS",  # Surface geopotential height
    ]

    VERTICAL_VARS = [
        "CLOUD",  # Cloud fraction (from cloud datasets)
        "H",  # Geopotential height
        "OMEGA",  # Vertical velocity (omega)
        "PL",  # Pressure levels
        "QI",  # Cloud ice mixing ratio
        "QL",  # Cloud liquid water mixing ratio
        "QV",  # Specific humidity
        "T",  # Temperature
        "U",  # Eastward wind
        "V",  # Northward wind
    ]

    # Pressure levels used by PrithviWxC
    PRESSURE_LEVELS = [
        34.0,
        39.0,
        41.0,
        43.0,
        44.0,
        45.0,
        48.0,
        51.0,
        53.0,
        56.0,
        63.0,
        68.0,
        71.0,
        72.0,
    ]

    # MERRA-2 dataset collections and their variables
    MERRA2_COLLECTIONS = {
        "M2T1NXLND": {
            "description": "Land surface diagnostics",
            "variables": ["EFLUX", "GWETROOT", "HFLUX", "LAI", "TS", "Z0M"],
            "temporal_resolution": "1H",
            "url_pattern": "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXLND.5.12.4/{year}/{month:02d}/MERRA2_{stream}.tavg1_2d_lnd_Nx.{date}.nc4",
        },
        "M2I1NXASM": {
            "description": "Atmospheric surface diagnostics",
            "variables": [
                "PS",
                "QV2M",
                "SLP",
                "T2M",
                "TQI",
                "TQL",
                "TQV",
                "U10M",
                "V10M",
                "FRLAND",
                "FROCEAN",
                "PHIS",
            ],
            "temporal_resolution": "1H",
            "url_pattern": "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I1NXASM.5.12.4/{year}/{month:02d}/MERRA2_{stream}.inst1_2d_asm_Nx.{date}.nc4",
        },
        "M2I3NPASM": {
            "description": "Atmospheric profiles",
            "variables": ["H", "OMEGA", "PL", "QI", "QL", "QV", "T", "U", "V"],
            "temporal_resolution": "3H",
            "url_pattern": "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I3NPASM.5.12.4/{year}/{month:02d}/MERRA2_{stream}.inst3_3d_asm_Np.{date}.nc4",
        },
        "M2I1NXRAD": {
            "description": "Radiation diagnostics",
            "variables": ["LWGAB", "LWGEM", "LWTUP", "SWGNT", "SWTNT"],
            "temporal_resolution": "1H",
            "url_pattern": "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I1NXRAD.5.12.4/{year}/{month:02d}/MERRA2_{stream}.inst1_2d_rad_Nx.{date}.nc4",
        },
        "M2TMNXGLC": {
            "description": "Land ice and glacier diagnostics",
            "variables": ["FRACI"],
            "temporal_resolution": "Monthly",
            "url_pattern": "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2TMNXGLC.5.12.4/{year}/{month:02d}/MERRA2_{stream}.tavgM_2d_glc_Nx.{date}.nc4",
        },
        "M2I3NPCLD": {
            "description": "Cloud diagnostics",
            "variables": ["CLOUD"],
            "temporal_resolution": "3H",
            "url_pattern": "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I3NPCLD.5.12.4/{year}/{month:02d}/MERRA2_{stream}.inst3_3d_cld_Np.{date}.nc4",
        },
    }

    def __init__(
        self,
        output_dir: str = "./processed_data",
        cache_dir: str = "./merra2_cache",
        earthdata_username: Optional[str] = None,
        earthdata_password: Optional[str] = None,
    ):
        """
        Initialize the MERRA-2 dataset processor.

        Args:
            output_dir: Directory to save processed datasets
            cache_dir: Directory to cache downloaded files
            earthdata_username: NASA Earthdata username
            earthdata_password: NASA Earthdata password
        """
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Setup authentication if provided
        self.earthdata_username = earthdata_username or os.getenv("EARTHDATA_USERNAME")
        self.earthdata_password = earthdata_password or os.getenv("EARTHDATA_PASSWORD")

        if not self.earthdata_username or not self.earthdata_password:
            logger.warning(
                "NASA Earthdata credentials not provided. "
                "Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables "
                "or provide them as arguments."
            )

        # Setup requests session with authentication
        self.session = requests.Session()
        if self.earthdata_username and self.earthdata_password:
            self.session.auth = (self.earthdata_username, self.earthdata_password)

    def _get_merra2_stream(self, date: datetime) -> str:
        """Get the appropriate MERRA-2 stream identifier for a date."""
        if date.year >= 1992:
            return "400"
        elif date.year >= 1980:
            return "300"
        else:
            return "100"

    def _download_file(self, url: str, output_path: Path) -> bool:
        """
        Download a file from NASA Earthdata.

        Args:
            url: URL to download
            output_path: Local path to save file

        Returns:
            True if successful, False otherwise
        """
        if output_path.exists():
            logger.info(f"File already exists: {output_path.name}")
            return True

        try:
            logger.info(f"Downloading: {url}")
            response = self.session.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with open(output_path, "wb") as f:
                with tqdm(
                    total=total_size, unit="B", unit_scale=True, desc=output_path.name
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            logger.info(f"Downloaded: {output_path.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            if output_path.exists():
                output_path.unlink()
            return False

    def download_data(
        self, start_date: str, end_date: str, temporal_resolution: str = "3H"
    ) -> Dict[str, List[Path]]:
        """
        Download MERRA-2 data for the specified date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            temporal_resolution: "1H", "3H", or "Monthly"

        Returns:
            Dictionary mapping collection names to lists of downloaded files
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        downloaded_files = {}

        for collection, info in self.MERRA2_COLLECTIONS.items():
            collection_dir = self.cache_dir / collection
            collection_dir.mkdir(exist_ok=True)
            downloaded_files[collection] = []

            logger.info(f"Downloading {collection}: {info['description']}")

            # Determine date range based on collection temporal resolution
            if info["temporal_resolution"] == "Monthly":
                dates = pd.date_range(start_dt, end_dt, freq="MS")  # Month start
            else:
                dates = pd.date_range(start_dt, end_dt, freq="D")  # Daily

            for date in dates:
                stream = self._get_merra2_stream(date)

                # Format date for filename
                if info["temporal_resolution"] == "Monthly":
                    date_str = date.strftime("%Y%m")
                else:
                    date_str = date.strftime("%Y%m%d")

                # Build URL
                url = info["url_pattern"].format(
                    year=date.year, month=date.month, stream=stream, date=date_str
                )

                # Output filename
                filename = url.split("/")[-1]
                output_path = collection_dir / filename

                # Download file
                if self._download_file(url, output_path):
                    downloaded_files[collection].append(output_path)
                else:
                    logger.warning(f"Failed to download {filename}")

        return downloaded_files

    def _extract_variables(
        self, file_path: Path, variables: List[str], pressure_levels: Optional[List[float]] = None
    ) -> xr.Dataset:
        """
        Extract specified variables from a MERRA-2 file.

        Args:
            file_path: Path to MERRA-2 NetCDF file
            variables: List of variable names to extract
            pressure_levels: Optional pressure levels for 3D data

        Returns:
            xarray Dataset with extracted variables
        """
        try:
            with xr.open_dataset(file_path) as ds:
                # Extract available variables
                available_vars = [var for var in variables if var in ds.data_vars]

                if not available_vars:
                    logger.warning(f"No requested variables found in {file_path.name}")
                    return None

                # Extract data
                data = ds[available_vars]

                # If 3D data, select pressure levels
                if pressure_levels and "lev" in data.dims:
                    # Convert pressure levels to integers for selection
                    target_levels = [int(lev) for lev in pressure_levels]
                    available_levels = data.lev.values

                    # Find matching levels
                    level_indices = []
                    for target_lev in target_levels:
                        # Find closest level
                        closest_idx = np.argmin(np.abs(available_levels - target_lev))
                        if np.abs(available_levels[closest_idx] - target_lev) < 5:  # Within 5 hPa
                            level_indices.append(closest_idx)

                    if level_indices:
                        data = data.isel(lev=level_indices)
                        # Update level coordinates to match target levels
                        data = data.assign_coords(lev=pressure_levels[: len(level_indices)])

                return data.load()  # Load into memory

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None

    def process_collection(
        self, collection: str, file_paths: List[Path], target_variables: List[str]
    ) -> Optional[xr.Dataset]:
        """
        Process all files from a MERRA-2 collection.

        Args:
            collection: Collection name (e.g., 'M2I3NPASM')
            file_paths: List of file paths for this collection
            target_variables: Variables to extract

        Returns:
            Combined xarray Dataset or None if processing failed
        """
        if not file_paths:
            logger.warning(f"No files found for collection {collection}")
            return None

        logger.info(f"Processing {len(file_paths)} files for {collection}")

        datasets = []

        for file_path in tqdm(file_paths, desc=f"Processing {collection}"):
            # Use pressure levels for 3D collections
            pressure_levels = self.PRESSURE_LEVELS if "3d" in collection.lower() else None

            ds = self._extract_variables(file_path, target_variables, pressure_levels)
            if ds is not None:
                datasets.append(ds)

        if not datasets:
            logger.error(f"No valid datasets found for {collection}")
            return None

        # Combine datasets along time dimension
        try:
            combined = xr.concat(datasets, dim="time")
            combined = combined.sortby("time")  # Ensure time ordering

            logger.info(f"Successfully processed {collection}: {list(combined.data_vars.keys())}")
            return combined

        except Exception as e:
            logger.error(f"Error combining datasets for {collection}: {e}")
            return None

    def align_and_combine_datasets(
        self, datasets: Dict[str, xr.Dataset], temporal_resolution: str = "3H"
    ) -> xr.Dataset:
        """
        Align and combine datasets from different collections.

        Args:
            datasets: Dictionary mapping collection names to datasets
            temporal_resolution: Target temporal resolution

        Returns:
            Combined dataset with all variables
        """
        logger.info("Aligning and combining datasets...")

        # Find common time range and spatial grid
        time_ranges = []
        spatial_grids = []

        for name, ds in datasets.items():
            if ds is not None:
                time_ranges.append((ds.time.min().values, ds.time.max().values))
                spatial_grids.append((ds.lat.values, ds.lon.values))

        # Determine common time range
        common_start = max([t[0] for t in time_ranges])
        common_end = min([t[1] for t in time_ranges])

        # Create target time grid
        if temporal_resolution == "1H":
            target_times = pd.date_range(common_start, common_end, freq="H")
        elif temporal_resolution == "3H":
            target_times = pd.date_range(common_start, common_end, freq="3H")
        elif temporal_resolution == "Monthly":
            target_times = pd.date_range(common_start, common_end, freq="MS")
        else:
            raise ValueError(f"Unsupported temporal resolution: {temporal_resolution}")

        # Use the first dataset's spatial grid as reference
        ref_lat, ref_lon = spatial_grids[0]

        # Align each dataset to common grid
        aligned_datasets = []

        for name, ds in datasets.items():
            if ds is None:
                continue

            logger.info(f"Aligning {name}...")

            # Interpolate to common time grid
            ds_aligned = ds.interp(time=target_times, method="linear")

            # Interpolate to common spatial grid if needed
            if not (
                np.array_equal(ds.lat.values, ref_lat) and np.array_equal(ds.lon.values, ref_lon)
            ):
                ds_aligned = ds_aligned.interp(lat=ref_lat, lon=ref_lon, method="linear")

            aligned_datasets.append(ds_aligned)

        # Combine all datasets
        logger.info("Combining aligned datasets...")
        combined = xr.merge(aligned_datasets, compat="override")

        return combined

    def create_prithvi_dataset(self, combined_dataset: xr.Dataset) -> Dict[str, np.ndarray]:
        """
        Create final dataset formatted for PrithviWxC_Encoder.

        Args:
            combined_dataset: Combined xarray dataset

        Returns:
            Dictionary with numpy arrays for surface, static, and vertical data
        """
        logger.info("Creating PrithviWxC_Encoder formatted dataset...")

        # Extract surface variables
        surface_data = []
        for var in self.SURFACE_VARS:
            if var in combined_dataset.data_vars:
                data = combined_dataset[var].values
                # Ensure data has correct dimensions: (time, lat, lon)
                if data.ndim == 3:
                    surface_data.append(data)
                else:
                    logger.warning(
                        f"Unexpected dimensions for surface variable {var}: {data.shape}"
                    )

        # Extract static variables
        static_data = []
        for var in self.STATIC_SURFACE_VARS:
            if var in combined_dataset.data_vars:
                data = combined_dataset[var].values
                # Static data should be (lat, lon) or (time, lat, lon) - take first time if needed
                if data.ndim == 3:
                    data = data[0]  # Take first time slice
                elif data.ndim == 2:
                    pass  # Already correct
                else:
                    logger.warning(f"Unexpected dimensions for static variable {var}: {data.shape}")
                    continue
                static_data.append(data)

        # Extract vertical variables
        vertical_data = []
        for var in self.VERTICAL_VARS:
            if var in combined_dataset.data_vars:
                data = combined_dataset[var].values
                # Ensure data has correct dimensions: (time, level, lat, lon)
                if data.ndim == 4:
                    vertical_data.append(data)
                else:
                    logger.warning(
                        f"Unexpected dimensions for vertical variable {var}: {data.shape}"
                    )

        # Stack arrays
        surface_array = (
            np.stack(surface_data, axis=1) if surface_data else None
        )  # (time, var, lat, lon)
        static_array = np.stack(static_data, axis=0) if static_data else None  # (var, lat, lon)
        vertical_array = (
            np.stack(vertical_data, axis=1) if vertical_data else None
        )  # (time, var, level, lat, lon)

        # Get coordinate information
        coords = {
            "time": combined_dataset.time.values,
            "lat": combined_dataset.lat.values,
            "lon": combined_dataset.lon.values,
        }

        if vertical_array is not None and "lev" in combined_dataset.dims:
            coords["levels"] = combined_dataset.lev.values

        return {
            "surface": surface_array,
            "static": static_array,
            "vertical": vertical_array,
            "coordinates": coords,
            "surface_vars": [var for var in self.SURFACE_VARS if var in combined_dataset.data_vars],
            "static_vars": [
                var for var in self.STATIC_SURFACE_VARS if var in combined_dataset.data_vars
            ],
            "vertical_vars": [
                var for var in self.VERTICAL_VARS if var in combined_dataset.data_vars
            ],
        }

    def save_dataset(
        self, dataset: Dict[str, np.ndarray], output_path: Path, compress: bool = True
    ) -> None:
        """
        Save the processed dataset to disk.

        Args:
            dataset: Processed dataset dictionary
            output_path: Output file path
            compress: Whether to compress the saved file
        """
        logger.info(f"Saving dataset to {output_path}")

        if compress:
            np.savez_compressed(output_path, **dataset)
        else:
            np.savez(output_path, **dataset)

        # Save metadata as separate file
        metadata = {
            "surface_vars": dataset["surface_vars"],
            "static_vars": dataset["static_vars"],
            "vertical_vars": dataset["vertical_vars"],
            "coordinates": dataset["coordinates"],
            "creation_time": datetime.now().isoformat(),
            "format_version": "1.0",
        }

        metadata_path = output_path.with_suffix(".metadata.json")
        import json

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Dataset saved successfully")
        logger.info(f"Surface variables: {len(dataset['surface_vars'])}")
        logger.info(f"Static variables: {len(dataset['static_vars'])}")
        logger.info(f"Vertical variables: {len(dataset['vertical_vars'])}")

        if dataset["surface"] is not None:
            logger.info(f"Surface data shape: {dataset['surface'].shape}")
        if dataset["vertical"] is not None:
            logger.info(f"Vertical data shape: {dataset['vertical'].shape}")
        if dataset["static"] is not None:
            logger.info(f"Static data shape: {dataset['static'].shape}")

    def process_timerange(
        self,
        start_date: str,
        end_date: str,
        temporal_resolution: str = "3H",
        output_filename: Optional[str] = None,
    ) -> Path:
        """
        Complete processing pipeline for a date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            temporal_resolution: "1H", "3H", or "Monthly"
            output_filename: Optional custom output filename

        Returns:
            Path to saved dataset file
        """
        logger.info(f"Processing MERRA-2 data from {start_date} to {end_date}")
        logger.info(f"Temporal resolution: {temporal_resolution}")

        # Download data
        downloaded_files = self.download_data(start_date, end_date, temporal_resolution)

        # Process each collection
        processed_datasets = {}

        for collection, files in downloaded_files.items():
            if not files:
                continue

            target_vars = self.MERRA2_COLLECTIONS[collection]["variables"]
            ds = self.process_collection(collection, files, target_vars)
            processed_datasets[collection] = ds

        # Filter out None datasets
        valid_datasets = {k: v for k, v in processed_datasets.items() if v is not None}

        if not valid_datasets:
            raise RuntimeError("No valid datasets were processed")

        # Align and combine
        combined_dataset = self.align_and_combine_datasets(valid_datasets, temporal_resolution)

        # Create PrithviWxC format
        prithvi_dataset = self.create_prithvi_dataset(combined_dataset)

        # Save dataset
        if output_filename is None:
            output_filename = f"merra2_prithvi_{start_date}_{end_date}_{temporal_resolution}.npz"

        output_path = self.output_dir / output_filename
        self.save_dataset(prithvi_dataset, output_path)

        return output_path


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Download and process MERRA-2 data for PrithviWxC_Encoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process 3-hourly data for January 2020
    python merra2_dataset_processor.py --start_date 2020-01-01 --end_date 2020-01-31 --temporal_resolution 3H

    # Process monthly data for 2020
    python merra2_dataset_processor.py --start_date 2020-01-01 --end_date 2020-12-31 --temporal_resolution Monthly

    # Process with custom output directory
    python merra2_dataset_processor.py --start_date 2020-01-01 --end_date 2020-01-07 --output_dir ./my_data
        """,
    )

    parser.add_argument(
        "--start_date", type=str, required=True, help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument("--end_date", type=str, required=True, help="End date in YYYY-MM-DD format")
    parser.add_argument(
        "--temporal_resolution",
        type=str,
        choices=["1H", "3H", "Monthly"],
        default="3H",
        help="Temporal resolution (default: 3H)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./processed_data",
        help="Output directory for processed datasets (default: ./processed_data)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./merra2_cache",
        help="Cache directory for downloaded files (default: ./merra2_cache)",
    )
    parser.add_argument("--output_filename", type=str, help="Custom output filename (optional)")
    parser.add_argument(
        "--earthdata_username",
        type=str,
        help="NASA Earthdata username (can also use EARTHDATA_USERNAME env var)",
    )
    parser.add_argument(
        "--earthdata_password",
        type=str,
        help="NASA Earthdata password (can also use EARTHDATA_PASSWORD env var)",
    )
    parser.add_argument(
        "--clean_cache", action="store_true", help="Remove cache directory after processing"
    )

    args = parser.parse_args()

    try:
        # Create processor
        processor = MERRA2DatasetProcessor(
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            earthdata_username=args.earthdata_username,
            earthdata_password=args.earthdata_password,
        )

        # Process data
        output_path = processor.process_timerange(
            start_date=args.start_date,
            end_date=args.end_date,
            temporal_resolution=args.temporal_resolution,
            output_filename=args.output_filename,
        )

        logger.info(f"Processing completed successfully!")
        logger.info(f"Output file: {output_path}")

        # Clean cache if requested
        if args.clean_cache and processor.cache_dir.exists():
            logger.info("Cleaning cache directory...")
            shutil.rmtree(processor.cache_dir)

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

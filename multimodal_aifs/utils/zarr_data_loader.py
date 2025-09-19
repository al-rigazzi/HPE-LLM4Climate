#!/usr/bin/env python3
"""
Zarr Data Loader for AIFS Multimodal System

This module provides utilities for loading Zarr format climate data
and converting it to the 5D tensor format expected by the AIFS multimodal model.

Features:
- Load Zarr datasets from local files or cloud storage
- Convert to standard AIFS format [B, T, V, H, W]
- Support for chunked/partial loading
- Integration with existing climate data utilities
- Xarray compatibility for metadata handling

Usage:
    from multimodal_aifs.utils.zarr_data_loader import ZarrClimateLoader

    loader = ZarrClimateLoader("path/to/data.zarr")
    data = loader.load_time_range("2024-01-01", "2024-01-07")
    tensor = loader.to_aifs_tensor(data)
"""

import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Try to import Zarr and Xarray
try:
    import xarray as xr

    ZARR_AVAILABLE = True
    print("📦 Zarr and Xarray successfully imported!")
except ImportError as e:
    ZARR_AVAILABLE = False
    warnings.warn(f"Zarr/Xarray not available: {e}. Install with: pip install zarr xarray")
    xr = None  # type: ignore

# Import climate data utilities
try:
    pass  # ClimateDataProcessor import removed as unused
except ImportError:
    # If climate utils not available, define basic variables
    CLIMATE_VARIABLES = ["temperature_2m", "surface_pressure", "geopotential_500"]


class ZarrClimateLoader:
    """
    Zarr data loader for AIFS multimodal climate system.

    Handles loading Zarr climate datasets and converting them to the
    5D tensor format [B, T, V, H, W] expected by AIFS models.
    """

    def __init__(
        self,
        zarr_path: str | Path,
        chunk_size: dict[str, int] | None = None,
        variables: list[str] | None = None,
    ):
        """
        Initialize Zarr climate data loader.

        Args:
            zarr_path: Path to Zarr dataset (local or cloud URL)
            chunk_size: Override default chunking (e.g., {"time": 24, "lat": 100})
            variables: list of variables to load (if None, loads all)
        """
        if not ZARR_AVAILABLE:
            raise ImportError("Zarr and Xarray are required. Install with: pip install zarr xarray")

        self.zarr_path = str(zarr_path)
        self.chunk_size = chunk_size or {}
        self.variables = variables

        # Initialize attributes with proper types
        self.spatial_dims: tuple[str, ...] = ()
        self.spatial_shape: tuple[int, ...] = ()
        self.is_aifs_format: bool = False
        self.time_dim: str = ""
        self.time_range: tuple[str, str] = ("", "")
        self.available_variables: list[str] = []
        self.ds: xr.Dataset | None = None

        # Load dataset
        self._load_dataset()

    def _load_dataset(self):
        """Load the Zarr dataset using Xarray."""
        try:
            print(f"Loading Zarr dataset: {self.zarr_path}")

            # Load with Xarray for easy metadata handling
            self.ds = xr.open_zarr(self.zarr_path, chunks=self.chunk_size)

            # Get dataset info
            self._analyze_dataset()

            print("Dataset loaded successfully")
            print(f"   Shape: {dict(self.ds.sizes)}")
            print(f"   🔢 Variables: {len(self.available_variables)}")
            print(f"   📅 Time range: {self.time_range[0]} to {self.time_range[1]}")

        except Exception as e:
            raise RuntimeError(f"Failed to load Zarr dataset: {e}") from e

    def _analyze_dataset(self):
        """Analyze the loaded dataset structure."""
        # Get available variables (exclude coordinates)
        assert self.ds is not None, "Dataset is not loaded"
        coords = set(self.ds.coords.keys())
        self.available_variables = [v for v in self.ds.data_vars.keys() if v not in coords]

        # Get spatial dimensions - handle both lat/lon and AIFS grid_point formats
        if "lat" in self.ds.dims and "lon" in self.ds.dims:
            self.spatial_dims = ("lat", "lon")
            self.spatial_shape = (self.ds.dims["lat"], self.ds.dims["lon"])
            self.is_aifs_format = False
        elif "latitude" in self.ds.dims and "longitude" in self.ds.dims:
            self.spatial_dims = ("latitude", "longitude")
            self.spatial_shape = (self.ds.sizes["latitude"], self.ds.sizes["longitude"])
            self.is_aifs_format = False
        elif "grid_point" in self.ds.dims:
            # AIFS-compatible format with flattened grid points
            self.spatial_dims = ("grid_point",)
            self.spatial_shape = (self.ds.sizes["grid_point"],)
            self.is_aifs_format = True
        else:
            raise ValueError("Could not identify spatial dimensions (lat/lon or grid_point)")

        # Get time dimension
        if "time" in self.ds.dims:
            self.time_dim = "time"
            time_values = self.ds[self.time_dim].values
            self.time_range = (str(time_values[0])[:19], str(time_values[-1])[:19])
        else:
            raise ValueError("Could not identify time dimension")

        # Filter variables if specified
        if self.variables:
            self.available_variables = [v for v in self.available_variables if v in self.variables]

    def load_time_range(
        self, start_time: str | None, end_time: str | None, variables: list[str] | None = None
    ) -> xr.Dataset:
        """
        Load data for a specific time range.

        Args:
            start_time: Start time (e.g., "2024-01-01") or None for all from beginning
            end_time: End time (e.g., "2024-01-07") or None for all to end
            variables: Variables to load (default: all available)

        Returns:
            Xarray dataset with selected time range and variables
        """
        # Select variables
        vars_to_load = variables or self.available_variables
        vars_to_load = [v for v in vars_to_load if v in self.available_variables]

        if not vars_to_load:
            raise ValueError(f"No valid variables found. Available: {self.available_variables}")

        # Format time range message
        if start_time is None and end_time is None:
            time_msg = "all time steps"
        elif start_time is None:
            time_msg = f"all time steps to {end_time}"
        elif end_time is None:
            time_msg = f"all time steps from {start_time}"
        else:
            time_msg = f"{start_time} to {end_time}"

        print(f"⏰ Loading time range: {time_msg}")
        print(f"🔢 Variables: {vars_to_load} ({len(vars_to_load)} total)")

        assert self.ds is not None, "Dataset is not loaded"

        # Select time range and variables
        if start_time is None and end_time is None:
            # Load all time steps
            subset = self.ds[vars_to_load]
        elif start_time is None:
            # Load from beginning to end_time
            subset = self.ds[vars_to_load].sel(time=slice(None, end_time))
        elif end_time is None:
            # Load from start_time to end
            subset = self.ds[vars_to_load].sel(time=slice(start_time, None))
        else:
            # Load specific range
            subset = self.ds[vars_to_load].sel(time=slice(start_time, end_time))

        # Load into memory (if needed)
        if hasattr(subset, "compute"):
            print("💭 Computing data (loading into memory)...")
            subset = subset.compute()

        print(f"Loaded data shape: {dict(subset.sizes)}")
        return subset

    def load_spatial_region(
        self,
        lat_range: tuple[float, float],
        lon_range: tuple[float, float],
        time_range: tuple[str, str] | None = None,
        variables: list[str] | None = None,
    ):
        """
        Load data for a specific spatial region and optional time range.

        Args:
            lat_range: tuple of (min_lat, max_lat) in degrees
            lon_range: tuple of (min_lon, max_lon) in degrees
            time_range: Optional tuple of (start_time, end_time) strings
            variables: Variables to load (default: all available)

        Returns:
            Xarray dataset with selected spatial region and variables
        """
        # Select variables
        vars_to_load = variables or self.available_variables
        vars_to_load = [v for v in vars_to_load if v in self.available_variables]

        if not vars_to_load:
            raise ValueError(f"No valid variables found. Available: {self.available_variables}")

        print("Loading spatial region:")
        print(f"   📍 Latitude: {lat_range[0]}° to {lat_range[1]}°")
        print(f"   📍 Longitude: {lon_range[0]}° to {lon_range[1]}°")
        print(f"   🔢 Variables: {vars_to_load} ({len(vars_to_load)} total)")

        assert self.ds is not None, "Dataset is not loaded"
        # Start with variable selection
        subset = self.ds[vars_to_load]

        # Apply spatial selection
        if len(self.spatial_dims) == 2:
            if len(self.spatial_dims) >= 2:
                lat_dim, lon_dim = self.spatial_dims[0], self.spatial_dims[1]
            else:
                raise ValueError("Spatial dimensions must contain at least latitude and longitude")

            # Handle longitude wrapping (e.g., -180 to 180 vs 0 to 360)
            lon_coords = subset[lon_dim].values
            lat_coords = subset[lat_dim].values

            # Select latitude range
            lat_mask = (lat_coords >= lat_range[0]) & (lat_coords <= lat_range[1])

            # Handle longitude range (accounting for potential wrapping)
            if lon_range[0] <= lon_range[1]:
                # Normal case: e.g., -120 to -60
                lon_mask = (lon_coords >= lon_range[0]) & (lon_coords <= lon_range[1])
            else:
                # Wrapped case: e.g., 170 to -170 (crossing 180°)
                lon_mask = (lon_coords >= lon_range[0]) | (lon_coords <= lon_range[1])

            # Apply spatial selection
            if lat_mask.any():
                subset = subset.isel({lat_dim: lat_mask})
            else:
                raise ValueError(f"No data found in latitude range {lat_range}")

            if lon_mask.any():
                subset = subset.isel({lon_dim: lon_mask})
            else:
                raise ValueError(f"No data found in longitude range {lon_range}")
        else:
            # AIFS format with single grid_point dimension - spatial selection not applicable
            print(
                "   ℹ️  AIFS format detected - spatial selection not applied "
                "(data already in grid point format)"
            )
            lat_dim = lon_dim = self.spatial_dims[0]  # For coordinate reporting only

        # Apply time selection if provided
        if time_range:
            start_time, end_time = time_range
            print(f"   ⏰ Time range: {start_time} to {end_time}")
            subset = subset.sel(time=slice(start_time, end_time))

        # Load into memory (if needed)
        if hasattr(subset, "compute"):
            print("💭 Computing regional data (loading into memory)...")
            subset = subset.compute()

        print(f"Regional data loaded: {dict(subset.dims)}")

        # Get actual coordinate ranges after selection
        actual_lat_range = (subset[lat_dim].min().item(), subset[lat_dim].max().item())
        actual_lon_range = (subset[lon_dim].min().item(), subset[lon_dim].max().item())

        print(f"   📍 Actual lat range: {actual_lat_range[0]:.2f}° to {actual_lat_range[1]:.2f}°")
        print(f"   📍 Actual lon range: {actual_lon_range[0]:.2f}° to {actual_lon_range[1]:.2f}°")

        return subset

    def to_aifs_tensor(
        self,
        data,
        batch_size: int = 1,
        normalize: bool = True,
        device: str = "cpu",
        use_fp16: bool = False,
    ) -> torch.Tensor:
        """
        Convert Xarray dataset to AIFS tensor format.
        For lat/lon format: [B, T, V, H, W]
        For AIFS grid_point format: [B, T, V, grid_points]

        Args:
            data: Xarray dataset
            batch_size: Batch size (for creating batches from time series)
            normalize: Whether to normalize the data
            device: Device to move tensor to ("cpu", "cuda", "mps", etc.)
            use_fp16: Whether to use FP16 (torch.float16) instead of FP32 (torch.float32)

        Returns:
            Tensor ready for AIFS model input
        """
        print("Converting to AIFS tensor format...")

        # Get data variables
        variables = list(data.data_vars.keys())

        # Check if data is already in AIFS format
        if "data" in variables and len(data.data.dims) == 5:
            # Data is already in AIFS format [batch, time, ensemble, grid_points, variables]
            print("Data already in AIFS format, using directly")
            if use_fp16:
                tensor = torch.from_numpy(data["data"].values).half()
            else:
                tensor = torch.from_numpy(data["data"].values).float()
            print(f"   AIFS tensor shape: {tensor.shape}")
            return tensor

        # Stack variables into single array
        arrays = []
        for var in variables:
            var_data = data[var].values
            if self.is_aifs_format:
                # AIFS format: [time, grid_points]
                if var_data.ndim == 2:
                    arrays.append(var_data)
                else:
                    raise ValueError(
                        f"AIFS format: Variable {var} has unexpected shape: {var_data.shape}"
                    )
            else:
                # Lat/lon format: [time, lat, lon]
                if var_data.ndim == 3:
                    arrays.append(var_data)
                else:
                    raise ValueError(
                        f"Lat/lon format: Variable {var} has unexpected shape: {var_data.shape}"
                    )

        # Stack variables
        if self.is_aifs_format:
            # AIFS format: [time, variables, grid_points]
            stacked = np.stack(arrays, axis=1)
        else:
            # Lat/lon format: [time, variables, lat, lon]
            stacked = np.stack(arrays, axis=1)

        # Convert to tensor
        if use_fp16:
            tensor = torch.from_numpy(stacked).half().to(device)
        else:
            tensor = torch.from_numpy(stacked).float().to(device)

        # Add batch dimension or create batches
        if batch_size == 1:
            # Single batch
            if self.is_aifs_format:
                # For AIFS format, add ensemble dimension: [batch, time, 1, grid_points, vars]
                tensor = tensor.unsqueeze(0)  # [1, time, vars, grid_points]
                tensor = tensor.unsqueeze(2)  # [1, time, 1, vars, grid_points]
                # Reorder to [batch, time, ensemble=1, grid_points, vars]
                tensor = tensor.permute(0, 1, 2, 4, 3)
            else:
                tensor = tensor.unsqueeze(0)
        else:
            # Create batches from time series
            time_steps = tensor.shape[0]
            if time_steps < batch_size:
                # Pad if needed
                padding = batch_size - time_steps
                tensor = torch.cat([tensor, tensor[-1:].repeat(padding, 1, 1, 1)], dim=0)

            # Reshape to batches
            time_per_batch = tensor.shape[0] // batch_size
            tensor = tensor[: batch_size * time_per_batch]
            if self.is_aifs_format:
                tensor = tensor.view(batch_size, time_per_batch, *tensor.shape[1:])
                # Add ensemble dimension for AIFS format
                tensor = tensor.unsqueeze(2)  # Add ensemble dimension
                # Reorder to [batch, time, ensemble=1, grid_points, vars]
                tensor = tensor.permute(0, 1, 2, 4, 3)
            else:
                tensor = tensor.view(batch_size, time_per_batch, *tensor.shape[1:])

        # Normalize if requested
        if normalize:
            # Simple normalization: standardize each variable
            for i, var in enumerate(variables):
                if self.is_aifs_format:
                    var_data = tensor[:, :, 0, :, i]  # [batch, time, ensemble=1, grid_points, vars]
                else:
                    var_data = tensor[:, :, i, :, :]  # [batch, time, vars, lat, lon]
                mean = var_data.mean()
                std = var_data.std() + 1e-8  # Avoid division by zero
                if self.is_aifs_format:
                    tensor[:, :, 0, :, i] = (var_data - mean) / std
                else:
                    tensor[:, :, i, :, :] = (var_data - mean) / std

        if self.is_aifs_format:
            print(
                f"   Format: [batch={tensor.shape[0]}, time={tensor.shape[1]}, "
                f"ensemble={tensor.shape[2]}, grid_points={tensor.shape[3]}, "
                f"vars={tensor.shape[4]}"
            )
        else:
            print(
                f"   Format: [batch={tensor.shape[0]}, time={tensor.shape[1]}, "
                f"vars={tensor.shape[2]}, height={tensor.shape[3]}, width={tensor.shape[4]}]"
            )

        tensor = tensor.to(device)
        return tensor

    def get_info(self) -> dict[str, Any]:
        """Get dataset information."""
        assert self.ds is not None, "Dataset is not loaded"
        return {
            "zarr_path": self.zarr_path,
            "dimensions": dict(self.ds.dims),
            "variables": self.available_variables,
            "spatial_shape": self.spatial_shape,
            "spatial_dims": self.spatial_dims,
            "time_range": self.time_range,
            "total_size_gb": self.ds.nbytes / 1e9,
        }


# Convenience functions
def load_zarr_for_aifs(
    zarr_path: str,
    start_time: str,
    end_time: str,
    variables: list[str] | None = None,
    batch_size: int = 2,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Convenience function to load Zarr data directly into AIFS tensor format.

    Args:
        zarr_path: Path to Zarr dataset
        start_time: Start time string
        end_time: End time string
        variables: Variables to load
        batch_size: Batch size for tensor
        normalize: Whether to normalize

    Returns:
        5D tensor ready for AIFS model
    """
    loader = ZarrClimateLoader(zarr_path, variables=variables)
    data = loader.load_time_range(start_time, end_time, variables)
    tensor = loader.to_aifs_tensor(data, batch_size, normalize)
    return tensor


def test_zarr_integration():
    """Test function for Zarr integration."""
    print("Testing Zarr Integration with AIFS")
    print("=" * 40)

    if not ZARR_AVAILABLE:
        print("Zarr not available. Install with: pip install zarr xarray")
        return False

    try:
        # Test with the test dataset
        test_path = "test_aifs_large.zarr"

        if not Path(test_path).exists():
            print(f"Test dataset not found: {test_path}")
            return False

        print(f"Testing with: {test_path}")

        # Load dataset
        loader = ZarrClimateLoader(test_path)

        # Get info
        info = loader.get_info()
        print(f"Dataset info: {info}")

        # Load time range
        data = loader.load_time_range("2024-01-01", "2024-01-01T12:00:00")

        # Convert to AIFS tensor
        tensor = loader.to_aifs_tensor(data, batch_size=2, normalize=True)

        print("Test successful!")
        print(f"   Final tensor shape: {tensor.shape}")
        print("   Expected format: [batch, time, vars, height, width]")

        return True

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


# Example usage
if __name__ == "__main__":
    test_zarr_integration()

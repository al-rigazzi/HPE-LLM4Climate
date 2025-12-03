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

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import constants
from multimodal_aifs.constants import ALL_AIFS_VARIABLES
from multimodal_aifs.utils.device_utils import (
    configure_device_for_max_perf,
    resolve_device,
    supports_amp,
)

# Try to import Zarr and Xarray
try:
    import xarray as xr

    ZARR_AVAILABLE = True
    print("📦 Zarr and Xarray successfully imported!")
except ImportError as e:
    ZARR_AVAILABLE = False
    warnings.warn(f"Zarr/Xarray not available: {e}. Install with: pip install zarr xarray")
    xr = None  # type: ignore

if TYPE_CHECKING:
    import xarray as xr_typing
else:
    xr_typing = Any

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
        self.time_dim: str = ""
        self.time_range: tuple[str, str] = ("", "")
        self.available_variables: list[str] = []
        self.ds: xr_typing.Dataset | None = None

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

        # Get spatial dimensions - assume AIFS grid_point format
        if "grid_point" in self.ds.dims:
            # AIFS-compatible format with flattened grid points
            self.spatial_dims = ("grid_point",)
            self.spatial_shape = (self.ds.sizes["grid_point"],)
        else:
            raise ValueError("Dataset must be in AIFS format with grid_point dimension")

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
    ) -> xr_typing.Dataset:
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
        device: str | torch.device | None = None,
        use_fp16: bool | None = None,
        runner=None,
        use_forcing_pipeline: bool = True,
    ) -> torch.Tensor:
        """
        Convert Xarray dataset to AIFS tensor format.

        This method uses SimpleRunner.prepare_input_tensor() to properly handle
        forcing variable computation (8 trigonometric + insolation). The runner
        is required for AIFS format data to ensure correct forcing computation.

        Args:
            data: Xarray dataset
            batch_size: Batch size (for creating batches from time series)
            normalize: Whether to normalize the data
            device: Device to move tensor to (auto-detected when None)
            use_fp16: Whether to use FP16 (auto-enabled for CUDA/MPS when None)
            runner: SimpleRunner instance required for AIFS format data
            use_forcing_pipeline: Whether to use SimpleRunner pipeline (default: True)

        Returns:
            Tensor ready for AIFS model input [batch, time, ensemble, grid_points, vars]
            Shape: [batch, time, ensemble=1, grid_points, 103]
            Contains 103 variables: 94 physics + 9 forcings computed by SimpleRunner

        Raises:
            ValueError: If runner is None for AIFS format data
            ValueError: If data is not in AIFS format (grid_point dimension required)
        """
        print("Converting to AIFS tensor format...")

        target_device = resolve_device(device)
        configure_device_for_max_perf(target_device)
        if use_fp16 is None:
            use_fp16 = supports_amp(target_device)
        tensor_dtype = torch.float16 if use_fp16 else torch.float32

        # Validate requirements for AIFS format
        if runner is None:
            raise ValueError(
                "SimpleRunner instance is required for AIFS format data. "
                "The runner handles forcing variable computation (9 variables: "
                "cos/sin lat/lon/time/julian_day/local_time + insolation). "
                "Pass a runner instance to to_aifs_tensor()."
            )

        print("   Using SimpleRunner.prepare_input_tensor()...")

        try:
            import pandas as pd

            # Extract physics variables only (94 variables)
            # SimpleRunner will add forcing variables internally
            fields = {}
            for var in ALL_AIFS_VARIABLES:
                if var in data.data_vars:
                    # Shape: [time, grid_points]
                    fields[var] = data[var].values
                else:
                    raise ValueError(
                        f"Required variable '{var}' not found in dataset. "
                        f"Available: {list(data.data_vars.keys())[:20]}..."
                    )

            # Get date from dataset (use first timestep)
            date = pd.Timestamp(data.time.values[0]).to_pydatetime()

            # Create input_state dict as expected by SimpleRunner
            input_state = {
                "date": date,
                "fields": fields,
                # Lat/lon will be added by SimpleRunner if not provided
            }

            # Initialize forcing inputs if not already done
            # These are normally set up during the first run() call
            if not hasattr(runner, "constant_forcings_inputs"):
                runner.constant_forcings_inputs = runner.create_constant_forcings_inputs(
                    input_state
                )
            if not hasattr(runner, "dynamic_forcings_inputs"):
                runner.dynamic_forcings_inputs = runner.create_dynamic_forcings_inputs(input_state)

            # Let SimpleRunner prepare the tensor with all forcings
            # Returns: [multi_step_input, number_of_features, grid_points]
            input_tensor_numpy = runner.prepare_input_tensor(input_state)

            print(f"   SimpleRunner created tensor: {input_tensor_numpy.shape}")

            # Convert to PyTorch and reshape to AIFS format
            # From: [timesteps, features, grid_points]
            # To: [batch=1, timesteps, ensemble=1, grid_points, features]
            tensor = torch.from_numpy(input_tensor_numpy).float()
            tensor = tensor.permute(0, 2, 1)  # [timesteps, grid_points, features]
            tensor = tensor.unsqueeze(0).unsqueeze(2)  # Add batch and ensemble dims

            print(
                f"   Format: [batch={tensor.shape[0]}, time={tensor.shape[1]}, "
                f"ensemble={tensor.shape[2]}, grid_points={tensor.shape[3]}, "
                f"vars={tensor.shape[4]}]"
            )

            should_log_stats = os.environ.get("AIFS_LOG_TENSOR_STATS", "false").lower() in {
                "1",
                "true",
                "yes",
            }
            max_abs_value = float(tensor.abs().max().item())

            fp16_limit = 65504.0
            force_fp32 = False

            if use_fp16 and max_abs_value > fp16_limit:
                print(
                    "⚠️  Tensor values exceed FP16 dynamic range "
                    f"(|value|={max_abs_value:.2f} > {fp16_limit}). Keeping FP32 inputs."
                )
                force_fp32 = True

            if force_fp32:
                use_fp16 = False
                tensor_dtype = torch.float32

            if should_log_stats:
                finite_mask = torch.isfinite(tensor)
                finite_ratio = finite_mask.float().mean().item()
                nan_ratio = 1.0 - finite_ratio
                print(f"   Tensor finite ratio: {finite_ratio:.6f} (nan ratio {nan_ratio:.6f})")

                finite_count = finite_mask.sum().item()
                if finite_count > 0:
                    finite_values = tensor[finite_mask]
                    finite_min = finite_values.min().item()
                    finite_max = finite_values.max().item()
                    finite_mean = finite_values.mean().item()
                    print(
                        "   Tensor finite stats: "
                        f"min={finite_min:.6f}, max={finite_max:.6f}, mean={finite_mean:.6f}"
                    )
                else:
                    print("   Tensor finite stats: no finite values available")

                nan_counts = (~finite_mask).reshape(-1, tensor.shape[-1]).sum(dim=0)
                if torch.any(nan_counts):
                    topk = min(5, tensor.shape[-1])
                    top_vals, top_idx = torch.topk(nan_counts.float(), k=topk)
                    top_idx = top_idx.tolist()
                    top_vals = top_vals.tolist()
                    print("   Vars with most NaNs (index -> count):")
                    for var_index, count in zip(top_idx, top_vals):
                        print(f"      [{var_index}] -> {int(count)}")

            # Move to device and convert to FP16 if needed
            tensor = tensor.to(target_device)
            tensor = tensor.to(tensor_dtype)

            return tensor

        except Exception as e:
            raise RuntimeError(f"Failed to prepare AIFS tensor using SimpleRunner: {e}") from e

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
        # Test with the real ECMWF dataset
        test_path = "data/real_ecmwf_latest.zarr"

        if not Path(test_path).exists():
            print(f"Real ECMWF dataset not found: {test_path}")
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

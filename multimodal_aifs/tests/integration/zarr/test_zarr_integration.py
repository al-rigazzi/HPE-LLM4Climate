#!/usr/bin/env python3
"""
Direct test of Zarr integration with AIFS multimodal system.
This script tests the complete pipeline: Zarr → 5D tensor → AIFS ready format.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("Testing Zarr Integration with AIFS Multimodal System")
print("=" * 60)

try:
    import xarray as xr
    import zarr

    print(f"Dependencies loaded successfully")
    print(f"   📦 zarr version: {zarr.__version__}")
    print(f"   📦 xarray version: {xr.__version__}")
    print(f"   📦 torch version: {torch.__version__}")
except ImportError as e:
    print(f"Missing dependencies: {e}")
    sys.exit(1)


def test_zarr_to_aifs_pipeline(zarr_dataset_path):
    """Test the complete Zarr → AIFS pipeline."""

    print("\nStep 1: Loading Zarr dataset")
    print("-" * 30)

    zarr_path = zarr_dataset_path
    if not Path(zarr_path).exists():
        print(f"Test dataset not found: {zarr_path}")
        pytest.fail(f"Test dataset not found: {zarr_path}")

    try:
        # Load dataset
        ds = xr.open_zarr(zarr_path)
        print(f"Dataset loaded: {zarr_path}")
        print(f"   Dimensions: {dict(ds.sizes)}")
        print(f"   🔢 Variables: {list(ds.data_vars.keys())}")

        # Get time range
        time_values = ds.time.values
        print(f"   📅 Time range: {str(time_values[0])[:19]} to {str(time_values[-1])[:19]}")

    except Exception as e:
        print(f"Failed to load dataset: {e}")
        pytest.fail(f"Failed to load dataset: {e}")

    print("\nStep 2: Converting to AIFS tensor format")
    print("-" * 30)

    try:
        # Select a subset of data (first 4 timesteps)
        subset = ds.isel(time=slice(0, 4))
        print(f"Selected subset: {dict(subset.sizes)}")

        # Get variables
        variables = list(subset.data_vars.keys())
        print(f"   🔢 Variables: {variables}")

        # Convert to numpy arrays and stack
        arrays = []
        is_aifs_format = "grid_point" in subset.dims or "grid_points" in subset.dims

        # Handle single "data" variable from conftest
        if len(variables) == 1 and variables[0] == "data":
            var_data = subset["data"].values
            if var_data.ndim == 5:
                # Conftest format: [batch, time, ensemble, grid_points, variables]
                # Extract the data and reshape to [time, variables, grid_points]
                var_data = var_data[0, :, 0, :, :]  # Remove batch and ensemble dims
                arrays.append(var_data)
                print(f"   data: {var_data.shape} (conftest AIFS format)")
                is_aifs_format = True
            else:
                raise ValueError(
                    f"Conftest data format: Variable data has unexpected shape: {var_data.shape}"
                )
        else:
            # Original logic for individual variables
            for var in variables:
                var_data = subset[var].values
                if is_aifs_format:
                    # AIFS format: [time, grid_points]
                    if var_data.ndim == 2:
                        arrays.append(var_data)
                        print(f"   {var}: {var_data.shape} (AIFS grid_point format)")
                    else:
                        raise ValueError(
                            f"AIFS format: Variable {var} has unexpected shape: {var_data.shape}"
                        )
                else:
                    # Lat/lon format: [time, lat, lon]
                    if var_data.ndim == 3:
                        arrays.append(var_data)
                        print(f"   {var}: {var_data.shape} (lat/lon format)")
                    else:
                        raise ValueError(
                            f"Lat/lon format: Variable {var} has unexpected shape: {var_data.shape}"
                        )

        # Stack variables
        if len(variables) == 1 and variables[0] == "data":
            # Conftest format: already in correct shape [time, variables, grid_points]
            stacked = arrays[0]
            print(f"Conftest data shape: {stacked.shape} (AIFS format)")
        elif is_aifs_format:
            # AIFS format: [time, variables, grid_points]
            stacked = np.stack(arrays, axis=1)
            print(f"Stacked shape: {stacked.shape} (AIFS format)")
        else:
            # Lat/lon format: [time, variables, lat, lon]
            stacked = np.stack(arrays, axis=1)
            print(f"Stacked shape: {stacked.shape} (lat/lon format)")

        # Convert to PyTorch tensor
        tensor = torch.from_numpy(stacked).float()

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        print(f"Final AIFS tensor shape: {tensor.shape}")
        if len(variables) == 1 and variables[0] == "data":
            print(
                f"   Format: [batch={tensor.shape[0]}, time={tensor.shape[1]}, "
                f"vars={tensor.shape[2]}, grid_points={tensor.shape[3]}] (conftest)"
            )
        elif is_aifs_format:
            print(
                f"   Format: [batch={tensor.shape[0]}, time={tensor.shape[1]}, "
                f"vars={tensor.shape[2]}, grid_points={tensor.shape[3]}]"
            )
        else:
            print(
                f"   Format: [batch={tensor.shape[0]}, time={tensor.shape[1]}, "
                f"vars={tensor.shape[2]}, height={tensor.shape[3]}, width={tensor.shape[4]}]"
            )

    except Exception as e:
        print(f"Failed tensor conversion: {e}")
        pytest.fail(f"Failed tensor conversion: {e}")

    print("\nStep 3: AIFS Integration Check")
    print("-" * 30)

    try:
        # Check tensor properties
        print(f"Tensor dtype: {tensor.dtype}")
        print(f"Tensor device: {tensor.device}")
        print(f"Memory usage: {tensor.numel() * tensor.element_size() / 1024:.2f} KB")

        # Check for NaN/Inf values
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        print(f"Data quality: NaN={has_nan}, Inf={has_inf}")

        # Basic statistics
        print(f"Data range: [{tensor.min().item():.3f}, {tensor.max().item():.3f}]")
        print(f"Data mean: {tensor.mean().item():.3f}")
        print(f"Data std: {tensor.std().item():.3f}")

        # Simulate AIFS tokenizer input
        batch_size, time_steps, num_vars = tensor.shape[0], tensor.shape[1], tensor.shape[2]
        if len(variables) == 1 and variables[0] == "data":
            grid_points = tensor.shape[3]
            print(f"\nAIFS Multimodal Integration (Conftest):")
            print(f"   Batch size: {batch_size} (ready for processing)")
            print(f"   Time steps: {time_steps} (temporal sequence)")
            print(f"   Variables: {num_vars} (climate features)")
            print(f"   Grid points: {grid_points} (flattened spatial)")
            print(f"   Total features: {num_vars * grid_points} per timestep")
            print(
                f"   Tensor format: [B={batch_size}, T={time_steps}, V={num_vars}, G={grid_points}] (conftest)"
            )
        elif is_aifs_format:
            grid_points = tensor.shape[3]
            print(f"\nAIFS Multimodal Integration:")
            print(f"   Batch size: {batch_size} (ready for processing)")
            print(f"   Time steps: {time_steps} (temporal sequence)")
            print(f"   Variables: {num_vars} (climate features)")
            print(f"   Grid points: {grid_points} (flattened spatial)")
            print(f"   Total features: {num_vars * grid_points} per timestep")
            print(
                f"   Tensor format: [B={batch_size}, T={time_steps}, V={num_vars}, G={grid_points}]"
            )
        else:
            height, width = tensor.shape[3], tensor.shape[4]
            print(f"\nAIFS Multimodal Integration:")
            print(f"   Batch size: {batch_size} (ready for processing)")
            print(f"   Time steps: {time_steps} (temporal sequence)")
            print(f"   Variables: {num_vars} (climate features)")
            print(f"   Spatial: {height}x{width} (grid resolution)")
            print(f"   Total features: {num_vars * height * width} per timestep")
            print(
                f"   Tensor format: [B={batch_size}, T={time_steps}, V={num_vars}, H={height}, W={width}]"
            )

    except Exception as e:
        print(f"Integration check failed: {e}")
        pytest.fail(f"Integration check failed: {e}")

    print("\nSuccess! Zarr → AIFS Pipeline Complete")
    print("=" * 60)
    if is_aifs_format:
        print("The zarr dataset can be successfully loaded and converted")
        print("to the AIFS tensor format [B,T,V,grid_points] expected by AIFS models.")
    else:
        print("The zarr dataset can be successfully loaded and converted")
        print("to the 5D tensor format [B,T,V,H,W] expected by AIFS models.")
    print("\nNext steps:")
    print("Use ZarrClimateLoader class for production workflows")
    print("Integrate with AIFS TimeSeries tokenizer")
    print("Feed tokenized data to Llama 3-8B model")
    print("Apply cross-attention fusion for multimodal processing")
    # Test passes by reaching this point without failures


if __name__ == "__main__":
    success = test_zarr_to_aifs_pipeline()
    sys.exit(0 if success else 1)

#!/usr/bin/env python
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
Test script demonstrating proper AIFS input preparation using SimpleRunner.

This script shows the correct approach: use SimpleRunner.prepare_input_tensor()
directly instead of manually computing forcing variables.
"""

import sys

sys.path.insert(0, "aifs-single-1.1")

import pandas as pd
import torch
import xarray as xr

from multimodal_aifs.constants import ALL_AIFS_VARIABLES


def test_input_preparation_with_simplerunner():
    """Test AIFS input preparation using SimpleRunner.prepare_input_tensor()."""
    print("=" * 80)
    print("AIFS Input Preparation with SimpleRunner Test")
    print("=" * 80)
    print()

    # Step 1: Load dataset
    print("Step 1: Loading dataset...")
    zarr_path = "data/real_ecmwf_latest.zarr"
    dataset = xr.open_zarr(zarr_path)
    print(f"✓ Dataset loaded: {dict(dataset.dims)}")
    print(f"  Variables: {len(dataset.data_vars)}")
    print()

    # Step 2: Load SimpleRunner
    print("Step 2: Loading AIFS SimpleRunner...")
    from anemoi.inference.runners.simple import SimpleRunner

    checkpoint = {"huggingface": "ecmwf/aifs-single-1.1"}
    runner = SimpleRunner(checkpoint, device="cpu")
    print(f"✓ SimpleRunner loaded")
    print(f"  Number of input features: {runner.checkpoint.number_of_input_features}")
    print()

    # Step 3: Prepare fields dict with 94 physics variables
    print("Step 3: Extracting 94 physics variables...")
    fields = {}
    for var in ALL_AIFS_VARIABLES:
        if var in dataset.data_vars:
            fields[var] = dataset[var].values
        else:
            print(f"  Warning: Variable '{var}' not found in dataset")
    print(f"✓ Extracted {len(fields)} physics variables")
    print()

    # Step 4: Create input_state
    print("Step 4: Creating input_state dict...")
    date = pd.Timestamp(dataset.time.values[0]).to_pydatetime()
    input_state = {
        "date": date,
        "fields": fields,
    }
    print(f"✓ input_state created")
    print(f"  Date: {date}")
    print(f"  Fields: {len(input_state['fields'])} variables")
    print()

    # Step 5: Initialize forcing inputs
    print("Step 5: Initializing forcing inputs...")
    if not hasattr(runner, "constant_forcings_inputs"):
        runner.constant_forcings_inputs = runner.create_constant_forcings_inputs(input_state)
    if not hasattr(runner, "dynamic_forcings_inputs"):
        runner.dynamic_forcings_inputs = runner.create_dynamic_forcings_inputs(input_state)
    print(f"✓ Forcing inputs initialized")
    print()

    # Step 6: Prepare input tensor
    print("Step 6: Calling SimpleRunner.prepare_input_tensor()...")
    print("  (SimpleRunner will add 9 forcing variables: 8 trig + insolation)")
    try:
        input_tensor_numpy = runner.prepare_input_tensor(input_state)
        print(f"✓ SimpleRunner created tensor: {input_tensor_numpy.shape}")
        print(f"  Format: [timesteps, features, grid_points]")
        print()

        # Step 7: Convert to PyTorch and reshape to AIFS format
        print("Step 7: Reshaping to AIFS format...")
        tensor = torch.from_numpy(input_tensor_numpy).float()
        tensor = tensor.permute(0, 2, 1)  # [timesteps, grid_points, features]
        tensor = tensor.unsqueeze(0).unsqueeze(2)  # Add batch and ensemble dims
        print(f"✓ Tensor reshaped: {tensor.shape}")
        print(f"  Format: [batch, time, ensemble, grid_points, vars]")
        print()

        # Success summary
        print("=" * 80)
        print("✓✓✓ SUCCESS ✓✓✓")
        print("=" * 80)
        print(f"Final tensor shape: {tensor.shape}")
        print(f"  Batch: {tensor.shape[0]}")
        print(f"  Time: {tensor.shape[1]}")
        print(f"  Ensemble: {tensor.shape[2]}")
        print(f"  Grid points: {tensor.shape[3]}")
        print(f"  Variables: {tensor.shape[4]} (94 physics + 9 forcings)")
        print()
        print("Key insight: SimpleRunner.prepare_input_tensor() handles all forcing")
        print("computation automatically. Just provide the 94 physics variables!")
        print()
        return True

    except Exception as e:
        print()
        print("=" * 80)
        print("✗✗✗ ERROR ✗✗✗")
        print("=" * 80)
        print(f"Error preparing tensor: {e}")
        import traceback

        traceback.print_exc()
        return False

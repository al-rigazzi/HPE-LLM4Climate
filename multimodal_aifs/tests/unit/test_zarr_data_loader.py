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
"""Unit tests for ``multimodal_aifs.utils.zarr_data_loader``."""

# pylint: disable=redefined-outer-name

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from multimodal_aifs.utils.zarr_data_loader import ZarrClimateLoader


@pytest.fixture()
def sample_zarr_dataset(tmp_path):
    """Create a minimal grid_point-format Zarr store for loader tests."""

    times = np.array(
        [
            np.datetime64("2024-01-01T00:00:00"),
            np.datetime64("2024-01-01T06:00:00"),
            np.datetime64("2024-01-01T12:00:00"),
        ]
    )
    grid_points = np.arange(8)
    rng = np.random.default_rng(42)

    dataset = xr.Dataset(
        {
            "t2m": (("time", "grid_point"), rng.normal(size=(len(times), len(grid_points)))),
            "u10": (("time", "grid_point"), rng.normal(size=(len(times), len(grid_points)))),
        },
        coords={"time": times, "grid_point": grid_points},
    )

    zarr_path = tmp_path / "sample.zarr"
    dataset.to_zarr(zarr_path)
    return zarr_path, dataset


def test_loader_initialization_and_time_range(sample_zarr_dataset):
    """Loader should expose variables and filter time windows."""

    path, dataset = sample_zarr_dataset
    loader = ZarrClimateLoader(str(path))

    assert set(loader.available_variables) == {"t2m", "u10"}

    start = str(dataset.time.values[0])
    end = str(dataset.time.values[1])
    subset = loader.load_time_range(start_time=start, end_time=end, variables=["t2m"])
    assert set(subset.data_vars) == {"t2m"}
    assert subset.sizes["time"] == 2


def test_loader_rejects_unknown_variables(sample_zarr_dataset):
    """Requesting missing variables should raise a ValueError."""

    path, _ = sample_zarr_dataset
    loader = ZarrClimateLoader(str(path))

    with pytest.raises(ValueError):
        loader.load_time_range(start_time=None, end_time=None, variables=["missing"])


def test_loader_spatial_region_on_grid_points(sample_zarr_dataset):
    """Spatial selection should no-op yet still return a dataset."""

    path, dataset = sample_zarr_dataset
    loader = ZarrClimateLoader(str(path))

    subset = loader.load_spatial_region(
        lat_range=(-30.0, 30.0),
        lon_range=(-60.0, 60.0),
        time_range=(str(dataset.time.values[0]), str(dataset.time.values[-1])),
    )
    assert subset.dims["time"] == dataset.dims["time"]
    assert set(subset.data_vars) == {"t2m", "u10"}

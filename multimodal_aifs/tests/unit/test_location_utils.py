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
"""Unit tests for ``multimodal_aifs.utils.location_utils``."""

from __future__ import annotations

import math

import pytest
import torch

from multimodal_aifs.utils.location_utils import (
    EARTH_RADIUS_KM,
    GridUtils,
    LocationUtils,
    SpatialEncoder,
)


def test_basic_geodesic_helpers():
    """Haversine, bearing, and destination helpers should be numerically stable."""

    # Quarter of Earth's circumference between (0, 0) and (0, 90)
    distance = LocationUtils.haversine_distance(0.0, 0.0, 0.0, 90.0)
    assert pytest.approx(distance, rel=1e-3) == math.pi * EARTH_RADIUS_KM / 2

    # Moving due east from the equator keeps latitude fixed and heading ≈ 90°
    bearing = LocationUtils.bearing(0.0, 0.0, 0.0, 90.0)
    assert pytest.approx(bearing, abs=1e-3) == 90.0

    lat, lon = LocationUtils.destination_point(0.0, 0.0, 90.0, 1000.0)
    assert abs(lat) < 1e-6  # stays on equator
    assert 0.0 < lon < 10.0  # roughly 9° for 1000 km

    norm_lat, norm_lon = LocationUtils.normalize_coordinates(200.0, 540.0)
    assert norm_lat == 90.0
    assert norm_lon == -180.0
    assert LocationUtils.validate_coordinates(norm_lat, norm_lon)


def test_grid_utils_coordinate_mappings_and_masks():
    """GridUtils should convert, crop, and weight spatial tensors consistently."""

    grid = GridUtils(lat_range=(-10.0, 10.0), lon_range=(-20.0, 20.0), resolution=1.0)
    lat_idx, lon_idx = grid.coordinates_to_indices(0.0, 0.0)
    lat, lon = grid.indices_to_coordinates(lat_idx, lon_idx)
    assert lat == 0.0
    assert lon == 0.0

    # Region extraction on both numpy and torch tensors
    torch_data = torch.arange(grid.lat_size * grid.lon_size, dtype=torch.float32).view(
        1, 1, grid.lat_size, grid.lon_size
    )
    region = grid.extract_region(torch_data, 0.0, 0.0, region_size_km=500.0)
    assert region.shape[-2] <= torch_data.shape[-2]
    assert region.shape[-1] <= torch_data.shape[-1]

    mask = grid.create_distance_mask(0.0, 0.0, max_distance_km=300.0)
    assert mask.shape == (grid.lat_size, grid.lon_size)
    assert mask[lat_idx, lon_idx]
    assert not mask[0, 0]

    weights = grid.get_spatial_weights(0.0, 0.0, decay_distance_km=500.0)
    assert weights[lat_idx, lon_idx] == pytest.approx(1.0, abs=1e-6)
    assert weights[0, 0] < weights[lat_idx, lon_idx]

    info = grid.get_grid_info()
    assert info["lat_size"] == grid.lat_size
    assert info["lon_size"] == grid.lon_size


def test_spatial_encoder_outputs_are_bounded():
    """SpatialEncoder encodings should be finite and repeatable for mirrored coordinates."""

    encoder = SpatialEncoder(encoding_dim=32)
    encoding = encoder.encode_coordinates(12.5, -45.0)
    assert encoding.shape[0] == 32
    assert torch.isfinite(encoding).all()

    relative = encoder.encode_relative_position(12.5, -45.0, -12.5, 135.0)
    assert relative.shape[0] == 32
    assert torch.isfinite(relative).all()
    assert (relative != 0).any()
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
Location mask generator for spatial climate data analysis.

This module provides functionality to generate masks for specific geographic
locations including continents, regions, countries, cities, states, and bodies of water.
"""

import os
import random
from dataclasses import dataclass
from typing import Literal

import torch

from ..utils.device_utils import resolve_device


@dataclass
class Location:
    """Represents a geographic location with its mask and metadata."""

    name: str
    location_type: Literal["continent", "region", "country", "city", "state", "body_of_water"]
    mask: torch.Tensor  # Boolean mask [grid_points]
    center_lat: float
    center_lon: float
    description: str
    # Bounding box corners: (lat, lon) tuples
    bbox_nw: tuple[float, float] | None = None  # Northwest corner (max_lat, min_lon)
    bbox_se: tuple[float, float] | None = None  # Southeast corner (min_lat, max_lon)

    def format_bbox_string(self) -> str:
        """Format bounding box as human-readable string with coordinates."""
        if self.bbox_nw is None or self.bbox_se is None:
            return ""

        def format_lat(lat: float) -> str:
            direction = "N" if lat >= 0 else "S"
            return f"{abs(lat):.1f}°{direction}"

        def format_lon(lon: float) -> str:
            direction = "E" if lon >= 0 else "W"
            return f"{abs(lon):.1f}°{direction}"

        nw_lat, nw_lon = self.bbox_nw
        se_lat, se_lon = self.bbox_se
        return f"from {format_lat(se_lat)} to {format_lat(nw_lat)}, {format_lon(nw_lon)} to {format_lon(se_lon)}"


class LocationMaskGenerator:
    """
    Generates spatial masks for different geographic locations.

    This class creates boolean masks that can be applied to AIFS grid data
    to isolate specific geographic regions for analysis.
    """

    def __init__(
        self,
        grid_points: int = 542080,
        seed: int | None = None,
        device: str | torch.device | None = None,
    ):
        """
        Initialize the location mask generator.

        Args:
            grid_points: Number of grid points in the AIFS model
            seed: Random seed for reproducibility
            device: Device for tensor operations (defaults to ANALYSIS_GPU env var, then 'cuda:0')
        """
        self.grid_points = grid_points
        self.rng = random.Random(seed)

        # Resolve device from ANALYSIS_GPU environment variable
        if device is None:
            analysis_gpu = os.environ.get("ANALYSIS_GPU", "0")
            if torch.cuda.is_available():
                device = f"cuda:{analysis_gpu}"
            else:
                device = "cpu"
        self.device = resolve_device(device)

        # Pre-defined location database
        self._init_location_database()

        # Pre-compute and cache the lat/lon grid
        self._lat_grid: torch.Tensor | None = None
        self._lon_grid: torch.Tensor | None = None

    def _init_location_database(self) -> None:
        """Initialize database of known locations with approximate coordinates."""
        self.locations = {
            # Continents
            "continents": [
                ("Africa", 0.0, 20.0, "the African continent"),
                ("Europe", 50.0, 10.0, "the European continent"),
                ("Asia", 30.0, 100.0, "the Asian continent"),
                ("North America", 45.0, -100.0, "North America"),
                ("South America", -15.0, -60.0, "South America"),
                ("Australia", -25.0, 135.0, "Australia"),
                ("Antarctica", -80.0, 0.0, "Antarctica"),
            ],
            # Regions
            "regions": [
                ("Mediterranean Basin", 40.0, 15.0, "the Mediterranean Basin region"),
                ("Sahara Desert", 23.0, 10.0, "the Sahara Desert"),
                ("Amazon Rainforest", -5.0, -60.0, "the Amazon Rainforest"),
                ("Siberia", 60.0, 105.0, "Siberia"),
                ("Middle East", 28.0, 45.0, "the Middle East"),
                ("Scandinavia", 62.0, 15.0, "Scandinavia"),
                ("Caribbean", 18.0, -75.0, "the Caribbean region"),
                ("Southeast Asia", 10.0, 105.0, "Southeast Asia"),
            ],
            # Countries
            "countries": [
                ("France", 46.6, 2.2, "France"),
                ("Germany", 51.2, 10.5, "Germany"),
                ("Italy", 42.8, 12.6, "Italy"),
                ("Spain", 40.5, -3.7, "Spain"),
                ("United Kingdom", 54.0, -2.0, "the United Kingdom"),
                ("United States", 38.0, -97.0, "the United States"),
                ("Canada", 56.0, -106.0, "Canada"),
                ("Brazil", -10.0, -55.0, "Brazil"),
                ("China", 35.0, 105.0, "China"),
                ("India", 20.0, 77.0, "India"),
                ("Japan", 36.0, 138.0, "Japan"),
                ("Australia", -27.0, 133.0, "Australia"),
            ],
            # Cities
            "cities": [
                ("Paris", 48.86, 2.35, "Paris, France"),
                ("London", 51.51, -0.13, "London, United Kingdom"),
                ("New York", 40.71, -74.01, "New York City, United States"),
                ("Tokyo", 35.68, 139.69, "Tokyo, Japan"),
                ("Beijing", 39.90, 116.40, "Beijing, China"),
                ("Mumbai", 19.08, 72.88, "Mumbai, India"),
                ("São Paulo", -23.55, -46.63, "São Paulo, Brazil"),
                ("Sydney", -33.87, 151.21, "Sydney, Australia"),
                ("Dubai", 25.20, 55.27, "Dubai, United Arab Emirates"),
                ("Moscow", 55.75, 37.62, "Moscow, Russia"),
            ],
            # US States
            "states": [
                ("California", 37.0, -120.0, "California"),
                ("Texas", 31.0, -100.0, "Texas"),
                ("Florida", 28.0, -82.0, "Florida"),
                ("New York State", 43.0, -75.0, "New York State"),
                ("Alaska", 64.0, -152.0, "Alaska"),
            ],
            # Bodies of Water
            "bodies_of_water": [
                ("Atlantic Ocean", 25.0, -30.0, "the Atlantic Ocean"),
                ("Pacific Ocean", 0.0, -160.0, "the Pacific Ocean"),
                ("Indian Ocean", -20.0, 70.0, "the Indian Ocean"),
                ("Mediterranean Sea", 35.0, 18.0, "the Mediterranean Sea"),
                ("Caribbean Sea", 15.0, -75.0, "the Caribbean Sea"),
                ("Gulf of Mexico", 25.0, -90.0, "the Gulf of Mexico"),
                ("Arabian Sea", 15.0, 65.0, "the Arabian Sea"),
                ("North Sea", 56.0, 3.0, "the North Sea"),
            ],
        }

    def _create_lat_lon_grid(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create latitude and longitude grids matching AIFS grid structure.

        Returns:
            Tuple of (latitude, longitude) tensors of shape [grid_points]
        """
        # Return cached grids if available
        if self._lat_grid is not None and self._lon_grid is not None:
            return self._lat_grid, self._lon_grid

        # AIFS uses a reduced Gaussian grid
        # For now, approximate with uniform random sampling
        # This should be replaced with actual AIFS grid coordinates

        # Generate random lat/lon coordinates for each grid point
        # This is a placeholder until we have the actual AIFS grid topology
        generator = torch.Generator(device="cpu").manual_seed(42)
        lat_flat = torch.empty(self.grid_points, device="cpu").uniform_(
            -90, 90, generator=generator
        )
        lon_flat = torch.empty(self.grid_points, device="cpu").uniform_(
            -180, 180, generator=generator
        )

        # Move to target device and cache
        self._lat_grid = lat_flat.to(self.device)
        self._lon_grid = lon_flat.to(self.device)

        return self._lat_grid, self._lon_grid

    def _create_mask_for_location(
        self,
        center_lat: float,
        center_lon: float,
        radius_deg: float,
        location_type: str,
    ) -> torch.Tensor:
        """
        Create a spatial mask for a location.

        Args:
            center_lat: Center latitude
            center_lon: Center longitude
            radius_deg: Radius in degrees
            location_type: Type of location (affects radius scaling)

        Returns:
            Boolean mask tensor [grid_points]
        """
        lat_grid, lon_grid = self._create_lat_lon_grid()

        # Adjust radius based on location type
        radius_scaling = {
            "continent": 25.0,
            "region": 15.0,
            "country": 10.0,
            "state": 8.0,
            "city": 2.0,
            "body_of_water": 12.0,
        }
        actual_radius = radius_deg * radius_scaling.get(location_type, 1.0)

        # Convert to radians (using torch)
        deg_to_rad = torch.pi / 180.0
        lat1 = center_lat * deg_to_rad
        lon1 = center_lon * deg_to_rad
        lat2 = lat_grid * deg_to_rad
        lon2 = lon_grid * deg_to_rad

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        cos_lat1 = torch.cos(torch.tensor(lat1, device=self.device))
        a = torch.sin(dlat / 2) ** 2 + cos_lat1 * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
        c = 2 * torch.arcsin(torch.sqrt(a))
        distance_deg = c * (180.0 / torch.pi)

        # Create mask
        mask = distance_deg <= actual_radius

        return mask

    def _compute_bounding_box(
        self, mask: torch.Tensor
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        Compute bounding box from mask.

        Args:
            mask: Boolean mask tensor [grid_points]

        Returns:
            Tuple of (bbox_nw, bbox_se) where each is (lat, lon)
        """
        lat_grid, lon_grid = self._create_lat_lon_grid()

        # Get coordinates of masked points
        masked_lats = lat_grid[mask]
        masked_lons = lon_grid[mask]

        if len(masked_lats) == 0:
            return (0.0, 0.0), (0.0, 0.0)

        # Compute bounds
        max_lat = float(masked_lats.max().item())
        min_lat = float(masked_lats.min().item())
        max_lon = float(masked_lons.max().item())
        min_lon = float(masked_lons.min().item())

        # NW corner: max_lat, min_lon
        # SE corner: min_lat, max_lon
        bbox_nw = (max_lat, min_lon)
        bbox_se = (min_lat, max_lon)

        return bbox_nw, bbox_se

    def get_random_location(self, location_type: str | None = None) -> Location:
        """
        Generate a random location with its mask.

        Args:
            location_type: Specific type to select, or None for random

        Returns:
            Location object with mask and metadata
        """
        if location_type is None:
            location_type = self.rng.choice(list(self.locations.keys()))

        # Select random location from category
        location_data = self.rng.choice(self.locations[location_type])
        name, lat, lon, description = location_data

        # Create mask
        mask = self._create_mask_for_location(
            lat, lon, radius_deg=1.0, location_type=location_type.rstrip("s")
        )

        # Map location_type to singular form
        type_mapping = {
            "continents": "continent",
            "regions": "region",
            "countries": "country",
            "cities": "city",
            "states": "state",
            "bodies_of_water": "body_of_water",
        }

        # Compute bounding box from mask
        bbox_nw, bbox_se = self._compute_bounding_box(mask)

        return Location(
            name=name,
            location_type=type_mapping[location_type],
            mask=mask,
            center_lat=lat,
            center_lon=lon,
            description=description,
            bbox_nw=bbox_nw,
            bbox_se=bbox_se,
        )

    def get_location_by_name(self, name: str) -> Location | None:
        """
        Get a specific location by name.

        Args:
            name: Name of the location

        Returns:
            Location object or None if not found
        """
        for location_type, locations in self.locations.items():
            for loc_data in locations:
                if loc_data[0].lower() == name.lower():
                    loc_name, lat, lon, description = loc_data

                    type_mapping = {
                        "continents": "continent",
                        "regions": "region",
                        "countries": "country",
                        "cities": "city",
                        "states": "state",
                        "bodies_of_water": "body_of_water",
                    }

                    mask = self._create_mask_for_location(
                        lat, lon, radius_deg=1.0, location_type=type_mapping[location_type]
                    )

                    # Compute bounding box from mask
                    bbox_nw, bbox_se = self._compute_bounding_box(mask)

                    return Location(
                        name=loc_name,
                        location_type=type_mapping[location_type],
                        mask=mask,
                        center_lat=lat,
                        center_lon=lon,
                        description=description,
                        bbox_nw=bbox_nw,
                        bbox_se=bbox_se,
                    )

        return None

    def list_available_locations(self) -> dict[str, list[str]]:
        """
        Get list of all available locations by category.

        Returns:
            Dictionary mapping location types to location names
        """
        return {loc_type: [loc[0] for loc in locs] for loc_type, locs in self.locations.items()}

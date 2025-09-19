"""
Location Utilities

This module provides utilities for handling geographic locations and spatial data
for AIFS multimodal analysis, including coordinate transformations, distance calculations,
and spatial region extraction.
"""

import math

import numpy as np
import torch

# Earth constants
EARTH_RADIUS_KM = 6371.0
EARTH_CIRCUMFERENCE_KM = 2 * math.pi * EARTH_RADIUS_KM

# Common coordinate systems
WGS84_SEMI_MAJOR = 6378137.0  # meters
WGS84_FLATTENING = 1 / 298.257223563

# Grid resolution constants
COMMON_RESOLUTIONS = {
    "0.25": 0.25,  # ECMWF ERA5
    "0.5": 0.5,  # Common reanalysis
    "1.0": 1.0,  # Coarse resolution
    "2.5": 2.5,  # Very coarse
}


class LocationUtils:
    """
    Utility class for handling geographic locations and spatial operations.
    """

    @staticmethod
    def degrees_to_radians(degrees: float | torch.Tensor) -> float | torch.Tensor:
        """Convert degrees to radians."""
        return degrees * math.pi / 180.0

    @staticmethod
    def radians_to_degrees(radians: float | torch.Tensor) -> float | torch.Tensor:
        """Convert radians to degrees."""
        return radians * 180.0 / math.pi

    @staticmethod
    def haversine_distance(
        lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float | torch.Tensor:
        """
        Calculate the great circle distance between two points on Earth.

        Args:
            lat1, lon1: First point coordinates (degrees)
            lat2, lon2: Second point coordinates (degrees)

        Returns:
            Distance in kilometers
        """
        # Convert to radians
        lat1_rad = LocationUtils.degrees_to_radians(lat1)
        lon1_rad = LocationUtils.degrees_to_radians(lon1)
        lat2_rad = LocationUtils.degrees_to_radians(lat2)
        lon2_rad = LocationUtils.degrees_to_radians(lon2)

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.asin(math.sqrt(a))

        return EARTH_RADIUS_KM * c

    @staticmethod
    def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float | torch.Tensor:
        """
        Calculate the initial bearing from point 1 to point 2.

        Args:
            lat1, lon1: First point coordinates (degrees)
            lat2, lon2: Second point coordinates (degrees)

        Returns:
            Bearing in degrees (0-360)
        """
        lat1_rad = LocationUtils.degrees_to_radians(lat1)
        lat2_rad = LocationUtils.degrees_to_radians(lat2)
        dlon_rad = LocationUtils.degrees_to_radians(lon2 - lon1)

        y = math.sin(dlon_rad) * math.cos(lat2_rad)
        x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(
            lat2_rad
        ) * math.cos(dlon_rad)

        bearing_rad = math.atan2(y, x)
        bearing_deg = LocationUtils.radians_to_degrees(bearing_rad)

        return (bearing_deg + 360) % 360

    @staticmethod
    def destination_point(
        lat: float, lon: float, bearing: float, distance_km: float
    ) -> tuple[float | torch.Tensor, float | torch.Tensor]:
        """
        Calculate destination point given start point, bearing, and distance.

        Args:
            lat, lon: Start point coordinates (degrees)
            bearing: Bearing in degrees
            distance_km: Distance in kilometers

        Returns:
            Destination latitude and longitude (degrees)
        """
        lat_rad = LocationUtils.degrees_to_radians(lat)
        lon_rad = LocationUtils.degrees_to_radians(lon)
        bearing_rad = LocationUtils.degrees_to_radians(bearing)

        angular_distance = distance_km / EARTH_RADIUS_KM

        lat2_rad = math.asin(
            math.sin(lat_rad) * math.cos(angular_distance)
            + math.cos(lat_rad) * math.sin(angular_distance) * math.cos(bearing_rad)
        )

        lon2_rad = lon_rad + math.atan2(
            math.sin(bearing_rad) * math.sin(angular_distance) * math.cos(lat_rad),
            math.cos(angular_distance) - math.sin(lat_rad) * math.sin(lat2_rad),
        )

        lat2 = LocationUtils.radians_to_degrees(lat2_rad)
        lon2 = LocationUtils.radians_to_degrees(lon2_rad)

        # Normalize longitude
        lon2 = ((lon2 + 540) % 360) - 180

        return lat2, lon2

    @staticmethod
    def normalize_coordinates(lat: float, lon: float) -> tuple[float, float]:
        """
        Normalize coordinates to standard ranges.

        Args:
            lat: Latitude (any range)
            lon: Longitude (any range)

        Returns:
            Normalized latitude (-90 to 90) and longitude (-180 to 180)
        """
        # Normalize latitude to [-90, 90]
        lat = max(-90, min(90, lat))

        # Normalize longitude to [-180, 180]
        lon = ((lon + 540) % 360) - 180

        return lat, lon

    @staticmethod
    def validate_coordinates(lat: float, lon: float) -> bool:
        """
        Validate if coordinates are within valid ranges.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            True if coordinates are valid
        """
        return -90 <= lat <= 90 and -180 <= lon <= 180


class GridUtils:
    """
    Utility class for handling gridded climate data operations.
    """

    def __init__(
        self,
        lat_range: tuple[float, float] = (-90, 90),
        lon_range: tuple[float, float] = (-180, 180),
        resolution: float = 0.25,
    ):
        """
        Initialize grid utilities.

        Args:
            lat_range: Latitude range (min, max)
            lon_range: Longitude range (min, max)
            resolution: Grid resolution in degrees
        """
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.resolution = resolution

        # Calculate grid dimensions
        self.lat_size = int((lat_range[1] - lat_range[0]) / resolution) + 1
        self.lon_size = int((lon_range[1] - lon_range[0]) / resolution) + 1

        # Create coordinate arrays
        self.lats = np.linspace(lat_range[0], lat_range[1], self.lat_size)
        self.lons = np.linspace(lon_range[0], lon_range[1], self.lon_size)

        # Create coordinate grids
        self.lat_grid, self.lon_grid = np.meshgrid(self.lats, self.lons, indexing="ij")

    def coordinates_to_indices(self, lat: float, lon: float) -> tuple[int, int]:
        """
        Convert coordinates to grid indices.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Grid indices (lat_idx, lon_idx)
        """
        lat_idx = int(round((lat - self.lat_range[0]) / self.resolution))
        lon_idx = int(round((lon - self.lon_range[0]) / self.resolution))

        # Clamp to valid ranges
        lat_idx = max(0, min(self.lat_size - 1, lat_idx))
        lon_idx = max(0, min(self.lon_size - 1, lon_idx))

        return lat_idx, lon_idx

    def indices_to_coordinates(self, lat_idx: int, lon_idx: int) -> tuple[float, float]:
        """
        Convert grid indices to coordinates.

        Args:
            lat_idx: Latitude index
            lon_idx: Longitude index

        Returns:
            Coordinates (lat, lon)
        """
        lat = self.lat_range[0] + lat_idx * self.resolution
        lon = self.lon_range[0] + lon_idx * self.resolution

        return lat, lon

    def extract_region(
        self,
        data: np.ndarray | torch.Tensor,
        center_lat: float,
        center_lon: float,
        region_size_km: float,
    ) -> np.ndarray | torch.Tensor:
        """
        Extract a square region around a center point.

        Args:
            data: Gridded data to extract from
            center_lat: Center latitude
            center_lon: Center longitude
            region_size_km: Size of region in kilometers

        Returns:
            Extracted region data
        """
        # Convert region size to degrees (approximate)
        lat_degrees = region_size_km / (111.32)  # 1 degree lat ≈ 111.32 km
        lon_degrees = region_size_km / (111.32 * abs(math.cos(math.radians(center_lat))))

        # Calculate region bounds
        lat_min = center_lat - lat_degrees / 2
        lat_max = center_lat + lat_degrees / 2
        lon_min = center_lon - lon_degrees / 2
        lon_max = center_lon + lon_degrees / 2

        # Convert to indices
        lat_min_idx, lon_min_idx = self.coordinates_to_indices(lat_min, lon_min)
        lat_max_idx, lon_max_idx = self.coordinates_to_indices(lat_max, lon_max)

        # Ensure proper ordering
        lat_min_idx, lat_max_idx = min(lat_min_idx, lat_max_idx), max(lat_min_idx, lat_max_idx)
        lon_min_idx, lon_max_idx = min(lon_min_idx, lon_max_idx), max(lon_min_idx, lon_max_idx)

        # Extract region
        if isinstance(data, torch.Tensor):
            return data[..., lat_min_idx : lat_max_idx + 1, lon_min_idx : lon_max_idx + 1]

        return data[..., lat_min_idx : lat_max_idx + 1, lon_min_idx : lon_max_idx + 1]

    def create_distance_mask(
        self, center_lat: float, center_lon: float, max_distance_km: float
    ) -> np.ndarray:
        """
        Create a distance mask from a center point.

        Args:
            center_lat: Center latitude
            center_lon: Center longitude
            max_distance_km: Maximum distance in kilometers

        Returns:
            Boolean mask array
        """
        distances = np.zeros_like(self.lat_grid)

        for i in range(self.lat_size):
            for j in range(self.lon_size):
                dist = LocationUtils.haversine_distance(
                    center_lat, center_lon, self.lat_grid[i, j], self.lon_grid[i, j]
                )
                distances[i, j] = dist

        return distances <= max_distance_km

    def get_spatial_weights(
        self, center_lat: float, center_lon: float, decay_distance_km: float = 1000.0
    ) -> np.ndarray:
        """
        Create spatial weights based on distance from center.

        Args:
            center_lat: Center latitude
            center_lon: Center longitude
            decay_distance_km: Distance at which weight becomes 1/e

        Returns:
            Weight array
        """
        weights = np.zeros_like(self.lat_grid)

        for i in range(self.lat_size):
            for j in range(self.lon_size):
                dist = LocationUtils.haversine_distance(
                    center_lat, center_lon, self.lat_grid[i, j], self.lon_grid[i, j]
                )
                weights[i, j] = np.exp(-dist / decay_distance_km)

        return weights

    def get_grid_info(self) -> dict:
        """Get information about the grid."""
        return {
            "lat_range": self.lat_range,
            "lon_range": self.lon_range,
            "resolution": self.resolution,
            "lat_size": self.lat_size,
            "lon_size": self.lon_size,
            "total_points": self.lat_size * self.lon_size,
            "coverage_area_km2": self.lat_size * self.lon_size * (111.32 * self.resolution) ** 2,
        }


class SpatialEncoder:
    """
    Encoder for spatial locations to create embeddings.
    """

    def __init__(self, encoding_dim: int = 64, max_distance_km: float = 20000.0):
        """
        Initialize spatial encoder.

        Args:
            encoding_dim: Dimension of encoded representations
            max_distance_km: Maximum distance for normalization
        """
        self.encoding_dim = encoding_dim
        self.max_distance_km = max_distance_km

    def encode_coordinates(self, lat: float, lon: float) -> torch.Tensor:
        """
        Encode coordinates using sinusoidal position encoding.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Encoded coordinate tensor
        """
        # Normalize coordinates
        lat_norm = lat / 90.0  # [-1, 1]
        lon_norm = lon / 180.0  # [-1, 1]

        # Create position encoding
        encoding = torch.zeros(self.encoding_dim)

        # Use half dimensions for each coordinate
        half_dim = self.encoding_dim // 4

        # Latitude encoding
        for i in range(half_dim):
            freq = 1.0 / (10000.0 ** (2 * i / half_dim))
            encoding[i * 2] = math.sin(lat_norm * freq)
            encoding[i * 2 + 1] = math.cos(lat_norm * freq)

        # Longitude encoding
        offset = half_dim * 2
        for i in range(half_dim):
            freq = 1.0 / (10000.0 ** (2 * i / half_dim))
            encoding[offset + i * 2] = math.sin(lon_norm * freq)
            encoding[offset + i * 2 + 1] = math.cos(lon_norm * freq)

        return encoding

    def encode_relative_position(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> torch.Tensor:
        """
        Encode relative position between two points.

        Args:
            lat1, lon1: First point
            lat2, lon2: Second point

        Returns:
            Relative position encoding
        """
        # Calculate distance and bearing
        distance = LocationUtils.haversine_distance(lat1, lon1, lat2, lon2)
        bearing = LocationUtils.bearing(lat1, lon1, lat2, lon2)

        # Normalize
        if isinstance(distance, torch.Tensor):
            distance_norm = torch.clamp(distance / self.max_distance_km, 0.0, 1.0)
        else:
            distance_norm = torch.tensor(min(distance / self.max_distance_km, 1.0))

        if isinstance(bearing, torch.Tensor):
            bearing_norm = bearing / 360.0
        else:
            bearing_norm = torch.tensor(bearing / 360.0)

        # Create encoding
        encoding = torch.zeros(self.encoding_dim)

        # Distance encoding (first half)
        half_dim = self.encoding_dim // 4
        for i in range(half_dim):
            freq = 1.0 / (10000.0 ** (2 * i / half_dim))
            encoding[i * 2] = math.sin(distance_norm * freq)
            encoding[i * 2 + 1] = math.cos(distance_norm * freq)

        # Bearing encoding (second half)
        offset = half_dim * 2
        for i in range(half_dim):
            freq = 1.0 / (10000.0 ** (2 * i / half_dim))
            encoding[offset + i * 2] = math.sin(bearing_norm * freq)
            encoding[offset + i * 2 + 1] = math.cos(bearing_norm * freq)

        return encoding


def test_location_utilities():
    """Test location utility functions."""
    print("Testing Location Utilities")
    print("=" * 40)

    # Test basic distance calculation
    london_lat, london_lon = 51.5074, -0.1278
    paris_lat, paris_lon = 48.8566, 2.3522

    distance = LocationUtils.haversine_distance(london_lat, london_lon, paris_lat, paris_lon)
    print(f"London to Paris distance: {distance:.1f} km")

    bearing = LocationUtils.bearing(london_lat, london_lon, paris_lat, paris_lon)
    print(f"London to Paris bearing: {bearing:.1f}°")

    # Test grid utilities
    grid = GridUtils(lat_range=(-90, 90), lon_range=(-180, 180), resolution=1.0)
    print(f"Grid info: {grid.get_grid_info()}")

    # Test coordinate conversion
    lat_idx, lon_idx = grid.coordinates_to_indices(london_lat, london_lon)
    lat_back, lon_back = grid.indices_to_coordinates(lat_idx, lon_idx)
    print(f"London coords: {london_lat}, {london_lon}")
    print(f"Grid indices: {lat_idx}, {lon_idx}")
    print(f"Back to coords: {lat_back:.1f}, {lon_back:.1f}")

    # Test spatial encoder
    encoder = SpatialEncoder(encoding_dim=64)
    coord_encoding = encoder.encode_coordinates(london_lat, london_lon)
    print(f"Coordinate encoding shape: {coord_encoding.shape}")
    print(f"Coordinate encoding norm: {coord_encoding.norm():.4f}")

    rel_encoding = encoder.encode_relative_position(london_lat, london_lon, paris_lat, paris_lon)
    print(f"Relative encoding shape: {rel_encoding.shape}")
    print(f"Relative encoding norm: {rel_encoding.norm():.4f}")

    # Test region extraction
    synthetic_data = torch.randn(1, 10, 181, 361)  # Global 1-degree grid
    region = grid.extract_region(synthetic_data, london_lat, london_lon, 500)  # 500km region
    print(f"Extracted region shape: {region.shape}")

    print("All location tests passed!")


if __name__ == "__main__":
    test_location_utilities()

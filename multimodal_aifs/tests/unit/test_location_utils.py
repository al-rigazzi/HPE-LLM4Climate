#!/usr/bin/env python3
"""
Unit Tests for Location Utils

This test module validates geographic and spatial operation utilities
for location-aware climate analysis with AIFS.

Usage:
    python multimodal_aifs/tests/unit/test_location_utils.py
    python -m pytest multimodal_aifs/tests/unit/test_location_utils.py -v
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from multimodal_aifs.utils.location_utils import GridUtils, LocationUtils, SpatialEncoder


class TestLocationUtils(unittest.TestCase):
    """Test suite for location utilities."""

    def test_coordinate_transformations(self):
        """Test coordinate transformation functions."""
        print("\\n🌍 Testing Coordinate Transformations")

        # Test degree/radian conversions
        test_degrees = [0, 30, 45, 90, 180, 270, 360]

        for deg in test_degrees:
            rad = LocationUtils.degrees_to_radians(deg)
            back_to_deg = LocationUtils.radians_to_degrees(rad)

            self.assertAlmostEqual(deg, back_to_deg, places=10)

        print("   ✅ Degree/radian conversions")

        # Test coordinate normalization
        test_cases = [
            (91, 181, 90, -179),  # Over limits
            (-91, -181, -90, 179),  # Under limits
            (45, 90, 45, 90),  # Within limits
            (0, 0, 0, 0),  # Zero case
        ]

        for lat_in, lon_in, lat_exp, lon_exp in test_cases:
            lat_out, lon_out = LocationUtils.normalize_coordinates(lat_in, lon_in)
            self.assertAlmostEqual(lat_out, lat_exp, places=5)
            self.assertAlmostEqual(lon_out, lon_exp, places=5)

        print("   ✅ Coordinate normalization")

    def test_distance_calculations(self):
        """Test distance calculation methods."""
        print("\\n📏 Testing Distance Calculations")

        # Test known distances
        test_cases = [
            # (lat1, lon1, lat2, lon2, expected_distance_km)
            (0, 0, 0, 1, 111.32),  # 1 degree longitude at equator
            (0, 0, 1, 0, 111.32),  # 1 degree latitude
            (51.5074, -0.1278, 48.8566, 2.3522, 344),  # London to Paris (approx)
            (40.7128, -74.0060, 34.0522, -118.2437, 3944),  # NYC to LA (approx)
        ]

        for lat1, lon1, lat2, lon2, expected_km in test_cases:
            distance = LocationUtils.haversine_distance(lat1, lon1, lat2, lon2)

            # Allow 10% tolerance for approximations
            tolerance = expected_km * 0.1
            self.assertAlmostEqual(distance, expected_km, delta=tolerance)

        print(f"   ✅ Distance calculations (within tolerance)")

        # Test zero distance
        same_point_distance = LocationUtils.haversine_distance(45, 90, 45, 90)
        self.assertAlmostEqual(same_point_distance, 0, places=10)

        print("   ✅ Zero distance case")

    def test_bearing_calculations(self):
        """Test bearing calculation methods."""
        print("\\n🧭 Testing Bearing Calculations")

        # Test cardinal directions
        test_cases = [
            # (lat1, lon1, lat2, lon2, expected_bearing)
            (0, 0, 1, 0, 0),  # North
            (0, 0, 0, 1, 90),  # East
            (0, 0, -1, 0, 180),  # South
            (0, 0, 0, -1, 270),  # West
        ]

        for lat1, lon1, lat2, lon2, expected_bearing in test_cases:
            bearing = LocationUtils.bearing(lat1, lon1, lat2, lon2)

            # Normalize expected bearing
            expected_bearing = expected_bearing % 360

            # Allow small tolerance for floating point precision
            diff = abs(bearing - expected_bearing)
            if diff > 180:
                diff = 360 - diff

            self.assertLess(diff, 1.0)  # Within 1 degree

        print("   ✅ Cardinal direction bearings")

    def test_destination_point(self):
        """Test destination point calculation."""
        print("\\n🎯 Testing Destination Point Calculation")

        # Test round trip
        start_lat, start_lon = 51.5074, -0.1278  # London
        bearing = 90  # East
        distance = 100  # km

        # Calculate destination
        dest_lat, dest_lon = LocationUtils.destination_point(
            start_lat, start_lon, bearing, distance
        )

        # Calculate distance back
        calculated_distance = LocationUtils.haversine_distance(
            start_lat, start_lon, dest_lat, dest_lon
        )

        # Should be close to original distance
        self.assertAlmostEqual(calculated_distance, distance, delta=1.0)

        print(f"   ✅ Round trip: {distance}km -> {calculated_distance:.1f}km")

        # Test bearing back
        back_bearing = LocationUtils.bearing(dest_lat, dest_lon, start_lat, start_lon)
        expected_back_bearing = (bearing + 180) % 360

        bearing_diff = abs(back_bearing - expected_back_bearing)
        if bearing_diff > 180:
            bearing_diff = 360 - bearing_diff

        self.assertLess(bearing_diff, 5.0)  # Within 5 degrees (some error expected)

        print(f"   ✅ Bearing consistency: {bearing}° -> {back_bearing:.1f}°")

    def test_coordinate_validation(self):
        """Test coordinate validation."""
        print("\\n✅ Testing Coordinate Validation")

        valid_cases = [(0, 0), (90, 180), (-90, -180), (45, 90), (-45, -90)]

        invalid_cases = [(91, 0), (-91, 0), (0, 181), (0, -181), (100, 200)]

        for lat, lon in valid_cases:
            self.assertTrue(LocationUtils.validate_coordinates(lat, lon))

        for lat, lon in invalid_cases:
            self.assertFalse(LocationUtils.validate_coordinates(lat, lon))

        print("   ✅ Coordinate validation")


class TestGridUtils(unittest.TestCase):
    """Test suite for grid utilities."""

    def setUp(self):
        """Set up test grid."""
        self.grid = GridUtils(lat_range=(-90, 90), lon_range=(-180, 180), resolution=1.0)

    def test_grid_initialization(self):
        """Test grid initialization."""
        print("\\n🗂️ Testing Grid Initialization")

        self.assertEqual(self.grid.lat_size, 181)  # -90 to 90 with 1-degree resolution
        self.assertEqual(self.grid.lon_size, 361)  # -180 to 180 with 1-degree resolution
        self.assertEqual(len(self.grid.lats), 181)
        self.assertEqual(len(self.grid.lons), 361)

        print(f"   ✅ Grid dimensions: {self.grid.lat_size} x {self.grid.lon_size}")

        # Test grid info
        info = self.grid.get_grid_info()
        self.assertIn("total_points", info)
        self.assertIn("coverage_area_km2", info)

        print(f"   ✅ Total grid points: {info['total_points']:,}")

    def test_coordinate_index_conversion(self):
        """Test coordinate to index conversion."""
        print("\\n🔢 Testing Coordinate/Index Conversion")

        test_coordinates = [
            (0, 0, 90, 180),  # Center point
            (90, 180, 180, 360),  # North-East corner
            (-90, -180, 0, 0),  # South-West corner
            (45, 90, 135, 270),  # Random point
        ]

        for lat, lon, exp_lat_idx, exp_lon_idx in test_coordinates:
            lat_idx, lon_idx = self.grid.coordinates_to_indices(lat, lon)

            # Check indices are within bounds
            self.assertGreaterEqual(lat_idx, 0)
            self.assertLess(lat_idx, self.grid.lat_size)
            self.assertGreaterEqual(lon_idx, 0)
            self.assertLess(lon_idx, self.grid.lon_size)

            # Convert back to coordinates
            lat_back, lon_back = self.grid.indices_to_coordinates(lat_idx, lon_idx)

            # Should be close to original (within grid resolution)
            self.assertAlmostEqual(lat, lat_back, delta=self.grid.resolution)
            self.assertAlmostEqual(lon, lon_back, delta=self.grid.resolution)

        print("   ✅ Coordinate/index round trip")

    def test_region_extraction(self):
        """Test spatial region extraction."""
        print("\\n✂️ Testing Region Extraction")

        # Create synthetic global data
        global_data = torch.randn(5, self.grid.lat_size, self.grid.lon_size)

        # Extract region around London
        london_lat, london_lon = 51.5, -0.1
        region_size_km = 500

        extracted = self.grid.extract_region(global_data, london_lat, london_lon, region_size_km)

        # Should have same number of variables but smaller spatial dimensions
        self.assertEqual(extracted.shape[0], global_data.shape[0])
        self.assertLess(extracted.shape[1], global_data.shape[1])
        self.assertLess(extracted.shape[2], global_data.shape[2])

        print(f"   ✅ Extracted region: {global_data.shape} -> {extracted.shape}")

        # Test with numpy array
        np_data = global_data.numpy()
        np_extracted = self.grid.extract_region(np_data, london_lat, london_lon, region_size_km)

        self.assertIsInstance(np_extracted, np.ndarray)
        self.assertEqual(np_extracted.shape, extracted.shape)

        print("   ✅ NumPy array compatibility")

    def test_distance_masks(self):
        """Test distance mask creation."""
        print("\\n🎭 Testing Distance Masks")

        center_lat, center_lon = 0, 0  # Equator, prime meridian
        max_distance_km = 1000

        mask = self.grid.create_distance_mask(center_lat, center_lon, max_distance_km)

        # Mask should have same shape as grid
        self.assertEqual(mask.shape, (self.grid.lat_size, self.grid.lon_size))

        # Should be boolean
        self.assertEqual(mask.dtype, bool)

        # Center point should be included
        center_lat_idx, center_lon_idx = self.grid.coordinates_to_indices(center_lat, center_lon)
        self.assertTrue(mask[center_lat_idx, center_lon_idx])

        # Count points within distance
        points_in_range = mask.sum()
        print(f"   ✅ Points within {max_distance_km}km: {points_in_range}")

        # Should have some points but not all
        self.assertGreater(points_in_range, 0)
        self.assertLess(points_in_range, mask.size)

    def test_spatial_weights(self):
        """Test spatial weight calculation."""
        print("\\n⚖️ Testing Spatial Weights")

        center_lat, center_lon = 45, 0
        decay_distance_km = 500

        weights = self.grid.get_spatial_weights(center_lat, center_lon, decay_distance_km)

        # Weights should have same shape as grid
        self.assertEqual(weights.shape, (self.grid.lat_size, self.grid.lon_size))

        # All weights should be between 0 and 1
        self.assertGreaterEqual(weights.min(), 0)
        self.assertLessEqual(weights.max(), 1)

        # Center should have highest weight
        center_lat_idx, center_lon_idx = self.grid.coordinates_to_indices(center_lat, center_lon)
        center_weight = weights[center_lat_idx, center_lon_idx]

        # Should be close to 1 (allowing for grid discretization)
        self.assertGreater(center_weight, 0.9)

        print(f"   ✅ Center weight: {center_weight:.3f}")
        print(f"   ✅ Weight range: [{weights.min():.3f}, {weights.max():.3f}]")


class TestSpatialEncoder(unittest.TestCase):
    """Test suite for spatial encoder."""

    def setUp(self):
        """Set up spatial encoder."""
        self.encoder = SpatialEncoder(encoding_dim=64)

    def test_coordinate_encoding(self):
        """Test coordinate encoding."""
        print("\\n🔢 Testing Coordinate Encoding")

        test_coordinates = [
            (0, 0),  # Origin
            (90, 180),  # Extreme values
            (-90, -180),  # Negative extreme
            (51.5, -0.1),  # London
            (40.7, -74.0),  # New York
        ]

        for lat, lon in test_coordinates:
            encoding = self.encoder.encode_coordinates(lat, lon)

            # Check output properties
            self.assertEqual(encoding.shape, (self.encoder.encoding_dim,))
            self.assertEqual(encoding.dtype, torch.float32)

            # Encoding should be deterministic
            encoding2 = self.encoder.encode_coordinates(lat, lon)
            torch.testing.assert_close(encoding, encoding2)

        print(f"   ✅ Coordinate encodings: dim={self.encoder.encoding_dim}")

        # Test that different coordinates produce different encodings
        london_encoding = self.encoder.encode_coordinates(51.5, -0.1)
        paris_encoding = self.encoder.encode_coordinates(48.9, 2.3)

        # Should be different
        self.assertFalse(torch.allclose(london_encoding, paris_encoding))

        print("   ✅ Different coordinates produce different encodings")

    def test_relative_position_encoding(self):
        """Test relative position encoding."""
        print("\\n↔️ Testing Relative Position Encoding")

        # Test known relationships
        london_lat, london_lon = 51.5, -0.1
        paris_lat, paris_lon = 48.9, 2.3

        rel_encoding = self.encoder.encode_relative_position(
            london_lat, london_lon, paris_lat, paris_lon
        )

        # Check output properties
        self.assertEqual(rel_encoding.shape, (self.encoder.encoding_dim,))
        self.assertEqual(rel_encoding.dtype, torch.float32)

        print(f"   ✅ Relative encoding: dim={self.encoder.encoding_dim}")

        # Test symmetry (reverse direction should be different)
        rev_encoding = self.encoder.encode_relative_position(
            paris_lat, paris_lon, london_lat, london_lon
        )

        self.assertFalse(torch.allclose(rel_encoding, rev_encoding))

        print("   ✅ Directional sensitivity")

        # Test zero distance
        same_point_encoding = self.encoder.encode_relative_position(
            london_lat, london_lon, london_lat, london_lon
        )

        # Should have specific pattern for zero distance
        self.assertEqual(same_point_encoding.shape, (self.encoder.encoding_dim,))

        print("   ✅ Zero distance encoding")


def run_location_utils_tests():
    """Run all location utility tests."""
    print("🧪 Running Location Utils Tests")
    print("=" * 50)

    # Create test suites
    test_classes = [TestLocationUtils, TestGridUtils, TestSpatialEncoder]
    suite = unittest.TestSuite()

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All location utils tests passed!")
    else:
        print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_location_utils_tests()
    sys.exit(0 if success else 1)

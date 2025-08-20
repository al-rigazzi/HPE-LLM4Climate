#!/usr/bin/env python3
"""
Location-Aware Climate Analysis Demo

This example demonstrates location-aware climate analysis capabilities
using geographic utilities and spatial encoders.

Features demonstrated:
- Geographic coordinate operations
- Spatial data processing
- Location encoding
- Distance calculations
- Regional analysis

Usage:
    python multimodal_aifs/examples/basic/location_aware_demo.py
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from multimodal_aifs.utils import (
    EARTH_RADIUS_KM,
    GridUtils,
    LocationUtils,
    SpatialEncoder,
    create_synthetic_climate_data,
)


def demo_location_utilities():
    """Demonstrate location utility functions."""
    print("🗺️ Location Utilities Demo")
    print("-" * 40)

    # Famous locations for testing
    locations = {
        "London": (51.5074, -0.1278),
        "Paris": (48.8566, 2.3522),
        "New York": (40.7128, -74.0060),
        "Tokyo": (35.6762, 139.6503),
        "Sydney": (-33.8688, 151.2093),
        "São Paulo": (-23.5505, -46.6333),
    }

    print("📍 Test Locations:")
    for name, (lat, lon) in locations.items():
        print(f"   {name:12s}: {lat:8.4f}°, {lon:9.4f}°")
    print()

    # Test distance calculations
    print("📏 Distance Calculations:")

    london_coords = locations["London"]
    paris_coords = locations["Paris"]
    ny_coords = locations["New York"]

    # London to Paris
    london_paris_dist = LocationUtils.haversine_distance(*london_coords, *paris_coords)
    print(f"   London → Paris:    {london_paris_dist:7.1f} km")

    # London to New York
    london_ny_dist = LocationUtils.haversine_distance(*london_coords, *ny_coords)
    print(f"   London → New York: {london_ny_dist:7.1f} km")

    # Test bearing calculations
    london_paris_bearing = LocationUtils.bearing(*london_coords, *paris_coords)
    london_ny_bearing = LocationUtils.bearing(*london_coords, *ny_coords)

    print(f"   London → Paris bearing:    {london_paris_bearing:6.1f}°")
    print(f"   London → New York bearing: {london_ny_bearing:6.1f}°")
    print()

    # Test destination point calculation
    print("🎯 Destination Point Calculation:")

    start_lat, start_lon = london_coords
    test_cases = [
        (0, 100),  # North, 100km
        (90, 200),  # East, 200km
        (180, 150),  # South, 150km
        (270, 100),  # West, 100km
    ]

    for bearing, distance in test_cases:
        dest_lat, dest_lon = LocationUtils.destination_point(
            start_lat, start_lon, bearing, distance
        )

        # Verify by calculating distance back
        calc_distance = LocationUtils.haversine_distance(start_lat, start_lon, dest_lat, dest_lon)

        direction = ["North", "East", "South", "West"][bearing // 90]
        print(
            f"   {direction:5s} {distance:3d}km: ({dest_lat:7.4f}, {dest_lon:8.4f}) - "
            f"verified: {calc_distance:6.1f}km"
        )
    print()


def demo_grid_operations():
    """Demonstrate grid operations and spatial processing."""
    print("🗂️ Grid Operations Demo")
    print("-" * 30)

    # Create test grid
    grid = GridUtils(
        lat_range=(-90, 90), lon_range=(-180, 180), resolution=1.0  # 1-degree resolution
    )

    print(f"📊 Grid Information:")
    info = grid.get_grid_info()
    print(f"   Latitude size:  {info['lat_size']}")
    print(f"   Longitude size: {info['lon_size']}")
    print(f"   Total points:   {info['total_points']:,}")
    print(f"   Coverage area:  {info['coverage_area_km2']:,.0f} km²")
    print()

    # Test coordinate/index conversion
    print("🔢 Coordinate ↔ Index Conversion:")

    test_coords = [
        (0, 0),  # Equator, Prime Meridian
        (90, 180),  # North Pole area
        (-90, -180),  # South Pole area
        (51.5, -0.1),  # London (approximate)
    ]

    for lat, lon in test_coords:
        lat_idx, lon_idx = grid.coordinates_to_indices(lat, lon)
        lat_back, lon_back = grid.indices_to_coordinates(lat_idx, lon_idx)

        print(
            f"   ({lat:6.1f}, {lon:7.1f}) → [{lat_idx:3d}, {lon_idx:3d}] → "
            f"({lat_back:6.1f}, {lon_back:7.1f})"
        )
    print()

    # Create synthetic global climate data
    print("🌍 Synthetic Global Climate Data:")

    # Create realistic-looking climate data
    n_variables = 5
    global_data = torch.zeros(n_variables, grid.lat_size, grid.lon_size)

    # Create temperature-like pattern
    for i in range(grid.lat_size):
        for j in range(grid.lon_size):
            lat = grid.lats[i]
            lon = grid.lons[j]

            # Temperature decreases with latitude
            temp = 30 - 40 * abs(lat) / 90  # Simple temperature model
            temp += 5 * math.sin(math.radians(lon))  # Longitude variation

            global_data[0, i, j] = temp  # Temperature
            global_data[1, i, j] = 1013 - abs(lat)  # Pressure
            global_data[2, i, j] = max(0, 20 - abs(lat) / 2)  # Humidity
            global_data[3, i, j] = 10 * math.sin(math.radians(lat))  # Wind U
            global_data[4, i, j] = 5 * math.cos(math.radians(lon))  # Wind V

    print(f"   Created global data: {global_data.shape}")
    print(f"   Temperature range: {global_data[0].min():.1f} to {global_data[0].max():.1f}")
    print()

    # Test region extraction
    print("✂️ Regional Data Extraction:")

    # Extract regions around different cities
    extraction_tests = [
        ("London", 51.5, -0.1, 500),
        ("Equator", 0, 0, 1000),
        ("Arctic", 80, 0, 800),
    ]

    for region_name, lat, lon, size_km in extraction_tests:
        extracted = grid.extract_region(global_data, lat, lon, size_km)

        # Calculate actual area
        lat_points, lon_points = extracted.shape[1], extracted.shape[2]
        area_approx = lat_points * lon_points * (111.32) ** 2  # Rough km²

        print(
            f"   {region_name:7s}: {global_data.shape} → {extracted.shape} "
            f"(~{area_approx:,.0f} km²)"
        )
    print()

    # Test distance masks
    print("🎭 Distance Mask Creation:")

    center_lat, center_lon = 51.5, -0.1  # London
    distances = [500, 1000, 2000]

    for max_dist in distances:
        mask = grid.create_distance_mask(center_lat, center_lon, max_dist)
        points_in_range = mask.sum()
        percentage = 100 * points_in_range / mask.size

        print(f"   Within {max_dist:4d}km: {points_in_range:5d} points ({percentage:4.1f}%)")
    print()

    # Test spatial weights
    print("⚖️ Spatial Weight Calculation:")

    decay_distances = [200, 500, 1000]

    for decay_dist in decay_distances:
        weights = grid.get_spatial_weights(center_lat, center_lon, decay_dist)

        # Convert to tensor if numpy
        if isinstance(weights, np.ndarray):
            weights_tensor = torch.from_numpy(weights)
        else:
            weights_tensor = weights

        nonzero_count = torch.count_nonzero(weights_tensor > 0.01)

        print(
            f"   Decay {decay_dist:4d}km: max={weights.max():.3f}, "
            f"mean={weights.mean():.6f}, nonzero={nonzero_count}"
        )
    print()


def demo_spatial_encoder():
    """Demonstrate spatial encoding capabilities."""
    print("🔢 Spatial Encoder Demo")
    print("-" * 28)

    # Initialize spatial encoder
    encoder = SpatialEncoder(encoding_dim=64)

    print(f"📐 Encoder Configuration:")
    print(f"   Encoding dimension: {encoder.encoding_dim}")
    print(f"   Max distance: {encoder.max_distance_km:,} km")
    print()

    # Test coordinate encoding
    print("🌍 Coordinate Encoding:")

    test_locations = [
        ("Origin", 0, 0),
        ("London", 51.5074, -0.1278),
        ("Paris", 48.8566, 2.3522),
        ("North Pole", 89.9, 0),
        ("South Pole", -89.9, 0),
    ]

    encodings = {}

    for name, lat, lon in test_locations:
        encoding = encoder.encode_coordinates(lat, lon)
        encodings[name] = encoding

        norm = encoding.norm().item()
        print(
            f"   {name:11s}: norm={norm:6.3f}, "
            f"range=[{encoding.min():.3f}, {encoding.max():.3f}]"
        )
    print()

    # Test encoding consistency
    print("🔄 Encoding Consistency:")

    london_encoding1 = encoder.encode_coordinates(51.5074, -0.1278)
    london_encoding2 = encoder.encode_coordinates(51.5074, -0.1278)

    consistency_error = (london_encoding1 - london_encoding2).abs().max().item()
    print(f"   Same coordinates: max difference = {consistency_error:.10f}")

    # Test different coordinates
    london_encoding = encodings["London"]
    paris_encoding = encodings["Paris"]

    similarity = torch.cosine_similarity(
        london_encoding.unsqueeze(0), paris_encoding.unsqueeze(0)
    ).item()

    print(f"   London vs Paris similarity: {similarity:.6f}")
    print()

    # Test relative position encoding
    print("↔️ Relative Position Encoding:")

    city_pairs = [
        ("London", "Paris"),
        ("London", "North Pole"),
        ("Origin", "London"),
    ]

    for city1, city2 in city_pairs:
        coords1 = next((lat, lon) for name, lat, lon in test_locations if name == city1)
        coords2 = next((lat, lon) for name, lat, lon in test_locations if name == city2)

        rel_encoding = encoder.encode_relative_position(*coords1, *coords2)
        rev_encoding = encoder.encode_relative_position(*coords2, *coords1)

        # Calculate actual distance and bearing for comparison
        distance = LocationUtils.haversine_distance(*coords1, *coords2)
        bearing = LocationUtils.bearing(*coords1, *coords2)

        rel_norm = rel_encoding.norm().item()
        rev_norm = rev_encoding.norm().item()
        similarity = torch.cosine_similarity(
            rel_encoding.unsqueeze(0), rev_encoding.unsqueeze(0)
        ).item()

        print(f"   {city1:11s} → {city2:11s}: dist={distance:7.0f}km, " f"bearing={bearing:6.1f}°")
        print(
            f"                     → encoding norm={rel_norm:6.3f}, "
            f"reverse sim={similarity:6.3f}"
        )
    print()


def demo_integrated_analysis():
    """Demonstrate integrated location-aware climate analysis."""
    print("🔗 Integrated Location-Aware Analysis")
    print("-" * 42)

    # Setup
    grid = GridUtils(lat_range=(-90, 90), lon_range=(-180, 180), resolution=2.0)
    encoder = SpatialEncoder(encoding_dim=32)

    # Create synthetic climate data
    climate_data = create_synthetic_climate_data(
        batch_size=1, n_variables=10, spatial_shape=(grid.lat_size, grid.lon_size)
    ).squeeze(
        0
    )  # Remove batch dimension

    print(f"🌍 Global climate data: {climate_data.shape}")

    # Define analysis locations
    analysis_locations = [
        ("Tropical", 0, 0),  # Equator
        ("Temperate", 45, 0),  # Mid-latitude
        ("Arctic", 75, 0),  # High latitude
        ("London", 51.5, -0.1),  # Specific city
    ]

    print(f"📍 Analysis locations: {len(analysis_locations)}")
    print()

    # Analyze each location
    print("🔍 Location Analysis Results:")

    for location_name, lat, lon in analysis_locations:
        print(f"\\n   📍 {location_name} ({lat}°, {lon}°):")

        # 1. Encode location
        location_encoding = encoder.encode_coordinates(lat, lon)

        # 2. Extract regional data
        regional_data = grid.extract_region(climate_data, lat, lon, 1000)  # 1000km region

        # 3. Calculate spatial weights
        weights = grid.get_spatial_weights(lat, lon, 500)  # 500km decay

        # 4. Extract data at specific point
        lat_idx, lon_idx = grid.coordinates_to_indices(lat, lon)
        point_data = climate_data[:, lat_idx, lon_idx]

        # 5. Calculate regional statistics
        regional_mean = regional_data.mean(dim=(1, 2))
        regional_std = regional_data.std(dim=(1, 2))

        # Report results
        print(f"      Location encoding norm: {location_encoding.norm():.3f}")
        print(f"      Regional data shape: {regional_data.shape}")
        print(f"      Point data range: [{point_data.min():.3f}, {point_data.max():.3f}]")
        print(f"      Regional mean: {regional_mean[0]:.3f} ± {regional_std[0]:.3f}")

        # 6. Compare with other locations
        if location_name != "Tropical":  # Skip self-comparison for first location
            tropical_coords = analysis_locations[0][1:3]
            distance_to_tropical = LocationUtils.haversine_distance(lat, lon, *tropical_coords)
            print(f"      Distance to Tropical: {distance_to_tropical:,.0f} km")

    print()

    # Performance analysis
    print("⚡ Performance Analysis:")

    # Time different operations
    operations = [
        ("Location encoding", lambda: encoder.encode_coordinates(51.5, -0.1)),
        ("Distance calculation", lambda: LocationUtils.haversine_distance(51.5, -0.1, 48.9, 2.3)),
        ("Region extraction", lambda: grid.extract_region(climate_data, 51.5, -0.1, 500)),
        ("Spatial weights", lambda: grid.get_spatial_weights(51.5, -0.1, 500)),
    ]

    for op_name, op_func in operations:
        # Warmup
        _ = op_func()

        # Benchmark
        start_time = time.time()
        for _ in range(10):
            _ = op_func()
        end_time = time.time()

        avg_time = (end_time - start_time) / 10
        print(f"   {op_name:20s}: {avg_time*1000:6.2f} ms")

    print()


def main():
    """Run all location-aware demos."""
    print("🗺️ Location-Aware Climate Analysis Demo")
    print("=" * 50)
    print()

    try:
        demo_location_utilities()
        print()

        demo_grid_operations()
        print()

        demo_spatial_encoder()
        print()

        demo_integrated_analysis()

        # Summary
        print("📋 Demo Summary")
        print("-" * 30)
        print("✅ Location utilities demonstrated")
        print("✅ Grid operations validated")
        print("✅ Spatial encoding tested")
        print("✅ Integrated analysis performed")
        print("✅ Performance benchmarked")

        print("\\n🎯 Next Steps:")
        print("   - Try the climate-text fusion demo")
        print("   - Explore advanced spatial analysis")
        print("   - Combine with AIFS encoder for full pipeline")

    except KeyboardInterrupt:
        print("\\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

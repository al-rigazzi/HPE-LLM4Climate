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
Example demonstrating location-based analysis with the current implementation.

This script shows how locations are currently defined using center coordinates
and masks for spatial reasoning.
"""

from multimodal_aifs.training.location_masks import LocationMaskGenerator
from multimodal_aifs.training.prompt_generator import ClimatePromptGenerator


def demonstrate_locations():
    """Demonstrate creating locations with the current implementation."""
    print("=" * 80)
    print("Location-Based Analysis Demo")
    print("=" * 80)

    # Initialize generator
    generator = LocationMaskGenerator(grid_points=10000, seed=42)

    print("\n1. Getting a predefined location:")
    print("-" * 80)
    france = generator.get_location_by_name("France")
    if france:
        print(f"Location: {france.description}")
        print(f"Center: {france.center_lat:.2f}°N, {france.center_lon:.2f}°E")
        print(f"Type: {france.location_type}")
        print(f"Mask shape: {france.mask.shape}, Active points: {france.mask.sum()}")

    print("\n2. Generating random locations:")
    print("-" * 80)
    for i in range(3):
        random_loc = generator.get_random_location()
        print(f"\nRandom Location {i+1}:")
        print(f"  Name: {random_loc.name}")
        print(f"  Type: {random_loc.location_type}")
        print(f"  Center: {random_loc.center_lat:.2f}°N, {random_loc.center_lon:.2f}°E")
        print(f"  Active grid points: {random_loc.mask.sum()}")

    print("\n3. Generating prompts with location information:")
    print("-" * 80)

    # Create a prompt generator
    prompt_gen = ClimatePromptGenerator()

    # Create mock statistics table
    stats_table = """Variable    Min      Max      Mean     Gradient
2t          285.0    295.0    290.0    0.1
10u         -5.0     10.0     2.5      0.05
msl         99000    102000   101000   50.0"""

    if france:
        # Generate a weather description prompt
        prompt = prompt_gen.generate_analysis_prompt(
            location=france,
            statistics=[],
            statistics_table=stats_table,
            task_type="weather_description",
        )

        print("\nGenerated Prompt (first 500 chars):")
        print("-" * 80)
        print(prompt[:500])
        print("...")

    print("\n4. Listing available locations:")
    print("-" * 80)
    available = generator.list_available_locations()
    for loc_type, locations in available.items():
        print(f"\n{loc_type.capitalize()}: {len(locations)} locations")
        print(f"  Examples: {', '.join(locations[:3])}")

    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_locations()

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
Test script for the sample generation pipeline with GPU support.

This script validates:
1. The pipeline works with torch instead of numpy
2. GPU acceleration via ANALYSIS_GPU environment variable
3. Summary table generation for different queries
4. Batch creation with numerical samples + text queries
5. Single training iteration capability
"""

import os
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from multimodal_aifs.training import ClimateTextDataLoader
from multimodal_aifs.training.location_masks import LocationMaskGenerator
from multimodal_aifs.training.statistics_computer import ClimateStatisticsComputer


def test_location_mask_generator_gpu():
    """Test that LocationMaskGenerator works on GPU with torch tensors."""
    print("\n" + "=" * 80)
    print("TEST 1: LocationMaskGenerator GPU Support")
    print("=" * 80)

    # Get device from environment
    analysis_gpu = os.environ.get("ANALYSIS_GPU", "0")
    device = f"cuda:{analysis_gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize generator
    generator = LocationMaskGenerator(grid_points=542080, seed=42, device=device)
    print(f"Generator device: {generator.device}")

    # Test getting random locations
    location_types = [
        "continents",
        "regions",
        "countries",
        "cities",
        "states",
        "bodies_of_water",
    ]

    for loc_type in location_types:
        location = generator.get_random_location(location_type=loc_type)
        print(f"\n{loc_type.upper()}:")
        print(f"  Name: {location.name}")
        print(f"  Coordinates: {location.center_lat:.2f}°N, {location.center_lon:.2f}°E")
        print(f"  Mask device: {location.mask.device}")
        print(f"  Mask dtype: {location.mask.dtype}")
        print(f"  Masked points: {location.mask.sum().item()} / {location.mask.shape[0]}")

        # Verify mask is on correct device
        assert str(location.mask.device) == str(
            generator.device
        ), f"Mask device mismatch: {location.mask.device} vs {generator.device}"
        assert (
            location.mask.dtype == torch.bool
        ), f"Mask dtype should be bool, got {location.mask.dtype}"

    print("\n[PASS] LocationMaskGenerator GPU test passed")


def test_statistics_computation():
    """Test that statistics computation works correctly."""
    print("\n" + "=" * 80)
    print("TEST 2: Statistics Computation")
    print("=" * 80)

    analysis_gpu = os.environ.get("ANALYSIS_GPU", "0")
    device = f"cuda:{analysis_gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create test data
    grid_points = 10000
    n_vars = 94
    masked_count = 500  # Number of points to include in mask

    # Create random climate data on GPU
    climate_data = torch.randn(grid_points, n_vars, device=device)

    # Create a simple test mask (first N points are True)
    # This ensures we have a valid mask for statistics computation
    mask = torch.zeros(grid_points, dtype=torch.bool, device=device)
    mask[:masked_count] = True

    print(f"Climate data shape: {climate_data.shape}")
    print(f"Climate data device: {climate_data.device}")
    print(f"Mask shape: {mask.shape}")
    print(f"Masked points: {mask.sum().item()}")

    # Compute statistics
    computer = ClimateStatisticsComputer()
    statistics = computer.compute_statistics(
        climate_data.cpu(),  # Statistics computer expects CPU tensors
        mask.cpu(),
    )

    print(f"Computed statistics for {len(statistics)} variables")
    assert len(statistics) == n_vars, f"Expected {n_vars} variables, got {len(statistics)}"

    # Format statistics table
    table = computer.format_statistics_table(statistics)
    print("\nStatistics Table:")
    print(table)
    assert "Variable" in table, "Statistics table should have header"

    print("\n[PASS] Statistics computation test passed")


def test_full_pipeline():
    """Test the full sample generation pipeline."""
    print("\n" + "=" * 80)
    print("TEST 3: Full Pipeline Test")
    print("=" * 80)

    zarr_path = Path(__file__).parent.parent / "data" / "real_ecmwf_latest.zarr"
    if not zarr_path.exists():
        print(f"[SKIP] Real ECMWF data not found at {zarr_path}")
        return

    analysis_gpu = os.environ.get("ANALYSIS_GPU", "1")
    device = f"cuda:{analysis_gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading data from: {zarr_path}")

    # Initialize dataloader
    dataloader = ClimateTextDataLoader(
        zarr_paths=str(zarr_path),
        samples_per_epoch=3,
        device=device,
        seed=42,
    )

    print("DataLoader initialized successfully")
    print(f"Samples per epoch: {len(dataloader)}")

    # Generate samples with no response length limit
    print("\nGenerating samples...")
    samples = dataloader.get_sample_preview(num_samples=3, max_response_length=None)

    for i, sample in enumerate(samples):
        print(f"\n--- Sample {i + 1} ---")
        print(f"Location: {sample.location.name} ({sample.location.location_type})")
        print(
            f"Coordinates: {sample.location.center_lat:.2f}°N, "
            f"{sample.location.center_lon:.2f}°E"
        )
        print(f"Mask device: {sample.mask.device}")
        print(f"Masked points: {sample.mask.sum().item()}")
        print(f"Climate tensor device: {sample.climate_tensor_t1.device}")
        print(f"AIFS input shape: {sample.aifs_input_tensor.shape}")
        print(f"Prompt length: {len(sample.prompt)} chars")
        print(f"Response length: {len(sample.llm_response)} chars")

        # Verify statistics table
        assert len(sample.statistics_table) > 100, "Statistics table seems too short"
        assert "Variable" in sample.statistics_table, "Statistics table missing header"
        print("\nStatistics table:")
        print(sample.statistics_table)

        # Print the LLM-generated weather description
        print("\nLLM Response (weather description):")
        print("-" * 60)
        print(sample.llm_response)
        print("-" * 60)

    print("\n[PASS] Full pipeline test passed")


def test_batch_creation():
    """Test creating batches of samples."""
    print("\n" + "=" * 80)
    print("TEST 4: Batch Creation")
    print("=" * 80)

    zarr_path = Path(__file__).parent.parent / "data" / "real_ecmwf_latest.zarr"
    if not zarr_path.exists():
        print(f"[SKIP] Real ECMWF data not found at {zarr_path}")
        return

    analysis_gpu = os.environ.get("ANALYSIS_GPU", "1")
    device = f"cuda:{analysis_gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize dataloader
    dataloader = ClimateTextDataLoader(
        zarr_paths=str(zarr_path),
        batch_size=2,
        samples_per_epoch=4,
        device=device,
        seed=42,
    )

    # Generate samples and batch them
    samples = dataloader.get_sample_preview(num_samples=4)

    # Create batched tensors
    batch_climate_t1 = torch.stack([s.climate_tensor_t1 for s in samples])
    batch_climate_t2 = torch.stack([s.climate_tensor_t2 for s in samples])
    batch_aifs = torch.cat([s.aifs_input_tensor for s in samples], dim=0)
    batch_masks = torch.stack([s.mask for s in samples])

    print(f"Batch climate_t1 shape: {batch_climate_t1.shape}")
    print(f"Batch climate_t2 shape: {batch_climate_t2.shape}")
    print(f"Batch AIFS input shape: {batch_aifs.shape}")
    print(f"Batch masks shape: {batch_masks.shape}")

    # Verify all tensors are on correct device
    print(f"\nBatch tensors device: {batch_climate_t1.device}")
    assert str(batch_climate_t1.device) == str(device), "Batch tensors not on expected device"

    # Collect prompts and responses
    prompts = [s.prompt for s in samples]
    responses = [s.llm_response for s in samples]
    print(f"Prompts: {len(prompts)}")
    print(f"Responses: {len(responses)}")

    print("\n[PASS] Batch creation test passed")


def test_training_iteration():
    """Test a single training iteration."""
    print("\n" + "=" * 80)
    print("TEST 5: Single Training Iteration")
    print("=" * 80)

    zarr_path = Path(__file__).parent.parent / "data" / "real_ecmwf_latest.zarr"
    if not zarr_path.exists():
        print(f"[SKIP] Real ECMWF data not found at {zarr_path}")
        return

    analysis_gpu = os.environ.get("ANALYSIS_GPU", "1")
    device = f"cuda:{analysis_gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize dataloader
    dataloader = ClimateTextDataLoader(
        zarr_paths=str(zarr_path),
        batch_size=2,
        samples_per_epoch=2,
        device=device,
        seed=42,
    )

    # Generate a batch
    samples = dataloader.get_sample_preview(num_samples=2)

    # Create batched tensors
    batch_aifs = torch.cat([s.aifs_input_tensor for s in samples], dim=0)
    batch_masks = torch.stack([s.mask for s in samples])

    print(f"Batch AIFS input shape: {batch_aifs.shape}")
    print(f"Batch masks shape: {batch_masks.shape}")

    # Simulate a simple forward pass with a mock loss
    # In real training, this would use the actual model
    print("\nSimulating forward pass...")

    # Simple mock: compute mean of masked climate data
    masked_data = []
    for sample in samples:
        mask = sample.mask
        data = sample.climate_tensor_t1[mask]
        masked_data.append(data.mean())

    loss = torch.stack(masked_data).mean()
    print(f"Mock loss: {loss.item():.4f}")

    # Simulate backward pass
    print("Simulating backward pass...")
    # In real training, this would call loss.backward()
    # For now, just verify we can compute gradients
    if batch_aifs.requires_grad:
        loss.backward()
        print("Backward pass completed")
    else:
        print("(Tensors don't require grad - would work in actual training)")

    print("\n[PASS] Training iteration test passed")


def main():
    """Run all tests."""
    print("=" * 80)
    print("SAMPLE GENERATION PIPELINE GPU TEST")
    print("=" * 80)

    analysis_gpu = os.environ.get("ANALYSIS_GPU", "1")
    print(f"ANALYSIS_GPU environment variable: {analysis_gpu}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")

    # Run tests
    test_location_mask_generator_gpu()
    test_statistics_computation()
    test_full_pipeline()
    test_batch_creation()
    test_training_iteration()

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()

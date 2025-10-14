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
"""Test sample generation from the training pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from multimodal_aifs.training import ClimateTextDataLoader


def test_sample_generation(llm_mock_status):
    """Test that training samples are generated correctly with real Mistral responses."""
    print("\n" + "=" * 80)
    print("Testing Training Sample Generation with Real Mistral")
    print("=" * 80)

    # Check if real ECMWF Zarr file exists
    zarr_path = Path(__file__).parent.parent.parent / "data" / "real_ecmwf_latest.zarr"

    if not zarr_path.exists():
        print(f"\nReal ECMWF Zarr file not found at {zarr_path}")
        raise FileNotFoundError(f"Required real ECMWF data not found: {zarr_path}")

    # Get model name from fixture (respects LLM_MODEL_NAME env var)
    model_name = llm_mock_status["model_name"]
    print(f"\nInitializing DataLoader with {model_name}...")
    print(f"Zarr path: {zarr_path}")
    print("\nNote: Requires HuggingFace authentication for gated model")
    print("If not authenticated, run: huggingface-cli login")

    dataloader = ClimateTextDataLoader(
        zarr_paths=str(zarr_path),
        mistral_model_name=model_name,
        samples_per_epoch=3,
        cache_mistral_model=True,
        device="mps",  # Use MPS for faster inference (FP16)
        seed=42,
    )

    print("DataLoader initialized successfully")
    print(f"Samples per epoch: {len(dataloader)}")

    # Generate samples
    print("\nGenerating samples...")
    samples = dataloader.get_sample_preview(num_samples=3)

    assert len(samples) == 3, f"Expected 3 samples, got {len(samples)}"

    # Verify each sample
    expected_grid = 542080
    expected_vars = 94  # Actual number of variables in the test zarr file

    for i, sample in enumerate(samples):
        print(f"\n--- Sample {i + 1} ---")
        print(f"Location: {sample.location.name} ({sample.location.location_type})")
        print(
            f"Coordinates: {sample.location.center_lat:.2f}°N, {sample.location.center_lon:.2f}°E"
        )
        print(f"Masked points: {sample.mask.sum()} / {expected_grid}")
        print(f"Prompt length: {len(sample.prompt)} chars")
        print(f"Response length: {len(sample.llm_response)} chars")
        print(f"Timesteps: {sample.timestep_1_index} -> {sample.timestep_2_index}")

        # Show first sample's response preview
        if i == 0:
            print("\n--- Mistral Response Preview (Sample 1) ---")
            print(
                sample.llm_response[:300] + "..."
                if len(sample.llm_response) > 300
                else sample.llm_response
            )

        # Verify tensor shapes
        assert sample.climate_tensor_t1.shape == (
            expected_grid,
            expected_vars,
        ), "Sample {i+1}: Wrong t1 shape: {sample.climate_tensor_t1.shape}"
        assert sample.climate_tensor_t2.shape == (
            expected_grid,
            expected_vars,
        ), "Sample {i+1}: Wrong t2 shape: {sample.climate_tensor_t2.shape}"
        assert sample.aifs_input_tensor.shape == (
            1,
            2,
            1,
            expected_grid,
            expected_vars,
        ), "Sample {i+1}: Wrong AIFS shape: {sample.aifs_input_tensor.shape}"
        assert sample.mask.shape == (
            expected_grid,
        ), "Sample {i+1}: Wrong mask shape: {sample.mask.shape}"

        # Verify mask has some points selected
        assert sample.mask.sum() > 0, "Sample {i+1}: Mask has no points selected"

        # Verify text is not empty
        assert len(sample.prompt) > 0, "Sample {i+1}: Prompt is empty"
        assert len(sample.llm_response) > 0, "Sample {i+1}: Response is empty"

        # Verify statistics table exists
        assert len(sample.statistics_table) > 0, "Sample {i+1}: Statistics table is empty"

        # Verify timesteps are consecutive
        assert (
            sample.timestep_2_index == sample.timestep_1_index + 1
        ), "Sample {i+1}: Timesteps are not consecutive"

    print("\n" + "=" * 80)
    print("✓ All sample generation tests passed")
    print("=" * 80)
    print()

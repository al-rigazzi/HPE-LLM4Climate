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
"""Unit tests for ClimateTextDataLoader using real components instead of mocks."""

# pylint: disable=protected-access

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import zarr

from multimodal_aifs.training import ClimateTextDataLoader
from multimodal_aifs.training.climate_dataloader import TrainingSample
from multimodal_aifs.training.location_masks import Location


@pytest.mark.skipif(
    not os.environ.get("HF_TOKEN"),
    reason="Requires HuggingFace token for accessing gated Mistral models",
)
class TestClimateTextDataLoaderReal:
    """Test suite using real components (minimal mocking)."""

    @pytest.fixture
    def real_test_zarr_path(self):
        """Create a realistic Zarr file for testing with proper climate data structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            zarr_path = Path(temp_dir) / "realistic_test.zarr"

            # Create realistic climate data structure
            root = zarr.open(str(zarr_path), mode="w")

            # Time dimension (small for testing)
            n_timesteps = 10
            root.create_array("time", data=np.arange(n_timesteps))

            # Grid points (consistent for all variables)
            grid_points = 2560  # Reduced from real size for testing

            # All variables must have the same grid_points dimension
            # Surface variables
            for var_name in ["2t", "10u", "10v", "msl", "tp"]:
                root.create_array(
                    var_name,
                    data=np.random.randn(n_timesteps, grid_points).astype(np.float32),
                    chunks=(1, grid_points),
                )

            # Pressure level variables (same grid_points)
            levels = [500, 700, 850]  # Fewer levels for testing
            for level in levels:
                for var in ["z", "t", "u", "v", "q", "w"]:
                    var_name = f"{var}_{level}"
                    root.create_array(
                        var_name,
                        data=np.random.randn(n_timesteps, grid_points).astype(np.float32),
                        chunks=(1, grid_points),
                    )

            # Grid point coordinates (same grid_points)
            root.create_array(
                "cos_latitude", data=np.random.uniform(-1, 1, grid_points).astype(np.float32)
            )
            root.create_array(
                "sin_latitude", data=np.random.uniform(-1, 1, grid_points).astype(np.float32)
            )
            root.create_array(
                "cos_longitude", data=np.random.uniform(-1, 1, grid_points).astype(np.float32)
            )
            root.create_array(
                "sin_longitude", data=np.random.uniform(-1, 1, grid_points).astype(np.float32)
            )

            yield str(zarr_path)

    @pytest.fixture
    def real_dataloader_no_model(self, real_test_zarr_path):
        """Create a real ClimateTextDataLoader but mock only the expensive model loading."""
        from unittest.mock import Mock

        with patch.object(ClimateTextDataLoader, "_initialize_mistral_model"):
            # Create real dataloader with real components (including real tokenizer)
            # Let it auto-detect device (will use MPS on macOS to catch compatibility issues)
            dataloader = ClimateTextDataLoader(
                zarr_paths=[real_test_zarr_path],
                mistral_model_name="mistralai/Ministral-8B-Instruct-2410",
                batch_size=2,
                samples_per_epoch=5,
                max_prompt_length=512,
                max_response_length=128,
                seed=42,
            )

            # Mock the model after initialization
            mock_model = Mock()
            mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
            dataloader.mistral_model = mock_model

            yield dataloader

    def test_initialization_basic(self, real_test_zarr_path):
        """Test basic initialization with minimal parameters."""
        with (
            patch("multimodal_aifs.training.climate_dataloader.AutoTokenizer"),
            patch.object(ClimateTextDataLoader, "_initialize_mistral_model"),
        ):

            dataloader = ClimateTextDataLoader(
                zarr_paths=[real_test_zarr_path], samples_per_epoch=3, device="cpu"
            )

            assert dataloader.batch_size == 1  # Default value
            assert dataloader.samples_per_epoch == 3
            assert dataloader.device == "cpu"
            assert len(dataloader.zarr_datasets) == 1

    def test_initialization_with_custom_parameters(self, real_test_zarr_path):
        """Test initialization with custom parameters."""
        with (
            patch("multimodal_aifs.training.climate_dataloader.AutoTokenizer"),
            patch.object(ClimateTextDataLoader, "_initialize_mistral_model"),
        ):

            dataloader = ClimateTextDataLoader(
                zarr_paths=[real_test_zarr_path],
                mistral_model_name="mistralai/Ministral-8B-Instruct-2410",
                batch_size=4,
                samples_per_epoch=20,
                max_prompt_length=1024,
                max_response_length=256,
                device="cuda",
                seed=123,
            )

            assert dataloader.batch_size == 4
            assert dataloader.samples_per_epoch == 20
            assert dataloader.max_prompt_length == 1024
            assert dataloader.max_response_length == 256
            assert dataloader.device == "cuda"

    def test_real_sample_generation(self, real_dataloader_no_model):
        """Test that we can generate real samples using real components."""
        # This is the key test - it uses ALL real components
        samples = real_dataloader_no_model.get_sample_preview(num_samples=2)

        assert len(samples) == 2

        for sample in samples:
            # Check that we got real TrainingSample objects with real data
            assert isinstance(sample, TrainingSample)
            assert isinstance(sample.location, Location)
            assert isinstance(sample.climate_tensor_t1, torch.Tensor)
            assert isinstance(sample.climate_tensor_t2, torch.Tensor)
            assert isinstance(sample.aifs_input_tensor, torch.Tensor)
            assert isinstance(sample.mask, torch.Tensor)
            assert isinstance(sample.prompt, str)
            assert isinstance(sample.llm_response, str)

            # Check tensor shapes are reasonable
            assert sample.climate_tensor_t1.shape[1] > 0  # Has variables
            assert sample.climate_tensor_t2.shape == sample.climate_tensor_t1.shape
            assert sample.mask.dtype == torch.bool
            assert sample.mask.sum() > 0  # Some points should be masked

            # Check that location has meaningful data
            assert hasattr(sample.location, "name")
            assert hasattr(sample.location, "center_lat")
            assert hasattr(sample.location, "center_lon")

            # Check that prompt contains useful information
            assert len(sample.prompt) > 50  # Should be substantial
            assert "climate" in sample.prompt.lower() or "weather" in sample.prompt.lower()

    def test_real_statistics_computation(self, real_dataloader_no_model):
        """Test that statistics computation works with real data."""
        sample = real_dataloader_no_model._generate_sample(sample_idx=0)

        # Verify that statistics are computed correctly
        assert hasattr(sample, "statistics_table")
        assert isinstance(sample.statistics_table, str)
        assert len(sample.statistics_table) > 0

        # Statistics should contain actual numbers
        assert any(char.isdigit() for char in sample.statistics_table)

    def test_real_location_masks(self, real_dataloader_no_model):
        """Test that location mask generation works correctly."""
        sample = real_dataloader_no_model._generate_sample(sample_idx=0)

        # Check mask properties
        assert isinstance(sample.mask, torch.Tensor)
        assert sample.mask.dtype == torch.bool

        # Mask should select a reasonable subset of points
        total_points = sample.mask.shape[0]
        selected_points = sample.mask.sum().item()

        assert 0 < selected_points < total_points  # Some but not all points selected
        assert selected_points >= 1  # At least some points (relaxed for small test data)

    def test_real_prompt_generation(self, real_dataloader_no_model):
        """Test that prompt generation works with real templates."""
        sample = real_dataloader_no_model._generate_sample(sample_idx=0)

        # Check prompt structure
        assert isinstance(sample.prompt, str)
        assert len(sample.prompt) > 100  # Should be substantial

        # Should contain location information
        assert sample.location.name in sample.prompt or "location" in sample.prompt.lower()

    def test_data_loader_iteration(self, real_dataloader_no_model):
        """Test that the dataloader can iterate correctly."""
        # Test iteration works
        sample_count = 0
        for sample in real_dataloader_no_model:
            sample_count += 1

            # Check sample structure
            assert isinstance(sample, TrainingSample)

            # Only check first sample to avoid long test
            break

        assert sample_count > 0

    def test_reproducibility(self, real_test_zarr_path):
        """Test that setting seed makes results reproducible."""
        from unittest.mock import Mock

        def create_dataloader():
            with patch.object(ClimateTextDataLoader, "_initialize_mistral_model"):
                dl = ClimateTextDataLoader(
                    zarr_paths=[real_test_zarr_path],
                    samples_per_epoch=2,
                    device="cpu",
                    seed=123,  # Fixed seed
                )
                # Mock the model
                mock_model = Mock()
                mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
                dl.mistral_model = mock_model
                return dl

        # Create two dataloaders with same seed
        dataloader1 = create_dataloader()
        dataloader2 = create_dataloader()

        # Generate samples
        sample1 = dataloader1._generate_sample(sample_idx=0)
        sample2 = dataloader2._generate_sample(sample_idx=0)

        # Should get same location (due to seed)
        assert sample1.location.name == sample2.location.name
        assert torch.allclose(sample1.climate_tensor_t1, sample2.climate_tensor_t1)

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            ClimateTextDataLoader(zarr_paths=["non_existent_file.zarr"], samples_per_epoch=1)


class TestTrainingSampleStructure:
    """Test the TrainingSample dataclass structure."""

    def test_training_sample_fields(self):
        """Test that TrainingSample has the expected fields."""
        # Create a location with the correct parameters
        location = Location(
            name="Test",
            location_type="city",
            description="Test location",
            center_lat=45.0,
            center_lon=-120.0,
            mask=torch.zeros(100, dtype=torch.bool),
        )

        sample = TrainingSample(
            climate_tensor_t1=torch.randn(100, 10),
            climate_tensor_t2=torch.randn(100, 10),
            aifs_input_tensor=torch.randn(1, 2, 1, 100, 10),
            location=location,
            mask=torch.zeros(100, dtype=torch.bool),
            statistics_table="Test stats",
            prompt="Test prompt",
            llm_response="Test response",
            prompt_tokens={"input_ids": torch.tensor([1, 2, 3])},
            response_tokens={"input_ids": torch.tensor([4, 5, 6])},
            timestep_1_index=0,
            timestep_2_index=1,
            zarr_file_path="/test/path.zarr",
        )

        # Verify all fields exist and have correct types
        assert isinstance(sample.climate_tensor_t1, torch.Tensor)
        assert isinstance(sample.climate_tensor_t2, torch.Tensor)
        assert isinstance(sample.aifs_input_tensor, torch.Tensor)
        assert isinstance(sample.location, Location)
        assert isinstance(sample.mask, torch.Tensor)
        assert isinstance(sample.statistics_table, str)
        assert isinstance(sample.prompt, str)
        assert isinstance(sample.llm_response, str)
        assert isinstance(sample.prompt_tokens, dict)
        assert isinstance(sample.response_tokens, dict)
        assert isinstance(sample.timestep_1_index, int)
        assert isinstance(sample.timestep_2_index, int)
        assert isinstance(sample.zarr_file_path, str)

"""Unit tests for ClimateTextDataLoader and complete training pipeline."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import zarr

from multimodal_aifs.core import ClimateStatisticsComputer, LocationMaskGenerator
from multimodal_aifs.core.aifs_climate_fusion import ClimatePromptGenerator
from multimodal_aifs.core.utils import Location
from multimodal_aifs.training import ClimateTextDataLoader
from multimodal_aifs.training.climate_dataloader import TrainingSample


class TestClimateTextDataLoader:
    """Test suite for ClimateTextDataLoader and training pipeline."""

    @pytest.fixture
    def real_test_zarr_path(self):
        """Create a realistic Zarr file for testing with proper climate data structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            zarr_path = Path(temp_dir) / "realistic_test.zarr"

            # Create realistic climate data structure
            root = zarr.open(str(zarr_path), mode='w')

            # Time dimension (small for testing)
            n_timesteps = 10
            root.create_array('time', data=np.arange(n_timesteps))

            # Grid points (realistic but small for testing)
            grid_points = 2560  # Reduced from real size for testing
            n_vars_per_level = 3  # Temperature, u-wind, v-wind

            # Surface variables
            for var_name in ['2t', '10u', '10v', 'msl', 'tp']:
                root.create_array(
                    var_name,
                    data=np.random.randn(n_timesteps, grid_points),
                    chunks=(1, grid_points)
                )

            # Pressure level variables
            levels = [500, 700, 850]  # Fewer levels for testing
            for level in levels:
                for var in ['z', 't', 'u', 'v', 'q', 'w']:
                    var_name = f"{var}_{level}"
                    root.create_array(
                        var_name,
                        data=np.random.randn(n_timesteps, grid_points),
                        chunks=(1, grid_points)
                    )

            # Grid point coordinates (constant)
            root.create_array('cos_latitude', data=np.random.uniform(-1, 1, grid_points))
            root.create_array('sin_latitude', data=np.random.uniform(-1, 1, grid_points))
            root.create_array('cos_longitude', data=np.random.uniform(-1, 1, grid_points))
            root.create_array('sin_longitude', data=np.random.uniform(-1, 1, grid_points))

            yield str(zarr_path)

    @pytest.fixture
    def real_dataloader_no_model(self, real_test_zarr_path):
        """Create a real ClimateTextDataLoader but mock only the expensive model loading."""
        with patch('multimodal_aifs.training.climate_dataloader.AutoTokenizer') as mock_tokenizer, \
             patch.object(ClimateTextDataLoader, '_initialize_mistral_model'):

            # Mock tokenizer to return dummy tokens
            mock_tokenizer_instance = mock_tokenizer.from_pretrained.return_value
            mock_tokenizer_instance.encode.return_value = [1, 2, 3, 4, 5]  # Dummy token IDs
            mock_tokenizer_instance.decode.return_value = "Dummy response"

            # Create real dataloader with real components
            dataloader = ClimateTextDataLoader(
                zarr_paths=[real_test_zarr_path],
                mistral_model_name="mistralai/Ministral-8B-Instruct-2410",
                batch_size=2,
                samples_per_epoch=5,
                max_prompt_length=512,
                max_response_length=128,
                device="cpu",
                seed=42
            )

            yield dataloader

    def test_initialization_basic(self, real_test_zarr_path):
        """Test basic initialization with minimal parameters."""
        with patch('multimodal_aifs.training.climate_dataloader.AutoTokenizer'), \
             patch.object(ClimateTextDataLoader, '_initialize_mistral_model'):

            dataloader = ClimateTextDataLoader(
                zarr_paths=[real_test_zarr_path],
                samples_per_epoch=3,
                device="cpu"
            )

            assert dataloader.batch_size == 1  # Default value
            assert dataloader.samples_per_epoch == 3
            assert dataloader.device == "cpu"
            assert len(dataloader.zarr_datasets) == 1

    def test_initialization_multiple_zarr_files(self, real_test_zarr_path):
        """Test initialization with multiple zarr files."""
        with patch('multimodal_aifs.training.climate_dataloader.AutoTokenizer'), \
             patch.object(ClimateTextDataLoader, '_initialize_mistral_model'):

            dataloader = ClimateTextDataLoader(
                zarr_paths=[real_test_zarr_path, real_test_zarr_path],  # Same file twice for test
                samples_per_epoch=5,
                device="cpu"
            )

            assert len(dataloader.zarr_datasets) == 2
            assert dataloader.samples_per_epoch == 5

    def test_initialization_with_custom_parameters(self, real_test_zarr_path):

    def test_initialization_multiple_zarr_files(self):
        """Test initialization with multiple Zarr files."""
        zarr_paths = ["file1.zarr", "file2.zarr"]

        with patch('multimodal_aifs.training.climate_dataloader.AutoTokenizer'), \
             patch.object(ClimateTextDataLoader, '_initialize_mistral_model'), \
             patch.object(ClimateTextDataLoader, '_load_zarr_metadata'), \
             patch('pathlib.Path.exists', return_value=True):

            dataloader = ClimateTextDataLoader(
                zarr_paths=zarr_paths,
                samples_per_epoch=5,
                cache_mistral_model=False,
                device="cpu"
            )

            assert len(dataloader.zarr_paths) == len(zarr_paths)
            for i, path in enumerate(dataloader.zarr_paths):
                assert str(path) == zarr_paths[i]

    def test_initialization_with_custom_parameters(self, mock_zarr_path):
        """Test initialization with custom parameters."""
        with patch('multimodal_aifs.training.climate_dataloader.AutoTokenizer'), \
             patch.object(ClimateTextDataLoader, '_initialize_mistral_model'), \
             patch.object(ClimateTextDataLoader, '_load_zarr_metadata'):

            dataloader = ClimateTextDataLoader(
                zarr_paths=mock_zarr_path,
                mistral_model_name="mistralai/Ministral-8B-Instruct-2410",
                batch_size=4,
                samples_per_epoch=20,
                max_prompt_length=1024,
                max_response_length=256,
                device="cuda",
                seed=123
            )

            assert dataloader.batch_size == 4
            assert dataloader.samples_per_epoch == 20
            assert dataloader.max_prompt_length == 1024
            assert dataloader.max_response_length == 256
            assert dataloader.device == "cuda"

    def test_get_sample_preview_basic(self, mock_dataloader_no_model):
        """Test basic sample preview generation."""
        # Mock the _generate_sample method
        mock_sample = Mock(spec=TrainingSample)
        mock_location = Mock()
        mock_location.name = "Test Location"
        mock_sample.location = mock_location
        mock_sample.llm_response = "Test response"
        mock_sample.prompt = "Test prompt"

        with patch.object(mock_dataloader_no_model, '_generate_sample', return_value=mock_sample):
            samples = mock_dataloader_no_model.get_sample_preview(num_samples=2)

            assert len(samples) == 2
            assert all(s.location.name == "Test Location" for s in samples)

    def test_get_sample_preview_with_custom_response_length(self, mock_dataloader_no_model):
        """Test sample preview with custom max_response_length."""
        original_max_response = mock_dataloader_no_model.max_response_length

        mock_sample = Mock(spec=TrainingSample)
        with patch.object(mock_dataloader_no_model, '_generate_sample', return_value=mock_sample):
            samples = mock_dataloader_no_model.get_sample_preview(
                num_samples=1,
                max_response_length=2048
            )

            # Should temporarily change max_response_length during generation
            assert len(samples) == 1
            # Should restore original value after
            assert mock_dataloader_no_model.max_response_length == original_max_response

    def test_generate_sample_integration(self, mock_dataloader_no_model):
        """Test sample generation integration."""
        # Mock the data loading components
        mock_climate_data = np.random.randn(1000, 10)
        mock_mask = np.zeros(1000, dtype=bool)
        mock_mask[100:200] = True

        with patch.object(mock_dataloader_no_model, '_load_climate_timesteps', return_value=mock_climate_data), \
             patch.object(mock_dataloader_no_model.location_generator, 'get_random_location') as mock_location, \
             patch.object(mock_dataloader_no_model.location_generator, '_create_mask_for_location', return_value=mock_mask):

            # Setup mock location
            mock_loc = Mock()
            mock_loc.name = "Test Location"
            mock_loc.description = "Test Location"
            mock_location.return_value = mock_loc

            sample = mock_dataloader_no_model._generate_sample(0)

            assert isinstance(sample, TrainingSample)
            assert sample.location.name == "Test Location"

    def test_training_sample_structure(self):
        """Test TrainingSample dataclass structure."""
        # Create a mock TrainingSample
        sample = TrainingSample(
            climate_tensor_t1=torch.randn(1000, 10),
            climate_tensor_t2=torch.randn(1000, 10),
            aifs_input_tensor=torch.randn(1, 2, 1, 1000, 10),
            location=Mock(),
            mask=torch.zeros(1000, dtype=torch.bool),
            prompt="Test prompt",
            llm_response="Test response",
            prompt_tokens={'input_ids': torch.tensor([[1, 2, 3]])},
            response_tokens={'input_ids': torch.tensor([[4, 5, 6]])},
            statistics_table="Test table",
            timestep_1_index=0,
            timestep_2_index=1,
            zarr_file_path="test.zarr"
        )

        # Check all required fields exist
        assert hasattr(sample, 'climate_tensor_t1')
        assert hasattr(sample, 'climate_tensor_t2')
        assert hasattr(sample, 'aifs_input_tensor')
        assert hasattr(sample, 'location')
        assert hasattr(sample, 'mask')
        assert hasattr(sample, 'prompt')
        assert hasattr(sample, 'llm_response')
        assert hasattr(sample, 'prompt_tokens')
        assert hasattr(sample, 'response_tokens')
        assert hasattr(sample, 'statistics_table')
        assert hasattr(sample, 'timestep_1_index')
        assert hasattr(sample, 'timestep_2_index')
        assert hasattr(sample, 'zarr_file_path')

    def test_dataloader_iteration(self, mock_dataloader_no_model):
        """Test DataLoader iteration protocol."""
        mock_sample = Mock(spec=TrainingSample)

        with patch.object(mock_dataloader_no_model, '_generate_sample', return_value=mock_sample):
            # Test __iter__
            samples = list(mock_dataloader_no_model)
            assert len(samples) == mock_dataloader_no_model.samples_per_epoch

            # Test __len__
            assert len(mock_dataloader_no_model) == mock_dataloader_no_model.samples_per_epoch

    def test_error_handling_in_iteration(self, mock_dataloader_no_model):
        """Test error handling during sample generation."""
        def mock_generate_sample(idx):
            if idx == 1:
                raise ValueError("Test error")
            return Mock(spec=TrainingSample)

        with patch.object(mock_dataloader_no_model, '_generate_sample', side_effect=mock_generate_sample), \
             patch('builtins.print'):  # Suppress error output

            samples = list(mock_dataloader_no_model)
            # Should continue despite error and return samples that succeeded
            assert len(samples) < mock_dataloader_no_model.samples_per_epoch

    def test_device_detection_and_model_placement(self, mock_zarr_path):
        """Test device detection and model placement logic."""
        with patch('multimodal_aifs.training.climate_dataloader.AutoTokenizer'), \
             patch.object(ClimateTextDataLoader, '_load_zarr_metadata'):

            # Mock model with device attributes
            mock_model = Mock()
            mock_model.device = torch.device('mps:0')

            with patch.object(ClimateTextDataLoader, '_initialize_mistral_model') as mock_init:
                dataloader = ClimateTextDataLoader(
                    zarr_paths=mock_zarr_path,
                    samples_per_epoch=1,
                    cache_mistral_model=True,
                    device="cpu"
                )

                # Mock the model after initialization
                dataloader.mistral_model = mock_model

                # Test device detection logic
                if hasattr(dataloader.mistral_model, 'device'):
                    detected_device = dataloader.mistral_model.device
                    assert detected_device == torch.device('mps:0')

    def test_print_sample_info(self, mock_dataloader_no_model):
        """Test print_sample_info method."""
        # Create a complete mock sample
        mock_sample = Mock(spec=TrainingSample)
        mock_location = Mock()
        mock_location.name = "Test Location"
        mock_location.location_type = "test"
        mock_location.center_lat = 45.0
        mock_location.center_lon = -120.0
        mock_sample.location = mock_location
        mock_sample.mask = torch.zeros(1000, dtype=torch.bool)
        mock_sample.mask[:100] = True
        mock_sample.timestep_1_index = 0
        mock_sample.timestep_2_index = 1
        mock_sample.climate_tensor_t1.shape = (1000, 10)
        mock_sample.climate_tensor_t2.shape = (1000, 10)
        mock_sample.aifs_input_tensor.shape = (1, 2, 1, 1000, 10)
        mock_sample.statistics_table = "Test statistics table"
        mock_sample.prompt = "Test prompt"
        mock_sample.llm_response = "Test response"
        mock_sample.prompt_tokens = {'input_ids': torch.tensor([[1, 2, 3]])}
        mock_sample.response_tokens = {'input_ids': torch.tensor([[4, 5, 6]])}
        mock_sample.zarr_file_path = "test.zarr"

        # Should not raise any errors
        with patch('builtins.print'):
            mock_dataloader_no_model.print_sample_info(mock_sample)

    def test_file_not_found_error(self):
        """Test error handling for missing Zarr files."""
        with pytest.raises(FileNotFoundError):
            ClimateTextDataLoader(
                zarr_paths="nonexistent.zarr",
                samples_per_epoch=1,
                cache_mistral_model=False
            )

    def test_string_zarr_path_conversion(self, mock_zarr_path):
        """Test that string Zarr paths are properly converted to lists."""
        with patch('multimodal_aifs.training.climate_dataloader.AutoTokenizer'), \
             patch.object(ClimateTextDataLoader, '_initialize_mistral_model'), \
             patch.object(ClimateTextDataLoader, '_load_zarr_metadata'):

            # Test single string path
            dataloader = ClimateTextDataLoader(
                zarr_paths=mock_zarr_path,  # String, not list
                samples_per_epoch=1,
                cache_mistral_model=False
            )

            assert isinstance(dataloader.zarr_paths, list)
            assert len(dataloader.zarr_paths) == 1
            assert str(dataloader.zarr_paths[0]) == mock_zarr_path


class TestTrainingPipelineIntegration:
    """Integration tests for the complete training pipeline."""

    def test_component_integration(self):
        """Test that all components integrate properly."""
        from multimodal_aifs.training.location_masks import LocationMaskGenerator
        from multimodal_aifs.training.prompt_generator import ClimatePromptGenerator
        from multimodal_aifs.training.statistics_computer import ClimateStatisticsComputer

        # Create components
        location_gen = LocationMaskGenerator(grid_points=1000, seed=42)
        stats_computer = ClimateStatisticsComputer()
        prompt_gen = ClimatePromptGenerator()

        # Test basic integration
        location = location_gen.get_random_location()
        assert location is not None

        # Mock data for statistics
        data = torch.tensor(np.random.randn(1000, 5), dtype=torch.float32)
        mask = torch.zeros(1000, dtype=torch.bool)
        mask[100:200] = True
        var_names = ["2t", "10u", "10v", "msl", "tp"]

        stats = stats_computer.compute_statistics(data, mask, var_names)
        table = stats_computer.format_statistics_table(stats)

        # Generate prompt
        prompt = prompt_gen.generate_analysis_prompt(
            location=location,
            statistics=stats,
            statistics_table=table,
            task_type="weather_description"
        )

        assert len(prompt) > 0
        assert location.name in prompt
        assert table in prompt

    def test_pipeline_reproducibility(self):
        """Test that pipeline is reproducible with same seed."""
        from multimodal_aifs.training.location_masks import LocationMaskGenerator

        # Same seed should produce same results
        gen1 = LocationMaskGenerator(grid_points=100, seed=42)
        gen2 = LocationMaskGenerator(grid_points=100, seed=42)

        loc1 = gen1.get_random_location()
        loc2 = gen2.get_random_location()

        assert loc1.name == loc2.name
        assert loc1.center_lat == loc2.center_lat
        assert loc1.center_lon == loc2.center_lon

    def test_template_and_data_integration(self):
        """Test that templates work correctly with real-like data."""
        from multimodal_aifs.training.location_masks import LocationMaskGenerator
        from multimodal_aifs.training.prompt_generator import ClimatePromptGenerator
        from multimodal_aifs.training.statistics_computer import ClimateStatisticsComputer

        # Create components
        prompt_gen = ClimatePromptGenerator()
        location_gen = LocationMaskGenerator(grid_points=100, seed=42)
        stats_computer = ClimateStatisticsComputer()

        # Get location
        location = location_gen.get_location_by_name("Australia")

        # Create realistic-looking climate data
        data = torch.tensor(np.random.randn(100, 8), dtype=torch.float32)
        data[:, 0] += 290  # Temperature-like (Kelvin)
        data[:, 1] *= 5    # Wind u
        data[:, 2] *= 5    # Wind v
        data[:, 3] = data[:, 3] * 1000 + 101000  # Pressure
        data[:, 4] = torch.abs(data[:, 4]) * 0.001   # Precipitation

        mask = torch.zeros(100, dtype=torch.bool)
        mask[20:50] = True

        var_names = ["2t", "10u", "10v", "msl", "tp", "z_500", "q_500", "t_850"]

        # Compute statistics
        stats = stats_computer.compute_statistics(data, mask, var_names)
        table = stats_computer.format_statistics_table(stats)

        # Test all prompt types
        prompt_types = ["weather_description", "forecast", "anomaly_detection"]

        for prompt_type in prompt_types:
            prompt = prompt_gen.generate_analysis_prompt(
                location=location,
                statistics=stats,
                statistics_table=table,
                task_type=prompt_type
            )

            # Basic validation
            assert len(prompt) > 100
            assert location.name in prompt
            assert table in prompt
            assert "Australia" in prompt

            # Format for Mistral
            formatted = prompt_gen.format_for_mistral(prompt)
            assert formatted.startswith("<s>[INST]")
            assert formatted.endswith("[/INST]")
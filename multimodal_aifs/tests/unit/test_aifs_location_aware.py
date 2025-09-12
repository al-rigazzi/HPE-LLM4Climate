#!/usr/bin/env python3
"""
Test script for AIFS Location-Aware Fusion Model
Tests geographic query processing, spatial cropping, and location-aware attention
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)


# Mock flash attention before any transformers imports
def setup_flash_attn_mock():
    """Mock flash attention components that cause import issues."""
    flash_attn_mock = MagicMock()
    flash_attn_mock.__version__ = "2.0.0"
    flash_attn_mock.__spec__ = MagicMock()  # Fix the __spec__ issue

    # Mock the functions that are typically imported
    flash_attn_mock.flash_attn_func = MagicMock()
    flash_attn_mock.flash_attn_supports_top_left_mask = MagicMock(return_value=True)

    sys.modules["flash_attn"] = flash_attn_mock
    sys.modules["flash_attn.flash_attn_interface"] = flash_attn_mock

    # Mock transformers flash attention utils
    with patch("transformers.utils.import_utils.is_flash_attn_2_available", return_value=False):
        with patch("transformers.utils.import_utils._is_package_available", return_value=False):
            pass


# Setup flash attention mock before imports
setup_flash_attn_mock()

try:
    from multimodal_aifs.core.aifs_location_aware_fusion import (
        AIFSLocationAwareFusion,
        GeographicResolver,
        SpatialCropper,
    )
    from multimodal_aifs.utils.text_utils import extract_location_keywords, parse_climate_query

    print("✅ Successfully imported location-aware components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class TestGeographicResolver:
    """Test geographic coordinate resolution"""

    def test_basic_location_resolution(self):
        """Test basic location name to coordinate resolution"""
        resolver = GeographicResolver()

        # Test known locations
        location_info = resolver.resolve_location("New York")
        assert location_info is not None
        assert "lat" in location_info
        assert "lon" in location_info
        assert "name" in location_info
        assert "bounds" in location_info
        assert isinstance(location_info["lat"], float)
        assert isinstance(location_info["lon"], float)
        print(f"✅ New York location info: {location_info}")

        # Test case insensitive
        location_info_lower = resolver.resolve_location("new york")
        assert location_info_lower == location_info
        print("✅ Case insensitive resolution works")

    def test_location_not_found(self):
        """Test handling of unknown locations"""
        resolver = GeographicResolver()
        location_info = resolver.resolve_location("NonExistentPlace12345")
        assert location_info is None
        print("✅ Unknown location handling works")


class TestSpatialCropper:
    """Test spatial data cropping functionality"""

    def test_spatial_cropping(self):
        """Test spatial cropping around coordinates"""
        grid_shape = (180, 360)  # lat, lon
        cropper = SpatialCropper(grid_shape)

        # Create mock weather data [batch, time, vars, lat, lon]
        weather_data = torch.randn(1, 1, 5, 180, 360)  # batch, time, channels, lat, lon

        # Test cropping around New York region
        bounds = {"north": 41.0, "south": 40.0, "east": -73.0, "west": -75.0}

        cropped, crop_info = cropper.crop_to_region(weather_data, bounds)

        assert cropped.shape[0] == 1  # batch preserved
        assert cropped.shape[1] == 1  # time preserved
        assert cropped.shape[2] == 5  # channels preserved
        assert cropped.shape[3] < 180  # lat dimension reduced
        assert cropped.shape[4] < 360  # lon dimension reduced
        print(f"✅ Spatial cropping: {weather_data.shape} -> {cropped.shape}")
        print(f"✅ Crop info: {crop_info}")

    def test_boundary_handling(self):
        """Test cropping near boundaries"""
        grid_shape = (180, 360)
        cropper = SpatialCropper(grid_shape)
        weather_data = torch.randn(1, 1, 5, 180, 360)

        # Test near North Pole
        bounds = {"north": 90.0, "south": 85.0, "east": 10.0, "west": -10.0}
        cropped, crop_info = cropper.crop_to_region(weather_data, bounds)
        assert cropped.shape[3] > 0 and cropped.shape[4] > 0
        print("✅ Boundary handling works (North Pole)")

        # Test near dateline
        bounds = {"north": 5.0, "south": -5.0, "east": -175.0, "west": 175.0}
        cropped, crop_info = cropper.crop_to_region(weather_data, bounds)
        assert cropped.shape[3] > 0 and cropped.shape[4] > 0
        print("✅ Boundary handling works (Dateline)")


class TestTextUtils:
    """Test text processing utilities"""

    def test_location_extraction(self):
        """Test location keyword extraction"""
        queries = [
            "What's the weather in New York?",
            "Tell me about rainfall in London today",
            "How hot is it in Tokyo right now?",
            "General climate patterns worldwide",
        ]

        for query in queries:
            locations = extract_location_keywords(query)
            print(f"Query: '{query}' -> Locations: {locations}")

        # Test specific extraction
        locations = extract_location_keywords("What's the weather in New York?")
        assert "new york" in locations or "New York" in locations
        print("✅ Location extraction works")

    def test_climate_query_parsing(self):
        """Test climate query parsing"""
        query = "What's the temperature and humidity in San Francisco?"
        parsed = parse_climate_query(query)

        assert "locations" in parsed  # Note: plural "locations"
        assert "variables" in parsed
        assert "temporal" in parsed

        print(f"✅ Parsed query: {parsed}")


class TestAIFSLocationAwareFusion(unittest.TestCase):
    """Test the main location-aware fusion model"""

    @patch("transformers.utils.import_utils.is_flash_attn_2_available", return_value=False)
    @patch("transformers.utils.import_utils._is_package_available", return_value=False)
    @patch("transformers.LlamaModel")
    @patch("transformers.LlamaTokenizer")
    def test_model_initialization(
        self, mock_tokenizer, mock_model, mock_pkg_available, mock_flash_available
    ):
        """Test model initialization with mocked components"""
        # Mock the tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token_id = 0
        mock_tokenizer_instance.vocab_size = 32000
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Mock the model
        mock_model_instance = MagicMock()
        mock_model_instance.config.hidden_size = 4096
        mock_model.from_pretrained.return_value = mock_model_instance

        try:
            model = AIFSLocationAwareFusion(
                llama_model_name="meta-llama/Meta-Llama-3-8B",
                grid_shape=(180, 360),
                use_mock_llama=True,  # Use mock for testing
                use_quantization=False,  # Disable for testing
            )
            print("✅ Model initialization successful")

            # Test model components
            assert hasattr(model, "geographic_resolver")
            assert hasattr(model, "spatial_cropper")
            assert hasattr(model, "spatial_encoder")
            print("✅ Model components initialized")

        except Exception as e:
            print(f"❌ Model initialization failed: {e}")
            raise

    @patch("transformers.utils.import_utils.is_flash_attn_2_available", return_value=False)
    @patch("transformers.utils.import_utils._is_package_available", return_value=False)
    @patch("transformers.LlamaModel")
    @patch("transformers.LlamaTokenizer")
    def test_forward_pass_mock(
        self, mock_tokenizer, mock_model, mock_pkg_available, mock_flash_available
    ):
        """Test forward pass configuration with mock data"""
        print("   Using mock LLaMA model for testing")

        try:
            model = AIFSLocationAwareFusion(
                llama_model_name="meta-llama/Meta-Llama-3-8B",
                grid_shape=(180, 360),
                use_mock_llama=True,
                use_quantization=False,
            )

            # Test model initialization and configuration
            self.assertIsNotNone(model.time_series_tokenizer)
            self.assertIsNotNone(model.llama_model)
            self.assertIsNotNone(model.geographic_resolver)
            self.assertIsNotNone(model.spatial_cropper)

            # Test tokenizer configuration
            tokenizer_info = model.time_series_tokenizer.get_tokenizer_info()
            self.assertEqual(tokenizer_info["temporal_modeling"], "transformer")
            self.assertEqual(tokenizer_info["spatial_dim"], 218)

            # Test spatial processing without full forward pass
            batch_size = 1
            weather_data = torch.randn(batch_size, 5, 3, 180, 360)
            text_queries = ["What's the weather in New York?"]

            # Test location extraction using the correct method
            location_info = model.process_location_query(text_queries[0])
            if location_info and location_info.get("location_name"):
                # Test spatial cropping functionality
                coordinates = location_info.get("coordinates", (40.7128, -74.0060))  # NYC default
                cropped_data = model.spatial_cropper.crop_to_region(
                    weather_data, coordinates[0], coordinates[1], crop_size_deg=2.0
                )
                print(
                    f"   🗺️ Cropped to {location_info.get('location_name', 'Unknown')}: {cropped_data.shape}"
                )

            print("✅ Model configuration and spatial processing validated")

        except Exception as e:
            print(f"❌ Configuration test failed: {e}")
            raise

    def test_real_llama_integration(self):
        """Test with real LLaMA 3-8B model (real model testing)"""
        print("\n🚀 Testing Real LLaMA 3-8B Integration...")

        try:
            # Initialize with real LLaMA model
            model = AIFSLocationAwareFusion(
                llama_model_name="meta-llama/Meta-Llama-3-8B",
                grid_shape=(180, 360),
                use_mock_llama=False,  # Use real LLaMA
                use_quantization=True,  # Enable quantization for memory efficiency
                device="cpu",  # Use CPU for compatibility
            )
            print("✅ Real LLaMA 3-8B model loaded successfully")

            # Test model components
            assert hasattr(model, "geographic_resolver")
            assert hasattr(model, "spatial_cropper")
            assert hasattr(model, "spatial_encoder")
            assert hasattr(model, "llama_model")
            assert hasattr(model, "llama_tokenizer")
            print("✅ All model components initialized")

            # Test tokenizer configuration
            tokenizer_info = model.time_series_tokenizer.get_tokenizer_info()
            self.assertEqual(tokenizer_info["temporal_modeling"], "transformer")
            self.assertEqual(tokenizer_info["spatial_dim"], 218)

            # Create test data for spatial processing
            batch_size = 1
            weather_data = torch.randn(batch_size, 5, 3, 180, 360)
            text_queries = ["What's the weather forecast for Tokyo?"]

            print("🌍 Testing spatial processing components...")
            # Test location extraction and spatial cropping (without full forward pass)
            location_info = model.process_location_query(text_queries[0])
            if location_info and location_info.get("location_name"):
                coordinates = location_info.get("coordinates", (35.6762, 139.6503))  # Tokyo default
                cropped_data = model.spatial_cropper.crop_to_region(
                    weather_data, coordinates[0], coordinates[1], crop_size_deg=2.0
                )
                print(
                    f"   🗺️ Cropped to {location_info.get('location_name', 'Unknown')}: {cropped_data.shape}"
                )

            print("✅ Real LLaMA spatial processing validated")

            # Test different queries for location extraction
            test_queries = [
                "What's the temperature in New York?",
                "How's the weather in London today?",
                "Is it raining in San Francisco?",
            ]

            print("🌐 Testing multiple location queries...")
            for query in test_queries:
                # Test location extraction without full forward pass
                single_location_info = model.process_location_query(query)
                location_name = (
                    single_location_info.get("location_name", "Unknown")
                    if single_location_info
                    else "Unknown"
                )
                print(f"   ✅ Query: '{query}' → Location: {location_name}")

            print("🎉 Real LLaMA 3-8B configuration test completed successfully!")

        except Exception as e:
            print(f"❌ Real LLaMA test failed: {e}")
            import traceback

            traceback.print_exc()
            raise


def test_integration_workflow():
    """Test the complete integration workflow"""
    print("\n🧪 Testing Location-Aware AIFS Integration Workflow")

    # Test 1: Geographic Resolution
    print("\n1. Testing Geographic Resolution...")
    resolver = GeographicResolver()
    location_info = resolver.resolve_location("London")
    if location_info:
        print(f"   ✅ London resolved to: {location_info}")
    else:
        print("   ⚠️ London not found in database")

    # Test 2: Text Processing
    print("\n2. Testing Text Processing...")
    query = "What's the temperature forecast for Tokyo tomorrow?"
    locations = extract_location_keywords(query)
    parsed = parse_climate_query(query)
    print(f"   ✅ Extracted locations: {locations}")
    print(f"   ✅ Parsed query: {parsed}")

    # Test 3: Spatial Cropping
    print("\n3. Testing Spatial Cropping...")
    grid_shape = (180, 360)
    cropper = SpatialCropper(grid_shape)
    weather_data = torch.randn(1, 1, 5, 180, 360)  # batch, time, vars, lat, lon

    if location_info:
        cropped, crop_info = cropper.crop_to_region(weather_data, location_info["bounds"])
        print(f"   ✅ Cropped data shape: {cropped.shape}")
        print(f"   ✅ Crop info: {crop_info}")

    print("\n✅ Integration workflow test completed!")


if __name__ == "__main__":
    print("🚀 Starting AIFS Location-Aware Fusion Tests")

    # Run individual component tests
    try:
        # Test geographic resolution
        test_geo = TestGeographicResolver()
        test_geo.test_basic_location_resolution()
        test_geo.test_location_not_found()

        # Test spatial cropping
        test_spatial = TestSpatialCropper()
        test_spatial.test_spatial_cropping()
        test_spatial.test_boundary_handling()

        # Test text utilities
        test_text = TestTextUtils()
        test_text.test_location_extraction()
        test_text.test_climate_query_parsing()

        # Test model initialization and forward pass
        test_model = TestAIFSLocationAwareFusion()
        test_model.test_model_initialization()
        test_model.test_forward_pass_mock()

        # Test integration workflow
        test_integration_workflow()

        print("\n🎉 All tests passed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

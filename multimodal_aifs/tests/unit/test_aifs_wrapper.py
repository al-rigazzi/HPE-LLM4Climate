#!/usr/bin/env python3
"""
Unit tests for AIFS wrapper.

This module tests the AIFSWrapper class and related functions for integrating
ECMWF AIFS as a climate AI backend.
"""

import unittest
from unittest.mock import patch, MagicMock, Mock
import os
import sys
from pathlib import Path

import numpy as np
import torch
import pytest

from multimodal_aifs.aifs_wrapper import (
    AIFSWrapper,
    _setup_flash_attn_mock,
    create_aifs_backend,
)


class TestFlashAttnMock(unittest.TestCase):
    """Test cases for flash attention mock functionality."""

    def test_setup_flash_attn_mock(self):
        """Test _setup_flash_attn_mock function."""
        # Clear existing mock if present
        if "flash_attn" in sys.modules:
            del sys.modules["flash_attn"]
        if "flash_attn.flash_attn_interface" in sys.modules:
            del sys.modules["flash_attn.flash_attn_interface"]

        with patch("platform.system", return_value="Darwin"), \
             patch("platform.machine", return_value="arm64"):

            _setup_flash_attn_mock()

            # Check that flash_attn module was created
            self.assertIn("flash_attn", sys.modules)
            self.assertIn("flash_attn.flash_attn_interface", sys.modules)

            # Test mock functionality
            flash_attn = sys.modules["flash_attn"]
            self.assertTrue(hasattr(flash_attn, "flash_attn_func"))
            self.assertTrue(hasattr(flash_attn, "FlashAttention"))

    def test_setup_flash_attn_mock_non_darwin(self):
        """Test that mock is not applied on non-Darwin systems."""
        with patch("platform.system", return_value="Linux"):
            # Clear any existing mock
            if "flash_attn" in sys.modules:
                del sys.modules["flash_attn"]

            _setup_flash_attn_mock()

            # Should not create mock on Linux
            # (This test assumes flash_attn is not actually installed)


class TestAIFSWrapper(unittest.TestCase):
    """Test cases for AIFSWrapper class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock paths
        self.mock_model_path = "/fake/path/to/model.ckpt"
        self.mock_config_path = "/fake/path/to/config.yaml"

    def test_initialization_default(self):
        """Test AIFSWrapper initialization with default parameters."""
        wrapper = AIFSWrapper()

        self.assertIsNone(wrapper.model_path)
        self.assertIsNone(wrapper.config_path)
        self.assertIsNone(wrapper.model)
        self.assertFalse(wrapper.is_loaded)

    def test_initialization_custom_paths(self):
        """Test AIFSWrapper initialization with custom paths."""
        wrapper = AIFSWrapper(
            model_path=self.mock_model_path,
            config_path=self.mock_config_path
        )

        self.assertEqual(wrapper.model_path, self.mock_model_path)
        self.assertEqual(wrapper.config_path, self.mock_config_path)
        self.assertIsNone(wrapper.model)
        self.assertFalse(wrapper.is_loaded)

    def test_check_availability_no_paths(self):
        """Test _check_availability when no paths are provided."""
        wrapper = AIFSWrapper()
        self.assertFalse(wrapper._check_availability())

    @patch("os.path.exists")
    def test_check_availability_with_paths(self, mock_exists):
        """Test _check_availability with valid paths."""
        wrapper = AIFSWrapper(
            model_path=self.mock_model_path,
            config_path=self.mock_config_path
        )

        # Test when both files exist
        mock_exists.return_value = True
        self.assertTrue(wrapper._check_availability())

        # Test when files don't exist
        mock_exists.return_value = False
        self.assertFalse(wrapper._check_availability())

    @patch("os.path.exists")
    def test_check_availability_model_only(self, mock_exists):
        """Test _check_availability with only model path."""
        wrapper = AIFSWrapper(model_path=self.mock_model_path)

        mock_exists.return_value = True
        self.assertTrue(wrapper._check_availability())

    @patch.object(AIFSWrapper, "_load_model")
    def test_load_model_success(self, mock_load_model):
        """Test successful model loading."""
        wrapper = AIFSWrapper()
        mock_load_model.return_value = None
        wrapper.is_loaded = True  # Simulate successful loading

        result = wrapper.load_model()

        self.assertIsNone(result)  # Should return None on success
        mock_load_model.assert_called_once()

    @patch.object(AIFSWrapper, "_load_model")
    def test_load_model_failure(self, mock_load_model):
        """Test model loading failure."""
        wrapper = AIFSWrapper()
        mock_load_model.side_effect = Exception("Loading failed")

        with self.assertRaises(Exception):
            wrapper.load_model()

    @patch("torch.load")
    @patch("os.path.exists")
    def test_private_load_model(self, mock_exists, mock_torch_load):
        """Test _load_model method."""
        # Mock a successful model load
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_torch_load.return_value = mock_model

        wrapper = AIFSWrapper(model_path=self.mock_model_path)
        wrapper._load_model()

        self.assertEqual(wrapper.model, mock_model)
        self.assertTrue(wrapper.is_loaded)
        mock_torch_load.assert_called_once()

    def test_predict_not_loaded(self):
        """Test predict method when model is not loaded."""
        wrapper = AIFSWrapper()

        test_data = torch.randn(1, 5, 32, 32)
        with self.assertRaises(RuntimeError):
            wrapper.predict(test_data)

    @patch.object(AIFSWrapper, "_load_model")
    def test_predict_with_loaded_model(self, mock_load_model):
        """Test predict method with loaded model."""
        wrapper = AIFSWrapper()

        # Mock a loaded model
        mock_model = MagicMock()
        mock_model.return_value = torch.randn(1, 5, 32, 32)
        wrapper.model = mock_model
        wrapper.is_loaded = True

        test_data = torch.randn(1, 5, 32, 32)
        result = wrapper.predict(test_data, steps=1)

        self.assertIsInstance(result, torch.Tensor)
        mock_model.assert_called()

    def test_predict_global_not_loaded(self):
        """Test predict_global method when model is not loaded."""
        wrapper = AIFSWrapper()

        with self.assertRaises(RuntimeError):
            wrapper.predict_global(steps=1)

    @patch.object(AIFSWrapper, "_load_model")
    def test_predict_global_with_loaded_model(self, mock_load_model):
        """Test predict_global method with loaded model."""
        wrapper = AIFSWrapper()

        # Mock a loaded model
        mock_model = MagicMock()
        mock_model.return_value = torch.randn(1, 5, 721, 1440)  # Global grid size
        wrapper.model = mock_model
        wrapper.is_loaded = True

        result = wrapper.predict_global(steps=2)

        self.assertIsInstance(result, torch.Tensor)
        mock_model.assert_called()

    def test_get_available_variables(self):
        """Test get_available_variables method."""
        wrapper = AIFSWrapper()
        variables = wrapper.get_available_variables()

        self.assertIsInstance(variables, list)
        self.assertGreater(len(variables), 0)

        # Check that common climate variables are included
        expected_vars = ["temperature_2m", "surface_pressure", "10m_u_component_of_wind"]
        for var in expected_vars:
            self.assertIn(var, variables)

    def test_get_model_info_not_loaded(self):
        """Test get_model_info when model is not loaded."""
        wrapper = AIFSWrapper()
        info = wrapper.get_model_info()

        self.assertIsInstance(info, dict)
        self.assertFalse(info["loaded"])
        self.assertIsNone(info["model"])

    @patch.object(AIFSWrapper, "_load_model")
    def test_get_model_info_loaded(self, mock_load_model):
        """Test get_model_info when model is loaded."""
        wrapper = AIFSWrapper(model_path=self.mock_model_path)

        # Mock a loaded model with parameters
        mock_model = MagicMock()
        mock_model.parameters.return_value = [torch.randn(100, 50), torch.randn(50)]
        wrapper.model = mock_model
        wrapper.is_loaded = True

        info = wrapper.get_model_info()

        self.assertIsInstance(info, dict)
        self.assertTrue(info["loaded"])
        self.assertEqual(info["model_path"], self.mock_model_path)
        self.assertIsInstance(info["parameters"], int)
        self.assertGreater(info["parameters"], 0)

    def test_predict_with_different_data_types(self):
        """Test predict method with different input data types."""
        wrapper = AIFSWrapper()

        # Mock a loaded model
        mock_model = MagicMock()
        mock_model.return_value = torch.randn(2, 5, 32, 32)
        wrapper.model = mock_model
        wrapper.is_loaded = True

        # Test with torch tensor
        tensor_data = torch.randn(2, 5, 32, 32)
        result_tensor = wrapper.predict(tensor_data)
        self.assertIsInstance(result_tensor, torch.Tensor)

        # Test with numpy array
        numpy_data = np.random.randn(2, 5, 32, 32).astype(np.float32)
        result_numpy = wrapper.predict(numpy_data)
        self.assertIsInstance(result_numpy, torch.Tensor)

    def test_predict_with_different_parameters(self):
        """Test predict method with different parameters."""
        wrapper = AIFSWrapper()

        # Mock a loaded model
        mock_model = MagicMock()
        mock_model.return_value = torch.randn(1, 5, 32, 32)
        wrapper.model = mock_model
        wrapper.is_loaded = True

        test_data = torch.randn(1, 5, 32, 32)

        # Test with different steps
        for steps in [1, 3, 6, 12]:
            result = wrapper.predict(test_data, steps=steps)
            self.assertIsInstance(result, torch.Tensor)

        # Test with different variables
        result_temp = wrapper.predict(test_data, variables=["temperature_2m"])
        self.assertIsInstance(result_temp, torch.Tensor)

        result_multi = wrapper.predict(test_data, variables=["temperature_2m", "surface_pressure"])
        self.assertIsInstance(result_multi, torch.Tensor)


class TestAIFSWrapperIntegration(unittest.TestCase):
    """Integration tests for AIFSWrapper."""

    def test_create_aifs_backend(self):
        """Test create_aifs_backend function."""
        backend = create_aifs_backend()

        self.assertIsInstance(backend, AIFSWrapper)
        self.assertIsNotNone(backend.model_path)
        self.assertIsNotNone(backend.config_path)

    @patch("torch.load")
    @patch("os.path.exists")
    def test_full_workflow_mock(self, mock_exists, mock_torch_load):
        """Test full workflow with mocked dependencies."""
        # Setup mocks
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_model.return_value = torch.randn(1, 5, 32, 32)
        mock_torch_load.return_value = mock_model

        # Create wrapper and test workflow
        wrapper = AIFSWrapper(model_path=self.mock_model_path)

        # Load model
        wrapper.load_model()
        self.assertTrue(wrapper.is_loaded)

        # Get model info
        info = wrapper.get_model_info()
        self.assertTrue(info["loaded"])

        # Make predictions
        test_data = torch.randn(2, 5, 32, 32)
        result = wrapper.predict(test_data, steps=3)
        self.assertIsInstance(result, torch.Tensor)

        # Test global prediction
        global_result = wrapper.predict_global(steps=1)
        self.assertIsInstance(global_result, torch.Tensor)

        # Get available variables
        variables = wrapper.get_available_variables()
        self.assertIsInstance(variables, list)
        self.assertGreater(len(variables), 0)


class TestAIFSWrapperErrorHandling(unittest.TestCase):
    """Test error handling in AIFSWrapper."""

    def test_invalid_model_path(self):
        """Test behavior with invalid model path."""
        wrapper = AIFSWrapper(model_path="/nonexistent/path/model.ckpt")
        self.assertFalse(wrapper._check_availability())

    def test_model_loading_error(self):
        """Test handling of model loading errors."""
        with patch("os.path.exists", return_value=True), \
             patch("torch.load", side_effect=Exception("Corrupted model file")):

            wrapper = AIFSWrapper(model_path="/fake/path/model.ckpt")

            with self.assertRaises(Exception):
                wrapper._load_model()

    def test_prediction_with_invalid_input(self):
        """Test prediction with invalid input data."""
        wrapper = AIFSWrapper()

        # Mock loaded model
        wrapper.model = MagicMock()
        wrapper.is_loaded = True

        # Test with wrong input shape
        with self.assertRaises((ValueError, RuntimeError)):
            wrapper.predict(torch.randn(2, 3))  # Wrong number of dimensions

    def test_prediction_parameters_validation(self):
        """Test validation of prediction parameters."""
        wrapper = AIFSWrapper()
        wrapper.model = MagicMock()
        wrapper.is_loaded = True

        test_data = torch.randn(1, 5, 32, 32)

        # Test with invalid steps
        with self.assertRaises(ValueError):
            wrapper.predict(test_data, steps=0)

        with self.assertRaises(ValueError):
            wrapper.predict(test_data, steps=-1)


if __name__ == "__main__":
    unittest.main()

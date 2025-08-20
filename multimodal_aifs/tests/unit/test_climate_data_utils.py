#!/usr/bin/env python3
"""
Simple working tests for climate data utilities.
"""

import unittest

import numpy as np
import torch

from multimodal_aifs.utils.climate_data_utils import (
    CLIMATE_VARIABLES,
    ClimateDataProcessor,
    create_synthetic_climate_data,
)


class TestClimateDataUtils(unittest.TestCase):
    """Test cases for climate data utilities."""

    def test_climate_data_processor_init(self):
        """Test ClimateDataProcessor initialization."""
        processor = ClimateDataProcessor()
        self.assertEqual(processor.normalization_method, "standard")
        self.assertEqual(processor.target_features, 218)
        self.assertFalse(processor.is_fitted)

    def test_climate_data_processor_fit_transform(self):
        """Test fit_transform method."""
        processor = ClimateDataProcessor()
        data = torch.randn(5, 10)  # Simple 2D data

        result = processor.fit_transform(data)
        self.assertIsInstance(result, torch.Tensor)
        self.assertTrue(processor.is_fitted)

    def test_create_synthetic_data(self):
        """Test create_synthetic_climate_data function."""
        data = create_synthetic_climate_data()
        self.assertIsInstance(data, torch.Tensor)
        self.assertEqual(len(data.shape), 4)  # Should be 4D tensor

    def test_create_synthetic_data_custom(self):
        """Test create_synthetic_climate_data with custom parameters."""
        data = create_synthetic_climate_data(batch_size=2, n_variables=3)
        self.assertIsInstance(data, torch.Tensor)
        self.assertEqual(data.shape[0], 2)
        self.assertEqual(data.shape[1], 3)

    def test_climate_variables_constants(self):
        """Test CLIMATE_VARIABLES constants."""
        self.assertIsInstance(CLIMATE_VARIABLES, dict)
        self.assertGreater(len(CLIMATE_VARIABLES), 0)

        for var_name, var_info in CLIMATE_VARIABLES.items():
            self.assertIsInstance(var_name, str)
            self.assertIsInstance(var_info, dict)
            self.assertIn("range", var_info)
            self.assertIn("units", var_info)

    def test_processor_get_stats(self):
        """Test get_stats method."""
        processor = ClimateDataProcessor()
        stats = processor.get_stats()
        self.assertIsInstance(stats, dict)

    def test_processor_normalization_methods(self):
        """Test different normalization methods."""
        data = torch.randn(10, 5)

        # Test standard normalization
        proc_std = ClimateDataProcessor(normalization_method="standard")
        result_std = proc_std.fit_transform(data)
        self.assertIsInstance(result_std, torch.Tensor)

        # Test minmax normalization
        proc_minmax = ClimateDataProcessor(normalization_method="minmax")
        result_minmax = proc_minmax.fit_transform(data)
        self.assertIsInstance(result_minmax, torch.Tensor)


if __name__ == "__main__":
    unittest.main()

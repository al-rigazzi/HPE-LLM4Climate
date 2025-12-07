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
Simple working tests for climate data utilities.
"""

import unittest

import torch

from multimodal_aifs.constants import AIFS_RAW_ENCODER_OUTPUT_DIM
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
        self.assertEqual(processor.target_features, AIFS_RAW_ENCODER_OUTPUT_DIM)
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

    def test_processor_requires_fit_before_transform(self):
        """Calling transform before fitting should raise a RuntimeError."""
        processor = ClimateDataProcessor()
        with self.assertRaises(RuntimeError):
            processor.transform(torch.randn(2, 4))

    def test_process_for_aifs_flattens_and_pads(self):
        """Spatial inputs are flattened and padded to the requested feature count."""
        processor = ClimateDataProcessor(target_features=16)
        spatial = torch.randn(2, 2, 2, 3)
        processor.fit(spatial.view(2, -1))
        processed = processor.process_for_aifs(spatial)
        self.assertEqual(processed.shape, (2, 16))
        self.assertTrue(torch.isfinite(processed).all())

    def test_inverse_transform_roundtrip_minmax(self):
        """Min-max normalization should approximately recover the original data."""
        data = torch.rand(4, 3)
        processor = ClimateDataProcessor(normalization_method="minmax")
        normalized = processor.fit_transform(data)
        recovered = processor.inverse_transform(normalized)
        self.assertTrue(torch.allclose(recovered, data, atol=1e-6))

    def test_get_stats_reports_fitted_values(self):
        """Stats dictionary should expose aggregate metrics after fitting."""
        data = torch.randn(4, 6)
        processor = ClimateDataProcessor()
        processor.fit(data)
        stats = processor.get_stats()
        self.assertTrue(stats["is_fitted"])
        self.assertIn("global_mean", stats)


if __name__ == "__main__":
    unittest.main()

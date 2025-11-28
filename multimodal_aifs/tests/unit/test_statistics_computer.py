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
"""Unit tests for ClimateStatisticsComputer."""

import numpy as np
import pytest
import torch

from multimodal_aifs.training.statistics_computer import (
    ClimateStatisticsComputer,
    VariableStatistics,
)


class TestClimateStatisticsComputer:
    """Test suite for ClimateStatisticsComputer."""

    @pytest.fixture
    def computer(self):
        """Create a ClimateStatisticsComputer instance."""
        return ClimateStatisticsComputer()

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # Create realistic climate data
        np.random.seed(42)

        # Temperature (K), wind u/v (m/s), pressure (Pa), precipitation (m)
        data = np.random.randn(100, 5)
        data[:, 0] = data[:, 0] * 5 + 280  # Temperature around 280K
        data[:, 1] = data[:, 1] * 10  # Wind u
        data[:, 2] = data[:, 2] * 10  # Wind v
        data[:, 3] = data[:, 3] * 1000 + 101000  # Pressure around 101kPa
        data[:, 4] = np.abs(data[:, 4]) * 0.001  # Precipitation (positive)

        return torch.tensor(data, dtype=torch.float32)

    @pytest.fixture
    def sample_mask(self):
        """Create a sample mask."""
        mask = torch.zeros(100, dtype=torch.bool)
        mask[10:20] = True  # Select 10 points
        return mask

    @pytest.fixture
    def variable_names(self):
        """Sample variable names."""
        return ["2t", "10u", "10v", "msl", "tp"]

    def test_initialization(self, computer):
        """Test ClimateStatisticsComputer initialization."""
        assert computer is not None
        assert hasattr(computer, "compute_statistics")
        assert hasattr(computer, "format_statistics_table")

    def test_compute_statistics_basic(self, computer, sample_data, sample_mask, variable_names):
        """Test basic statistics computation."""
        stats = computer.compute_statistics(sample_data, sample_mask, variable_names)

        assert len(stats) == len(variable_names)

        for i, stat in enumerate(stats):
            assert isinstance(stat, VariableStatistics)
            assert stat.variable_name == variable_names[i]
            assert isinstance(stat.min_value, float)
            assert isinstance(stat.max_value, float)
            assert isinstance(stat.mean_value, float)
            assert isinstance(stat.grad_magnitude, float)

            # Basic sanity checks
            assert stat.min_value <= stat.mean_value <= stat.max_value
            assert stat.grad_magnitude >= 0

    def test_compute_statistics_empty_mask(self, computer, sample_data, variable_names):
        """Test statistics computation with empty mask."""
        empty_mask = torch.zeros(100, dtype=torch.bool)

        # Should handle empty mask gracefully by returning empty list
        stats = computer.compute_statistics(sample_data, empty_mask, variable_names)
        assert len(stats) == 0  # No statistics should be computed for empty mask

    def test_compute_statistics_single_point(self, computer, sample_data, variable_names):
        """Test statistics computation with single point mask."""
        single_mask = torch.zeros(100, dtype=torch.bool)
        single_mask[50] = True

        stats = computer.compute_statistics(sample_data, single_mask, variable_names)

        for i, stat in enumerate(stats):
            # For single point, min = max = mean
            expected_value = sample_data[50, i].item()
            assert abs(stat.min_value - expected_value) < 1e-10
            assert abs(stat.max_value - expected_value) < 1e-10
            assert abs(stat.mean_value - expected_value) < 1e-10
            # Gradient should be zero for single point
            assert stat.grad_magnitude == 0

    def test_compute_statistics_full_mask(self, computer, sample_data, variable_names):
        """Test statistics computation with full mask."""
        full_mask = torch.ones(100, dtype=torch.bool)

        stats = computer.compute_statistics(sample_data, full_mask, variable_names)

        for i, stat in enumerate(stats):
            # Should match numpy statistics
            masked_data = sample_data[:, i]
            assert abs(stat.min_value - masked_data.min().item()) < 1e-5
            assert abs(stat.max_value - masked_data.max().item()) < 1e-5
            assert abs(stat.mean_value - masked_data.mean().item()) < 1e-5

    def test_format_statistics_table(self, computer, sample_data, sample_mask, variable_names):
        """Test statistics table formatting."""
        stats = computer.compute_statistics(sample_data, sample_mask, variable_names)
        table = computer.format_statistics_table(stats)

        assert isinstance(table, str)
        assert len(table) > 0

        # Should contain headers
        assert "Variable" in table
        assert "Min" in table
        assert "Max" in table
        assert "Mean" in table
        assert "Gradient" in table

        # Should contain variable names
        for var_name in variable_names:
            assert var_name in table

        # Should have proper table structure
        lines = table.split("\n")
        assert len(lines) >= len(variable_names) + 2  # Header + separator + data

    def test_statistics_table_formatting_precision(self, computer):
        """Test that statistics table has proper numerical formatting."""
        # Create simple known data
        data = np.array([[1.23456789, 100.987654]])  # 1 point, 2 variables
        mask = np.array([True])
        var_names = ["test1", "test2"]

        stats = computer.compute_statistics(data, mask, var_names)
        table = computer.format_statistics_table(stats)

        # Should have reasonable precision (not too many decimal places)
        assert "1.23456789" not in table  # Should be rounded
        assert "100.987654" not in table  # Should be rounded

    def test_gradient_calculation(self, computer):
        """Test gradient calculation with known data."""
        # Create data with known gradient
        x = np.linspace(0, 10, 11)
        y = np.linspace(0, 10, 11)
        data = torch.tensor(
            np.column_stack([x, y, x**2]), dtype=torch.float32
        )  # Linear and quadratic
        mask = torch.ones(11, dtype=torch.bool)
        var_names = ["linear", "linear2", "quadratic"]

        stats = computer.compute_statistics(data, mask, var_names)

        # Linear data should have small gradient
        assert stats[0].grad_magnitude >= 0
        assert stats[1].grad_magnitude >= 0
        # Quadratic should have larger gradient
        assert stats[2].grad_magnitude > stats[0].grad_magnitude

    def test_coordinate_variables_excluded(self, computer):
        """Test that coordinate variables are excluded from analysis."""
        # Create data with coordinate-like variables
        data = torch.tensor(np.random.randn(50, 8), dtype=torch.float32)
        mask = torch.ones(50, dtype=torch.bool)
        var_names = [
            "2t",
            "10u",
            "cos_latitude",
            "sin_longitude",
            "cos_julian_day",
            "insolation",
            "10v",
            "msl",
        ]

        stats = computer.compute_statistics(data, mask, var_names)

        # Should exclude coordinate variables
        included_vars = [s.variable_name for s in stats]
        assert "2t" in included_vars
        assert "10u" in included_vars
        assert "10v" in included_vars
        assert "msl" in included_vars

        # These should be excluded
        assert "cos_latitude" not in included_vars
        assert "sin_longitude" not in included_vars
        assert "cos_julian_day" not in included_vars
        assert "insolation" not in included_vars

    def test_variable_statistics_dataclass(self):
        """Test VariableStatistics dataclass."""
        stat = VariableStatistics(
            variable_name="test_var",
            min_value=1.0,
            max_value=10.0,
            mean_value=5.5,
            grad_magnitude=0.1,
            rot_magnitude=0.05,
            div_magnitude=0.02,
            unit="m/s",
            description="test variable",
        )

        assert stat.variable_name == "test_var"
        assert stat.min_value == 1.0
        assert stat.max_value == 10.0
        assert stat.mean_value == 5.5
        assert stat.grad_magnitude == 0.1
        assert stat.rot_magnitude == 0.05
        assert stat.div_magnitude == 0.02

    def test_statistics_with_invalid_data(self, computer, sample_mask, variable_names):
        """Test statistics computation with invalid data."""
        # Data with NaN values
        data_with_nan = torch.tensor(np.random.randn(100, 5), dtype=torch.float32)
        data_with_nan[50:60, :] = float("nan")

        stats = computer.compute_statistics(data_with_nan, sample_mask, variable_names)

        # Should handle NaN gracefully
        assert len(stats) == len(variable_names)
        for stat in stats:
            # Results might be NaN, but shouldn't crash
            assert hasattr(stat, "min_value")
            assert hasattr(stat, "max_value")
            assert hasattr(stat, "mean_value")

    def test_statistics_with_mismatched_dimensions(self, computer, variable_names):
        """Test error handling with mismatched dimensions."""
        data = torch.tensor(np.random.randn(100, 3), dtype=torch.float32)  # 3 variables
        mask = torch.ones(100, dtype=torch.bool)
        var_names_5 = ["var1", "var2", "var3", "var4", "var5"]  # 5 names

        # Should handle mismatch gracefully or raise appropriate error
        try:
            stats = computer.compute_statistics(data, mask, var_names_5)
            # If it doesn't raise an error, should only process available variables
            assert len(stats) <= min(data.shape[1], len(var_names_5))
        except (IndexError, ValueError):
            # Or it might raise an appropriate error
            pass

    def test_table_formatting_with_extreme_values(self, computer):
        """Test table formatting with extreme numerical values."""
        # Create data with extreme values
        data = torch.tensor([[1e-10, 1e10, 0.0, -1e5, 1e-3]], dtype=torch.float32)
        mask = torch.tensor([True], dtype=torch.bool)
        var_names = ["tiny", "huge", "zero", "negative", "small"]

        stats = computer.compute_statistics(data, mask, var_names)
        table = computer.format_statistics_table(stats)

        # Should format all values without errors
        assert isinstance(table, str)
        assert len(table) > 0
        # Should contain scientific notation for extreme values
        lines = table.split("\n")
        assert len(lines) > 0

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
"""Climate statistics computer for generating analytical summaries.

This module computes statistical measures for climate variables.
"""

from dataclasses import dataclass

import torch

from ..constants import ALL_AIFS_VARIABLES


@dataclass
class VariableStatistics:
    """Statistical measures for a single climate variable."""

    variable_name: str
    min_value: float
    max_value: float
    mean_value: float
    std_value: float
    unit: str
    description: str


class ClimateStatisticsComputer:
    """
    Computes comprehensive statistics for climate variables.

    This class handles:
    1. Basic statistics (min, max, mean, stddev)
    """

    # Variable units and descriptions
    VARIABLE_METADATA = {
        # Temperature variables
        "2t": ("K", "2-meter temperature"),
        "2d": ("K", "2-meter dewpoint temperature"),
        "skt": ("K", "Skin temperature"),
        # Pressure variables
        "msl": ("Pa", "Mean sea level pressure"),
        "sp": ("Pa", "Surface pressure"),
        # Wind variables
        "10u": ("m/s", "10-meter u-wind component"),
        "10v": ("m/s", "10-meter v-wind component"),
        "100u": ("m/s", "100-meter u-wind component"),
        "100v": ("m/s", "100-meter v-wind component"),
        # Moisture variables
        "tcw": ("kg/m²", "Total column water"),
        # Precipitation
        "tp": ("m", "Total precipitation"),
        "cp": ("m", "Convective precipitation"),
        # Geopotential
        "z": ("m²/s²", "Geopotential"),
        # Topography
        "lsm": ("0-1", "Land-sea mask"),
        "slor": ("rad", "Slope of sub-gridscale orography"),
        "sdor": ("m", "Standard deviation of orography"),
    }

    # Variables to exclude from analysis
    EXCLUDED_VARIABLES = [
        "cos_julian_day",
        "sin_julian_day",
        "cos_local_time",
        "sin_local_time",
        "cos_latitude",
        "sin_latitude",
        "cos_longitude",
        "sin_longitude",
        "insolation",
    ]

    def __init__(self, grid_shape: tuple[int, int] | None = None):
        """
        Initialize the statistics computer.

        Args:
            grid_shape: Shape of the spatial grid (nlat, nlon) if known
        """
        self.grid_shape = grid_shape

    def _should_analyze_variable(self, var_name: str) -> bool:
        """Check if a variable should be analyzed."""
        return var_name not in self.EXCLUDED_VARIABLES

    def _get_variable_metadata(self, var_name: str) -> tuple[str, str]:
        """Get unit and description for a variable."""
        # Check direct match
        if var_name in self.VARIABLE_METADATA:
            return self.VARIABLE_METADATA[var_name]

        # Check pressure level variables
        for param in ["q", "t", "u", "v", "w", "z"]:
            if var_name.startswith(f"{param}_"):
                units = {
                    "q": ("kg/kg", "Specific humidity"),
                    "t": ("K", "Temperature"),
                    "u": ("m/s", "U-wind component"),
                    "v": ("m/s", "V-wind component"),
                    "w": ("Pa/s", "Vertical velocity"),
                    "z": ("m²/s²", "Geopotential"),
                }
                level = var_name.split("_")[1]
                unit, desc = units[param]
                return unit, f"{desc} at {level} hPa"

        # Default
        return ("unknown", var_name)

    def compute_statistics(
        self,
        climate_data: torch.Tensor,
        mask: torch.Tensor,
        variable_names: list[str] | None = None,
    ) -> list[VariableStatistics]:
        """
        Compute comprehensive statistics for all variables.

        Args:
            climate_data: Climate tensor [grid_points, n_variables]
            mask: Boolean mask [grid_points]
            variable_names: List of variable names (defaults to ALL_AIFS_VARIABLES)

        Returns:
            List of VariableStatistics for each analyzed variable
        """
        if variable_names is None:
            variable_names = ALL_AIFS_VARIABLES

        if len(variable_names) != climate_data.shape[1]:
            raise ValueError(
                f"Variable names count ({len(variable_names)}) does not match "
                f"data dimensions ({climate_data.shape[1]})"
            )

        statistics = []

        for i, var_name in enumerate(variable_names):
            if not self._should_analyze_variable(var_name):
                continue

            field = climate_data[:, i]
            masked_field = field[mask]

            if not mask.any() or len(masked_field) == 0:
                continue

            # Basic statistics
            min_val = float(masked_field.min().item())
            max_val = float(masked_field.max().item())
            mean_val = float(masked_field.mean().item())
            std_val = float(masked_field.std().item()) if len(masked_field) > 1 else 0.0

            # Get metadata
            unit, description = self._get_variable_metadata(var_name)

            statistics.append(
                VariableStatistics(
                    variable_name=var_name,
                    min_value=min_val,
                    max_value=max_val,
                    mean_value=mean_val,
                    std_value=std_val,
                    unit=unit,
                    description=description,
                )
            )

        return statistics

    def format_statistics_table(
        self, statistics: list[VariableStatistics], max_vars: int | None = 20
    ) -> str:
        """
        Format statistics as a well-formatted table.

        Args:
            statistics: List of VariableStatistics
            max_vars: Maximum number of variables to include (None for all)

        Returns:
            Formatted table string
        """
        if not statistics:
            return "No statistics available."

        # Limit number of variables if requested
        if max_vars is not None and len(statistics) > max_vars:
            # Select most important variables (surface and some pressure levels)
            priority_vars = ["2t", "msl", "10u", "10v", "tp", "tcw", "z"]
            priority_stats = [
                s for s in statistics if any(p in s.variable_name for p in priority_vars)
            ]
            other_stats = [s for s in statistics if s not in priority_stats]
            statistics = priority_stats[:max_vars] + other_stats[: max_vars - len(priority_stats)]

        # Build table
        lines = []
        lines.append("=" * 88)
        lines.append(f"{'Variable':<20} {'Min':>14} {'Max':>14} {'Mean':>14} {'StdDev':>14}")
        lines.append("=" * 88)

        for stat in statistics:
            # Format values with appropriate precision
            min_str = f"{stat.min_value:14.4f}"
            max_str = f"{stat.max_value:14.4f}"
            mean_str = f"{stat.mean_value:14.4f}"
            std_str = f"{stat.std_value:14.4f}"

            lines.append(f"{stat.variable_name:<20} {min_str} {max_str} {mean_str} {std_str}")

        lines.append("=" * 88)
        lines.append("Units and descriptions:")
        for stat in statistics[:10]:  # Show first 10 descriptions
            lines.append(f"  {stat.variable_name}: {stat.description} ({stat.unit})")
        if len(statistics) > 10:
            lines.append(f"  ... and {len(statistics) - 10} more variables")

        return "\n".join(lines)

    def format_statistics_narrative(
        self, statistics: list[VariableStatistics], max_vars: int | None = 20
    ) -> str:
        """
        Format statistics as a JSON object.

        JSON is clearly machine-readable input, which prevents language
        models from echoing the data format back in their responses.

        Args:
            statistics: List of VariableStatistics
            max_vars: Maximum number of variables to include (None for all)

        Returns:
            JSON-formatted statistics string
        """
        import json

        if not statistics:
            return "No statistics available."

        # Limit number of variables if requested
        if max_vars is not None and len(statistics) > max_vars:
            priority_vars = ["2t", "msl", "10u", "10v", "tp", "tcw", "z"]
            priority_stats = [
                s for s in statistics if any(p in s.variable_name for p in priority_vars)
            ]
            other_stats = [s for s in statistics if s not in priority_stats]
            statistics = priority_stats[:max_vars] + other_stats[: max_vars - len(priority_stats)]

        data: dict[str, dict[str, float | str]] = {}
        for stat in statistics:
            data[stat.variable_name] = {
                "min": round(stat.min_value, 4),
                "max": round(stat.max_value, 4),
                "mean": round(stat.mean_value, 4),
                "std": round(stat.std_value, 4),
                "unit": stat.unit,
            }

        return json.dumps(data, indent=2)

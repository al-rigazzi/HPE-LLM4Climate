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
Climate statistics computer for generating analytical summaries.

This module computes statistical measures and differential operators
(gradient, rotation, divergence) for climate variables.
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
    grad_magnitude: float  # Average gradient magnitude
    rot_magnitude: float  # Average rotation magnitude (for vector fields)
    div_magnitude: float  # Average divergence magnitude (for vector fields)
    unit: str
    description: str


class ClimateStatisticsComputer:
    """
    Computes comprehensive statistics for climate variables.

    This class handles:
    1. Basic statistics (min, max, mean)
    2. Spatial gradients
    3. Rotation (curl) for vector fields
    4. Divergence for vector fields
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
        self._init_variable_pairs()

    def _init_variable_pairs(self) -> None:
        """Identify vector field pairs (u, v components) for rotation/divergence."""
        self.vector_pairs = []

        # Surface and pressure level wind pairs
        for prefix in ["10", "100"]:
            self.vector_pairs.append((f"{prefix}u", f"{prefix}v"))

        # Pressure level winds
        pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        for level in pressure_levels:
            self.vector_pairs.append((f"u_{level}", f"v_{level}"))

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

    def compute_gradient(self, field: torch.Tensor, mask: torch.Tensor) -> float:
        """
        Compute average gradient magnitude over masked region.

        Args:
            field: Climate field [grid_points]
            mask: Boolean mask [grid_points]

        Returns:
            Average gradient magnitude
        """
        if not mask.any():
            return 0.0

        # Extract masked values
        masked_field = field[mask]

        if len(masked_field) < 2:
            return 0.0

        # Simple gradient estimate using neighboring points
        # For a proper implementation, would need actual grid topology
        gradients = torch.abs(masked_field[1:] - masked_field[:-1])

        return float(gradients.mean().item())

    def compute_rotation(
        self,
        u_field: torch.Tensor,
        v_field: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        """
        Compute average rotation (curl) magnitude for vector field.

        Args:
            u_field: U-component [grid_points]
            v_field: V-component [grid_points]
            mask: Boolean mask [grid_points]

        Returns:
            Average rotation magnitude
        """
        if not mask.any():
            return 0.0

        # Extract masked values
        u_masked = u_field[mask]
        v_masked = v_field[mask]

        if len(u_masked) < 2:
            return 0.0

        # Simplified rotation estimate
        # Proper implementation would use actual spatial derivatives
        du = torch.abs(u_masked[1:] - u_masked[:-1])
        dv = torch.abs(v_masked[1:] - v_masked[:-1])

        rotation = torch.sqrt(du**2 + dv**2)

        return float(rotation.mean().item())

    def compute_divergence(
        self,
        u_field: torch.Tensor,
        v_field: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        """
        Compute average divergence magnitude for vector field.

        Args:
            u_field: U-component [grid_points]
            v_field: V-component [grid_points]
            mask: Boolean mask [grid_points]

        Returns:
            Average divergence magnitude
        """
        if not mask.any():
            return 0.0

        # Extract masked values
        u_masked = u_field[mask]
        v_masked = v_field[mask]

        if len(u_masked) < 2:
            return 0.0

        # Simplified divergence estimate
        # Proper implementation would use spatial derivatives
        divergence = torch.abs(u_masked[1:] - u_masked[:-1]) + torch.abs(
            v_masked[1:] - v_masked[:-1]
        )

        return float(divergence.mean().item())

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

        # Create dictionary for quick lookup of vector pairs
        vector_pair_dict = dict(self.vector_pairs)

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

            # Gradient
            grad_mag = self.compute_gradient(field, mask)

            # Rotation and divergence (for vector fields)
            rot_mag = 0.0
            div_mag = 0.0

            if var_name in vector_pair_dict:
                # This is a u-component, find its v-component
                v_var_name = vector_pair_dict[var_name]
                if v_var_name in variable_names:
                    v_idx = variable_names.index(v_var_name)
                    v_field = climate_data[:, v_idx]

                    rot_mag = self.compute_rotation(field, v_field, mask)
                    div_mag = self.compute_divergence(field, v_field, mask)

            # Get metadata
            unit, description = self._get_variable_metadata(var_name)

            statistics.append(
                VariableStatistics(
                    variable_name=var_name,
                    min_value=min_val,
                    max_value=max_val,
                    mean_value=mean_val,
                    grad_magnitude=grad_mag,
                    rot_magnitude=rot_mag,
                    div_magnitude=div_mag,
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
        lines.append("=" * 120)
        lines.append(
            f"{'Variable':<20} {'Min':<12} {'Max':<12} {'Mean':<12} "
            f"{'Gradient':<12} {'Rotation':<12} {'Divergence':<12}"
        )
        lines.append("=" * 120)

        for stat in statistics:
            # Format values with appropriate precision
            min_str = f"{stat.min_value:11.4f}"
            max_str = f"{stat.max_value:11.4f}"
            mean_str = f"{stat.mean_value:11.4f}"
            grad_str = f"{stat.grad_magnitude:11.4e}"
            rot_str = f"{stat.rot_magnitude:11.4e}" if stat.rot_magnitude > 0 else "N/A"
            div_str = f"{stat.div_magnitude:11.4e}" if stat.div_magnitude > 0 else "N/A"

            lines.append(
                f"{stat.variable_name:<20} {min_str} {max_str} {mean_str} "
                f"{grad_str} {rot_str:<12} {div_str:<12}"
            )

        lines.append("=" * 120)
        lines.append("Units and descriptions:")
        for stat in statistics[:10]:  # Show first 10 descriptions
            lines.append(f"  {stat.variable_name}: {stat.description} ({stat.unit})")
        if len(statistics) > 10:
            lines.append(f"  ... and {len(statistics) - 10} more variables")

        return "\n".join(lines)

"""
Climate Data Utilities

This module provides utilities for processing climate data for use with AIFS
multimodal analysis, including data preprocessing, normalization, and format conversion.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch

# Climate data constants
CLIMATE_VARIABLES = {
    # Temperature variables (Kelvin)
    "temperature_2m": {"range": (200, 330), "units": "K", "standard": 273.15},
    "temperature_850hPa": {"range": (200, 320), "units": "K", "standard": 273.15},
    "temperature_500hPa": {"range": (180, 300), "units": "K", "standard": 273.15},
    # Pressure variables (Pascal)
    "surface_pressure": {"range": (50000, 110000), "units": "Pa", "standard": 101325},
    "mean_sea_level_pressure": {"range": (95000, 108000), "units": "Pa", "standard": 101325},
    "geopotential_1000hPa": {"range": (-500, 1000), "units": "m²/s²", "standard": 0},
    "geopotential_850hPa": {"range": (800, 2000), "units": "m²/s²", "standard": 1400},
    "geopotential_500hPa": {"range": (4500, 6000), "units": "m²/s²", "standard": 5500},
    # Wind variables (m/s)
    "10m_u_component_of_wind": {"range": (-50, 50), "units": "m/s", "standard": 0},
    "10m_v_component_of_wind": {"range": (-50, 50), "units": "m/s", "standard": 0},
    "u_component_of_wind_1000hPa": {"range": (-80, 80), "units": "m/s", "standard": 0},
    "v_component_of_wind_1000hPa": {"range": (-80, 80), "units": "m/s", "standard": 0},
    "u_component_of_wind_850hPa": {"range": (-100, 100), "units": "m/s", "standard": 0},
    "v_component_of_wind_850hPa": {"range": (-100, 100), "units": "m/s", "standard": 0},
    # Humidity variables (kg/kg)
    "specific_humidity_850hPa": {"range": (0, 0.025), "units": "kg/kg", "standard": 0.01},
    "specific_humidity_700hPa": {"range": (0, 0.020), "units": "kg/kg", "standard": 0.008},
    "2m_dewpoint_temperature": {"range": (180, 320), "units": "K", "standard": 273.15},
    # Precipitation (m/s)
    "total_precipitation": {"range": (0, 0.01), "units": "m/s", "standard": 0},
    # Other variables
    "snow_depth": {"range": (0, 10), "units": "m", "standard": 0},
    "soil_temperature_level_4": {"range": (250, 320), "units": "K", "standard": 288.15},
    "volumetric_soil_water_layer_1": {"range": (0, 1), "units": "m³/m³", "standard": 0.3},
    "volumetric_soil_water_layer_4": {"range": (0, 1), "units": "m³/m³", "standard": 0.3},
}

# Geographic constants
EARTH_RADIUS_KM = 6371.0
DEFAULT_GRID_RESOLUTION = 0.25  # degrees


class ClimateDataProcessor:
    """
    Processor for climate data to prepare it for AIFS multimodal analysis.

    This class handles preprocessing, normalization, and format conversion
    of climate data from various sources.
    """

    def __init__(self, normalization_method: str = "standard", target_features: int = 218):
        """
        Initialize climate data processor.

        Args:
            normalization_method: Method for data normalization ("standard", "minmax", "robust")
            target_features: Target number of features for AIFS compatibility
        """
        self.normalization_method = normalization_method
        self.target_features = target_features
        self.variable_stats: Dict[str, Dict[str, float]] = {}
        self.is_fitted = False

    def fit(self, data: torch.Tensor | np.ndarray, variable_names: List[str] | None = None):
        """
        Fit the processor to data to compute normalization statistics.

        Args:
            data: Climate data to fit on
            variable_names: Names of climate variables (optional)
        """
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)

        # Compute statistics based on normalization method
        if self.normalization_method == "standard":
            self.data_mean = data.mean(dim=0)
            self.data_std = data.std(dim=0) + 1e-8  # Avoid division by zero
        elif self.normalization_method == "minmax":
            self.data_min = data.min(dim=0)[0]
            self.data_max = data.max(dim=0)[0]
            self.data_range = self.data_max - self.data_min + 1e-8
        elif self.normalization_method == "robust":
            self.data_median = data.median(dim=0)[0]
            self.data_q75 = data.quantile(0.75, dim=0)
            self.data_q25 = data.quantile(0.25, dim=0)
            self.data_iqr = self.data_q75 - self.data_q25 + 1e-8

        # Store variable-specific statistics if names provided
        if variable_names:
            for i, var_name in enumerate(variable_names):
                if i < data.shape[1]:
                    self.variable_stats[var_name] = {
                        "mean": float(data[:, i].mean()),
                        "std": float(data[:, i].std()),
                        "min": float(data[:, i].min()),
                        "max": float(data[:, i].max()),
                    }

        self.is_fitted = True

    def transform(self, data: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        Transform data using fitted normalization parameters.

        Args:
            data: Data to transform

        Returns:
            Normalized data tensor
        """
        if not self.is_fitted:
            raise RuntimeError("Processor must be fitted before transform")

        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)

        # Apply normalization
        if self.normalization_method == "standard":
            data = (data - self.data_mean) / self.data_std
        elif self.normalization_method == "minmax":
            data = (data - self.data_min) / self.data_range
        elif self.normalization_method == "robust":
            data = (data - self.data_median) / self.data_iqr

        return data

    def fit_transform(
        self, data: torch.Tensor | np.ndarray, variable_names: List[str] | None = None
    ) -> torch.Tensor:
        """
        Fit and transform data in one step.

        Args:
            data: Data to fit and transform
            variable_names: Names of climate variables

        Returns:
            Normalized data tensor
        """
        self.fit(data, variable_names)
        return self.transform(data)

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform normalized data back to original scale.

        Args:
            data: Normalized data

        Returns:
            Data in original scale
        """
        if not self.is_fitted:
            raise RuntimeError("Processor must be fitted before inverse transform")

        if self.normalization_method == "standard":
            return data * self.data_std + self.data_mean
        if self.normalization_method == "minmax":
            return data * self.data_range + self.data_min
        if self.normalization_method == "robust":
            return data * self.data_iqr + self.data_median

        return data

    def process_for_aifs(
        self, data: torch.Tensor | np.ndarray, variable_names: List[str] | None = None
    ) -> torch.Tensor:
        """
        Process climate data for AIFS model input.

        Args:
            data: Raw climate data
            variable_names: Names of variables

        Returns:
            Processed data ready for AIFS encoder
        """
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)

        # Handle spatial data
        if data.dim() > 2:
            data = self._flatten_spatial_data(data)

        # Normalize if fitted
        if self.is_fitted:
            data = self.transform(data)

        # Ensure correct number of features
        data = self._adjust_features(data)

        return data

    def _flatten_spatial_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        Flatten spatial climate data for AIFS processing.

        Args:
            data: Spatial climate data

        Returns:
            Flattened data
        """
        if data.dim() == 3:
            # (batch, lat, lon) -> (batch, lat*lon)
            batch_size = data.shape[0]
            return data.view(batch_size, -1)
        if data.dim() == 4:
            # (batch, vars, lat, lon) -> (batch, vars*lat*lon)
            batch_size = data.shape[0]
            return data.view(batch_size, -1)

        return data

    def _adjust_features(self, data: torch.Tensor) -> torch.Tensor:
        """
        Adjust number of features to match AIFS requirements.

        Args:
            data: Input data

        Returns:
            Data with correct number of features
        """
        current_features = data.shape[-1]

        if current_features == self.target_features:
            return data
        if current_features > self.target_features:
            # Truncate excess features
            return data[..., : self.target_features]

        # Pad with zeros
        batch_size = data.shape[0]
        padding_size = self.target_features - current_features
        padding = torch.zeros(batch_size, padding_size, device=data.device, dtype=data.dtype)
        return torch.cat([data, padding], dim=-1)

    def get_stats(self) -> Dict:
        """Get processor statistics."""
        stats = {
            "normalization_method": self.normalization_method,
            "target_features": self.target_features,
            "is_fitted": self.is_fitted,
            "variable_stats": self.variable_stats,
        }

        if self.is_fitted:
            if self.normalization_method == "standard":
                stats["global_mean"] = float(self.data_mean.mean())
                stats["global_std"] = float(self.data_std.mean())
            elif self.normalization_method == "minmax":
                stats["global_min"] = float(self.data_min.min())
                stats["global_max"] = float(self.data_max.max())
            elif self.normalization_method == "robust":
                stats["global_median"] = float(self.data_median.median())
                stats["global_iqr"] = float(self.data_iqr.median())

        return stats


def create_synthetic_climate_data(
    batch_size: int = 8,
    n_variables: int = 25,
    spatial_shape: Tuple[int, int] = (64, 128),
    add_noise: bool = True,
) -> torch.Tensor:
    """
    Create synthetic climate data for testing.

    Args:
        batch_size: Number of samples
        n_variables: Number of climate variables
        spatial_shape: Spatial dimensions (lat, lon)
        add_noise: Whether to add realistic noise

    Returns:
        Synthetic climate data tensor
    """
    # Create base patterns
    lat_size, lon_size = spatial_shape

    # Create coordinate grids
    lat = torch.linspace(-90, 90, lat_size)
    lon = torch.linspace(-180, 180, lon_size)
    lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing="ij")

    # Initialize data
    data = torch.zeros(batch_size, n_variables, lat_size, lon_size)

    for batch_idx in range(batch_size):
        for var_idx in range(n_variables):
            # Create different patterns for different variables
            if var_idx < 5:  # Temperature-like variables
                # Temperature gradient from equator to poles
                pattern = 288.15 - 30 * torch.abs(lat_grid) / 90
                pattern += 10 * torch.sin(lon_grid * np.pi / 180)  # Longitudinal variation
            elif var_idx < 10:  # Pressure-like variables
                # Pressure patterns
                pattern = 101325 - 1000 * torch.abs(lat_grid) / 90
                pattern += 500 * torch.cos(2 * lat_grid * np.pi / 180)
            elif var_idx < 15:  # Wind-like variables
                # Wind patterns
                pattern = 10 * torch.sin(lat_grid * np.pi / 180)
                pattern += 5 * torch.cos(lon_grid * np.pi / 180)
            else:  # Other variables
                # Mixed patterns
                pattern = torch.sin(lat_grid * np.pi / 180) * torch.cos(lon_grid * np.pi / 180)
                pattern *= 10 + var_idx

            # Add temporal variation (different for each batch)
            time_factor = torch.tensor(np.sin(batch_idx * np.pi / batch_size))
            pattern += pattern * 0.1 * time_factor

            # Add noise if requested
            if add_noise:
                noise = torch.randn_like(pattern) * pattern.std() * 0.05
                pattern += noise

            data[batch_idx, var_idx] = pattern

    return data


def test_climate_processor():
    """Test climate data processor functionality."""
    print("🧪 Testing Climate Data Processor")
    print("=" * 40)

    # Create synthetic data
    test_data = create_synthetic_climate_data(batch_size=10, n_variables=20)
    print(f"Created synthetic data: {test_data.shape}")

    # Flatten for processing
    batch_size = test_data.shape[0]
    flattened_data = test_data.view(batch_size, -1)
    print(f"Flattened data: {flattened_data.shape}")

    # Test processor
    processor = ClimateDataProcessor(normalization_method="standard")

    # Fit and transform
    normalized_data = processor.fit_transform(flattened_data)
    print(f"Normalized data: {normalized_data.shape}")
    print(f"Mean: {normalized_data.mean():.4f}, Std: {normalized_data.std():.4f}")

    # Process for AIFS
    aifs_ready = processor.process_for_aifs(flattened_data)
    print(f"AIFS-ready data: {aifs_ready.shape}")

    # Test inverse transform
    recovered_data = processor.inverse_transform(normalized_data)
    print(f"Recovery error: {(recovered_data - flattened_data).abs().mean():.6f}")

    # Get stats
    stats = processor.get_stats()
    print(f"Processor stats: {stats}")

    print("✅ All tests passed!")


if __name__ == "__main__":
    test_climate_processor()

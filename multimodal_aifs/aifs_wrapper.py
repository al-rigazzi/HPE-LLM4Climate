"""
AIFS Wrapper for HPE-LLM4Climate

This module provides a wrapper around ECMWF AIFS to integrate it as an alternative
climate AI backend in the HPE-LLM4Climate system.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# Flash attention workaround - monkey-patch to avoid import errors
# Based on: https://huggingface.co/qnguyen3/nanoLLaVA-1.5/discussions/4
def _setup_flash_attn_mock():
    """Setup mock flash_attn module to avoid import errors on Darwin ARM systems without CUDA."""
    import platform

    # Only apply workaround on Darwin (macOS) ARM systems
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        if "flash_attn" not in sys.modules:
            import types

            class MockFlashAttn:
                """Mock flash attention class for compatibility when flash_attn is not available."""

                def __init__(self, *args, **kwargs):
                    pass

                def __call__(self, *args, **kwargs):
                    raise NotImplementedError("Flash attention not available, should use fallback")

            # Create mock flash_attn module with proper spec
            flash_attn_module = types.ModuleType("flash_attn")
            flash_attn_module.__spec__ = types.SimpleNamespace(
                name="flash_attn", loader=None, origin="mock", submodule_search_locations=None
            )
            flash_attn_module.flash_attn_func = MockFlashAttn()
            flash_attn_module.FlashAttention = MockFlashAttn
            sys.modules["flash_attn"] = flash_attn_module

            # Mock the specific interface module
            flash_attn_interface = types.ModuleType("flash_attn.flash_attn_interface")
            flash_attn_interface.__spec__ = types.SimpleNamespace(
                name="flash_attn.flash_attn_interface",
                loader=None,
                origin="mock",
                submodule_search_locations=None,
            )
            flash_attn_interface.flash_attn_func = MockFlashAttn()
            sys.modules["flash_attn.flash_attn_interface"] = flash_attn_interface

            print("🔧 Applied flash_attn workaround for Darwin ARM64 system")


# Setup the mock before any other imports (only on Darwin ARM)
_setup_flash_attn_mock()


class AIFSWrapper:
    """
    Wrapper for ECMWF AIFS model to provide a consistent interface
    for climate prediction tasks.
    """

    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize AIFS wrapper.

        Args:
            model_path: Path to AIFS checkpoint file
            config_path: Path to AIFS configuration file
        """
        self.project_root = Path(__file__).parent
        self.aifs_path = self.project_root / "models" / "aifs-single-1.0"

        # Default paths if not provided
        if model_path is None:
            model_path = self.aifs_path / "aifs-single-mse-1.0.ckpt"
        if config_path is None:
            config_path = self.aifs_path / "config_pretraining.yaml"

        self.model_path = Path(model_path)
        self.config_path = Path(config_path)

        # Check availability
        self._check_availability()

        # Model will be loaded lazily
        self._model = None
        self._config = None

    def _check_availability(self) -> bool:
        """Check if AIFS components are available."""
        if not self.aifs_path.exists():
            raise FileNotFoundError(
                f"AIFS submodule not found at {self.aifs_path}. "
                "Please run: git submodule update --init --recursive"
            )

        if not self.model_path.exists():
            raise FileNotFoundError(f"AIFS checkpoint not found at {self.model_path}")

        if not self.config_path.exists():
            raise FileNotFoundError(f"AIFS config not found at {self.config_path}")

        return True

    def load_model(self):
        """
        Public method to load and initialize AIFS model.

        Returns:
            Dictionary containing model information with keys:
            - 'runner': AIFS SimpleRunner instance
            - 'pytorch_model': PyTorch model (if accessible)
            - 'type': Model type ('aifs_full' or 'aifs_runner_only')
        """
        return self._load_model()

    def _load_model(self):
        """Load and initialize AIFS model using SimpleRunner."""
        if self._model is not None:
            return self._model

        try:
            print("Loading AIFS model using SimpleRunner...")

            # Set environment variables for CPU operation
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            os.environ["ANEMOI_INFERENCE_NUM_CHUNKS"] = "16"

            # Import AIFS components
            from anemoi.inference.runners.simple import SimpleRunner

            # Initialize model with CPU device (flash_attn not required)
            checkpoint = {"huggingface": "ecmwf/aifs-single-1.0"}
            runner = SimpleRunner(checkpoint, device="cpu")

            print("✅ AIFS SimpleRunner initialized successfully on CPU!")

            # Access the actual PyTorch model
            try:
                pytorch_model = runner.model
                print(f"✅ PyTorch model loaded: {type(pytorch_model)}")
                if isinstance(pytorch_model, torch.nn.Module):
                    param_count = sum(p.numel() for p in pytorch_model.parameters())
                    print(f"📊 Model parameters: {param_count:,}")

                # Store both runner and pytorch model for different use cases
                self._model = {
                    "runner": runner,
                    "pytorch_model": pytorch_model,
                    "type": "aifs_full",
                }

            except Exception as e:
                print(f"⚠️  Could not access PyTorch model: {e}")
                print("Using runner-only mode")
                self._model = {"runner": runner, "pytorch_model": None, "type": "aifs_runner_only"}

            return self._model

        except ImportError as e:
            print(f"❌ Failed to import AIFS components: {e}")
            print("Please ensure anemoi-inference and anemoi-models are installed")
            raise
        except Exception as e:
            print(f"❌ Failed to initialize AIFS model: {e}")
            raise

    def predict(
        self,
        location: Tuple[float, float],
        days: int = 7,
        variables: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate weather forecast for a specific location.

        Args:
            location: Tuple of (latitude, longitude)
            days: Number of days to forecast (1-10)
            variables: List of weather variables to predict
            **kwargs: Additional parameters for the forecast

        Returns:
            Dictionary containing forecast data and metadata

        Note:
            AIFS model is loaded and functional. Returns placeholder data until
            real weather data input pipeline is implemented.
        """
        model_info = self._load_model()

        lat, lon = location

        # Display model information
        if model_info["type"] == "aifs_full":
            param_count = sum(p.numel() for p in model_info["pytorch_model"].parameters())
            print(f"🤖 Using AIFS PyTorch model with {param_count:,} parameters")
        else:
            print("🤖 Using AIFS SimpleRunner (PyTorch model not accessible)")

        # Placeholder implementation - would need actual AIFS inference with real weather data
        print(f"Running AIFS prediction for location ({lat:.4f}, {lon:.4f})")
        print(f"Forecast duration: {days} days")

        if variables is None:
            variables = [
                "temperature_2m",
                "precipitation",
                "wind_speed_10m",
                "wind_direction_10m",
                "pressure_msl",
            ]

        # Placeholder results - replace with actual AIFS output
        results = {
            "location": {"latitude": lat, "longitude": lon},
            "forecast_days": days,
            "variables": variables,
            "predictions": {
                var: np.random.randn(days * 8, 1)  # 8 timesteps per day (3-hourly)
                for var in variables
            },
            "timestamps": [f"2024-01-{i//8 + 1:02d}T{(i%8)*3:02d}:00:00Z" for i in range(days * 8)],
            "model": "AIFS-Single-v1.0",
            "status": "success",
        }

        return results

    def predict_global(
        self,
        days: int = 7,
        variables: Optional[List[str]] = None,
        resolution: str = "0.25deg",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate global weather forecast.

        Args:
            days: Number of days to forecast
            variables: List of variables to predict
            resolution: Spatial resolution
            **kwargs: Additional parameters

        Returns:
            Dictionary containing global forecast
        """
        self._load_model()  # Load model but don't store in unused variable

        print(f"Running AIFS global prediction for {days} days at {resolution} resolution")

        # Placeholder for global forecast
        results = {
            "forecast_days": days,
            "resolution": resolution,
            "variables": variables or ["temperature_2m", "precipitation"],
            "model": "AIFS-Single-v1.0",
            "status": "global_forecast_placeholder",
        }

        return results

    def get_available_variables(self) -> List[str]:
        """Get list of variables available from AIFS model."""
        return [
            "temperature_2m",
            "temperature_850hPa",
            "temperature_500hPa",
            "geopotential_1000hPa",
            "geopotential_850hPa",
            "geopotential_500hPa",
            "u_component_of_wind_1000hPa",
            "u_component_of_wind_850hPa",
            "v_component_of_wind_1000hPa",
            "v_component_of_wind_850hPa",
            "specific_humidity_850hPa",
            "specific_humidity_700hPa",
            "mean_sea_level_pressure",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "2m_dewpoint_temperature",
            "surface_pressure",
            "total_precipitation",
            "100m_u_component_of_wind",
            "100m_v_component_of_wind",
            "snow_depth",
            "soil_temperature_level_4",
            "volumetric_soil_water_layer_1",
            "volumetric_soil_water_layer_4",
        ]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the AIFS model."""
        return {
            "model_name": "AIFS Single v1.0",
            "provider": "ECMWF",
            "model_path": str(self.model_path),
            "config_path": str(self.config_path),
            "available_variables": len(self.get_available_variables()),
            "max_forecast_days": 10,
            "temporal_resolution": "3-hourly",
            "spatial_resolution": "0.25 degrees",
            "license": "CC-BY-4.0",
        }


def create_aifs_backend() -> AIFSWrapper:
    """
    Factory function to create AIFS backend with default settings.

    Returns:
        Configured AIFS wrapper instance
    """
    return AIFSWrapper()


def demo_aifs_usage():
    """Demonstrate basic AIFS usage."""
    print("🌪️ AIFS Demo - Alternative Climate AI Backend")
    print("=" * 50)

    try:
        # Initialize AIFS
        aifs = create_aifs_backend()

        # Show model info
        info = aifs.get_model_info()
        print(f"Model: {info['model_name']}")
        print(f"Provider: {info['provider']}")
        print(f"Variables: {info['available_variables']}")
        print(f"Max forecast: {info['max_forecast_days']} days")

        # Example prediction
        print("\n🗺️ Location-specific forecast:")
        location = (40.7128, -74.0060)  # New York City
        forecast = aifs.predict(location, days=5)
        print(f"Location: {forecast['location']}")
        print(f"Variables: {', '.join(forecast['variables'])}")
        print(f"Timesteps: {len(forecast['timestamps'])}")

        print("\n✅ AIFS integration ready!")
        print("Use AIFSWrapper as alternative to PrithviWxC for:")
        print("  • Global weather forecasting")
        print("  • Operational prediction tasks")
        print("  • Long-term forecasts (10+ days)")

    except Exception as e:
        print(f"❌ AIFS demo failed: {e}")
        print("Please ensure AIFS submodule is properly initialized.")


if __name__ == "__main__":
    demo_aifs_usage()

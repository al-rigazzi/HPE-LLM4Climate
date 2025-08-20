"""
Multimodal AIFS - Location-Aware Climate Analysis

This package provides multi    # Utility modules
    "AIFSEncoderWrapper",
    "ClimateDataProcessor",
    "GeographicUtilities",l climate analysis capabilities using ECMWF AIFS
(AI Integrated Forecasting System) combined with large language models for
location-aware climate understanding and analysis.

Key Components:
- AIFS Climate Encoder: Uses ECMWF AIFS model for climate data encoding
- Location-Aware Fusion: Geographic-specific climate-text integration
- Multimodal Analysis: Combined climate data and natural language processing
- Spatial Attention: Region-specific climate pattern analysis

Architecture:
    AIFS Climate Data → Graph Neural Network Encoder → Hidden Representations
                                    ↓
    Text Queries → Language Model → Text Embeddings
                                    ↓
    Location Context → Geographic Resolver → Spatial Masks
                                    ↓
              Location-Aware Fusion → Climate Analysis

Usage:
    from multimodal_aifs import AIFSClimateAnalysis

    # Initialize analysis system
    analyzer = AIFSClimateAnalysis(
        aifs_model_path="path/to/aifs/model",
        llm_model="microsoft/DialoGPT-medium"
    )

    # Analyze climate for specific location
    result = analyzer.analyze(
        query="What is the temperature trend in New York?",
        location=(40.7128, -74.0060),  # NYC coordinates
        climate_data=climate_tensor
    )

Features:
- Integration with ECMWF AIFS operational weather model
- Location-aware attention mechanisms
- Geographic query processing
- Spatial climate data cropping
- Multimodal fusion strategies
- Climate assessment generation

Requirements:
- PyTorch >= 1.9.0
- transformers >= 4.21.0
- anemoi-inference >= 0.4.9
- numpy >= 1.21.0
"""

__version__ = "0.1.0"
__author__ = "HPE-LLM4Climate Team"
__email__ = "climate-ai@hpe.com"

# Core imports
# Core module imports - TODO: Implement these modules
# from .core.aifs_climate_fusion import AIFSClimateTextFusion
# from .core.aifs_location_aware import (
#     AIFSGeographicResolver,
#     AIFSLocationAwareAttention,
#     AIFSSpatialCropper,
# )
# from .core.aifs_location_aware_fusion import (
#     AIFSFusionMode,
#     AIFSLocationAwareClimateAnalysis,
# )# Utility imports
from .utils.aifs_encoder_utils import AIFSEncoderWrapper
from .utils.climate_data_utils import ClimateDataProcessor
from .utils.location_utils import GridUtils, LocationUtils

__all__ = [
    # Utility modules
    "AIFSEncoderWrapper",
    "ClimateDataProcessor",
    "LocationUtils",
    "GridUtils",
    # Package metadata
    "__version__",
    "__author__",
    "__email__",
]

# Package-level configuration
AIFS_DEFAULT_MODEL_PATH = "aifs-single-1.0/aifs-single-mse-1.0.ckpt"
DEFAULT_LLM_MODEL = "microsoft/DialoGPT-medium"
DEFAULT_CLIMATE_RESOLUTION = (721, 1440)  # AIFS standard resolution
DEFAULT_ATTENTION_HEADS = 16
DEFAULT_HIDDEN_DIM = 1024  # AIFS encoder output dimension

# Geographic constants
EARTH_RADIUS_KM = 6371.0
DEFAULT_SPATIAL_RADIUS_KM = 100.0
DEFAULT_GRID_RESOLUTION_DEG = 0.25


def get_package_info():
    """Get package information."""
    return {
        "name": "multimodal_aifs",
        "version": __version__,
        "description": "Location-aware multimodal climate analysis with ECMWF AIFS",
        "author": __author__,
        "email": __email__,
        "core_modules": len(__all__),
        "aifs_integration": True,
        "location_aware": True,
        "multimodal_fusion": True,
    }


def check_requirements():
    """Check if required packages are available."""
    requirements = {
        "torch": "PyTorch for neural networks",
        "transformers": "Hugging Face transformers for LLM",
        "numpy": "Numerical computing",
        "anemoi.inference": "AIFS model inference (optional)",
    }

    missing = []
    for package, description in requirements.items():
        try:
            if "." in package:
                # Handle nested imports like anemoi.inference
                parts = package.split(".")
                module = __import__(parts[0])
                for part in parts[1:]:
                    module = getattr(module, part)
            else:
                __import__(package)
        except ImportError:
            missing.append(f"{package}: {description}")

    if missing:
        print("⚠️  Missing optional dependencies:")
        for item in missing:
            print(f"   - {item}")
        print("\nInstall with: pip install torch transformers numpy anemoi-inference")
    else:
        print("✅ All requirements satisfied")

    return len(missing) == 0


# Initialize package
if __name__ != "__main__":
    # Only run checks when imported, not when executed directly
    pass

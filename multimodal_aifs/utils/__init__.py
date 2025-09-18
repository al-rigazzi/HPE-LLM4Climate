"""
Utilities Package for AIFS Multimodal Analysis

This package provides utility modules for handling climate data processing,
location-aware operations, and text processing for multimodal climate analysis.
"""

from .climate_data_utils import (
    CLIMATE_VARIABLES,
    ClimateDataProcessor,
    create_synthetic_climate_data,
)
from .location_utils import (
    COMMON_RESOLUTIONS,
    EARTH_RADIUS_KM,
    GridUtils,
    LocationUtils,
    SpatialEncoder,
)
from .text_utils import (
    CLIMATE_KEYWORDS,
    CLIMATE_PHRASES,
    ClimateTextProcessor,
    TextEmbeddingUtils,
)

__version__ = "0.1.0"

__all__ = [
    # Climate data utilities
    "ClimateDataProcessor",
    "CLIMATE_VARIABLES",
    "create_synthetic_climate_data",
    # Location utilities
    "LocationUtils",
    "GridUtils",
    "SpatialEncoder",
    "EARTH_RADIUS_KM",
    "COMMON_RESOLUTIONS",
    # Text utilities
    "ClimateTextProcessor",
    "TextEmbeddingUtils",
    "CLIMATE_KEYWORDS",
    "CLIMATE_PHRASES",
]  # Utility constants
UTILS_INFO = {
    "package": "multimodal_aifs.utils",
    "version": __version__,
    "components": {
        "aifs_encoder": "AIFS model integration and encoding",
        "climate_data": "Climate data preprocessing and normalization",
        "location": "Geographic and spatial operations",
        "text": "Climate text processing and embedding",
    },
}


def get_utils_info():
    """Get information about available utilities."""
    return UTILS_INFO


def test_all_utils():
    """Run tests for all utility modules."""
    print("🔧 Testing All AIFS Multimodal Utilities")
    print("=" * 50)

    try:
        # Test climate data utils
        print("\n1. Testing Climate Data Utils...")
        from .climate_data_utils import test_climate_processor

        test_climate_processor()

        # Test location utils
        print("\n2. Testing Location Utils...")
        from .location_utils import test_location_utilities

        test_location_utilities()

        # Test text utils
        print("\n3. Testing Text Utils...")
        from .text_utils import test_text_processing

        test_text_processing()

        print("\n" + "=" * 50)
        print("✅ All utility tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    # Display package information
    info = get_utils_info()
    print(f"Package: {info['package']} v{info['version']}")
    print("Components:")
    for name, desc in info["components"].items():
        print(f"  - {name}: {desc}")

    # Run all tests
    test_all_utils()

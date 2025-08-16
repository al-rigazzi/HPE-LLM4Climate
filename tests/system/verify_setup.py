#!/usr/bin/env python3
"""
Setup Verification Script for HPE-LLM4Climate

This script verifies that all required dependencies are properly installed
and that the location-aware climate analysis system is working correctly.

Usage:
    python verify_setup.py
"""

import sys
import os
import warnings
from typing import List, Tuple

# Add project root to Python path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

def test_import(package_name: str, import_statement: str) -> Tuple[bool, str]:
    """Test if a package can be imported."""
    try:
        exec(import_statement)
        return True, "OK"
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {e}"

def main():
    """Run all verification tests."""
    print("🔍 HPE-LLM4Climate Setup Verification")
    print("=" * 50)

    # Core requirements
    core_tests = [
        ("NumPy", "import numpy"),
        ("PyTorch", "import torch"),
        ("Transformers", "import transformers"),
        ("HuggingFace Hub", "import huggingface_hub"),
        ("Accelerate", "import accelerate"),
        ("GeoPy", "import geopy"),
        ("Requests", "import requests"),
    ]

    print("\n📦 Testing Core Dependencies:")
    all_passed = True
    for name, import_stmt in core_tests:
        success, message = test_import(name, import_stmt)
        status = "✅" if success else "❌"
        print(f"  {status} {name:20} : {message}")
        if not success:
            all_passed = False

    # Multimodal system tests
    print("\n🤖 Testing Multimodal System:")
    multimodal_tests = [
        ("Encoder Extractor", "from multimodal.utils.encoder_extractor import PrithviWxC_Encoder"),
        ("Climate-Text Fusion", "from multimodal.core.climate_text_fusion import ClimateTextFusion"),
        ("Location-Aware Core", "from multimodal.core.location_aware import GeographicResolver, SpatialCropper"),
        ("Location-Aware Fusion", "from multimodal.core.location_aware_fusion import LocationAwareClimateAnalysis"),
    ]

    for name, import_stmt in multimodal_tests:
        success, message = test_import(name, import_stmt)
        status = "✅" if success else "❌"
        print(f"  {status} {name:20} : {message}")
        if not success:
            all_passed = False

    # Functional tests
    print("\n🧪 Testing Functionality:")

    try:
        # Test geographic resolution
        from multimodal.core.location_aware import GeographicResolver

        # Test local backend
        resolver_local = GeographicResolver(backend='local')
        result = resolver_local.resolve_location("sweden")
        if result:
            print(f"  ✅ Local geo resolution : Found {result.name} ({result.location_type})")
        else:
            print(f"  ❌ Local geo resolution : Failed")
            all_passed = False

        # Test GeoPy backend
        resolver_geopy = GeographicResolver(backend='geopy')
        result = resolver_geopy.resolve_location("Paris, France")
        if result:
            print(f"  ✅ GeoPy resolution    : Found {result.name}")
        else:
            print(f"  ⚠️  GeoPy resolution    : No result (network issue?)")

        # Test location-aware analysis
        from multimodal.core.location_aware_fusion import LocationAwareClimateAnalysis
        import torch

        # Try different Prithvi encoder files
        encoder_files = [
            "data/weights/prithvi_encoder_fixed.pt",  # Try the one with 25 layers first
            "data/weights/prithvi_encoder_corrected.pt",
            "data/weights/prithvi_encoder.pt"
        ]

        model = None
        for prithvi_path in encoder_files:
            if os.path.exists(prithvi_path):
                try:
                    model = LocationAwareClimateAnalysis(prithvi_encoder_path=prithvi_path)
                    print(f"  ✅ Successfully loaded Prithvi encoder: {prithvi_path}")
                    break
                except Exception as e:
                    print(f"  ⚠️  Failed to load {prithvi_path}: {str(e)[:100]}...")
                    continue

        if model is None:
            model = LocationAwareClimateAnalysis()
            print(f"  ⚠️  All Prithvi encoders failed, using demo mode")

        # Create properly sized climate data for the fusion model
        if hasattr(model, 'climate_text_fusion') and model.climate_text_fusion:
            # Use actual climate encoder dimensions (2560)
            climate_dim = model.climate_text_fusion.climate_dim
            climate_data = torch.randn(1, 50, climate_dim)
        else:
            # Demo mode with default dimensions
            climate_data = torch.randn(1, 50, 768)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.analyze_location_query(
                climate_data,
                "Climate trends in Stockholm, Sweden",
                return_visualization=False
            )

        if result and 'location' in result:
            print(f"  ✅ Climate analysis    : {result['location']} -> {result['climate_risk']}")
        else:
            print(f"  ❌ Climate analysis    : Failed")
            all_passed = False

    except Exception as e:
        print(f"  ❌ Functionality test  : {e}")
        all_passed = False

    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Setup is complete and functional.")
        print("\n🚀 Ready to use:")
        print("   • Location-aware climate analysis")
        print("   • Multi-backend geographic resolution")
        print("   • Spatial attention masking")
        print("   • Multimodal climate-text fusion")
        return 0
    else:
        print("❌ Some tests failed. Please install missing dependencies:")
        print("   pip install -r requirements.txt")
        print("   pip install -r multimodal/requirements-geo.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())

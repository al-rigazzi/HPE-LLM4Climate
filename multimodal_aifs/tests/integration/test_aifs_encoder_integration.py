#!/usr/bin/env python3
"""
Test script for the new AIFS Encoder Utils

This script tests the basic functionality of the aifs_encoder_utils module.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test that all imports work correctly."""
    print("Testing AIFS Encoder Utils Imports")
    print("=" * 50)

    # Test core imports
    from multimodal_aifs.core.aifs_encoder_utils import (
        AIFSCompleteEncoder,
        check_aifs_dependencies,
        create_aifs_encoder,
        get_checkpoint_info,
        get_default_checkpoint_path,
        load_aifs_encoder,
        save_aifs_encoder,
        validate_checkpoint,
    )

    print("Core module imports successful")

    # Test package-level imports
    from multimodal_aifs import AIFSCompleteEncoder as PackageEncoder
    from multimodal_aifs import create_aifs_encoder as package_create
    from multimodal_aifs import load_aifs_encoder as package_load
    from multimodal_aifs import save_aifs_encoder as package_save

    print("Package-level imports successful")

    # Test functions
    deps_available = check_aifs_dependencies()
    default_path = get_default_checkpoint_path()

    print(f"AIFS dependencies available: {deps_available}")
    print(f"Default checkpoint path: {default_path}")

    print("\nAll imports and basic functions work correctly!")
    # Assert that imports worked (if we got here, they did)
    assert True


def test_class_definition():
    """Test that the AIFSCompleteEncoder class is properly defined."""
    print("\nTesting AIFSCompleteEncoder Class Definition")
    print("=" * 55)

    from multimodal_aifs.core.aifs_encoder_utils import AIFSCompleteEncoder

    # Check class attributes
    assert hasattr(AIFSCompleteEncoder, "__init__"), "Missing __init__ method"
    assert hasattr(AIFSCompleteEncoder, "forward"), "Missing forward method"

    print("AIFSCompleteEncoder class definition is correct")
    print("📋 Required methods: __init__, forward")


def test_utility_functions():
    """Test utility functions."""
    print("\n🛠️  Testing Utility Functions")
    print("=" * 35)

    from multimodal_aifs.core.aifs_encoder_utils import (
        DEFAULT_CHECKPOINT_DIR,
        DEFAULT_CHECKPOINT_NAME,
        EXPECTED_INPUT_SHAPE,
        EXPECTED_OUTPUT_SHAPE,
        check_aifs_dependencies,
        get_default_checkpoint_path,
    )

    # Test constants
    assert isinstance(DEFAULT_CHECKPOINT_DIR, str), "DEFAULT_CHECKPOINT_DIR must be string"
    assert isinstance(DEFAULT_CHECKPOINT_NAME, str), "DEFAULT_CHECKPOINT_NAME must be string"
    assert isinstance(EXPECTED_INPUT_SHAPE, list), "EXPECTED_INPUT_SHAPE must be list"
    assert isinstance(EXPECTED_OUTPUT_SHAPE, list), "EXPECTED_OUTPUT_SHAPE must be list"

    # Test functions
    default_path = get_default_checkpoint_path()
    deps_available = check_aifs_dependencies()

    print("Utility functions work correctly")
    print(f"Default path: {default_path}")
    print(f"Dependencies: {deps_available}")
    print(f"Expected input shape: {EXPECTED_INPUT_SHAPE}")
    print(f"Expected output shape: {EXPECTED_OUTPUT_SHAPE}")


def main():
    """Run all tests."""
    print("AIFS Encoder Utils Test Suite")
    print("=" * 40)

    tests = [
        ("Import Tests", test_imports),
        ("Class Definition Tests", test_class_definition),
        ("Utility Function Tests", test_utility_functions),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"{test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{status} - {test_name}")
        if success:
            passed += 1

    print(f"\nResults: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("All tests passed! AIFS Encoder Utils is ready to use.")
        return 0
    else:
        print("Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    EXIT_CODE = main()
    sys.exit(EXIT_CODE)

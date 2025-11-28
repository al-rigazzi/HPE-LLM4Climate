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
    # pylint: disable=unused-import
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

    # pylint: enable=unused-import

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

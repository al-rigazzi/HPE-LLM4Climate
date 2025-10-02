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
Multimodal AIFS Core Module

This module contains the core functionality for AIFS-based multimodal climate models,
including encoder extraction, saving, loading, and fusion capabilities.
"""

from .aifs_encoder_utils import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_CHECKPOINT_NAME,
    EXPECTED_INPUT_SHAPE,
    EXPECTED_OUTPUT_SHAPE,
    AIFSCompleteEncoder,
    check_aifs_dependencies,
    create_aifs_encoder,
    get_checkpoint_info,
    get_default_checkpoint_path,
    load_aifs_encoder,
    save_aifs_encoder,
    validate_checkpoint,
)

# Additional modules available for import:
# from .aifs_climate_fusion import AIFSClimateFusion

__all__ = [
    # AIFS Encoder utilities
    "AIFSCompleteEncoder",
    "save_aifs_encoder",
    "load_aifs_encoder",
    "create_aifs_encoder",
    "get_checkpoint_info",
    "validate_checkpoint",
    "get_default_checkpoint_path",
    "check_aifs_dependencies",
    "DEFAULT_CHECKPOINT_DIR",
    "DEFAULT_CHECKPOINT_NAME",
    "EXPECTED_INPUT_SHAPE",
    "EXPECTED_OUTPUT_SHAPE",
    # Additional modules available:
    # "AIFSClimateFusion",
]

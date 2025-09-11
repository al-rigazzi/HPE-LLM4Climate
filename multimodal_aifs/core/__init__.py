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

# TODO: Import these when the modules are ready
# from .aifs_climate_fusion import AIFSClimateFusion
# from .aifs_location_aware import AIFSLocationAware
# from .aifs_location_aware_fusion import AIFSLocationAwareFusion

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
    # TODO: Add these when the modules are ready
    # "AIFSClimateFusion",
    # "AIFSLocationAware",
    # "AIFSLocationAwareFusion"
]

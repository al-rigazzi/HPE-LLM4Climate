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
Constants used throughout the multimodal AIFS codebase.

This module contains all read-only constants that are ubiquitous across the project,
including AIFS model specifications, tensor shapes, and climate variable definitions.
"""

from typing import Final

# ===================== AIFS MODEL SPECIFICATIONS =====================

# AIFS grid and data dimensions
AIFS_GRID_POINTS: Final[int] = (
    542080  # Number of spatial grid points in AIFS model (N320 reduced Gaussian grid)
)

# AIFS input variable counts
# - AIFS_INPUT_VARIABLES: Total features model receives after SimpleRunner adds forcings
# - AIFS_PHYSICS_VARIABLES: Physics variables user provides (defined later after variable lists)
# User provides 94 physics variables → SimpleRunner adds 9 forcings → Model receives 103 total
AIFS_INPUT_VARIABLES: Final[int] = 103  # Total: 94 physics + 9 forcings (8 trig + insolation)

AIFS_RAW_ENCODER_OUTPUT_DIM: Final[int] = 102  # Raw AIFS encoder output dimension

# AIFS temporal specifications
AIFS_DEFAULT_TIME_STEPS: Final[int] = 2  # Default number of time steps
AIFS_DEFAULT_ENSEMBLE_SIZE: Final[int] = 1  # Default ensemble size

# ===================== TENSOR SHAPES =====================

# AIFS input tensor shapes
AIFS_SAMPLE_SHAPE: Final[tuple[int, ...]] = (2, 1, AIFS_GRID_POINTS, AIFS_INPUT_VARIABLES)
"""Shape of a single AIFS sample: [time, ensemble, grid, vars]"""

AIFS_BATCH_SHAPE: Final[tuple[int, ...]] = (1, 2, 1, AIFS_GRID_POINTS, AIFS_INPUT_VARIABLES)
"""Shape of a one-sample batch: [batch, time, ensemble, grid, vars]"""

# AIFS output tensor shapes
AIFS_ENCODER_OUTPUT_SHAPE: Final[tuple[int, ...]] = (
    AIFS_GRID_POINTS,
    AIFS_RAW_ENCODER_OUTPUT_DIM,
)
"""Expected encoder output shape: [grid_points, embedding_dim]"""

# ===================== CLIMATE VARIABLES =====================

# Surface variables (12 total)
SURFACE_VARIABLES: Final[list[str]] = [
    "10u",  # 10m u-component of wind
    "10v",  # 10m v-component of wind
    "2d",  # 2m dewpoint temperature
    "2t",  # 2m temperature
    "msl",  # Mean sea level pressure
    "skt",  # Skin temperature
    "sp",  # Surface pressure
    "tcw",  # Total column water
    "lsm",  # Land-sea mask
    "z",  # Geopotential
    "slor",  # Slope of sub-gridscale orography
    "sdor",  # Standard deviation of orography
]

# ===================== ECMWF OPEN DATA API CONSTANTS =====================
# These constants define parameter names and mappings specific to the ECMWF Open Data API.
# They are used when downloading data from ECMWF's operational forecast service.
# Note: ECMWF API parameter names differ from AIFS internal naming conventions.

# Surface parameters for ECMWF Open Data API requests
ECMWF_PARAM_SFC: Final[list[str]] = [
    "10u",  # 10m u-component of wind
    "10v",  # 10m v-component of wind
    "2d",  # 2m dewpoint temperature
    "2t",  # 2m temperature
    "msl",  # Mean sea level pressure
    "skt",  # Skin temperature
    "sp",  # Surface pressure
    "tcw",  # Total column water
    "lsm",  # Land-sea mask
    "z",  # Geopotential at surface
    "slor",  # Slope of sub-gridscale orography
    "sdor",  # Standard deviation of orography
]

# Soil parameters for ECMWF Open Data API requests
ECMWF_PARAM_SOIL: Final[list[str]] = [
    "vsw",  # Volumetric soil water (maps to swvl1/swvl2)
    "sot",  # Soil temperature (maps to stl1/stl2)
]

# Pressure level parameters for ECMWF Open Data API requests
ECMWF_PARAM_PL: Final[list[str]] = [
    "gh",  # Geopotential height (converted to geopotential z via multiplication by 9.80665)
    "t",  # Temperature
    "u",  # U-component of wind
    "v",  # V-component of wind
    "w",  # Vertical velocity
    "q",  # Specific humidity
]

# Pressure levels in hPa for ECMWF Open Data API requests
ECMWF_LEVELS: Final[list[int]] = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

# Soil levels for ECMWF Open Data API requests
ECMWF_SOIL_LEVELS: Final[list[int]] = [1, 2]

# Mapping from ECMWF soil parameter names to AIFS variable names
# ECMWF uses generic names (vsw, sot) with level suffixes,
# while AIFS uses specific names (swvl1, stl1, etc.)
ECMWF_SOIL_MAPPING: Final[dict[str, str]] = {
    "sot_1": "stl1",  # Soil temperature level 1
    "sot_2": "stl2",  # Soil temperature level 2
    "vsw_1": "swvl1",  # Volumetric soil water level 1
    "vsw_2": "swvl2",  # Volumetric soil water level 2
}

# Soil variables (4 total)
SOIL_VARIABLES: Final[list[str]] = [
    "stl1",  # Soil temperature level 1
    "stl2",  # Soil temperature level 2
    "swvl1",  # Soil water level 1
    "swvl2",  # Soil water level 2
]

# Pressure levels (hPa) - MUST match AIFS training data order (high to low pressure)
PRESSURE_LEVELS: Final[list[int]] = [
    1000,  # Surface
    925,
    850,
    700,
    600,
    500,
    400,
    300,
    250,
    200,
    150,
    100,
    50,  # Upper atmosphere
]

# Pressure level variables (6 parameters × 13 levels = 78 total)
# Order MUST match AIFS training data: t, u, v, w, q, z (from gh conversion)
PRESSURE_LEVEL_PARAMETERS: Final[list[str]] = ["t", "u", "v", "w", "q", "z"]
"""Parameters available at pressure levels: temperature, u-wind, v-wind,
vertical velocity, specific humidity, geopotential (from gh conversion)"""

# Complete variable list (94 total - physics variables ONLY for user input)
# CRITICAL: Variable order MUST match AIFS training data preparation
# From AIFS v1.1 notebook (run_AIFS_v1.1.ipynb):
#   1. Surface variables (12): 10u, 10v, 2d, 2t, msl, skt, sp, tcw, lsm, z, slor, sdor
#   2. Soil variables (4): stl1, stl2, swvl1, swvl2
#   3. Pressure level variables (78): 6 params × 13 levels, organized by parameter then level
#
# IMPORTANT: Forcing variables are computed by SimpleRunner and added automatically:
#   - Trigonometric (8): cos_latitude, sin_latitude, cos_longitude, sin_longitude,
#                       cos_julian_day, sin_julian_day, cos_local_time, sin_local_time
#   - Insolation (1): Computed from date/time and solar geometry
#   These 9 forcing variables should NOT be in ALL_AIFS_VARIABLES!
#
# NOTE: Variables like 100u, 100v, insolation that appear in v1.1 README are OUTPUT
#       variables, not inputs. The model produces them, users don't provide them.
ALL_AIFS_VARIABLES: Final[list[str]] = (
    SURFACE_VARIABLES  # 12 variables (includes constant forcings: lsm, z, sdor, slor)
    + SOIL_VARIABLES  # 4 variables
    # Pressure level variables (78): for each param, iterate through all levels
    + [f"{param}_{level}" for param in PRESSURE_LEVEL_PARAMETERS for level in PRESSURE_LEVELS]
    # Total: 12 + 4 + 78 = 94 physics variables (forcings added by SimpleRunner)
)  # ===================== MODEL CONFIGURATIONS =====================

# Fusion model dimensions
FUSION_DEFAULT_DIM: Final[int] = 512  # Default fusion dimension
TEXT_DEFAULT_DIM: Final[int] = 768  # Default text embedding dimension

# Temporal modeling configurations
TEMPORAL_HIDDEN_DIM: Final[int] = 256  # Default hidden dimension for temporal models
# Transformer projection dimension (divisible by common head counts)
TRANSFORMER_PROJECTION_DIM: Final[int] = 216

# Attention configurations
DEFAULT_NUM_HEADS: Final[int] = 8  # Default number of attention heads

# File system configurations
DEFAULT_CHECKPOINT_DIR: Final[str] = "multimodal_aifs/models/extracted_models"
DEFAULT_CHECKPOINT_NAME: Final[str] = "aifs_complete_encoder.pth"

# Expected input/output shapes for validation
# [batch, time, ensemble, grid, vars]
EXPECTED_INPUT_SHAPE: Final[list[int]] = [1, 2, 1, AIFS_GRID_POINTS, AIFS_INPUT_VARIABLES]
# [grid_points, embedding_dim]
EXPECTED_OUTPUT_SHAPE: Final[list[int]] = [AIFS_GRID_POINTS, AIFS_RAW_ENCODER_OUTPUT_DIM]

# ===================== VALIDATION CONSTANTS =====================

# Expected tensor validation
MIN_BATCH_SIZE: Final[int] = 1
MAX_BATCH_SIZE: Final[int] = 64  # Reasonable upper limit for memory constraints

# Grid validation
EXPECTED_GRID_SIZE: Final[int] = AIFS_GRID_POINTS

# Variable count validation
EXPECTED_VARIABLE_COUNT: Final[int] = AIFS_INPUT_VARIABLES

# Embedding dimension validation
EXPECTED_EMBEDDING_DIM: Final[int] = AIFS_RAW_ENCODER_OUTPUT_DIM

# ===================== UTILITY FUNCTIONS =====================


def validate_aifs_input_shape(tensor_shape: tuple[int, ...]) -> bool:
    """
    Validate that a tensor shape matches expected AIFS input format.

    Args:
        tensor_shape: Shape tuple to validate

    Returns:
        True if shape is valid AIFS input format
    """
    if len(tensor_shape) != 5:
        return False

    batch, time, ensemble, grid, variables = tensor_shape

    return (
        MIN_BATCH_SIZE <= batch <= MAX_BATCH_SIZE
        and time >= 1
        and ensemble >= 1
        and grid == EXPECTED_GRID_SIZE
        and variables == EXPECTED_VARIABLE_COUNT
    )


def validate_aifs_output_shape(tensor_shape: tuple[int, ...]) -> bool:
    """
    Validate that a tensor shape matches expected AIFS encoder output format.

    Args:
        tensor_shape: Shape tuple to validate

    Returns:
        True if shape is valid AIFS encoder output format
    """
    return (
        len(tensor_shape) == 2
        and tensor_shape[0] == EXPECTED_GRID_SIZE
        and tensor_shape[1] == EXPECTED_EMBEDDING_DIM
    )


def get_variable_info() -> dict[str, int]:
    """
    Get information about variable counts by category.

    Returns:
        Dictionary with variable counts
    """
    return {
        "surface_variables": len(SURFACE_VARIABLES),
        "soil_variables": len(SOIL_VARIABLES),
        "pressure_level_variables": len(PRESSURE_LEVEL_PARAMETERS) * len(PRESSURE_LEVELS),
        "total_variables": AIFS_INPUT_VARIABLES,
    }


# ===================== ASSERTIONS FOR VALIDATION =====================

# Validate that our variable lists sum to the expected total
_CALCULATED_TOTAL = (
    len(SURFACE_VARIABLES)
    + len(SOIL_VARIABLES)
    + len(PRESSURE_LEVEL_PARAMETERS) * len(PRESSURE_LEVELS)
)

# ALL_AIFS_VARIABLES contains only the 94 physics variables (user-provided)
# AIFS_INPUT_VARIABLES is the total after SimpleRunner adds 9 forcing variables (103 total)
AIFS_PHYSICS_VARIABLES: Final[int] = 94  # User-provided physics variables only

assert (
    len(ALL_AIFS_VARIABLES) == AIFS_PHYSICS_VARIABLES
), f"Variable list length mismatch: {len(ALL_AIFS_VARIABLES)} != {AIFS_PHYSICS_VARIABLES}"

assert (
    _CALCULATED_TOTAL == AIFS_PHYSICS_VARIABLES
), f"Calculated total mismatch: {_CALCULATED_TOTAL} != {AIFS_PHYSICS_VARIABLES}"

assert AIFS_INPUT_VARIABLES == AIFS_PHYSICS_VARIABLES + 9, (
    f"Expected AIFS_INPUT_VARIABLES to be {AIFS_PHYSICS_VARIABLES + 9} "
    f"(94 physics + 9 forcings), got {AIFS_INPUT_VARIABLES}"
)

assert len(SURFACE_VARIABLES) == 12, f"Expected 12 surface variables, got {len(SURFACE_VARIABLES)}"
assert len(SOIL_VARIABLES) == 4, f"Expected 4 soil variables, got {len(SOIL_VARIABLES)}"
assert len(PRESSURE_LEVELS) == 13, f"Expected 13 pressure levels, got {len(PRESSURE_LEVELS)}"
assert (
    len(PRESSURE_LEVEL_PARAMETERS) == 6
), f"Expected 6 pressure level parameters, got {len(PRESSURE_LEVEL_PARAMETERS)}"

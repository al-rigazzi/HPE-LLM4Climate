"""
Constants used throughout the multimodal AIFS codebase.

This module contains all read-only constants that are ubiquitous across the project,
including AIFS model specifications, tensor shapes, and climate variable definitions.
"""

from typing import Final

# ===================== AIFS MODEL SPECIFICATIONS =====================

# AIFS grid and data dimensions
AIFS_GRID_POINTS: Final[int] = 542080  # Number of spatial grid points in AIFS model
AIFS_INPUT_VARIABLES: Final[int] = 103  # Total input variables (90 prognostic + 13 forcing)
AIFS_RAW_ENCODER_OUTPUT_DIM: Final[int] = 102  # Raw AIFS encoder output dimension
AIFS_PROJECTED_ENCODER_OUTPUT_DIM: Final[int] = 218  # Projected encoder output dimension

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
    AIFS_PROJECTED_ENCODER_OUTPUT_DIM,
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

# Soil variables (4 total)
SOIL_VARIABLES: Final[list[str]] = [
    "stl1",  # Soil temperature level 1
    "stl2",  # Soil temperature level 2
    "swvl1",  # Soil water level 1
    "swvl2",  # Soil water level 2
]

# Pressure levels (hPa)
PRESSURE_LEVELS: Final[list[int]] = [
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    850,
    925,
    1000,
]

# Pressure level variables (6 parameters × 13 levels = 78 total)
PRESSURE_LEVEL_PARAMETERS: Final[list[str]] = ["q", "t", "u", "v", "w", "z"]
"""Parameters available at pressure levels: specific humidity, temperature,
u-wind, v-wind, vertical velocity, geopotential"""

# Additional derived variables (9 total)
DERIVED_VARIABLES: Final[list[str]] = [
    "cp",  # Convective precipitation
    "tp",  # Total precipitation
    "cos_latitude",  # Cosine of latitude
    "cos_longitude",  # Cosine of longitude
    "sin_latitude",  # Sine of latitude
    "sin_longitude",  # Sine of longitude
    "cos_julian_day",  # Cosine of julian day
    "cos_local_time",  # Cosine of local time
    "sin_julian_day",  # Sine of julian day
    "sin_local_time",  # Sine of local time
    "insolation",  # Solar insolation
    "100u",  # 100m u-component of wind
    "100v",  # 100m v-component of wind
    "hcc",  # High cloud cover
    "lcc",  # Low cloud cover
    "mcc",  # Medium cloud cover
    "ro",  # Runoff
    "sf",  # Snowfall
    "ssrd",  # Surface solar radiation downwards
    "strd",  # Surface thermal radiation downwards
    "tcc",  # Total cloud cover
]

# Complete variable list (103 total: 78 pressure level + 25 surface/derived/soil)
ALL_AIFS_VARIABLES: Final[list[str]] = (
    # Pressure level variables first (78 total: 6 parameters × 13 levels)
    [f"{param}_{level}" for param in PRESSURE_LEVEL_PARAMETERS for level in PRESSURE_LEVELS]
    +
    # Surface and derived variables (25 total to reach 103)
    SURFACE_VARIABLES  # 12 variables
    + [
        "cp",
        "tp",
        "cos_latitude",
        "sin_latitude",
        "cos_longitude",
        "sin_longitude",
        "cos_julian_day",
        "sin_julian_day",
        "cos_local_time",
        "sin_local_time",
        "insolation",
        "100u",
        "100v",
    ]  # 13 additional variables (12 + 13 = 25)
)

# ===================== MODEL CONFIGURATIONS =====================

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
EXPECTED_OUTPUT_SHAPE: Final[list[int]] = [AIFS_GRID_POINTS, AIFS_PROJECTED_ENCODER_OUTPUT_DIM]

# ===================== VALIDATION CONSTANTS =====================

# Expected tensor validation
MIN_BATCH_SIZE: Final[int] = 1
MAX_BATCH_SIZE: Final[int] = 64  # Reasonable upper limit for memory constraints

# Grid validation
EXPECTED_GRID_SIZE: Final[int] = AIFS_GRID_POINTS

# Variable count validation
EXPECTED_VARIABLE_COUNT: Final[int] = AIFS_INPUT_VARIABLES

# Embedding dimension validation
EXPECTED_EMBEDDING_DIM: Final[int] = AIFS_PROJECTED_ENCODER_OUTPUT_DIM

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
        "derived_variables": len(DERIVED_VARIABLES),
        "total_variables": AIFS_INPUT_VARIABLES,
    }


# ===================== ASSERTIONS FOR VALIDATION =====================

# Validate that our variable lists sum to the expected total
_CALCULATED_TOTAL = (
    len(SURFACE_VARIABLES)
    + len(SOIL_VARIABLES)
    + len(PRESSURE_LEVEL_PARAMETERS) * len(PRESSURE_LEVELS)
    + len([v for v in DERIVED_VARIABLES if v not in SURFACE_VARIABLES + SOIL_VARIABLES])
)

assert (
    len(ALL_AIFS_VARIABLES) == AIFS_INPUT_VARIABLES
), f"Variable list length mismatch: {len(ALL_AIFS_VARIABLES)} != {AIFS_INPUT_VARIABLES}"

assert len(SURFACE_VARIABLES) == 12, f"Expected 12 surface variables, got {len(SURFACE_VARIABLES)}"
assert len(SOIL_VARIABLES) == 4, f"Expected 4 soil variables, got {len(SOIL_VARIABLES)}"
assert len(PRESSURE_LEVELS) == 13, f"Expected 13 pressure levels, got {len(PRESSURE_LEVELS)}"
assert (
    len(PRESSURE_LEVEL_PARAMETERS) == 6
), f"Expected 6 pressure level parameters, got {len(PRESSURE_LEVEL_PARAMETERS)}"

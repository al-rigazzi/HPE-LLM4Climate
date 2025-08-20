"""Core multimodal fusion modules for climate-text integration."""

# Core multimodal fusion modules
from .climate_text_fusion import ClimateTextFusion
from .location_aware import GeographicResolver, LocationAwareAttention, SpatialCropper
from .location_aware_fusion import LocationAwareClimateAnalysis

__all__ = [  # pylint: disable=duplicate-code
    "ClimateTextFusion",  # Core fusion model
    "GeographicResolver",  # Location resolution
    "SpatialCropper",  # Spatial processing
    "LocationAwareAttention",  # Attention mechanism
    "LocationAwareClimateAnalysis",  # Location-aware analysis
]

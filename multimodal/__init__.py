"""
Multimodal module for PrithviWxC extensions.

This module contains utilities for extracting and working with components
of the PrithviWxC model for multimodal applications.
"""

from .core.climate_text_fusion import ClimateTextFusion
from .core.location_aware import GeographicResolver, LocationAwareAttention, SpatialCropper
from .core.location_aware_fusion import LocationAwareClimateAnalysis
from .utils.encoder_extractor import PrithviWxC_Encoder, extract_encoder_weights

__all__ = [  # pylint: disable=duplicate-code
    "PrithviWxC_Encoder",
    "extract_encoder_weights",
    "ClimateTextFusion",
    "GeographicResolver",
    "SpatialCropper",
    "LocationAwareAttention",
    "LocationAwareClimateAnalysis",
]

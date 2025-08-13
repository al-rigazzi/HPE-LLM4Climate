"""
Multimodal module for PrithviWxC extensions.

This module contains utilities for extracting and working with components
of the PrithviWxC model for multimodal applications.
"""

from .encoder_extractor import PrithviWxC_Encoder, extract_encoder_weights

__all__ = ['PrithviWxC_Encoder', 'extract_encoder_weights']

"""
Example usage of the extracted PrithviWxC encoder.

This script demonstrates how to load and use the extracted encoder
for feature extraction from climate data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
import yaml

from multimodal.encoder_extractor import PrithviWxC_Encoder


def load_encoder(encoder_weights_path: str, device: str = 'cpu'):
    """
    Load the extracted encoder model.

    Args:
        encoder_weights_path: Path to the saved encoder weights
        device: Device to load the model on ('cpu' or 'cuda')

    Returns:
        Loaded encoder model
    """
    # Load the saved encoder
    checkpoint = torch.load(encoder_weights_path, map_location=device)
    config = checkpoint['config']

    # Extract model parameters from config
    params = config['params']

    # Create dummy scalers (in practice, load from saved scalers)
    in_mu = torch.zeros(params['in_channels'])
    in_sig = torch.ones(params['in_channels'])
    static_mu = torch.zeros(params['in_channels_static'])
    static_sig = torch.ones(params['in_channels_static'])

    # Create encoder model
    encoder = PrithviWxC_Encoder(
        in_channels=params['in_channels'],
        input_size_time=params['input_size_time'],
        in_channels_static=params['in_channels_static'],
        input_scalers_mu=in_mu,
        input_scalers_sigma=in_sig,
        input_scalers_epsilon=params['input_scalers_epsilon'],
        static_input_scalers_mu=static_mu,
        static_input_scalers_sigma=static_sig,
        static_input_scalers_epsilon=params['static_input_scalers_epsilon'],
        n_lats_px=params['n_lats_px'],
        n_lons_px=params['n_lons_px'],
        patch_size_px=params['patch_size_px'],
        mask_unit_size_px=params['mask_unit_size_px'],
        mask_ratio_inputs=0.0,  # No masking for inference
        embed_dim=params['embed_dim'],
        n_blocks_encoder=params['n_blocks_encoder'],
        mlp_multiplier=params['mlp_multiplier'],
        n_heads=params['n_heads'],
        dropout=0.0,  # No dropout for inference
        drop_path=0.0,  # No drop path for inference
        parameter_dropout=0.0,  # No parameter dropout for inference
        residual="climate",
        masking_mode="global",
        positional_encoding="fourier",
        encoder_shifting=False,
        checkpoint_encoder=[],
    )

    # Load the weights
    encoder.load_state_dict(checkpoint['model_state_dict'])
    encoder.to(device)
    encoder.eval()

    return encoder


def extract_features(encoder, climate_data, static_data, input_time, lead_time):
    """
    Extract features from climate data using the encoder.

    Args:
        encoder: Loaded encoder model
        climate_data: Climate input data [batch, time, channels, lat, lon]
        static_data: Static input data [batch, channels, lat, lon]
        input_time: Input time stamps [batch]
        lead_time: Lead time for prediction [batch]

    Returns:
        Encoded features [batch, n_tokens, local_seq, embed_dim]
    """
    # Prepare batch
    batch = {
        'x': climate_data,
        'static': static_data,
        'climate': climate_data[:, -1],  # Use last time step as climate
        'input_time': input_time,
        'lead_time': lead_time,
    }

    # Extract features
    with torch.no_grad():
        features = encoder(batch)

    return features


def example_usage():
    """
    Example of how to use the encoder for feature extraction.
    """
    print("PrithviWxC Encoder Usage Example")
    print("=" * 40)

    # Example parameters (adjust based on your model)
    batch_size = 2
    n_times = 2
    n_channels = 160
    n_static_channels = 4
    n_lats = 720
    n_lons = 1440

    # Create sample data
    print("Creating sample climate data...")

    climate_data = torch.randn(batch_size, n_times, n_channels, n_lats, n_lons)
    static_data = torch.randn(batch_size, n_static_channels, n_lats, n_lons)
    input_time = torch.tensor([0.0, 3.0])  # Hours
    lead_time = torch.tensor([18.0, 18.0])  # Hours

    print(f"Climate data shape: {climate_data.shape}")
    print(f"Static data shape: {static_data.shape}")

    # Note: In practice, you would load a real encoder like this:
    # encoder = load_encoder('/path/to/encoder_weights.pt', device='cpu')

    print("\nTo extract features with a real encoder:")
    print("1. Extract encoder: python multimodal/encoder_extractor.py ...")
    print("2. Load encoder: encoder = load_encoder('encoder_weights.pt')")
    print("3. Extract features: features = extract_features(encoder, data, ...)")
    print("4. Use features for downstream tasks (classification, regression, etc.)")

    print("\nPotential applications:")
    print("- Climate pattern recognition")
    print("- Extreme weather detection")
    print("- Climate model comparison")
    print("- Multimodal fusion with satellite imagery")
    print("- Transfer learning to regional models")


def multimodal_fusion_example():
    """
    Example of how the encoder could be used for multimodal fusion.
    """
    print("\nMultimodal Fusion Example")
    print("=" * 30)

    print("""
    The extracted encoder can be combined with other modalities:

    1. Climate + Satellite Imagery:
       climate_features = prithvi_encoder(climate_data)
       satellite_features = satellite_encoder(satellite_data)
       fused_features = fusion_layer([climate_features, satellite_features])

    2. Climate + Text (weather reports):
       climate_features = prithvi_encoder(climate_data)
       text_features = text_encoder(weather_reports)
       combined = multimodal_transformer([climate_features, text_features])

    3. Climate + Topography:
       climate_features = prithvi_encoder(climate_data)
       topo_features = topo_encoder(elevation_data)
       prediction = downstream_model([climate_features, topo_features])
    """)


if __name__ == "__main__":
    example_usage()
    multimodal_fusion_example()

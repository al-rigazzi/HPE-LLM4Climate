"""
Test script for the PrithviWxC encoder extractor.

This script demonstrates how to use the encoder extractor and tests its functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from pathlib import Path
import numpy as np

from multimodal.encoder_extractor import PrithviWxC_Encoder, extract_encoder_weights
from PrithviWxC.model import PrithviWxC
from PrithviWxC.dataloaders.merra2 import input_scalers, static_input_scalers


def test_encoder_extraction():
    """Test the encoder extraction functionality."""
    
    print("Testing PrithviWxC Encoder Extraction...")
    
    # Define test configuration
    config = {
        "params": {
            "in_channels": 160,  # Example value
            "input_size_time": 2,
            "in_channels_static": 4,
            "input_scalers_epsilon": 1e-5,
            "static_input_scalers_epsilon": 1e-5,
            "n_lats_px": 720,
            "n_lons_px": 1440,
            "patch_size_px": [8, 8],
            "mask_unit_size_px": [8, 8],
            "embed_dim": 1024,
            "n_blocks_encoder": 24,
            "n_blocks_decoder": 8,
            "mlp_multiplier": 4.0,
            "n_heads": 16,
            "dropout": 0.0,
            "drop_path": 0.0,
            "parameter_dropout": 0.0,
        }
    }
    
    # Create dummy scalers
    in_channels = config["params"]["in_channels"]
    in_channels_static = config["params"]["in_channels_static"]
    
    in_mu = torch.randn(in_channels)
    in_sig = torch.ones(in_channels)
    static_mu = torch.randn(in_channels_static)
    static_sig = torch.ones(in_channels_static)
    output_scalers = torch.ones(in_channels)
    
    print("Creating encoder model...")
    
    # Create encoder model
    encoder_model = PrithviWxC_Encoder(
        in_channels=config["params"]["in_channels"],
        input_size_time=config["params"]["input_size_time"],
        in_channels_static=config["params"]["in_channels_static"],
        input_scalers_mu=in_mu,
        input_scalers_sigma=in_sig,
        input_scalers_epsilon=config["params"]["input_scalers_epsilon"],
        static_input_scalers_mu=static_mu,
        static_input_scalers_sigma=static_sig,
        static_input_scalers_epsilon=config["params"]["static_input_scalers_epsilon"],
        n_lats_px=config["params"]["n_lats_px"],
        n_lons_px=config["params"]["n_lons_px"],
        patch_size_px=config["params"]["patch_size_px"],
        mask_unit_size_px=config["params"]["mask_unit_size_px"],
        mask_ratio_inputs=0.0,
        embed_dim=config["params"]["embed_dim"],
        n_blocks_encoder=config["params"]["n_blocks_encoder"],
        mlp_multiplier=config["params"]["mlp_multiplier"],
        n_heads=config["params"]["n_heads"],
        dropout=config["params"]["dropout"],
        drop_path=config["params"]["drop_path"],
        parameter_dropout=config["params"]["parameter_dropout"],
        residual="climate",
        masking_mode="global",
        positional_encoding="fourier",
        encoder_shifting=False,
        checkpoint_encoder=[],
    )
    
    print("Creating test input...")
    
    # Create test input
    batch_size = 1
    test_batch = {
        'x': torch.randn(batch_size, config["params"]["input_size_time"], 
                        config["params"]["in_channels"], 
                        config["params"]["n_lats_px"], 
                        config["params"]["n_lons_px"]),
        'static': torch.randn(batch_size, config["params"]["in_channels_static"], 
                             config["params"]["n_lats_px"], 
                             config["params"]["n_lons_px"]),
        'climate': torch.randn(batch_size, config["params"]["in_channels"], 
                              config["params"]["n_lats_px"], 
                              config["params"]["n_lons_px"]),
        'input_time': torch.tensor([0.0]),
        'lead_time': torch.tensor([18.0]),
    }
    
    print("Testing forward pass...")
    
    # Test forward pass
    encoder_model.eval()
    with torch.no_grad():
        try:
            output = encoder_model(test_batch)
            print(f"✓ Forward pass successful!")
            print(f"  Input shape: {test_batch['x'].shape}")
            print(f"  Output shape: {output.shape}")
            
            # Calculate model size
            total_params = sum(p.numel() for p in encoder_model.parameters())
            print(f"  Total parameters: {total_params:,}")
            
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            return False
    
    print("✓ Encoder extraction test completed successfully!")
    return True


def demonstrate_usage():
    """Demonstrate how to use the encoder extractor with real data paths."""
    
    print("\n" + "="*60)
    print("USAGE DEMONSTRATION")
    print("="*60)
    
    print("""
To extract the encoder from a full PrithviWxC model, use:

python multimodal/encoder_extractor.py \\
    --config_path /data/config.yaml \\
    --weights_path /data/weights/prithvi.wxc.2300m.v1.pt \\
    --output_path /data/weights/prithvi_encoder.pt \\
    --surf_scaler_path /data/climatology/musigma_surface.nc \\
    --vert_scaler_path /data/climatology/musigma_vertical.nc

The extracted encoder can then be used for:
1. Feature extraction from climate data
2. Pretraining for downstream tasks
3. Multimodal fusion with other data types
4. Transfer learning to new domains

The encoder preserves all the preprocessing, embedding, and 
transformer capabilities of the original model.
    """)


if __name__ == "__main__":
    test_encoder_extraction()
    demonstrate_usage()

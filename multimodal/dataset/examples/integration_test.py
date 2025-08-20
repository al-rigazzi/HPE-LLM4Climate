#!/usr/bin/env python3
"""
Integration Test: MERRA-2 Dataset with PrithviWxC_Encoder

This script demonstrates full integration between the processed MERRA-2
datasets and the PrithviWxC_Encoder model.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from data_loader import PrithviMERRA2Dataset, MERRA2DataLoader
from multimodal.utils.encoder_extractor import PrithviWxC_Encoder

# Import MERRA-2 scalers if available
try:
    from PrithviWxC.dataloaders.merra2 import input_scalers, static_input_scalers
    SCALERS_AVAILABLE = True
except ImportError:
    SCALERS_AVAILABLE = False
    print("⚠️  PrithviWxC scalers not available, using dummy scalers")


def create_dummy_encoder_config():
    """Create a configuration for PrithviWxC_Encoder compatible with our data."""

    # Calculate input channels based on our processed data
    # Surface: 20 variables
    # Vertical: 10 variables * 14 levels = 140 variables
    # Total input channels: 20 + 140 = 160

    config = {
        "in_channels": 160,  # 20 surface + 10*14 vertical
        "input_size_time": 2,  # Number of input time steps
        "in_channels_static": 4,  # 4 static variables
        "n_lats_px": 361,  # MERRA-2 latitude dimension
        "n_lons_px": 576,  # MERRA-2 longitude dimension
        "patch_size_px": (8, 8),  # Patch size for tokenization
        "mask_unit_size_px": (16, 16),  # Mask unit size
        "mask_ratio_inputs": 0.0,  # No masking for inference
        "embed_dim": 512,  # Embedding dimension
        "n_blocks_encoder": 6,  # Number of encoder blocks
        "mlp_multiplier": 4.0,  # MLP multiplier
        "n_heads": 8,  # Number of attention heads
        "dropout": 0.1,  # Dropout rate
        "drop_path": 0.1,  # DropPath rate
        "parameter_dropout": 0.0,  # Parameter dropout
        "residual": "climate",  # Residual mode
        "masking_mode": "global",  # Masking mode
        "positional_encoding": "fourier",  # Position encoding
        "encoder_shifting": False,  # No SWIN shifting
        "checkpoint_encoder": None,  # No gradient checkpointing
        "input_scalers_epsilon": 1e-5,  # Scaler epsilon
        "static_input_scalers_epsilon": 1e-5,  # Static scaler epsilon
    }

    return config


def create_dummy_scalers(config):
    """Create dummy scalers when real ones are not available."""

    in_channels = config["in_channels"]
    in_channels_static = config["in_channels_static"]

    # Create dummy scalers (mean=0, std=1)
    input_scalers_mu = torch.zeros(in_channels)
    input_scalers_sigma = torch.ones(in_channels)
    static_input_scalers_mu = torch.zeros(in_channels_static)
    static_input_scalers_sigma = torch.ones(in_channels_static)

    return input_scalers_mu, input_scalers_sigma, static_input_scalers_mu, static_input_scalers_sigma


def load_real_scalers(config):
    """Load real MERRA-2 scalers if available."""
    if not SCALERS_AVAILABLE:
        return create_dummy_scalers(config)

    # Define variables matching our processed data
    surface_vars = [
        "EFLUX", "GWETROOT", "HFLUX", "LAI", "LWGAB", "LWGEM", "LWTUP",
        "PS", "QV2M", "SLP", "SWGNT", "SWTNT", "T2M", "TQI", "TQL",
        "TQV", "TS", "U10M", "V10M", "Z0M"
    ]
    static_surface_vars = ["FRACI", "FRLAND", "FROCEAN", "PHIS"]
    vertical_vars = ["CLOUD", "H", "OMEGA", "PL", "QI", "QL", "QV", "T", "U", "V"]
    levels = [34.0, 39.0, 41.0, 43.0, 44.0, 45.0, 48.0, 51.0, 53.0, 56.0, 63.0, 68.0, 71.0, 72.0]

    try:
        # Try to load real scalers
        surf_scaler_path = Path("data/climatology/musigma_surface.nc")
        vert_scaler_path = Path("data/climatology/musigma_vertical.nc")

        if surf_scaler_path.exists() and vert_scaler_path.exists():
            in_mu, in_sig = input_scalers(surface_vars, vertical_vars, levels, surf_scaler_path, vert_scaler_path)
            static_mu, static_sig = static_input_scalers(surf_scaler_path, static_surface_vars)
            return in_mu, in_sig, static_mu, static_sig
        else:
            print("⚠️  Scaler files not found, using dummy scalers")
            return create_dummy_scalers(config)

    except Exception as e:
        print(f"⚠️  Error loading scalers: {e}, using dummy scalers")
        return create_dummy_scalers(config)


def reshape_data_for_encoder(batch, config):
    """
    Reshape dataset batch to match PrithviWxC_Encoder expected format.

    Args:
        batch: Batch from PrithviMERRA2Dataset
        config: Encoder configuration

    Returns:
        Reformatted batch for encoder
    """
    encoder_batch = {}

    # Input data: (batch, time, vars, lat, lon) -> (batch, time, vars, lat, lon)
    if 'x' in batch:
        encoder_batch['x'] = batch['x']

    # Static data: (batch, vars, lat, lon) -> (batch, vars, lat, lon)
    if 'static' in batch:
        encoder_batch['static'] = batch['static']

    # Time information
    if 'input_time' in batch:
        encoder_batch['input_time'] = batch['input_time'].squeeze(-1)  # Remove extra dimension
    if 'lead_time' in batch:
        encoder_batch['lead_time'] = batch['lead_time'].squeeze(-1)  # Remove extra dimension

    # Climate data (if using residual mode)
    if config["residual"] == "climate":
        # For demo, use the same as input data
        # In practice, this would be climatological data
        if 'x' in batch:
            # Use mean across time dimension as pseudo-climate
            encoder_batch['climate'] = batch['x'].mean(dim=1)  # (batch, vars, lat, lon)

    return encoder_batch


def test_encoder_integration(dataset_path: str, config: dict):
    """
    Test full integration between dataset and encoder.

    Args:
        dataset_path: Path to processed MERRA-2 dataset
        config: Encoder configuration
    """
    print("=== Testing PrithviWxC_Encoder Integration ===\n")

    # 1. Load dataset
    print("1. Loading processed MERRA-2 dataset...")
    try:
        dataset = PrithviMERRA2Dataset(
            dataset_path=dataset_path,
            input_time_steps=config["input_size_time"],
            time_step_hours=6,
            lead_time_hours=6
        )
        print(f"✅ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False

    # 2. Create data loader
    print("\n2. Creating PyTorch DataLoader...")
    dataloader = MERRA2DataLoader.create_dataloader(
        dataset_path=dataset_path,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        input_time_steps=config["input_size_time"]
    )
    print("✅ DataLoader created")

    # 3. Load scalers
    print("\n3. Loading input scalers...")
    input_mu, input_sig, static_mu, static_sig = load_real_scalers(config)
    print(f"✅ Scalers loaded: input({input_mu.shape}), static({static_mu.shape})")

    # 4. Create encoder model
    print("\n4. Creating PrithviWxC_Encoder model...")
    try:
        encoder = PrithviWxC_Encoder(
            in_channels=config["in_channels"],
            input_size_time=config["input_size_time"],
            in_channels_static=config["in_channels_static"],
            input_scalers_mu=input_mu,
            input_scalers_sigma=input_sig,
            input_scalers_epsilon=config["input_scalers_epsilon"],
            static_input_scalers_mu=static_mu,
            static_input_scalers_sigma=static_sig,
            static_input_scalers_epsilon=config["static_input_scalers_epsilon"],
            n_lats_px=config["n_lats_px"],
            n_lons_px=config["n_lons_px"],
            patch_size_px=config["patch_size_px"],
            mask_unit_size_px=config["mask_unit_size_px"],
            mask_ratio_inputs=config["mask_ratio_inputs"],
            embed_dim=config["embed_dim"],
            n_blocks_encoder=config["n_blocks_encoder"],
            mlp_multiplier=config["mlp_multiplier"],
            n_heads=config["n_heads"],
            dropout=config["dropout"],
            drop_path=config["drop_path"],
            parameter_dropout=config["parameter_dropout"],
            residual=config["residual"],
            masking_mode=config["masking_mode"],
            positional_encoding=config["positional_encoding"],
            encoder_shifting=config["encoder_shifting"],
            checkpoint_encoder=config["checkpoint_encoder"]
        )

        total_params = sum(p.numel() for p in encoder.parameters())
        print(f"✅ Encoder created: {total_params:,} parameters")

    except Exception as e:
        print(f"❌ Failed to create encoder: {e}")
        return False

    # 5. Test forward pass
    print("\n5. Testing forward pass...")
    encoder.eval()

    try:
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                print(f"   Testing batch {i + 1}:")

                # Show input shapes
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"     {key}: {value.shape}")

                # Reshape for encoder
                encoder_batch = reshape_data_for_encoder(batch, config)

                # Forward pass
                try:
                    encoded_features = encoder(encoder_batch)
                    print(f"     ✅ Forward pass successful")
                    print(f"     Output shape: {encoded_features.shape}")
                    print(f"     Output range: [{encoded_features.min():.3f}, {encoded_features.max():.3f}]")

                except Exception as e:
                    print(f"     ❌ Forward pass failed: {e}")
                    return False

                if i >= 2:  # Test first 3 batches
                    break

    except Exception as e:
        print(f"❌ Error during forward pass testing: {e}")
        return False

    print("\n✅ Integration test completed successfully!")
    print("\nThe processed MERRA-2 dataset is fully compatible with PrithviWxC_Encoder")

    return True


def test_multimodal_fusion_pipeline(dataset_path: str, config: dict):
    """
    Test the encoder in a multimodal fusion context.

    Args:
        dataset_path: Path to processed dataset
        config: Encoder configuration
    """
    print("\n=== Testing Multimodal Fusion Pipeline ===\n")

    # Create a simple fusion model that uses the encoder
    class SimpleMultimodalModel(nn.Module):
        def __init__(self, encoder_config):
            super().__init__()

            # Load scalers
            input_mu, input_sig, static_mu, static_sig = load_real_scalers(encoder_config)

            # Create encoder
            self.climate_encoder = PrithviWxC_Encoder(
                in_channels=encoder_config["in_channels"],
                input_size_time=encoder_config["input_size_time"],
                in_channels_static=encoder_config["in_channels_static"],
                input_scalers_mu=input_mu,
                input_scalers_sigma=input_sig,
                input_scalers_epsilon=encoder_config["input_scalers_epsilon"],
                static_input_scalers_mu=static_mu,
                static_input_scalers_sigma=static_sig,
                static_input_scalers_epsilon=encoder_config["static_input_scalers_epsilon"],
                n_lats_px=encoder_config["n_lats_px"],
                n_lons_px=encoder_config["n_lons_px"],
                patch_size_px=encoder_config["patch_size_px"],
                mask_unit_size_px=encoder_config["mask_unit_size_px"],
                mask_ratio_inputs=encoder_config["mask_ratio_inputs"],
                embed_dim=encoder_config["embed_dim"],
                n_blocks_encoder=encoder_config["n_blocks_encoder"],
                mlp_multiplier=encoder_config["mlp_multiplier"],
                n_heads=encoder_config["n_heads"],
                dropout=encoder_config["dropout"],
                drop_path=encoder_config["drop_path"],
                parameter_dropout=encoder_config["parameter_dropout"],
                residual=encoder_config["residual"],
                masking_mode=encoder_config["masking_mode"],
                positional_encoding=encoder_config["positional_encoding"],
                encoder_shifting=encoder_config["encoder_shifting"],
                checkpoint_encoder=encoder_config["checkpoint_encoder"]
            )

            # Simple text encoder (dummy)
            self.text_encoder = nn.Linear(512, encoder_config["embed_dim"])

            # Fusion layer
            self.fusion = nn.MultiheadAttention(
                embed_dim=encoder_config["embed_dim"],
                num_heads=encoder_config["n_heads"],
                batch_first=True
            )

            # Output projection
            self.output_proj = nn.Linear(encoder_config["embed_dim"], 128)

        def forward(self, climate_batch, text_features):
            # Encode climate data
            climate_encoded = self.climate_encoder(climate_batch)  # (batch, seq, embed_dim)

            # Encode text (dummy processing)
            text_encoded = self.text_encoder(text_features)  # (batch, embed_dim)
            text_encoded = text_encoded.unsqueeze(1)  # (batch, 1, embed_dim)

            # Fusion via cross-attention
            fused_features, _ = self.fusion(
                query=text_encoded,
                key=climate_encoded,
                value=climate_encoded
            )  # (batch, 1, embed_dim)

            # Output projection
            output = self.output_proj(fused_features.squeeze(1))  # (batch, 128)

            return output

    # Test the multimodal model
    print("1. Creating multimodal fusion model...")
    try:
        model = SimpleMultimodalModel(config)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Multimodal model created: {total_params:,} parameters")
    except Exception as e:
        print(f"❌ Failed to create multimodal model: {e}")
        return False

    print("\n2. Testing multimodal forward pass...")

    # Load dataset
    dataloader = MERRA2DataLoader.create_dataloader(
        dataset_path=dataset_path,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        input_time_steps=config["input_size_time"]
    )

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            try:
                # Prepare climate data
                climate_batch = reshape_data_for_encoder(batch, config)

                # Create dummy text features
                batch_size = batch['x'].shape[0]
                text_features = torch.randn(batch_size, 512)

                # Forward pass
                output = model(climate_batch, text_features)

                print(f"   Batch {i + 1}:")
                print(f"     Climate input: {batch['x'].shape}")
                print(f"     Text input: {text_features.shape}")
                print(f"     Fused output: {output.shape}")
                print(f"     ✅ Multimodal forward pass successful")

            except Exception as e:
                print(f"❌ Multimodal forward pass failed: {e}")
                return False

            if i >= 1:  # Test 2 batches
                break

    print("\n✅ Multimodal fusion pipeline test completed successfully!")

    return True


def main():
    """Run comprehensive integration tests."""

    print("=== MERRA-2 / PrithviWxC_Encoder Integration Test ===\n")

    # Configuration
    config = create_dummy_encoder_config()

    # Test dataset path (modify as needed)
    dataset_path = "./example_output/merra2_prithvi_2020-01-01_2020-01-07_3H.npz"

    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please run basic_usage.py first to create a test dataset")
        return

    print(f"📂 Using dataset: {dataset_path}")
    print(f"🔧 Encoder configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()

    # Run tests
    success = True

    # Test 1: Basic encoder integration
    success &= test_encoder_integration(dataset_path, config)

    # Test 2: Multimodal fusion pipeline
    if success:
        success &= test_multimodal_fusion_pipeline(dataset_path, config)

    if success:
        print("\n🎉 All integration tests passed!")
        print("\nYour MERRA-2 dataset processing pipeline is ready for:")
        print("- Training PrithviWxC_Encoder models")
        print("- Multimodal climate-text fusion applications")
        print("- Large-scale climate data analysis")
    else:
        print("\n❌ Some integration tests failed")
        print("Please check the error messages above and fix any issues")


if __name__ == "__main__":
    main()

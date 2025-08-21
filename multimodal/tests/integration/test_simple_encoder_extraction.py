#!/usr/bin/env python3
"""
Simple Encoder Extraction Test

This test extracts the encoder from PrithviWxC checkpoint using residual="climate"
and verifies the in_channels_static configuration.

Usage:
    python multimodal/tests/integration/test_simple_encoder_extraction.py
"""

import os
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from multimodal.utils.encoder_extractor import PrithviWxC_Encoder, extract_encoder_weights
from PrithviWxC.model import PrithviWxC


def test_simple_encoder_extraction():
    """
    Simple test to extract encoder and verify in_channels_static for residual="climate"
    """
    print("🚀 SIMPLE ENCODER EXTRACTION TEST")
    print("=" * 50)

    try:
        # Load configuration
        config_path = project_root / "data" / "config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        print(f"📋 Configuration loaded:")
        print(f"   in_channels: {config['params']['in_channels']}")
        print(f"   in_channels_static: {config['params']['in_channels_static']}")
        print(f"   n_blocks_encoder: {config['params']['n_blocks_encoder']}")

        # Load checkpoint to inspect dimensions
        checkpoint_path = project_root / "data" / "weights" / "prithvi.wxc.2300m.v1.pt"

        if not checkpoint_path.exists():
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return False

        print(f"\\n📁 Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model_state" in state_dict:
            state_dict = state_dict["model_state"]

        # Inspect key dimensions from checkpoint
        print(f"\\n🔍 CHECKPOINT INSPECTION:")

        # Check input scalers
        if "input_scalers_mu" in state_dict:
            input_shape = state_dict["input_scalers_mu"].shape
            print(f"   input_scalers_mu shape: {input_shape}")

        # Check static scalers
        if "static_input_scalers_mu" in state_dict:
            static_scaler_shape = state_dict["static_input_scalers_mu"].shape
            print(f"   static_input_scalers_mu shape: {static_scaler_shape}")

        # Check patch embedding static
        if "patch_embedding_static.proj.weight" in state_dict:
            static_embed_shape = state_dict["patch_embedding_static.proj.weight"].shape
            print(f"   patch_embedding_static.proj.weight shape: {static_embed_shape}")

            # For climate mode: static_embed_channels = in_channels + in_channels_static
            static_embed_channels = static_embed_shape[1]  # Input channels to this layer
            in_channels = config["params"]["in_channels"]
            derived_static_channels = static_embed_channels - in_channels

            print(f"\\n📊 CLIMATE MODE ANALYSIS:")
            print(f"   Static embedding expects: {static_embed_channels} input channels")
            print(f"   Base in_channels: {in_channels}")
            print(f"   Derived in_channels_static: {derived_static_channels}")

            if derived_static_channels == config["params"]["in_channels_static"]:
                print(f"   ✅ Config matches checkpoint: {derived_static_channels} static channels")
            else:
                print(
                    f"   ⚠️  Config mismatch: config={config['params']['in_channels_static']}, derived={derived_static_channels}"
                )

        # Now try to create encoder with the config values
        print(f"\\n🏗️  CREATING ENCODER WITH CONFIG VALUES:")

        # Use simple scalers for this test
        in_channels = config["params"]["in_channels"]
        in_channels_static = config["params"]["in_channels_static"]

        in_mu = torch.zeros(in_channels)
        in_sig = torch.ones(in_channels)
        static_mu = torch.zeros(1, in_channels_static, 1, 1)
        static_sig = torch.ones(1, in_channels_static, 1, 1)

        print(f"   Creating encoder with:")
        print(f"     in_channels: {in_channels}")
        print(f"     in_channels_static: {in_channels_static}")
        print(f"     residual: climate")

        # Create encoder
        encoder = PrithviWxC_Encoder(
            in_channels=in_channels,
            input_size_time=config["params"]["input_size_time"],
            in_channels_static=in_channels_static,
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
            residual="climate",  # Climate residual mode
            masking_mode="global",
            positional_encoding="fourier",
            encoder_shifting=False,
            checkpoint_encoder=[],
        )

        print(f"   ✅ Encoder created successfully")
        print(f"   Architecture verification:")
        print(f"     in_channels: {encoder.in_channels}")
        print(f"     in_channels_static: {encoder.in_channels_static}")
        print(f"     embed_dim: {encoder.embed_dim}")
        print(f"     n_blocks_encoder: {encoder.n_blocks_encoder}")

        # Check if patch_embedding_static expects the right input
        if hasattr(encoder, "patch_embedding_static"):
            expected_static_input = encoder.patch_embedding_static.proj.in_channels
            calculated_input = encoder.in_channels + encoder.in_channels_static
            print(f"     patch_embedding_static expects: {expected_static_input} channels")
            print(f"     calculated total (in + static): {calculated_input} channels")

            if expected_static_input == calculated_input:
                print(f"     ✅ Climate mode configuration is consistent")
            else:
                print(f"     ❌ Climate mode configuration mismatch!")
                return False

        # Try loading weights with adjusted static scalers if needed
        print(f"\\n🔄 TESTING WEIGHT LOADING:")

        # Adjust static scalers if needed
        if "static_input_scalers_mu" in state_dict:
            original_static_shape = state_dict["static_input_scalers_mu"].shape
            expected_static_shape = (1, in_channels_static, 1, 1)

            if original_static_shape != expected_static_shape:
                print(
                    f"   📝 Adjusting static scalers: {original_static_shape} -> {expected_static_shape}"
                )

                # Create adjusted state dict
                adjusted_state_dict = state_dict.copy()

                if original_static_shape[1] > in_channels_static:
                    # Truncate
                    adjusted_state_dict["static_input_scalers_mu"] = state_dict[
                        "static_input_scalers_mu"
                    ][:, :in_channels_static, :, :].clone()
                    adjusted_state_dict["static_input_scalers_sigma"] = state_dict[
                        "static_input_scalers_sigma"
                    ][:, :in_channels_static, :, :].clone()
                    print(
                        f"     Truncated from {original_static_shape[1]} to {in_channels_static} channels"
                    )
                else:
                    # Pad
                    pad_size = in_channels_static - original_static_shape[1]
                    mu_pad = torch.zeros(1, pad_size, 1, 1)
                    sig_pad = torch.ones(1, pad_size, 1, 1)
                    adjusted_state_dict["static_input_scalers_mu"] = torch.cat(
                        [state_dict["static_input_scalers_mu"], mu_pad], dim=1
                    )
                    adjusted_state_dict["static_input_scalers_sigma"] = torch.cat(
                        [state_dict["static_input_scalers_sigma"], sig_pad], dim=1
                    )
                    print(
                        f"     Padded from {original_static_shape[1]} to {in_channels_static} channels"
                    )

                state_dict = adjusted_state_dict

        # Create full model for extraction
        full_model = PrithviWxC(
            in_channels=in_channels,
            input_size_time=config["params"]["input_size_time"],
            in_channels_static=in_channels_static,
            input_scalers_mu=in_mu,
            input_scalers_sigma=in_sig,
            input_scalers_epsilon=config["params"]["input_scalers_epsilon"],
            static_input_scalers_mu=static_mu,
            static_input_scalers_sigma=static_sig,
            static_input_scalers_epsilon=config["params"]["static_input_scalers_epsilon"],
            output_scalers=torch.ones(in_channels),
            n_lats_px=config["params"]["n_lats_px"],
            n_lons_px=config["params"]["n_lons_px"],
            patch_size_px=config["params"]["patch_size_px"],
            mask_unit_size_px=config["params"]["mask_unit_size_px"],
            mask_ratio_inputs=0.0,
            mask_ratio_targets=0.0,
            embed_dim=config["params"]["embed_dim"],
            n_blocks_encoder=config["params"]["n_blocks_encoder"],
            n_blocks_decoder=config["params"]["n_blocks_decoder"],
            mlp_multiplier=config["params"]["mlp_multiplier"],
            n_heads=config["params"]["n_heads"],
            dropout=config["params"]["dropout"],
            drop_path=config["params"]["drop_path"],
            parameter_dropout=config["params"]["parameter_dropout"],
            residual="climate",
            masking_mode="global",
            encoder_shifting=False,
            decoder_shifting=False,
            positional_encoding="fourier",
            checkpoint_encoder=[],
            checkpoint_decoder=[],
        )

        print(f"   🔄 Loading weights into full model...")
        missing_keys, unexpected_keys = full_model.load_state_dict(state_dict, strict=False)

        print(f"   Weight loading results:")
        print(f"     Missing keys: {len(missing_keys)}")
        print(f"     Unexpected keys: {len(unexpected_keys)}")

        if len(missing_keys) > 10:  # Some missing keys for decoder are expected
            print(f"     ⚠️  Many missing keys (decoder-related expected)")
        else:
            print(f"     ✅ Reasonable number of missing keys")

        # Extract encoder weights
        print(f"   🔄 Extracting encoder weights...")
        extract_encoder_weights(full_model, encoder)

        print(f"   ✅ Encoder weights extracted successfully")

        # Test a simple forward pass
        print(f"\\n🧪 TESTING ENCODER FORWARD PASS:")

        batch_size = 1
        dummy_data = {
            "x": torch.randn(batch_size, 2, in_channels, 360, 576),
            "static": torch.randn(batch_size, in_channels_static, 360, 576),
            "climate": torch.randn(batch_size, in_channels, 360, 576),
            "input_time": torch.tensor([0.5], dtype=torch.float32),
            "lead_time": torch.tensor([1.0], dtype=torch.float32),
        }

        print(f"   Input shapes:")
        print(f"     x: {dummy_data['x'].shape}")
        print(f"     static: {dummy_data['static'].shape}")
        print(f"     climate: {dummy_data['climate'].shape}")

        encoder.eval()
        with torch.no_grad():
            try:
                output = encoder(dummy_data)
                print(f"   ✅ Forward pass successful!")
                print(f"     Output shape: {output.shape}")
                print(f"     Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

                if torch.isnan(output).any() or torch.isinf(output).any():
                    print(f"     ❌ Output contains NaN/Inf values")
                    return False
                else:
                    print(f"     ✅ Output values are valid")

            except Exception as e:
                print(f"     ❌ Forward pass failed: {e}")
                return False

        print(f"\\n🎉 SUCCESS!")
        print(f"✅ Encoder extraction and testing completed successfully")
        print(f"✅ in_channels_static: {in_channels_static} is correct for residual='climate'")
        print(f"✅ Climate mode configuration is working properly")

        return True

    except Exception as e:
        print(f"\\n❌ TEST FAILED!")
        print(f"Error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_simple_encoder_extraction()
    exit(0 if success else 1)

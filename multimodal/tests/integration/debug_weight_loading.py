#!/usr/bin/env python3
"""
Debug Weight Loading Issues

This script investigates the missing keys issue in encoder weight loading.
"""

import os
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from multimodal.utils.encoder_extractor import PrithviWxC_Encoder
from PrithviWxC.model import PrithviWxC


def debug_weight_loading():
    """Debug the weight loading process step by step."""
    print("🔍 DEBUGGING WEIGHT LOADING ISSUES")
    print("=" * 60)

    # Load configuration
    config_path = project_root / "data" / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"📋 Configuration loaded:")
    print(f"   in_channels: {config['params']['in_channels']}")
    print(f"   in_channels_static: {config['params']['in_channels_static']}")
    print(f"   n_blocks_encoder: {config['params']['n_blocks_encoder']}")

    # Load checkpoint
    checkpoint_path = project_root / "data" / "weights" / "prithvi.wxc.2300m.v1.pt"
    print(f"\n📁 Loading checkpoint: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state" in state_dict:
        state_dict = state_dict["model_state"]

    print(f"\n🔍 CHECKPOINT ANALYSIS:")
    print(f"   Total keys in checkpoint: {len(state_dict.keys())}")

    # Analyze key patterns
    encoder_keys = [k for k in state_dict.keys() if not k.startswith("decoder")]
    decoder_keys = [k for k in state_dict.keys() if k.startswith("decoder")]
    other_keys = [
        k for k in state_dict.keys() if not k.startswith("encoder") and not k.startswith("decoder")
    ]

    print(
        f"   Keys starting with 'encoder': {len([k for k in state_dict.keys() if k.startswith('encoder')])}"
    )
    print(f"   Keys starting with 'decoder': {len(decoder_keys)}")
    print(f"   Other keys: {len(other_keys)}")

    print(f"\n📊 OTHER KEYS (non-encoder/decoder):")
    for key in sorted(other_keys)[:20]:  # Show first 20
        print(
            f"     {key}: {state_dict[key].shape if torch.is_tensor(state_dict[key]) else type(state_dict[key])}"
        )
    if len(other_keys) > 20:
        print(f"     ... and {len(other_keys) - 20} more")

    # Detect architecture from checkpoint
    actual_in_channels = config["params"]["in_channels"]  # 160
    if "patch_embedding_static.proj.weight" in state_dict:
        static_embed_channels = state_dict["patch_embedding_static.proj.weight"].shape[1]
        actual_static_channels = static_embed_channels - actual_in_channels
        print(f"\n🔍 ARCHITECTURE DETECTION:")
        print(f"   Static embedding expects: {static_embed_channels} total channels")
        print(f"   Base in_channels: {actual_in_channels}")
        print(f"   Derived in_channels_static: {actual_static_channels}")
    else:
        actual_static_channels = config["params"]["in_channels_static"]
        print(f"\n⚠️  No static embedding found, using config: {actual_static_channels}")

    # Handle scalers
    if "static_input_scalers_mu" in state_dict:
        scaler_static_channels = state_dict["static_input_scalers_mu"].shape[1]
        print(f"\n📊 STATIC SCALERS:")
        print(f"   Scaler channels: {scaler_static_channels}")
        print(f"   Target channels: {actual_static_channels}")

        if scaler_static_channels != actual_static_channels:
            print(f"   ⚠️  Mismatch detected - need adjustment")

    # Create scalers for model initialization
    if "input_scalers_mu" in state_dict:
        in_mu = state_dict["input_scalers_mu"].clone()
        in_sig = state_dict["input_scalers_sigma"].clone()
        if in_mu.dim() > 1:
            in_mu = in_mu.squeeze()
        if in_sig.dim() > 1:
            in_sig = in_sig.squeeze()
    else:
        in_mu = torch.zeros(actual_in_channels)
        in_sig = torch.ones(actual_in_channels)

    static_mu = state_dict.get(
        "static_input_scalers_mu", torch.zeros(1, actual_static_channels, 1, 1)
    ).clone()
    static_sig = state_dict.get(
        "static_input_scalers_sigma", torch.ones(1, actual_static_channels, 1, 1)
    ).clone()

    # Create the FULL model first
    print(f"\n🏗️  CREATING FULL PRITHVIWXC MODEL:")
    full_model = PrithviWxC(
        in_channels=actual_in_channels,
        input_size_time=config["params"]["input_size_time"],
        in_channels_static=actual_static_channels,
        input_scalers_mu=in_mu,
        input_scalers_sigma=in_sig,
        input_scalers_epsilon=config["params"]["input_scalers_epsilon"],
        static_input_scalers_mu=static_mu,
        static_input_scalers_sigma=static_sig,
        static_input_scalers_epsilon=config["params"]["static_input_scalers_epsilon"],
        output_scalers=torch.ones(actual_in_channels),
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

    print(f"   ✅ Full model created")

    # Get the full model's expected keys
    full_model_keys = set(full_model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())

    print(f"\n📊 KEY ANALYSIS:")
    print(f"   Full model expects: {len(full_model_keys)} keys")
    print(f"   Checkpoint provides: {len(checkpoint_keys)} keys")

    missing_in_checkpoint = full_model_keys - checkpoint_keys
    unexpected_in_checkpoint = checkpoint_keys - full_model_keys

    print(f"   Missing from checkpoint: {len(missing_in_checkpoint)}")
    print(f"   Unexpected in checkpoint: {len(unexpected_in_checkpoint)}")

    if missing_in_checkpoint:
        print(f"\n❌ MISSING KEYS (first 20):")
        for key in sorted(list(missing_in_checkpoint))[:20]:
            print(f"     {key}")
        if len(missing_in_checkpoint) > 20:
            print(f"     ... and {len(missing_in_checkpoint) - 20} more")

    if unexpected_in_checkpoint:
        print(f"\n⚠️  UNEXPECTED KEYS (first 20):")
        for key in sorted(list(unexpected_in_checkpoint))[:20]:
            print(f"     {key}")
        if len(unexpected_in_checkpoint) > 20:
            print(f"     ... and {len(unexpected_in_checkpoint) - 20} more")

    # Now test loading
    print(f"\n🔄 TESTING WEIGHT LOADING:")
    try:
        missing_keys, unexpected_keys = full_model.load_state_dict(state_dict, strict=False)
        print(f"   Missing keys: {len(missing_keys)}")
        print(f"   Unexpected keys: {len(unexpected_keys)}")

        if len(missing_keys) > 0:
            print(f"\n❌ MISSING KEYS DURING LOADING (first 20):")
            for key in missing_keys[:20]:
                print(f"     {key}")
            if len(missing_keys) > 20:
                print(f"     ... and {len(missing_keys) - 20} more")

        if len(unexpected_keys) > 0:
            print(f"\n⚠️  UNEXPECTED KEYS DURING LOADING (first 20):")
            for key in unexpected_keys[:20]:
                print(f"     {key}")
            if len(unexpected_keys) > 20:
                print(f"     ... and {len(unexpected_keys) - 20} more")

    except Exception as e:
        print(f"   ❌ Loading failed: {e}")
        return False

    # Now create encoder and compare
    print(f"\n🏗️  CREATING ENCODER MODEL:")
    encoder_model = PrithviWxC_Encoder(
        in_channels=actual_in_channels,
        input_size_time=config["params"]["input_size_time"],
        in_channels_static=actual_static_channels,
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

    print(f"   ✅ Encoder created")

    encoder_keys = set(encoder_model.state_dict().keys())
    print(f"   Encoder expects: {len(encoder_keys)} keys")

    # Compare encoder keys with full model keys
    encoder_in_full = encoder_keys.intersection(full_model_keys)
    encoder_not_in_full = encoder_keys - full_model_keys

    print(f"   Encoder keys also in full model: {len(encoder_in_full)}")
    print(f"   Encoder keys NOT in full model: {len(encoder_not_in_full)}")

    if encoder_not_in_full:
        print(f"\n⚠️  ENCODER KEYS NOT IN FULL MODEL:")
        for key in sorted(encoder_not_in_full):
            print(f"     {key}")

    print(f"\n🎯 CONCLUSION:")
    if len(missing_keys) > 50:  # Too many missing keys
        print(f"   ❌ Too many missing keys ({len(missing_keys)}) - likely architectural mismatch")
        print(f"   🔍 Check if full model configuration matches checkpoint")
    else:
        print(
            f"   ✅ Missing keys count acceptable ({len(missing_keys)}) - likely just decoder keys"
        )

    return True


if __name__ == "__main__":
    debug_weight_loading()

#!/usr/bin/env python3
"""
Minimal test to debug the tuple index out of range error
"""

import torch
import sys
import os
import yaml

# Add the project root to Python path
sys.path.append('/Users/arigazzi/Documents/DeepLearning/LLM for climate/HPE-LLM4Climate')

from multimodal.utils.encoder_extractor import PrithviWxC_Encoder

def debug_forward_pass():
    """Debug the specific forward pass error."""

    print("🔍 DEBUGGING FORWARD PASS ERROR")
    print("=" * 50)

    # Load config
    config_path = "/Users/arigazzi/Documents/DeepLearning/LLM for climate/HPE-LLM4Climate/data/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Basic parameters from config
    in_channels = config["params"]["in_channels"]  # 160
    in_channels_static = config["params"]["in_channels_static"]  # 8

    print(f"Config: in_channels={in_channels}, in_channels_static={in_channels_static}")

    # Create simple scalers
    in_mu = torch.zeros(in_channels)
    in_sig = torch.ones(in_channels)
    static_mu = torch.zeros(1, in_channels_static, 1, 1)
    static_sig = torch.ones(1, in_channels_static, 1, 1)

    print("📋 Creating encoder...")

    # Create encoder with simplified parameters
    try:
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
            mask_ratio_inputs=0.0,  # Default value
            embed_dim=config["params"]["embed_dim"],
            n_blocks_encoder=config["params"]["n_blocks_encoder"],
            mlp_multiplier=config["params"]["mlp_multiplier"],
            n_heads=config["params"]["n_heads"],
            dropout=config["params"]["dropout"],
            drop_path=config["params"]["drop_path"],
            parameter_dropout=config["params"]["parameter_dropout"],
            residual="climate",
            positional_encoding="absolute",  # Default value
            masking_mode="global",  # Default value
            encoder_shifting=False,  # Default value
            checkpoint_encoder=config["params"]["checkpoint_encoder"]
        )
        print("✅ Encoder created successfully")

        # Debug shape calculations
        print(f"\\n🔍 Shape calculations:")
        print(f"   n_lats_px: {encoder.n_lats_px}")
        print(f"   n_lons_px: {encoder.n_lons_px}")
        print(f"   mask_unit_size_px: {encoder.mask_unit_size_px}")
        print(f"   global_shape_mu: {encoder.global_shape_mu}")
        print(f"   _nglobal_mu: {encoder._nglobal_mu}")
        print(f"   _global_idx shape: {encoder._global_idx.shape}")

    except Exception as e:
        print(f"❌ Encoder creation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test forward pass step by step
    print(f"\\n🧪 Testing forward pass step by step...")

    batch_size = 1
    dummy_data = {
        'x': torch.randn(batch_size, 2, in_channels, 360, 576),
        'static': torch.randn(batch_size, in_channels_static, 360, 576),
        'climate': torch.randn(batch_size, in_channels, 360, 576),
        'input_time': torch.tensor([0.5], dtype=torch.float32),
        'lead_time': torch.tensor([1.0], dtype=torch.float32)
    }

    print(f"✅ Test data created")

    encoder.eval()
    with torch.no_grad():
        try:
            print("🔍 Step 1: Input validation...")
            assert dummy_data["x"].shape[2] == encoder.in_channels
            assert dummy_data["x"].shape[3] == encoder.n_lats_px
            assert dummy_data["x"].shape[4] == encoder.n_lons_px
            print("✅ Input validation passed")

            print("🔍 Step 2: Input rescaling...")
            x_rescaled = (dummy_data["x"] - encoder.input_scalers_mu) / (
                encoder.input_scalers_sigma + encoder.input_scalers_epsilon
            )
            print("✅ Input rescaling passed")

            print("🔍 Step 3: Static input processing...")
            x_static = (dummy_data["static"] - encoder.static_input_scalers_mu) / (
                encoder.static_input_scalers_sigma + encoder.static_input_scalers_epsilon
            )
            print("✅ Static input processing passed")

            print("🔍 Step 4: Climate data processing...")
            climate_scaled = (
                dummy_data["climate"] - encoder.input_scalers_mu.view(1, -1, 1, 1)
            ) / (
                encoder.input_scalers_sigma.view(1, -1, 1, 1) + encoder.input_scalers_epsilon
            )
            print("✅ Climate data processing passed")

            print("🔍 Step 5: Generate mask...")
            indices_masked, indices_unmasked = encoder.generate_mask(
                (batch_size, encoder._nglobal_mu)
            )
            print(f"✅ Mask generation passed")
            print(f"   indices_masked shape: {indices_masked.shape}")
            print(f"   indices_unmasked shape: {indices_unmasked.shape}")

            print("🔍 Step 6: Flatten and apply parameter dropout...")
            x_rescaled_flat = x_rescaled.flatten(1, 2)
            x_rescaled_dropout = encoder.parameter_dropout(x_rescaled_flat)
            print(f"✅ Flattening passed: {x_rescaled_flat.shape} -> {x_rescaled_dropout.shape}")

            print("🔍 Step 7: Patch embedding...")
            x_embedded = encoder.patch_embedding(x_rescaled_dropout)
            print(f"✅ Patch embedding passed: {x_embedded.shape}")

            print("🔍 Step 8: Static patch embedding...")
            static_concat = torch.cat((x_static, climate_scaled), dim=1)
            print(f"   Static concat shape: {static_concat.shape}")
            static_embedded = encoder.patch_embedding_static(static_concat)
            print(f"✅ Static patch embedding passed: {static_embedded.shape}")

            print("🔍 Step 9: Convert to patching...")
            x_embedded_patch = encoder.to_patching(x_embedded)
            static_embedded_patch = encoder.to_patching(static_embedded)
            print(f"✅ To patching passed: x={x_embedded_patch.shape}, static={static_embedded_patch.shape}")

            print("🔍 Step 10: Time encoding...")
            time_encoding = encoder.time_encoding(dummy_data["input_time"], dummy_data["lead_time"])
            print(f"✅ Time encoding passed: {time_encoding.shape}")

            print("🔍 Step 11: Combine embeddings...")
            tokens = x_embedded_patch + static_embedded_patch + time_encoding
            print(f"✅ Token combination passed: {tokens.shape}")

            print("🔍 Step 11b: Transpose for encoder...")
            tokens = tokens.transpose(1, 2)  # [batch, spatial_tokens, embed_dim]
            print(f"✅ Token transpose passed: {tokens.shape}")

            print("🔍 Step 12: Apply masking...")
            indices_unmasked = indices_unmasked.to(device=tokens.device)
            maskdim = indices_unmasked.ndim
            print(f"   maskdim: {maskdim}, tokens.ndim: {tokens.ndim}")
            print(f"   indices_unmasked shape: {indices_unmasked.shape}")
            print(f"   tokens shape: {tokens.shape}")

            # This is where the error likely occurs
            unmask_view = (*indices_unmasked.shape, *[1] * (tokens.ndim - maskdim))
            print(f"   unmask_view: {unmask_view}")

            unmasked = torch.gather(
                tokens,
                dim=maskdim - 1,
                index=indices_unmasked.view(*unmask_view).expand(
                    *indices_unmasked.shape, *tokens.shape[maskdim:]
                ),
            )
            print(f"✅ Masking passed: {unmasked.shape}")

            print("🔍 Step 13: Encoder forward...")
            x_encoded = encoder.encoder(unmasked)
            print(f"✅ Encoder forward passed: {x_encoded.shape}")

            print("\\n🎉 Full forward pass completed successfully!")

        except Exception as e:
            print(f"❌ Forward pass failed at step: {e}")
            import traceback
            traceback.print_exc()
            return    print("\\n🎉 All steps completed successfully!")

if __name__ == "__main__":
    debug_forward_pass()

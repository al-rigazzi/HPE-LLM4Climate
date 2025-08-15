"""
Fix Prithvi Encoder Extraction

This script re-extracts the encoder with the correct configuration inferred
from the actual model weights, fixing the static channel size mismatch.
"""

import torch
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def fix_encoder_extraction():
    """Re-extract the encoder with correct configuration."""

    print("🔧 Fixing Prithvi Encoder Extraction")
    print("=" * 50)

    # Load original model
    original_path = "data/weights/prithvi.wxc.2300m.v1.pt"
    output_path = "data/weights/prithvi_encoder.pt"

    print(f"📁 Loading original model from: {original_path}")
    original = torch.load(original_path, map_location='cpu')
    model_state = original['model_state']

    # Infer correct configuration from actual model weights
    print("\n🔍 Inferring correct configuration from model weights...")

    # Get static channel dimensions from actual tensors
    static_mu_shape = model_state['static_input_scalers_mu'].shape
    actual_static_channels = static_mu_shape[1]

    # Get other dimensions from patch embeddings
    patch_embed_weight = model_state['patch_embedding.proj.weight']
    patch_embed_static_weight = model_state['patch_embedding_static.proj.weight']

    # Infer in_channels from patch embedding
    # patch_embedding takes [time * in_channels + in_channels_static] channels
    total_input_channels = patch_embed_weight.shape[1] // 4  # Divide by 4 for 2x2 patches
    actual_in_channels = total_input_channels - actual_static_channels

    print(f"   ✅ Actual static channels: {actual_static_channels} (config said 8)")
    print(f"   ✅ Actual input channels: {actual_in_channels}")
    print(f"   ✅ Total input channels: {total_input_channels}")

    # Create corrected configuration
    corrected_config = {
        'params': {
            'in_channels': actual_in_channels,
            'input_size_time': 2,
            'in_channels_static': actual_static_channels,  # CORRECTED!
            'input_scalers_epsilon': 0.0,
            'static_input_scalers_epsilon': 0.0,
            'n_lats_px': 360,
            'n_lons_px': 576,
            'patch_size_px': [2, 2],
            'mask_unit_size_px': [30, 32],
            'embed_dim': 2560,
            'n_blocks_encoder': 12,
            'n_blocks_decoder': 2,
            'mlp_multiplier': 4,
            'n_heads': 16,
            'dropout': 0.0,
            'drop_path': 0.0,
            'parameter_dropout': 0.0,
            'checkpoint_encoder': [],
            'checkpoint_decoder': []
        }
    }

    print(f"\n📝 Corrected configuration:")
    print(f"   in_channels: {corrected_config['params']['in_channels']}")
    print(f"   in_channels_static: {corrected_config['params']['in_channels_static']}")

    # Extract encoder-specific weights
    print(f"\n📦 Extracting encoder weights...")
    encoder_weights = {}

    encoder_keys = [
        'mask_token',
        'input_scalers_mu',
        'input_scalers_sigma',
        'static_input_scalers_mu',
        'static_input_scalers_sigma',
        'patch_embedding.proj.weight',
        'patch_embedding.proj.bias',
        'patch_embedding_static.proj.weight',
        'patch_embedding_static.proj.bias',
        'input_time_embedding.weight',
        'input_time_embedding.bias'
    ]

    # Add encoder transformer blocks
    for i in range(12):  # n_blocks_encoder
        encoder_keys.extend([
            f'encoder.lgl_block.transformers.{i}.attention.0.weight',
            f'encoder.lgl_block.transformers.{i}.attention.0.bias',
            f'encoder.lgl_block.transformers.{i}.attention.1.qkv_layer.weight',
            f'encoder.lgl_block.transformers.{i}.attention.1.w_layer.weight',
            f'encoder.lgl_block.transformers.{i}.ff.0.weight',
            f'encoder.lgl_block.transformers.{i}.ff.0.bias',
            f'encoder.lgl_block.transformers.{i}.ff.1.net.0.weight',
            f'encoder.lgl_block.transformers.{i}.ff.1.net.0.bias',
            f'encoder.lgl_block.transformers.{i}.ff.1.net.3.weight',
            f'encoder.lgl_block.transformers.{i}.ff.1.net.3.bias'
        ])

    # Extract the weights
    for key in encoder_keys:
        if key in model_state:
            encoder_weights[key] = model_state[key]
            print(f"   ✅ {key}: {model_state[key].shape}")
        else:
            print(f"   ⚠️  Missing: {key}")

    print(f"\n💾 Saving corrected encoder to: {output_path}")

    # Save with corrected configuration
    torch.save({
        'model_state_dict': encoder_weights,
        'config': corrected_config
    }, output_path)

    print(f"\n✅ Successfully saved corrected encoder!")
    print(f"   📊 Total weights: {len(encoder_weights)}")
    print(f"   🔧 Corrected static channels: {actual_static_channels}")

    # Verify the fix
    print(f"\n🧪 Verifying the fix...")
    checkpoint = torch.load(output_path, map_location='cpu')

    config_static = checkpoint['config']['params']['in_channels_static']
    actual_static_shape = checkpoint['model_state_dict']['static_input_scalers_mu'].shape[1]

    if config_static == actual_static_shape:
        print(f"   ✅ FIXED! Config static channels ({config_static}) matches actual tensors ({actual_static_shape})")
    else:
        print(f"   ❌ Still broken: Config ({config_static}) != Actual ({actual_static_shape})")

    return True

if __name__ == "__main__":
    fix_encoder_extraction()

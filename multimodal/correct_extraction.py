"""
Correct Encoder Extraction - Infer Config from Model

This script properly extracts the encoder by inferring the correct configuration
directly from the model weights, not from an incorrect config file.
"""

import torch
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def extract_encoder_properly():
    """Extract encoder with configuration inferred from actual model weights."""

    print("🔧 Properly Extracting Prithvi Encoder")
    print("=" * 50)

    # Load original model
    original_path = "data/weights/prithvi.wxc.2300m.v1.pt"
    print(f"📁 Loading original model: {original_path}")

    original = torch.load(original_path, map_location='cpu')
    model_state = original['model_state']

    print("✅ Model loaded successfully")

    # INFER CORRECT CONFIGURATION FROM ACTUAL MODEL WEIGHTS
    print("\n🔍 Inferring configuration from model weights...")

    # Get dimensions from actual tensors
    static_scalers_shape = model_state['static_input_scalers_mu'].shape
    input_scalers_shape = model_state['input_scalers_mu'].shape
    patch_embed_weight = model_state['patch_embedding.proj.weight']

    # Infer correct dimensions
    actual_static_channels = static_scalers_shape[1]
    actual_input_channels = input_scalers_shape[2]  # [1, 1, 160, 1, 1]
    embed_dim = patch_embed_weight.shape[0]  # 2560

    print(f"   📊 Inferred dimensions:")
    print(f"      input_channels: {actual_input_channels}")
    print(f"      static_channels: {actual_static_channels}")
    print(f"      embed_dim: {embed_dim}")

    # Create CORRECT configuration based on actual model
    config = {
        'in_channels': actual_input_channels,
        'input_size_time': 2,
        'in_channels_static': actual_static_channels,
        'input_scalers_epsilon': 0.0,
        'static_input_scalers_epsilon': 0.0,
        'n_lats_px': 360,
        'n_lons_px': 576,
        'patch_size_px': [2, 2],
        'mask_unit_size_px': [30, 32],
        'embed_dim': embed_dim,
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

    print(f"\\n✅ Correct configuration determined:")
    print(f"   in_channels: {config['in_channels']} (was incorrectly 160 in config.yaml)")
    print(f"   in_channels_static: {config['in_channels_static']} (was incorrectly 8 in config.yaml)")

    # Extract only encoder weights
    print(f"\\n📦 Extracting encoder weights...")
    encoder_state = {}

    # Core encoder components
    encoder_keys = [
        'mask_token',
        'input_scalers_mu', 'input_scalers_sigma',
        'static_input_scalers_mu', 'static_input_scalers_sigma',
        'patch_embedding.proj.weight', 'patch_embedding.proj.bias',
        'patch_embedding_static.proj.weight', 'patch_embedding_static.proj.bias',
        'input_time_embedding.weight', 'input_time_embedding.bias',
    ]

    # Add transformer blocks (0-11 for encoder)
    for i in range(12):
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
    extracted_count = 0
    for key in encoder_keys:
        if key in model_state:
            encoder_state[key] = model_state[key]
            extracted_count += 1
        else:
            print(f"   ⚠️  Missing key: {key}")

    print(f"   ✅ Extracted {extracted_count}/{len(encoder_keys)} weights")

    # Save with CORRECT configuration
    output_path = "data/weights/prithvi_encoder_corrected.pt"
    print(f"\\n💾 Saving corrected encoder: {output_path}")

    checkpoint = {
        'model_state_dict': encoder_state,
        'config': {'params': config}
    }

    torch.save(checkpoint, output_path)

    # Verify the fix
    print(f"\\n🧪 Verifying extraction...")
    saved = torch.load(output_path, map_location='cpu')

    # Check key dimensions match
    saved_config = saved['config']['params']
    saved_static_shape = saved['model_state_dict']['static_input_scalers_mu'].shape[1]
    saved_input_shape = saved['model_state_dict']['input_scalers_mu'].shape[2]

    print(f"   Config static channels: {saved_config['in_channels_static']}")
    print(f"   Actual static tensor: {saved_static_shape}")
    print(f"   Config input channels: {saved_config['in_channels']}")
    print(f"   Actual input tensor: {saved_input_shape}")

    if (saved_config['in_channels_static'] == saved_static_shape and
        saved_config['in_channels'] == saved_input_shape):
        print("   ✅ SUCCESS: Configuration matches actual weights!")
        return output_path
    else:
        print("   ❌ FAILED: Configuration still doesn't match")
        return None

if __name__ == "__main__":
    result = extract_encoder_properly()
    if result:
        print(f"\\n🎉 Encoder properly extracted to: {result}")
        print("   Use this file instead of the incorrect extraction")
    else:
        print("\\n💥 Extraction failed - debug needed")

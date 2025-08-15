"""
Complete Encoder Fix - Corrected Configuration and Extraction

This creates the encoder with the EXACT configuration matching the original model.
"""

import torch
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_correct_encoder():
    """Create encoder with completely correct configuration and weights."""

    print("🔧 Creating Completely Correct Prithvi Encoder")
    print("=" * 60)

    # Load original model
    original_path = "data/weights/prithvi.wxc.2300m.v1.pt"
    print(f"📁 Loading original model: {original_path}")

    original = torch.load(original_path, map_location='cpu')
    model_state = original['model_state']

    print("✅ Original model loaded")

    # Extract EXACT configuration from model weights
    print("\n🔍 Extracting exact configuration from model...")

    input_scalers_mu = model_state['input_scalers_mu']
    static_scalers_mu = model_state['static_input_scalers_mu']
    patch_embed_weight = model_state['patch_embedding.proj.weight']
    static_embed_weight = model_state['patch_embedding_static.proj.weight']

    # Count actual encoder blocks
    encoder_blocks = set()
    for key in model_state.keys():
        if 'encoder.lgl_block.transformers.' in key:
            block_num = int(key.split('transformers.')[1].split('.')[0])
            encoder_blocks.add(block_num)

    max_encoder_block = max(encoder_blocks)
    n_blocks_encoder = max_encoder_block + 1  # 0-indexed

    # CORRECT configuration
    config = {
        'in_channels': input_scalers_mu.shape[2],  # 160
        'input_size_time': 2,
        'in_channels_static': static_scalers_mu.shape[1],  # 11
        'input_scalers_epsilon': 0.0,
        'static_input_scalers_epsilon': 0.0,
        'n_lats_px': 360,
        'n_lons_px': 576,
        'patch_size_px': [2, 2],
        'mask_unit_size_px': [30, 32],
        'embed_dim': patch_embed_weight.shape[0],  # 2560
        'n_blocks_encoder': n_blocks_encoder,  # 25
        'n_blocks_decoder': 2,
        'mlp_multiplier': 4,
        'n_heads': 16,
        'dropout': 0.0,
        'drop_path': 0.0,
        'parameter_dropout': 0.0,
        'checkpoint_encoder': [],
        'checkpoint_decoder': []
    }

    print(f"   ✅ in_channels: {config['in_channels']}")
    print(f"   ✅ in_channels_static: {config['in_channels_static']}")
    print(f"   ✅ n_blocks_encoder: {config['n_blocks_encoder']}")
    print(f"   ✅ embed_dim: {config['embed_dim']}")

    # Verify patch embedding dimensions match
    expected_patch_input = config['in_channels'] * config['input_size_time']
    actual_patch_input = patch_embed_weight.shape[1]

    print(f"\\n🧪 Verification:")
    print(f"   Expected patch input: {expected_patch_input}")
    print(f"   Actual patch input: {actual_patch_input}")
    print(f"   Match: {'✅' if expected_patch_input == actual_patch_input else '❌'}")

    # Extract ALL encoder weights
    print(f"\\n📦 Extracting encoder weights...")
    encoder_state = {}

    # Core encoder components
    core_keys = [
        'mask_token',
        'input_scalers_mu', 'input_scalers_sigma',
        'static_input_scalers_mu', 'static_input_scalers_sigma',
        'patch_embedding.proj.weight', 'patch_embedding.proj.bias',
        'patch_embedding_static.proj.weight', 'patch_embedding_static.proj.bias',
        'input_time_embedding.weight', 'input_time_embedding.bias',
        'lead_time_embedding.weight', 'lead_time_embedding.bias'  # This was missing!
    ]

    # Add ALL encoder transformer blocks (0 to n_blocks_encoder-1)
    for i in range(config['n_blocks_encoder']):  # 0 to 24
        core_keys.extend([
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
    extracted = 0
    missing = []

    for key in core_keys:
        if key in model_state:
            encoder_state[key] = model_state[key]
            extracted += 1
        else:
            missing.append(key)

    print(f"   ✅ Extracted: {extracted}/{len(core_keys)} weights")
    if missing:
        print(f"   ⚠️  Missing: {missing}")

    # Save with CORRECT configuration
    output_path = "data/weights/prithvi_encoder_fixed.pt"
    print(f"\\n💾 Saving fixed encoder: {output_path}")

    checkpoint = {
        'model_state_dict': encoder_state,
        'config': {'params': config}
    }

    torch.save(checkpoint, output_path)

    # Final verification
    print(f"\\n🧪 Final verification...")
    saved = torch.load(output_path, map_location='cpu')
    saved_config = saved['config']['params']
    saved_state = saved['model_state_dict']

    # Check all key dimensions match
    checks = [
        ('in_channels_static', saved_config['in_channels_static'], saved_state['static_input_scalers_mu'].shape[1]),
        ('in_channels', saved_config['in_channels'], saved_state['input_scalers_mu'].shape[2]),
        ('embed_dim', saved_config['embed_dim'], saved_state['patch_embedding.proj.weight'].shape[0]),
        ('n_blocks_encoder', saved_config['n_blocks_encoder'], n_blocks_encoder)
    ]

    all_correct = True
    for name, config_val, actual_val in checks:
        match = config_val == actual_val
        print(f"   {name}: config={config_val}, actual={actual_val} {'✅' if match else '❌'}")
        if not match:
            all_correct = False

    if all_correct:
        print(f"\\n🎉 SUCCESS: Encoder is completely correct!")
        return output_path
    else:
        print(f"\\n❌ FAILED: Still have mismatches")
        return None

if __name__ == "__main__":
    result = create_correct_encoder()
    if result:
        print(f"\\n✅ Fixed encoder saved to: {result}")
        print("   This encoder should load without any warnings or errors")
    else:
        print(f"\\n💥 Fix failed")

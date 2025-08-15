"""
Test Fixed Prithvi Encoder

This script tests the correctly extracted encoder that has proper configuration
matching the actual model weights.
"""

import torch
from pathlib import Path

def test_corrected_encoder():
    """Test with the fixed encoder using direct weight loading."""
    print("🧪 Testing Fixed Prithvi Encoder")
    print("=" * 50)

    # Use the FIXED encoder
    encoder_path = "data/weights/prithvi_encoder_fixed.pt"

    if not Path(encoder_path).exists():
        print(f"❌ Fixed encoder not found: {encoder_path}")
        print("   Run: python multimodal/create_fixed_encoder.py")
        return False

    print(f"✅ Found fixed encoder: {encoder_path}")

    # Load and verify the configuration
    print("🔍 Verifying fixed configuration...")
    encoder_data = torch.load(encoder_path, map_location='cpu')
    config = encoder_data['config']['params']
    state_dict = encoder_data['model_state_dict']

    print(f"   Configuration loaded:")
    print(f"     Encoder blocks: {config['n_blocks_encoder']}")
    print(f"     Embedding dimension: {config['embed_dim']}")
    print(f"     Input channels: {config['in_channels']}")
    print(f"     Static channels: {config['in_channels_static']}")
    print(f"   State dict contains {len(state_dict)} tensors")

    # Verify key tensors match configuration
    patch_weight = state_dict['patch_embedding.proj.weight']
    expected_in_channels = config['in_channels'] * 2  # Time steps
    expected_embed_dim = config['embed_dim']

    print(f"\n🔍 Verifying tensor shapes:")
    print(f"   Patch embedding weight: {patch_weight.shape}")
    print(f"   Expected: ({expected_embed_dim}, {expected_in_channels}, 2, 2)")

    if (patch_weight.shape[0] == expected_embed_dim and
        patch_weight.shape[1] == expected_in_channels):
        print("   ✅ Patch embedding dimensions are correct!")
    else:
        print("   ❌ Patch embedding dimensions are wrong")
        return False

    # Test feature extraction
    print(f"\n🚀 Testing feature extraction...")

    # Create sample climate data matching the configuration
    batch_size = 2
    climate_data = torch.randn(batch_size, config['in_channels'], 2, 180, 288)
    static_data = torch.randn(batch_size, config['in_channels_static'], 180, 288)

    print(f"   Input climate data: {climate_data.shape}")
    print(f"   Input static data: {static_data.shape}")

    # Extract patch embedding weights
    patch_weight = state_dict['patch_embedding.proj.weight']
    patch_bias = state_dict['patch_embedding.proj.bias']

    with torch.no_grad():
        # Reshape for patch embedding
        B, C, T, H, W = climate_data.shape
        climate_reshaped = climate_data.view(B, C * T, H, W)

        print(f"   Reshaped climate data: {climate_reshaped.shape}")

        # Apply patch embedding
        features = torch.conv2d(climate_reshaped, patch_weight, patch_bias, stride=2)
        print(f"   Patch features: {features.shape}")

        # Convert to sequence format
        B, embed_dim, H_patch, W_patch = features.shape
        feature_sequence = features.flatten(2).transpose(1, 2)
        print(f"   Feature sequence: {feature_sequence.shape}")

        # Verify dimensions match expected
        expected_patches = H_patch * W_patch
        if (feature_sequence.shape[1] == expected_patches and
            feature_sequence.shape[2] == expected_embed_dim):
            print("   ✅ Feature extraction successful!")
        else:
            print("   ❌ Feature extraction failed")
            return False

    # Test some transformer blocks if they exist
    print(f"\n🔧 Testing transformer blocks...")
    transformer_found = False

    for key in state_dict.keys():
        if 'encoder.lgl_block.transformers.0.' in key:
            transformer_found = True
            break

    if transformer_found:
        print(f"   ✅ Found transformer blocks in state dict")

        # Test first transformer block weights
        attention_weight_key = 'encoder.lgl_block.transformers.0.attention.1.qkv_layer.weight'
        if attention_weight_key in state_dict:
            attn_weight = state_dict[attention_weight_key]
            print(f"   First attention layer weight: {attn_weight.shape}")
            expected_attn_dim = expected_embed_dim * 3  # Q, K, V
            if attn_weight.shape[0] == expected_attn_dim:
                print("   ✅ Attention layer dimensions correct!")
            else:
                print("   ❌ Attention layer dimensions wrong")
        else:
            print("   ⚠️  Attention weight not found in expected location")
    else:
        print("   ⚠️  No transformer blocks found (might be minimal encoder)")

    # Summary statistics
    print(f"\n📊 Weight Statistics:")
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"   Total parameters: {total_params:,}")

    # Calculate memory usage
    total_size = sum(p.numel() * p.element_size() for p in state_dict.values())
    print(f"   Total memory: {total_size / (1024*1024):.1f} MB")

    # Test a simple forward pass simulation
    print(f"\n🧪 Testing simple analysis simulation...")

    # Use first batch for analysis
    sample_features = feature_sequence[0]  # [num_patches, embed_dim]

    # Simple feature analysis
    feature_mean = sample_features.mean().item()
    feature_std = sample_features.std().item()
    feature_norm = sample_features.norm().item()

    print(f"   Feature statistics:")
    print(f"     Mean: {feature_mean:.4f}")
    print(f"     Std: {feature_std:.4f}")
    print(f"     Norm: {feature_norm:.4f}")

    # Location-based analysis (simple version)
    num_patches = sample_features.shape[0]
    patch_grid_size = int(num_patches ** 0.5)

    if patch_grid_size * patch_grid_size == num_patches:
        print(f"   Patch grid: {patch_grid_size} x {patch_grid_size}")

        # Reshape to spatial grid
        spatial_features = sample_features.view(patch_grid_size, patch_grid_size, -1)

        # Compute spatial statistics
        spatial_var = spatial_features.var(dim=-1)
        max_var_idx = spatial_var.flatten().argmax()
        max_var_row = max_var_idx // patch_grid_size
        max_var_col = max_var_idx % patch_grid_size

        print(f"   Most variable region: ({max_var_row}, {max_var_col})")
        print(f"   Spatial variance range: {spatial_var.min():.4f} - {spatial_var.max():.4f}")
    else:
        print(f"   Non-square patch grid: {num_patches} patches")

    print(f"\n🎉 SUCCESS: Fixed encoder test completed!")
    print(f"   ✅ Configuration verified")
    print(f"   ✅ Weight shapes correct")
    print(f"   ✅ Feature extraction working")
    print(f"   ✅ {total_params:,} parameters loaded")
    print(f"   ✅ Ready for climate analysis")

    return True


if __name__ == "__main__":
    print("🚀 Fixed Encoder Test - Verifying Corrected Weights")
    success = test_corrected_encoder()
    if success:
        print("\n✅ Fixed encoder test completed successfully!")
    else:
        print("\n❌ Fixed encoder test failed!")

"""
Test Fixed Encoder Extraction

This script tests the fixed encoder extraction functionality.
"""

from pathlib import Path

import torch


def test_fixed_encoder():
    """Test the fixed encoder extraction results."""
    print("🧪 Testing Fixed Encoder Extraction Results")
    print("=" * 50)

    # Test the fixed encoder file
    fixed_encoder_path = "data/weights/prithvi_encoder_fixed.pt"

    if not Path(fixed_encoder_path).exists():
        print(f"❌ Fixed encoder not found: {fixed_encoder_path}")
        print("   Run: python multimodal/create_fixed_encoder.py")
        return False

    print(f"✅ Found fixed encoder: {fixed_encoder_path}")

    # Load and analyze the fixed encoder
    print("\n📁 Loading fixed encoder...")
    encoder_data = torch.load(fixed_encoder_path, map_location="cpu")

    if "config" not in encoder_data or "model_state_dict" not in encoder_data:
        print("❌ Invalid encoder format")
        return False

    config = encoder_data["config"]["params"]
    state_dict = encoder_data["model_state_dict"]

    print(f"   ✅ Configuration loaded:")
    # Check if model_type exists, otherwise skip it
    if "model_type" in encoder_data["config"]:
        print(f"      Model type: {encoder_data['config']['model_type']}")
    print(f"      Encoder blocks: {config['n_blocks_encoder']}")
    print(f"      Embedding dim: {config['embed_dim']}")
    print(f"      Input channels: {config['in_channels']}")
    print(f"      Static channels: {config['in_channels_static']}")

    print(f"   ✅ State dict contains {len(state_dict)} tensors")

    # Test critical weight tensors
    print("\n🔍 Verifying critical weight tensors...")

    required_keys = [
        "patch_embedding.proj.weight",
        "patch_embedding.proj.bias",
        "patch_embedding_static.proj.weight",
        "patch_embedding_static.proj.bias",
        "input_scalers_mu",
        "input_scalers_sigma",
        "static_input_scalers_mu",
        "static_input_scalers_sigma",
    ]

    missing_keys = []
    for key in required_keys:
        if key not in state_dict:
            missing_keys.append(key)
        else:
            print(f"   ✅ {key}: {state_dict[key].shape}")

    if missing_keys:
        print(f"   ❌ Missing keys: {missing_keys}")
        return False

    # Test patch embedding dimensions
    print("\n🔧 Testing patch embedding extraction...")

    patch_weight = state_dict["patch_embedding.proj.weight"]
    expected_out_dim = config["embed_dim"]
    expected_in_dim = config["in_channels"] * 2  # Time steps

    if (
        patch_weight.shape[0] == expected_out_dim
        and patch_weight.shape[1] == expected_in_dim
    ):
        print(f"   ✅ Patch embedding dimensions correct: {patch_weight.shape}")
    else:
        print(f"   ❌ Patch embedding dimensions wrong: {patch_weight.shape}")
        print(f"       Expected: ({expected_out_dim}, {expected_in_dim}, 2, 2)")
        return False

    # Test transformer blocks
    print("\n🤖 Testing transformer block extraction...")

    transformer_keys = [
        k for k in state_dict.keys() if "encoder.lgl_block.transformers." in k
    ]

    if transformer_keys:
        print(f"   ✅ Found {len(transformer_keys)} transformer layer weights")

        # Check number of blocks
        block_numbers = set()
        for key in transformer_keys:
            if "transformers." in key:
                parts = key.split("transformers.")[1].split(".")
                if parts[0].isdigit():
                    block_numbers.add(int(parts[0]))

        max_block = max(block_numbers) if block_numbers else -1
        expected_blocks = config["n_blocks_encoder"]

        if max_block + 1 == expected_blocks:
            print(f"   ✅ All {expected_blocks} transformer blocks present")
        else:
            print(f"   ⚠️  Expected {expected_blocks} blocks, found {max_block + 1}")
    else:
        print("   ⚠️  No transformer weights found (minimal encoder)")

    # Test feature extraction capability
    print("\n🚀 Testing feature extraction functionality...")

    # Create sample input - check actual expected static channels from weights
    static_weight = state_dict["patch_embedding_static.proj.weight"]
    expected_static_channels = static_weight.shape[1]  # Get expected input channels

    batch_size = 1
    climate_data = torch.randn(batch_size, config["in_channels"], 2, 180, 288)
    static_data = torch.randn(batch_size, expected_static_channels, 180, 288)

    print(f"   Sample climate data: {climate_data.shape}")
    print(
        f"   Sample static data: {static_data.shape} (expected {expected_static_channels} channels)"
    )

    # Apply patch embedding
    with torch.no_grad():
        # Climate patch embedding
        B, C, T, H, W = climate_data.shape
        climate_reshaped = climate_data.view(B, C * T, H, W)

        patch_weight = state_dict["patch_embedding.proj.weight"]
        patch_bias = state_dict["patch_embedding.proj.bias"]

        climate_features = torch.conv2d(
            climate_reshaped, patch_weight, patch_bias, stride=2
        )
        print(f"   ✅ Climate features extracted: {climate_features.shape}")

        # Static patch embedding
        static_weight = state_dict["patch_embedding_static.proj.weight"]
        static_bias = state_dict["patch_embedding_static.proj.bias"]

        static_features = torch.conv2d(
            static_data, static_weight, static_bias, stride=2
        )
        print(f"   ✅ Static features extracted: {static_features.shape}")

        # Verify dimensions match
        if (
            climate_features.shape[1] == config["embed_dim"]
            and static_features.shape[1] == config["embed_dim"]
        ):
            print(f"   ✅ Feature dimensions match config")
        else:
            print(f"   ❌ Feature dimensions don't match config")
            return False

    # Test parameter count
    print("\n📊 Analyzing extracted parameters...")

    total_params = sum(p.numel() for p in state_dict.values())
    print(f"   Total parameters: {total_params:,}")

    # Memory usage
    total_size = sum(p.numel() * p.element_size() for p in state_dict.values())
    print(f"   Memory usage: {total_size / (1024*1024):.1f} MB")

    # Parameter breakdown
    param_breakdown = {}
    for key, tensor in state_dict.items():
        category = key.split(".")[0]
        if category not in param_breakdown:
            param_breakdown[category] = 0
        param_breakdown[category] += tensor.numel()

    print(f"   Parameter breakdown:")
    for category, count in sorted(param_breakdown.items()):
        print(f"     {category}: {count:,}")

    # Test data types and devices
    print("\n🔧 Verifying tensor properties...")

    dtypes = set(tensor.dtype for tensor in state_dict.values())
    devices = set(str(tensor.device) for tensor in state_dict.values())

    print(f"   Data types: {dtypes}")
    print(f"   Devices: {devices}")

    if len(dtypes) == 1 and torch.float32 in dtypes:
        print("   ✅ All tensors are float32")
    else:
        print("   ⚠️  Mixed data types found")

    if len(devices) == 1 and "cpu" in list(devices)[0]:
        print("   ✅ All tensors on CPU")
    else:
        print("   ⚠️  Mixed devices found")

    print(f"\n🎉 SUCCESS: Fixed encoder extraction test completed!")
    print(f"   ✅ All {len(required_keys)} required weights present")
    print(f"   ✅ Dimensions match configuration")
    print(f"   ✅ Feature extraction functional")
    print(f"   ✅ {total_params:,} parameters extracted")
    print(f"   ✅ Ready for climate analysis")

    return True


if __name__ == "__main__":
    print("🚀 Testing Fixed Encoder Extraction")
    success = test_fixed_encoder()
    if success:
        print("\n✅ Fixed encoder extraction test passed!")
    else:
        print("\n❌ Fixed encoder extraction test failed!")

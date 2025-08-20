#!/usr/bin/env python3
"""
Test AIFS Encoder Capability for 5-D Time Series Tensors

This script investigates whether the AIFS encoder can process 5-D tensors
representing time series climate data with dimensions:
[batch, time, variables, height, width]

Author: GitHub Copilot
Date: August 20, 2025
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_5d_tensor_tokenization():
    """Test AIFS encoder capability for 5-D time series tensors."""

    print("🧪 Testing AIFS Encoder for 5-D Time Series Tokenization")
    print("=" * 60)

    # Test tensor shapes for time series climate data
    test_shapes = [
        # [batch, time, vars, height, width]
        (2, 4, 3, 32, 32),  # Small test case
        (1, 8, 5, 64, 64),  # Medium time series
        (4, 12, 7, 128, 128),  # Larger spatial resolution
        (1, 24, 10, 256, 256),  # Very high resolution
    ]

    print("Test tensor shapes (batch, time, vars, height, width):")
    for i, shape in enumerate(test_shapes):
        print(f"  {i+1}. {shape}")
    print()

    # Try to load AIFS encoder
    try:
        from multimodal_aifs.utils.aifs_encoder_utils import AIFSEncoderWrapper

        # Initialize with dummy encoder for testing
        encoder = AIFSEncoderWrapper(encoder_path=None, device="cpu")
        encoder_info = encoder.get_encoder_info()

        print("✅ AIFS Encoder Loaded Successfully")
        print(f"   Type: {encoder_info['encoder_type']}")
        print(f"   Input dim: {encoder_info['input_dim']}")
        print(f"   Output dim: {encoder_info['output_dim']}")
        print(f"   Parameters: {encoder_info['parameters']:,}")
        print()

    except Exception as e:
        print(f"❌ Failed to load AIFS encoder: {e}")
        return

    # Test each tensor shape
    for i, shape in enumerate(test_shapes):
        print(f"🔍 Test {i+1}: Shape {shape}")

        # Create random 5-D tensor
        tensor_5d = torch.randn(*shape)
        print(f"   Created 5-D tensor: {tensor_5d.shape}")

        try:
            # Test direct encoding (this will likely fail)
            print("   Attempting direct 5-D encoding...")
            try:
                encoded_direct = encoder.encode_climate_data(tensor_5d)
                print(
                    f"   ✅ Direct encoding succeeded: {tensor_5d.shape} -> {encoded_direct.shape}"
                )
            except Exception as e:
                print(f"   ❌ Direct encoding failed: {e}")

                # Test reshape strategies
                print("   Trying reshape strategies...")

                # Strategy 1: Flatten time and variables
                batch, time, vars, h, w = shape

                # 1a. Reshape to (batch, time*vars, h, w) - treat as multi-channel
                reshaped_1 = tensor_5d.view(batch, time * vars, h, w)
                print(f"      Strategy 1a: {tensor_5d.shape} -> {reshaped_1.shape}")
                try:
                    encoded_1a = encoder.encode_climate_data(reshaped_1)
                    print(f"      ✅ Strategy 1a succeeded: -> {encoded_1a.shape}")
                except Exception as e:
                    print(f"      ❌ Strategy 1a failed: {e}")

                # 1b. Reshape to (batch*time, vars, h, w) - treat time as batch
                reshaped_1b = tensor_5d.view(batch * time, vars, h, w)
                print(f"      Strategy 1b: {tensor_5d.shape} -> {reshaped_1b.shape}")
                try:
                    encoded_1b = encoder.encode_climate_data(reshaped_1b)
                    # Reshape back to include time dimension
                    encoded_1b = encoded_1b.view(batch, time, -1)
                    print(f"      ✅ Strategy 1b succeeded: -> {encoded_1b.shape}")
                except Exception as e:
                    print(f"      ❌ Strategy 1b failed: {e}")

                # 1c. Flatten spatial dimensions first
                reshaped_1c = tensor_5d.view(batch, time, vars * h * w)
                print(f"      Strategy 1c: {tensor_5d.shape} -> {reshaped_1c.shape}")
                try:
                    encoded_1c = encoder.encode_climate_data(reshaped_1c)
                    print(f"      ✅ Strategy 1c succeeded: -> {encoded_1c.shape}")
                except Exception as e:
                    print(f"      ❌ Strategy 1c failed: {e}")

                # 1d. Complete flattening to 2D
                reshaped_1d = tensor_5d.view(batch, -1)
                print(f"      Strategy 1d: {tensor_5d.shape} -> {reshaped_1d.shape}")
                try:
                    encoded_1d = encoder.encode_climate_data(reshaped_1d)
                    print(f"      ✅ Strategy 1d succeeded: -> {encoded_1d.shape}")
                except Exception as e:
                    print(f"      ❌ Strategy 1d failed: {e}")

        except Exception as e:
            print(f"   ❌ Test failed completely: {e}")

        print()

    # Test sequential processing for time series
    print("🔄 Testing Sequential Time Series Processing")
    print("-" * 40)

    # Use a moderate size for sequential testing
    batch, time, vars, h, w = 2, 8, 5, 64, 64
    tensor_5d = torch.randn(batch, time, vars, h, w)

    print(f"Sequential processing test: {tensor_5d.shape}")

    try:
        # Process each timestep separately
        timestep_encodings = []

        for t in range(time):
            timestep_data = tensor_5d[:, t, :, :, :]  # Shape: (batch, vars, h, w)
            print(f"   Processing timestep {t}: {timestep_data.shape}")

            try:
                encoded_timestep = encoder.encode_climate_data(timestep_data)
                timestep_encodings.append(encoded_timestep)
                print(f"   ✅ Timestep {t} encoded: -> {encoded_timestep.shape}")
            except Exception as e:
                print(f"   ❌ Timestep {t} failed: {e}")
                break

        if timestep_encodings:
            # Combine timestep encodings
            sequence_encoding = torch.stack(timestep_encodings, dim=1)
            print(
                f"✅ Sequential encoding complete: {tensor_5d.shape} -> {sequence_encoding.shape}"
            )
            print(
                f"   Final shape interpretation: (batch={sequence_encoding.shape[0]}, "
                f"time={sequence_encoding.shape[1]}, features={sequence_encoding.shape[2]})"
            )
        else:
            print("❌ Sequential encoding failed completely")

    except Exception as e:
        print(f"❌ Sequential processing failed: {e}")

    print()
    print("📊 Summary and Recommendations")
    print("=" * 60)
    print("AIFS encoder analysis for 5-D time series tensors:")
    print()
    print("🔍 Input Requirements:")
    print(f"   • AIFS expects {encoder_info['input_dim']} input features")
    print(f"   • Output dimension: {encoder_info['output_dim']}")
    print("   • Designed for weather field data (spatial patterns)")
    print()
    print("🎯 Recommended Approaches for 5-D Time Series:")
    print("   1. Sequential Processing: Process each timestep separately")
    print("   2. Temporal Batching: Reshape time as additional batch dimension")
    print("   3. Feature Engineering: Extract meaningful spatial features first")
    print("   4. Hybrid Approach: Combine AIFS spatial encoding with temporal modeling")
    print()
    print("⚡ Best Strategy for Time Series Tokenization:")
    print("   • Use AIFS encoder for spatial feature extraction per timestep")
    print("   • Apply separate temporal model (e.g., LSTM, Transformer) for sequence modeling")
    print("   • This preserves AIFS spatial understanding while adding temporal capabilities")


if __name__ == "__main__":
    test_5d_tensor_tokenization()

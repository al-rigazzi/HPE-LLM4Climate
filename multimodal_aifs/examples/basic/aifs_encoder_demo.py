#!/usr/bin/env python3
"""
AIFS Encoder Demonstration

This example demonstrates basic usage of the AIFS encoder wrapper
for climate data encoding and analysis.

Features demonstrated:
- AIFS encoder initialization
- Climate data preprocessing
- Batch encoding
- Performance measurement
- Output analysis

Usage:
    python multimodal_aifs/examples/basic/aifs_encoder_demo.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from multimodal_aifs.utils import AIFSEncoderWrapper, create_synthetic_climate_data


def demo_aifs_encoder():
    """Demonstrate AIFS encoder functionality."""
    print("🌍 AIFS Encoder Demonstration")
    print("=" * 50)

    # Check for AIFS model
    aifs_model_path = (
        project_root / "multimodal_aifs" / "models" / "extracted_models" / "aifs_encoder_full.pth"
    )
    has_real_model = aifs_model_path.exists()

    print(f"📍 Project root: {project_root}")
    print(f"🤖 AIFS model path: {aifs_model_path}")
    print(f"✅ Real model available: {has_real_model}")
    print()

    # Initialize encoder
    print("🔧 Initializing AIFS Encoder...")

    if has_real_model:
        encoder = AIFSEncoderWrapper(encoder_path=str(aifs_model_path), device="cpu")
        print("   ✅ Real AIFS model loaded")
    else:
        encoder = AIFSEncoderWrapper(encoder_path=None, device="cpu")
        print("   ⚠️  No real model - using demonstration mode")

    # Get encoder information
    if encoder.encoder is not None:
        print(f"   📊 Model parameters: {sum(p.numel() for p in encoder.parameters()):,}")
        print(f"   📐 Input dimension: {encoder.input_dim}")
        print(f"   📐 Output dimension: {encoder.output_dim}")
        print(f"   💻 Device: {encoder.device}")
        if encoder.encoder_info:
            print(f"   🔧 Model type: {encoder.encoder_info.get('type', 'unknown')}")
    else:
        print("   ❌ No encoder loaded")
    print()

    # Create synthetic climate data
    print("🌡️ Creating Synthetic Climate Data...")

    # Method 1: Simple random data
    batch_size = 8
    input_features = 218  # AIFS expected input size

    simple_data = torch.randn(batch_size, input_features)
    print(f"   📊 Simple data shape: {simple_data.shape}")
    print(f"   📈 Data range: [{simple_data.min():.3f}, {simple_data.max():.3f}]")

    # Method 2: Realistic synthetic climate data
    realistic_data = create_synthetic_climate_data(
        batch_size=batch_size, n_variables=25, spatial_shape=(64, 128), add_noise=True
    )

    # Flatten for AIFS input
    flattened_data = realistic_data.view(batch_size, -1)

    # Adjust to correct input size
    if flattened_data.shape[1] > input_features:
        flattened_data = flattened_data[:, :input_features]
    elif flattened_data.shape[1] < input_features:
        padding = torch.zeros(batch_size, input_features - flattened_data.shape[1])
        flattened_data = torch.cat([flattened_data, padding], dim=1)

    print(f"   🌍 Realistic data shape: {realistic_data.shape}")
    print(f"   📏 Flattened shape: {flattened_data.shape}")
    print()

    # Test encoding (only if real model available)
    if encoder.encoder is not None:
        print("🔄 Testing Climate Data Encoding...")

        # Test simple data
        start_time = time.time()
        encoded_simple = encoder.encode_climate_data(simple_data)
        simple_time = time.time() - start_time

        print(f"   ⚡ Simple encoding: {simple_data.shape} -> {encoded_simple.shape}")
        print(f"   ⏱️  Encoding time: {simple_time:.4f}s")
        print(f"   🎯 Throughput: {batch_size/simple_time:.1f} samples/s")

        # Test realistic data
        start_time = time.time()
        encoded_realistic = encoder.encode_climate_data(flattened_data)
        realistic_time = time.time() - start_time

        print(f"   🌍 Realistic encoding: {flattened_data.shape} -> {encoded_realistic.shape}")
        print(f"   ⏱️  Encoding time: {realistic_time:.4f}s")

        # Analyze output characteristics
        print(f"   📊 Output statistics:")
        print(f"      Mean: {encoded_realistic.mean():.4f}")
        print(f"      Std:  {encoded_realistic.std():.4f}")
        print(f"      Min:  {encoded_realistic.min():.4f}")
        print(f"      Max:  {encoded_realistic.max():.4f}")
        print()

        # Test batch encoding
        print("📦 Testing Batch Encoding...")

        batch_results = encoder.encode_batch(flattened_data)

        print(f"   📊 Batch results:")
        print(f"      Encoded shape: {batch_results.shape}")
        print(f"      Processing successful: {batch_results.numel() > 0}")
        print()

    else:
        print("⚠️  Skipping encoding tests (no real model available)")
        print("   To see encoding in action, place AIFS model at:")
        print(f"   {aifs_model_path}")
        print()

    # Test preprocessing capabilities
    print("🔄 Testing Data Preprocessing...")

    # Test various input formats
    test_cases = [
        ("Correct size", torch.randn(4, 218)),
        ("Too small", torch.randn(4, 100)),
        ("Too large", torch.randn(4, 500)),
        ("NumPy input", np.random.randn(4, 218).astype(np.float32)),
    ]

    for case_name, test_data in test_cases:
        if isinstance(test_data, np.ndarray):
            tensor_data = torch.tensor(test_data)
        else:
            tensor_data = test_data

        processed = encoder._preprocess_climate_data(tensor_data)

        print(f"   {case_name}: {tuple(test_data.shape)} -> {tuple(processed.shape)}")

    print()

    # Performance benchmarking
    if encoder.encoder is not None:
        print("⚡ Performance Benchmarking...")

        batch_sizes = [1, 4, 8, 16, 32]

        for bs in batch_sizes:
            if bs > 32:  # Skip very large batches for demo
                continue

            data = torch.randn(bs, 218)

            # Warmup
            if bs == batch_sizes[0]:
                _ = encoder.encode_climate_data(data)

            # Benchmark
            start_time = time.time()
            encoded = encoder.encode_climate_data(data)
            end_time = time.time()

            encoding_time = end_time - start_time
            throughput = bs / encoding_time

            print(f"   Batch {bs:2d}: {encoding_time:.4f}s, {throughput:6.1f} samples/s")

        print()

    # Feature analysis
    print("🔍 Feature Analysis...")

    if encoder.encoder is not None:
        # Encode different types of data to see feature differences
        temp_data = torch.ones(1, 218) * 0.5  # Warm temperature pattern
        cold_data = torch.ones(1, 218) * -0.5  # Cold temperature pattern
        random_data = torch.randn(1, 218)  # Random pattern

        temp_features = encoder.encode_climate_data(temp_data)
        cold_features = encoder.encode_climate_data(cold_data)
        random_features = encoder.encode_climate_data(random_data)

        # Compute similarities
        temp_cold_sim = torch.cosine_similarity(temp_features, cold_features)
        temp_random_sim = torch.cosine_similarity(temp_features, random_features)
        cold_random_sim = torch.cosine_similarity(cold_features, random_features)

        print(f"   🌡️  Temperature vs Cold similarity: {temp_cold_sim.item():.4f}")
        print(f"   🎲 Temperature vs Random similarity: {temp_random_sim.item():.4f}")
        print(f"   ❄️  Cold vs Random similarity: {cold_random_sim.item():.4f}")

    else:
        print("   ⚠️  Feature analysis requires real model")

    print()

    # Summary
    print("📋 Demo Summary")
    print("-" * 30)
    print(f"✅ AIFS encoder {'loaded' if encoder.encoder is not None else 'simulated'}")
    print(f"✅ Data preprocessing demonstrated")
    print(f"✅ Input/output shapes validated")
    if encoder.encoder is not None:
        print(f"✅ Real encoding performed")
        print(f"✅ Performance benchmarked")
        print(f"✅ Feature analysis completed")
    else:
        print(f"⚠️  Encoding simulation only")

    print("\\n🎯 Next Steps:")
    print("   - Try the climate-text fusion demo")
    print("   - Explore location-aware examples")
    print("   - Run with real AIFS model for full functionality")


if __name__ == "__main__":
    try:
        demo_aifs_encoder()
    except KeyboardInterrupt:
        print("\\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python3
"""
AIFS Time Series + LLaMA Integration Example

This example demonstrates how to use the AIFSTimeSeriesTokenizer
with LLaMA for climate-language understanding tasks.

Usage:
    python multimodal_aifs/examples/aifs_llama_example.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the fusion wrapper for backward compatibility with the example interface
from multimodal_aifs.conftest import AIFSClimateTextFusionWrapper
from multimodal_aifs.utils.aifs_time_series_tokenizer import AIFSTimeSeriesTokenizer


def create_sample_climate_data():
    """Create sample climate time series data."""
    print("Creating sample climate data...")

    # Simulate 5-day weather data for 2 locations
    batch_size = 2  # 2 locations
    time_steps = 5  # 5 days
    variables = 4  # temperature, humidity, pressure, wind

    # AIFS expects specific grid dimensions - use a compatible size
    # For demo purposes, we'll create data that can be reshaped to AIFS format
    # AIFS grid is 542080 points, we'll use a smaller demo size
    height, width = 32, 32  # 1024 total points (much smaller than AIFS 542080)

    climate_data = torch.zeros(batch_size, time_steps, variables, height, width)

    for location in range(batch_size):
        for day in range(time_steps):
            # Temperature (variable 0): seasonal pattern + spatial variation
            base_temp = 20 + 5 * np.sin(2 * np.pi * day / 365)  # seasonal
            lat_gradient = torch.linspace(-10, 10, height).unsqueeze(1)
            lon_gradient = torch.linspace(-5, 5, width).unsqueeze(0)
            climate_data[location, day, 0] = (
                base_temp + lat_gradient + lon_gradient + torch.randn(height, width) * 2
            )

            # Humidity (variable 1): correlated with temperature
            climate_data[location, day, 1] = (
                60 - 0.5 * climate_data[location, day, 0] + torch.randn(height, width) * 5
            )

            # Pressure (variable 2): more stable
            climate_data[location, day, 2] = 1013 + torch.randn(height, width) * 8

            # Wind speed (variable 3): random with some spatial correlation
            climate_data[location, day, 3] = 5 + torch.randn(height, width) * 3

    print(f"   Climate data shape: {climate_data.shape}")
    print(
        f"   Temperature range: {climate_data[:, :, 0].min():.1f}°C "
        "to {climate_data[:, :, 0].max():.1f}°C"
    )
    print(
        f"   Humidity range: {climate_data[:, :, 1].min():.1f}% "
        "to {climate_data[:, :, 1].max():.1f}%"
    )
    print("   Note: Using demo grid size (1024) instead of full AIFS grid (542080)")

    return climate_data


def create_climate_analysis_prompts():
    """Create climate analysis prompts for LLaMA."""
    prompts = [
        "Analyze the temperature patterns in this 5-day climate"
        " data and provide insights about potential weather trends.",
        "Based on the humidity and pressure data, what can you"
        " tell me about the atmospheric conditions in this region?",
    ]

    print(f"Created {len(prompts)} climate analysis prompts")
    return prompts


def demonstrate_aifs_llama_fusion():
    """Demonstrate AIFS-LLaMA fusion for climate analysis."""
    print("\nAIFS-LLaMA Climate Analysis Demo")
    print("=" * 60)

    # Create sample data
    climate_data = create_sample_climate_data()
    text_prompts = create_climate_analysis_prompts()

    print("\nInitializing AIFS-LLaMA Fusion Model...")

    # First, try to load AIFS model
    print("   Loading AIFS model...")
    aifs_model = None
    try:
        # Setup flash attention mocking (MacOS only)
        import platform
        import types

        if platform.system() == "Darwin":
            flash_attn_mock = types.ModuleType("flash_attn")
            flash_attn_interface_mock = types.ModuleType("flash_attn_interface")
            flash_attn_interface_mock.flash_attn_func = lambda *args, **kwargs: None
            flash_attn_interface_mock.flash_attn_varlen_func = lambda *args, **kwargs: None
            flash_attn_mock.flash_attn_interface = flash_attn_interface_mock

            sys.modules["flash_attn"] = flash_attn_mock
            sys.modules["flash_attn.flash_attn_interface"] = flash_attn_interface_mock
            os.environ["USE_FLASH_ATTENTION"] = "false"
            print("   Flash attention mock enabled for MacOS")
        else:
            print("   Using real flash attention (non-MacOS system)")

        from anemoi.inference.runners.simple import SimpleRunner

        checkpoint = {"huggingface": "ecmwf/aifs-single-1.0"}
        runner = SimpleRunner(checkpoint, device="cpu")
        aifs_model = runner.model
        print("   Real AIFS model loaded")
    except Exception as e:
        print(f"   AIFS model not available ({e}), using mock")
        aifs_model = None

    # Initialize fusion model using our production wrapper
    model = AIFSClimateTextFusionWrapper(
        model=aifs_model,
        device_str="cpu",
        fusion_dim=512,
        use_mock_llama=True,  # Use mock LLaMA for demo
        verbose=True,
    )

    print("Model initialized successfully!")

    # Test different tasks
    tasks = [
        ("embedding", "Multimodal Embedding Extraction"),
        ("generation", "Climate-Conditioned Text Generation"),
        ("classification", "Climate Pattern Classification"),
    ]

    print("\nTesting Different Tasks:")
    print("-" * 40)

    for task_name, task_description in tasks:
        print(f"\n{task_description}")

        try:
            # Process with real model when available
            outputs = model(climate_data=climate_data, text_inputs=text_prompts, task=task_name)

            # Display results
            if task_name == "embedding":
                embeddings = outputs["embeddings"]
                print(f"   Generated embeddings: {embeddings.shape}")
                print(f"   Embedding dimension: {embeddings.shape[-1]}")
                print(
                    f"   Embedding stats: mean={embeddings.mean():.4f}, "
                    f"std={embeddings.std():.4f}"
                )

            elif task_name == "generation":
                logits = outputs["logits"]
                print(f"   Generated logits: {logits.shape}")
                print(f"   Vocabulary size: {logits.shape[-1]}")
                print(
                    "   Sample next token probabilities: "
                    f"{torch.softmax(logits[0, 0, :10], dim=0)}"
                )

            elif task_name == "classification":
                class_logits = outputs["classification_logits"]
                predictions = torch.softmax(class_logits, dim=1)
                print(f"   Classification logits: {class_logits.shape}")
                print(f"   Predicted classes: {torch.argmax(predictions, dim=1)}")
                print(f"   Confidence scores: {torch.max(predictions, dim=1)[0]}")

        except Exception as e:
            print(f"   Task failed: {e}")
            raise RuntimeError(f"Real model processing failed for task '{task_name}': {e}") from e

    print("\nModel Architecture Analysis:")
    print("-" * 40)

    # Analyze the model components
    print("AIFS Time Series Tokenizer:")
    print(f"   Temporal modeling: {model.time_series_tokenizer.temporal_modeling}")
    print(f"   Hidden dimension: {model.time_series_tokenizer.hidden_dim}")
    print(f"   Spatial dimension: {model.time_series_tokenizer.spatial_dim}")

    print("\nLLaMA Integration:")
    print(f"   Hidden size: {model.llama_hidden_size}")
    print(f"   Fusion strategy: {model.fusion_strategy}")
    print(f"   Device: {model.device}")

    print("\nIntegration Benefits:")
    print("   Rich climate representation via AIFS spatial encoding")
    print("   Temporal dynamics captured by transformer")
    print("   Natural language understanding via LLaMA")
    print("   Cross-modal attention for climate-text fusion")
    print("   End-to-end trainable for climate-language tasks")


def demonstrate_compression_analysis(aifs_model=None):
    """Demonstrate compression analysis of AIFS tokenization."""
    print("\nAIFS Tokenization Compression Analysis")
    print("=" * 60)

    # Test different data sizes
    test_configs = [
        ("Small Regional", 1, 3, 3, (16, 16)),
        ("Medium Regional", 2, 7, 5, (32, 32)),
        ("Large Continental", 1, 14, 7, (64, 64)),
        ("Extended Forecast", 1, 30, 5, (32, 32)),
    ]

    if aifs_model is None:
        print("No AIFS model available for compression analysis")
        print("Skipping compression analysis - requires AIFS model")
        return

    try:
        tokenizer = AIFSTimeSeriesTokenizer(
            aifs_model=aifs_model, temporal_modeling="transformer", device="cpu"
        )
    except Exception as e:
        print(f"Could not create tokenizer: {e}")
        print("Skipping compression analysis")
        return

    print("\nCompression Analysis:")
    print("-" * 50)

    for config_name, batch, time, variables, (h, w) in test_configs:
        # Create test data
        data = torch.randn(batch, time, variables, h, w)

        # Tokenize
        try:
            tokens = tokenizer.tokenize_time_series(data)

            # Calculate metrics
            input_size = data.numel() * 4  # float32 bytes
            output_size = tokens.numel() * 4  # float32 bytes
            compression_ratio = input_size / output_size

            input_mb = input_size / (1024 * 1024)
            output_mb = output_size / (1024 * 1024)

            print(f"{config_name}:")
            print(f"   Input:  {data.shape} ({input_mb:.2f} MB)")
            print(f"   Output: {tokens.shape} ({output_mb:.2f} MB)")
            print(f"   Compression: {compression_ratio:.1f}x")
            print()
        except Exception as e:
            print(f"{config_name}: Failed - {e}")


def main():
    """Main demonstration function."""
    print("AIFS Time Series + LLaMA Integration Demo")
    print("=" * 60)
    print("This demo shows how to integrate AIFS spatial-temporal")
    print("climate tokenization with LLaMA language models for")
    print(" climate-language understanding tasks.")
    print()

    # Load AIFS model for use in multiple functions
    aifs_model = None
    try:
        # Setup flash attention mocking (MacOS only)
        import platform
        import types

        if platform.system() == "Darwin":
            flash_attn_mock = types.ModuleType("flash_attn")
            flash_attn_interface_mock = types.ModuleType("flash_attn_interface")
            flash_attn_interface_mock.flash_attn_func = lambda *args, **kwargs: None
            flash_attn_interface_mock.flash_attn_varlen_func = lambda *args, **kwargs: None
            flash_attn_mock.flash_attn_interface = flash_attn_interface_mock

            sys.modules["flash_attn"] = flash_attn_mock
            sys.modules["flash_attn.flash_attn_interface"] = flash_attn_interface_mock
            os.environ["USE_FLASH_ATTENTION"] = "false"
            print("Flash attention mock enabled for MacOS")
        else:
            print("Using real flash attention (non-MacOS system)")

        from anemoi.inference.runners.simple import SimpleRunner

        checkpoint = {"huggingface": "ecmwf/aifs-single-1.0"}
        runner = SimpleRunner(checkpoint, device="cpu")
        aifs_model = runner.model
        print("AIFS model loaded for demo")
    except Exception as e:
        print(f"AIFS model not available for demo: {e}")

    # Run demonstrations
    demonstrate_aifs_llama_fusion()
    demonstrate_compression_analysis(aifs_model)

    print("\nDemo completed successfully!")
    print("\nNext Steps:")
    print("   Install transformers package for real LLaMA integration")
    print("   Train on real climate-text paired datasets")
    print("   Fine-tune for specific climate analysis tasks")
    print("   Scale to larger spatial-temporal resolutions")
    print("   Integrate with real-time weather data APIs")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
AIFS Integration Example

This script demonstrates how to integrate ECMWF AIFS with the HPE-LLM4Climate system
for comparative climate analysis, benchmarking, and time series tokenization.

Features demonstrated:
- AIFS encoder integration for spatial climate data
- AIFSTimeSeriesTokenizer for 5-D temporal data
- Multimodal fusion of climate time series with text

Requirements:
- aifs-single-1.0 submodule initialized
- HPE-LLM4Climate dependencies installed
- AIFS model dependencies (see aifs-single-1.0/README.md)
"""

import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import AIFS components
try:
    from multimodal_aifs.utils.aifs_encoder_utils import AIFSEncoderWrapper
    from multimodal_aifs.utils.aifs_time_series_tokenizer import AIFSTimeSeriesTokenizer

    AIFS_TIMESERIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  AIFS TimeSeriesTokenizer not available: {e}")
    AIFS_TIMESERIES_AVAILABLE = False


def check_aifs_availability():
    """Check if AIFS submodule is available and accessible."""
    aifs_path = project_root / "aifs-single-1.0"

    if not aifs_path.exists():
        print("❌ AIFS submodule not found!")
        print("Please initialize submodules:")
        print("  git submodule update --init --recursive")
        return False

    checkpoint_path = aifs_path / "aifs-single-mse-1.0.ckpt"
    if not checkpoint_path.exists():
        print("❌ AIFS checkpoint not found!")
        print("Please ensure the AIFS submodule is properly downloaded.")
        return False

    print("✅ AIFS submodule found and accessible")
    return True


def demonstrate_timeseries_tokenization():
    """Demonstrate time series tokenization with AIFS."""
    print("\n🕐 Time Series Tokenization Example")
    print("=" * 50)

    if not AIFS_TIMESERIES_AVAILABLE:
        print("❌ AIFSTimeSeriesTokenizer not available")
        return

    print("Creating sample 5-D climate time series data...")

    # Create sample time series: [batch, time, variables, height, width]
    batch_size = 2
    time_steps = 8  # 8 hours of data
    n_variables = 5  # temperature, pressure, humidity, wind_u, wind_v
    spatial_shape = (64, 64)  # 64x64 spatial grid

    # Generate synthetic climate time series
    sample_data = torch.randn(batch_size, time_steps, n_variables, *spatial_shape)

    print(f"✅ Generated sample data: {sample_data.shape}")
    print(
        f"   Interpretation: {batch_size} samples, {time_steps} timesteps, {n_variables} variables, {spatial_shape[0]}x{spatial_shape[1]} spatial"
    )

    # Test different temporal modeling approaches
    temporal_models = ["transformer", "lstm", "none"]

    for model_type in temporal_models:
        try:
            print(f"\n🔧 Testing {model_type.upper()} temporal modeling:")

            # Initialize tokenizer
            tokenizer = AIFSTimeSeriesTokenizer(
                temporal_modeling=model_type, hidden_dim=512, device="cpu"
            )

            # Tokenize the time series
            tokens = tokenizer.tokenize_time_series(sample_data)

            print(f"   ✅ Tokenization: {sample_data.shape} -> {tokens.shape}")
            print(
                f"   📊 Output interpretation: {tokens.shape[0]} samples, {tokens.shape[1]} timesteps, {tokens.shape[2]} features"
            )

            # Calculate compression ratio
            input_size = sample_data.numel() * 4  # float32 bytes
            output_size = tokens.numel() * 4
            compression = input_size / output_size

            print(
                f"   💾 Data compression: {compression:.1f}x ({input_size/1024:.1f}KB -> {output_size/1024:.1f}KB)"
            )

        except Exception as e:
            print(f"   ❌ {model_type.upper()} failed: {e}")

    print("\n💡 Use Cases for Time Series Tokenization:")
    print("   • Multi-timestep weather forecasting")
    print("   • Climate pattern analysis over time")
    print("   • Temporal anomaly detection")
    print("   • Multimodal climate-text fusion")


def demonstrate_data_pipeline():
    """Show how to integrate AIFS with existing data pipeline."""
    print("\n🔄 Data Pipeline Integration")
    print("=" * 50)

    # Check for MERRA-2 processed data
    dataset_path = project_root / "dataset" / "example_output"
    if dataset_path.exists():
        merra2_files = list(dataset_path.glob("*.npz"))
        if merra2_files:
            print(f"✅ Found {len(merra2_files)} processed MERRA-2 datasets")
            for file in merra2_files:
                print(f"  • {file.name}")
        else:
            print("⚠️  No processed MERRA-2 data found")
    else:
        print("⚠️  MERRA-2 dataset directory not found")

    print("\n📈 Integration workflow:")
    print("  1. Load MERRA-2 data using existing data_loader.py")
    print("  2. Convert to AIFS-compatible format")
    print("  3. Run AIFS inference for global forecasts")


def show_usage_examples():
    """Display usage examples for different integration scenarios."""
    print("\n💡 Usage Examples")
    print("=" * 50)

    examples = [
        {
            "title": "Time Series Tokenization",
            "code": """
# Tokenize 5-D climate time series with AIFS
from multimodal_aifs.utils.aifs_time_series_tokenizer import AIFSTimeSeriesTokenizer

tokenizer = AIFSTimeSeriesTokenizer(temporal_modeling="transformer")
climate_data_5d = torch.randn(2, 8, 5, 64, 64)  # [batch, time, vars, h, w]
tokens = tokenizer.tokenize_time_series(climate_data_5d)
print(f"Tokenized: {climate_data_5d.shape} -> {tokens.shape}")
            """,
        },
    ]

    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}:")
        print(example["code"])


def main():
    """Main integration demonstration."""
    print("🌍 ECMWF AIFS + HPE-LLM4Climate Integration")
    print("=" * 60)

    # Check system status
    aifs_ok = check_aifs_availability()

    if not aifs_ok:
        print("\n⚠️  AIFS not available. Please setup submodule first.")
        print("Documentation: docs/aifs_integration.md")
        return

    # Show AIFS integration capabilities
    demonstrate_timeseries_tokenization()

    # Show integration capabilities
    demonstrate_data_pipeline()
    show_usage_examples()

    print("\n🎯 Next Steps:")
    print("  1. Try the multimodal time series demo:")
    print("     python multimodal_aifs/examples/multimodal_timeseries_demo.py")
    print("  2. Review docs/aifs_integration.md for detailed setup")
    print("  3. Explore aifs-single-1.0/run_AIFS_v1.ipynb for AIFS examples")

    print("\n✨ Integration complete! Ready for comparative climate AI analysis.")


if __name__ == "__main__":
    main()

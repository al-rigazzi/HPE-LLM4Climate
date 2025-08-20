#!/usr/bin/env python3
"""
AIFS Integration Example

This script demonstrates how to integrate ECMWF AIFS with the HPE-LLM4Climate system
for comparative climate analysis, benchmarking, and time series tokenization.

Features demonstrated:
- AIFS encoder integration for spatial climate data
- AIFSTimeSeriesTokenizer for 5-D temporal data
- Comparative analysis between AIFS and PrithviWxC
- Multimodal fusion of climate time series with text

Requirements:
- aifs-single-1.0 submodule initialized
- HPE-LLM4Climate dependencies installed
- AIFS model dependencies (see aifs-single-1.0/README.md)
"""

import os
import sys
from pathlib import Path

import numpy as np
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


def check_prithvi_availability():
    """Check if PrithviWxC components are available."""
    try:
        # Try importing from the actual multimodal location
        from multimodal.core.models import MultimodalTransformer

        print("✅ PrithviWxC multimodal components available")
        return True
    except ImportError:
        try:
            # Alternative import path
            from multimodal.location_aware_fusion import LocationAwareFusion

            print("✅ PrithviWxC multimodal components available")
            return True
        except ImportError as e:
            print(f"❌ PrithviWxC components not available: {e}")
            print("   Note: This demo can still show AIFS time series tokenization")
            return False


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


def demonstrate_comparative_analysis():
    """Demonstrate comparative analysis between AIFS and PrithviWxC."""
    print("\n🔬 Comparative Analysis Example")
    print("=" * 50)

    # Example locations for analysis
    test_locations = [
        {"name": "New York City", "lat": 40.7128, "lon": -74.0060},
        {"name": "London", "lat": 51.5074, "lon": -0.1278},
        {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503},
        {"name": "Sydney", "lat": -33.8688, "lon": 151.2093},
    ]

    print("Test locations for comparative analysis:")
    for i, loc in enumerate(test_locations, 1):
        print(f"  {i}. {loc['name']}: ({loc['lat']:.4f}, {loc['lon']:.4f})")

    print("\n📊 Analysis capabilities:")
    print("  • AIFS: Global 10+ day weather forecasts")
    print("  • PrithviWxC: Regional climate analysis with multimodal fusion")
    print("  • Comparison: Skill assessment across different time scales")
    print("  • Ensemble: Combined predictions for improved accuracy")


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
    print("  4. Run PrithviWxC for regional climate analysis")
    print("  5. Compare and ensemble results")


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
        {
            "title": "Benchmark Comparison",
            "code": """
# Compare AIFS and PrithviWxC predictions
from aifs_integration import AIFSWrapper, compare_models

aifs = AIFSWrapper('aifs-single-1.0/aifs-single-mse-1.0.ckpt')
prithvi = LocationAwareFusion()

results = compare_models(aifs, prithvi, location=(40.7128, -74.0060))
print(f"RMSE - AIFS: {results['aifs_rmse']:.3f}, PrithviWxC: {results['prithvi_rmse']:.3f}")
            """,
        },
        {
            "title": "Ensemble Forecasting",
            "code": """
# Create ensemble predictions
ensemble_forecast = create_ensemble([
    aifs.predict(location, days=10),
    prithvi.predict(location, days=10)
], weights=[0.6, 0.4])  # Weight AIFS higher for longer forecasts
            """,
        },
        {
            "title": "Multi-scale Analysis",
            "code": """
# Global context from AIFS, regional details from PrithviWxC
global_forecast = aifs.predict_global(days=7)
regional_analysis = prithvi.analyze_region(bbox, climate_query="drought risk")

combined_insight = merge_global_regional(global_forecast, regional_analysis)
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
    prithvi_ok = check_prithvi_availability()

    if not aifs_ok:
        print("\n⚠️  AIFS not available. Please setup submodule first.")
        print("Documentation: docs/aifs_integration.md")
        return

    # Show AIFS integration capabilities even without PrithviWxC
    demonstrate_timeseries_tokenization()

    if not prithvi_ok:
        print("\n⚠️  PrithviWxC not available, but AIFS time series tokenization is working!")
        print("   You can still use:")
        print("   • AIFSTimeSeriesTokenizer for 5-D climate data")
        print("   • AIFS encoder for spatial climate encoding")
        print("   • Multimodal fusion demonstrations")
        print("\n🎯 Try the full multimodal demo:")
        print("   python multimodal_aifs/examples/multimodal_timeseries_demo.py")
        return

    # Show integration capabilities
    demonstrate_comparative_analysis()
    demonstrate_data_pipeline()
    show_usage_examples()

    print("\n🎯 Next Steps:")
    print("  1. Try the multimodal time series demo:")
    print("     python multimodal_aifs/examples/multimodal_timeseries_demo.py")
    print("  2. Review docs/aifs_integration.md for detailed setup")
    print("  3. Explore aifs-single-1.0/run_AIFS_v1.ipynb for AIFS examples")
    print("  4. Run multimodal examples for PrithviWxC demonstrations")
    print("  5. Implement comparative analysis for your use case")

    print("\n✨ Integration complete! Ready for comparative climate AI analysis.")


if __name__ == "__main__":
    main()

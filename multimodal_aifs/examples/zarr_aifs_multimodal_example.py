#!/usr/bin/env python3
"""
AIFS Multimodal with Zarr Data Example

This example demonstrates how to use Zarr format climate data
with the AIFS multimodal model (AIFS + Llama 3-8B integration).

Features:
- Load Zarr climate datasets
- Convert to AIFS-compatible 5D tensors
- Process with AIFS TimeSeries tokenizer
- Integrate with Llama 3-8B for multimodal analysis

Usage:
    python zarr_aifs_multimodal_example.py --zarr-path /path/to/climate.zarr
    python zarr_aifs_multimodal_example.py --zarr-url s3://bucket/climate.zarr
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from multimodal_aifs.utils.aifs_time_series_tokenizer import AIFSTimeSeriesTokenizer
    from multimodal_aifs.utils.zarr_data_loader import ZarrClimateLoader, load_zarr_for_aifs

    ZARR_LOADER_AVAILABLE = True
except ImportError as e:
    ZARR_LOADER_AVAILABLE = False
    print(f"⚠️  Zarr loader not available: {e}")

# Try to import Llama integration
try:
    sys.path.append(str(project_root / "multimodal_aifs" / "tests" / "integration"))
    from test_aifs_llama_integration import AIFSLlamaFusionModel

    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    print("⚠️  Llama integration not available")


def demonstrate_zarr_to_aifs(
    zarr_path: str,
    start_time: str = "2024-01-01",
    end_time: str = "2024-01-07",
    variables: list[str] | None = None,
):
    """Demonstrate loading Zarr data and processing with AIFS."""

    print("🌍 AIFS Multimodal with Zarr Data Demo")
    print("=" * 45)

    if not ZARR_LOADER_AVAILABLE:
        print("❌ Zarr loader not available. Please install: pip install zarr xarray")
        return

    # Step 1: Load Zarr Dataset
    print("\n📁 Step 1: Loading Zarr Climate Dataset")
    try:
        loader = ZarrClimateLoader(zarr_path, variables=variables)
        dataset_info = loader.get_info()

        print("✅ Dataset loaded successfully!")
        print(f"   📊 Variables: {len(dataset_info['variables'])}")
        print(f"   🌍 Spatial: {dataset_info['spatial_shape']}")
        print(
            f"   📅 Time range: {dataset_info['time_range'][0]} to {dataset_info['time_range'][1]}"
        )
        print(f"   💾 Size: {dataset_info['total_size_gb']:.2f} GB")

    except Exception as e:
        print(f"❌ Failed to load Zarr dataset: {e}")
        return

    # Step 2: Load Specific Time Range
    print(f"\n⏰ Step 2: Loading Time Range ({start_time} to {end_time})")
    try:
        data = loader.load_time_range(start_time, end_time, variables)
        print(f"✅ Time range loaded: {dict(data.dims)}")

    except Exception as e:
        print(f"❌ Failed to load time range: {e}")
        return

    # Step 3: Convert to AIFS Tensor Format
    print("\n🔄 Step 3: Converting to AIFS 5D Tensor Format")
    try:
        # Convert to [B, T, V, H, W] format
        aifs_tensor = loader.to_aifs_tensor(data, batch_size=2, normalize=True)
        print(f"✅ AIFS tensor created: {aifs_tensor.shape}")
        print(
            f"   📊 Format: [batch={aifs_tensor.shape[0]}, time={aifs_tensor.shape[1]}, "
            f"vars={aifs_tensor.shape[2]}, height={aifs_tensor.shape[3]}, width={aifs_tensor.shape[4]}]"
        )
        print(f"   💾 Memory: {aifs_tensor.element_size() * aifs_tensor.nelement() / 1e6:.1f} MB")

    except Exception as e:
        print(f"❌ Failed to convert to AIFS tensor: {e}")
        return

    # Step 4: AIFS TimeSeries Tokenization
    print("\n🤖 Step 4: AIFS TimeSeries Tokenization")
    try:
        # Initialize AIFS tokenizer
        tokenizer = AIFSTimeSeriesTokenizer(
            temporal_modeling="transformer", hidden_dim=512, device="cpu"
        )

        # Tokenize the climate data
        climate_tokens = tokenizer(aifs_tensor)
        print(f"✅ Climate tokenization successful!")
        print(f"   🎯 Token shape: {climate_tokens.shape}")
        print(
            f"   📊 Format: [batch={climate_tokens.shape[0]}, seq_len={climate_tokens.shape[1]}, hidden={climate_tokens.shape[2]}]"
        )

    except Exception as e:
        print(f"❌ Failed AIFS tokenization: {e}")
        climate_tokens = None

    # Step 5: Multimodal Integration (if available)
    if LLAMA_AVAILABLE and climate_tokens is not None:
        print("\n🔗 Step 5: AIFS-Llama Multimodal Integration")
        try:
            # Initialize multimodal fusion model
            model = AIFSLlamaFusionModel(
                llm_model_name="meta-llama/Meta-Llama-3-8B",
                time_series_dim=512,
                fusion_strategy="cross_attention",
                device="cpu",
                use_mock_llama=True,  # Use mock for demo
            )

            # Create sample text input
            sample_text = [
                "Analyze temperature patterns in this climate data",
                "What are the weather trends shown in the data?",
            ]

            # Process multimodal input
            result = model.process_climate_text(climate_tokens, sample_text)

            print(f"✅ Multimodal processing successful!")
            print(f"   🎯 Output shape: {result['fused_output'].shape}")
            print(f"   📝 Sample analysis: {result.get('generated_text', 'Analysis complete')}")

        except Exception as e:
            print(f"❌ Multimodal integration failed: {e}")
    else:
        print(f"\n⚠️  Skipping multimodal integration (Llama not available)")

    # Summary
    print(f"\n📋 Demo Summary")
    print("=" * 20)
    print("✅ Zarr data successfully loaded and processed!")
    print("✅ AIFS tensor format conversion complete")
    if climate_tokens is not None:
        print("✅ AIFS TimeSeries tokenization successful")
    if LLAMA_AVAILABLE:
        print("✅ Multimodal AIFS-Llama integration ready")

    return {
        "zarr_path": zarr_path,
        "dataset_info": dataset_info,
        "aifs_tensor_shape": aifs_tensor.shape,
        "climate_tokens_shape": climate_tokens.shape if climate_tokens is not None else None,
        "multimodal_ready": LLAMA_AVAILABLE,
    }


def demonstrate_zarr_spatial_region(
    zarr_path: str, lat_range: tuple = (30, 60), lon_range: tuple = (-120, -60)
):
    """Demonstrate loading a specific spatial region from Zarr."""

    print("\n🌍 Spatial Region Loading Demo")
    print("=" * 35)

    if not ZARR_LOADER_AVAILABLE:
        print("❌ Zarr loader not available")
        return

    try:
        loader = ZarrClimateLoader(zarr_path)

        print(f"📍 Loading region:")
        print(f"   Latitude: {lat_range[0]}° to {lat_range[1]}°")
        print(f"   Longitude: {lon_range[0]}° to {lon_range[1]}°")

        # Load spatial region
        regional_data = loader.load_spatial_region(
            lat_range=lat_range, lon_range=lon_range, time_range=("2024-01-01", "2024-01-03")
        )

        # Convert to AIFS format
        regional_tensor = loader.to_aifs_tensor(regional_data, batch_size=1)

        print(f"✅ Regional data loaded!")
        print(f"   📊 Tensor shape: {regional_tensor.shape}")
        print(f"   🌍 Spatial resolution: {regional_tensor.shape[3]}×{regional_tensor.shape[4]}")

        return regional_tensor

    except Exception as e:
        print(f"❌ Regional loading failed: {e}")
        return None


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="AIFS Multimodal with Zarr Data Demo")
    parser.add_argument("--zarr-path", required=True, help="Path to Zarr dataset")
    parser.add_argument("--start-time", default="2024-01-01", help="Start time")
    parser.add_argument("--end-time", default="2024-01-07", help="End time")
    parser.add_argument("--variables", nargs="+", help="Specific variables to load")
    parser.add_argument("--spatial-demo", action="store_true", help="Run spatial region demo")

    args = parser.parse_args()

    print("🚀 AIFS Multimodal Zarr Integration Demo")
    print("=" * 50)
    print(f"📁 Zarr path: {args.zarr_path}")
    print(f"⏰ Time range: {args.start_time} to {args.end_time}")
    if args.variables:
        print(f"🔢 Variables: {args.variables}")

    # Main demo
    result = demonstrate_zarr_to_aifs(
        zarr_path=args.zarr_path,
        start_time=args.start_time,
        end_time=args.end_time,
        variables=args.variables,
    )

    # Spatial demo (optional)
    if args.spatial_demo:
        demonstrate_zarr_spatial_region(args.zarr_path)

    if result:
        print(f"\n🎉 Demo completed successfully!")
        print(f"   Ready for AIFS multimodal processing with Zarr data!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright 2025 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
AIFS Multimodal with Zarr Data Example

This example demonstrates how to use Zarr format climate data
with the AIFS multimodal model (AIFS + Mistral-7B-Instruct integration).

Features:
- Load Zarr climate datasets
- Convert to AIFS-compatible 5D tensors
- Process with AIFS TimeSeries tokenizer
- Integrate with Mistral-7B-Instruct for multimodal analysis

Usage:
    python zarr_aifs_multimodal_example.py --zarr-path /path/to/climate.zarr
    python zarr_aifs_multimodal_example.py --zarr-url s3://bucket/climate.zarr
"""

import argparse
import sys
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from multimodal_aifs.utils.aifs_time_series_tokenizer import AIFSTimeSeriesTokenizer
    from multimodal_aifs.utils.zarr_data_loader import ZarrClimateLoader

    ZARR_LOADER_AVAILABLE = True
except ImportError as e:
    ZARR_LOADER_AVAILABLE = False
    print(f"Zarr loader not available: {e}")

# Try to import Mistral integration
try:
    sys.path.append(str(project_root / "multimodal_aifs" / "tests" / "integration"))
    from test_aifs_mistral_integration import AIFSMistralFusionModel

    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    print("Mistral integration not available")


def demonstrate_zarr_to_aifs(
    zarr_path: str,
    start_time: str = "2024-01-01",
    end_time: str = "2024-01-07",
    variables: list[str] | None = None,
) -> dict[str, Any] | None:
    """Demonstrate loading Zarr data and processing with AIFS."""

    print("AIFS Multimodal with Zarr Data Demo")
    print("=" * 45)

    if not ZARR_LOADER_AVAILABLE:
        print("Zarr loader not available. Please install: pip install zarr xarray")
        return None

    # Step 1: Load Zarr Dataset
    print("\nStep 1: Loading Zarr Climate Dataset")
    try:
        loader = ZarrClimateLoader(zarr_path, variables=variables)
        dataset_info = loader.get_info()

        print("Dataset loaded successfully!")
        print(f"   Variables: {len(dataset_info['variables'])}")
        print(f"   Spatial: {dataset_info['spatial_shape']}")
        print(
            f"   📅 Time range: {dataset_info['time_range'][0]} to {dataset_info['time_range'][1]}"
        )
        print(f"   Size: {dataset_info['total_size_gb']:.2f} GB")

    except Exception as e:
        print(f"Failed to load Zarr dataset: {e}")
        return None

    # Step 2: Load Specific Time Range
    print(f"\n⏰ Step 2: Loading Time Range ({start_time} to {end_time})")
    try:
        data = loader.load_time_range(start_time, end_time, variables)
        print(f"Time range loaded: {dict(data.dims)}")

    except Exception as e:
        print(f"Failed to load time range: {e}")
        return None

    # Step 3: Convert to AIFS Tensor Format
    print("\nStep 3: Converting to AIFS 5D Tensor Format")
    try:
        # Convert to [B, T, V, H, W] format
        aifs_tensor = loader.to_aifs_tensor(data, batch_size=2, normalize=True)
        print(f"AIFS tensor created: {aifs_tensor.shape}")
        print(
            f"   Format: [batch={aifs_tensor.shape[0]}, time={aifs_tensor.shape[1]}, "
            f"vars={aifs_tensor.shape[2]}, height={aifs_tensor.shape[3]}, "
            f"width={aifs_tensor.shape[4]}]"
        )
        print(f"   Memory: {aifs_tensor.element_size() * aifs_tensor.nelement() / 1e6:.1f} MB")

    except Exception as e:
        print(f"Failed to convert to AIFS tensor: {e}")
        return None

    # Step 4: AIFS TimeSeries Tokenization
    print("\n🤖 Step 4: AIFS TimeSeries Tokenization")
    try:
        # Initialize AIFS tokenizer
        tokenizer = AIFSTimeSeriesTokenizer(hidden_dim=512, device="cpu")

        # Tokenize the climate data
        climate_tokens = tokenizer(aifs_tensor)
        print("Climate tokenization successful!")
        print(f"   Token shape: {climate_tokens.shape}")
        print(
            f"   Format: [batch={climate_tokens.shape[0]}, "
            f"seq_len={climate_tokens.shape[1]}, hidden={climate_tokens.shape[2]}]"
        )

    except Exception as e:
        print(f"Failed AIFS tokenization: {e}")
        climate_tokens = None

    # Step 5: Multimodal Integration (if available)
    if MISTRAL_AVAILABLE and climate_tokens is not None:
        print("\n🔗 Step 5: AIFS-Mistral Multimodal Integration")
        try:
            # Initialize multimodal fusion model
            model = AIFSMistralFusionModel(
                llm_model_name="mistralai/Mistral-7B-Instruct-v0.3",
                time_series_dim=512,
                fusion_strategy="cross_attention",
                device="cpu",
                use_mock_mistral=True,  # Use mock for demo
            )

            # Create sample text input
            sample_text = [
                "Analyze temperature patterns in this climate data",
                "What are the weather trends shown in the data?",
            ]

            # Process multimodal input
            result = model.process_climate_text(climate_tokens, sample_text)

            print("Multimodal processing successful!")
            print(f"   Output shape: {result['fused_output'].shape}")
            print(f"   Sample analysis: {result.get('generated_text', 'Analysis complete')}")

        except Exception as e:
            print(f"Multimodal integration failed: {e}")
    else:
        print("\nSkipping multimodal integration (Mistral not available)")

    # Summary
    print("\n📋 Demo Summary")
    print("=" * 20)
    print("Zarr data successfully loaded and processed!")
    print("AIFS tensor format conversion complete")
    if climate_tokens is not None:
        print("AIFS TimeSeries tokenization successful")
    if MISTRAL_AVAILABLE:
        print("Multimodal AIFS-Mistral integration ready")

    return {
        "zarr_path": zarr_path,
        "dataset_info": dataset_info,
        "aifs_tensor_shape": aifs_tensor.shape,
        "climate_tokens_shape": climate_tokens.shape if climate_tokens is not None else None,
        "multimodal_ready": MISTRAL_AVAILABLE,
    }


def demonstrate_zarr_spatial_region(
    zarr_path: str, lat_range: tuple = (30, 60), lon_range: tuple = (-120, -60)
):
    """Demonstrate loading a specific spatial region from Zarr."""

    print("\nSpatial Region Loading Demo")
    print("=" * 35)

    if not ZARR_LOADER_AVAILABLE:
        print("Zarr loader not available")
        return None

    try:
        loader = ZarrClimateLoader(zarr_path)

        print("📍 Loading region:")
        print(f"   Latitude: {lat_range[0]}° to {lat_range[1]}°")
        print(f"   Longitude: {lon_range[0]}° to {lon_range[1]}°")

        # Load spatial region
        regional_data = loader.load_spatial_region(
            lat_range=lat_range, lon_range=lon_range, time_range=("2024-01-01", "2024-01-03")
        )

        # Convert to AIFS format
        regional_tensor = loader.to_aifs_tensor(regional_data, batch_size=1)

        print("Regional data loaded!")
        print(f"   Tensor shape: {regional_tensor.shape}")
        print(f"   Spatial resolution: {regional_tensor.shape[3]}×{regional_tensor.shape[4]}")

        return regional_tensor

    except Exception as e:
        print(f"Regional loading failed: {e}")
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

    print("AIFS Multimodal Zarr Integration Demo")
    print("=" * 50)
    print(f"Zarr path: {args.zarr_path}")
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
        print("\nDemo completed successfully!")
        print("   Ready for AIFS multimodal processing with Zarr data!")


if __name__ == "__main__":
    main()

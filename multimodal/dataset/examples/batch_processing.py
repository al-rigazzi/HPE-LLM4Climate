#!/usr/bin/env python3
"""
Batch Processing Example for MERRA-2 Dataset Processor

This script demonstrates how to efficiently process multiple time periods
and create datasets for different temporal resolutions.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data_loader import MERRA2DataLoader, validate_dataset_compatibility
from merra2_dataset_processor import MERRA2DatasetProcessor


def process_single_period(args: Tuple[str, str, str, str, str]) -> Tuple[str, bool, str]:
    """
    Process a single time period.

    Args:
        args: Tuple of (start_date, end_date, temporal_resolution, output_dir, cache_dir)

    Returns:
        Tuple of (period_id, success, output_path_or_error)
    """
    start_date, end_date, temporal_resolution, output_dir, cache_dir = args
    period_id = f"{start_date}_{end_date}_{temporal_resolution}"

    try:
        processor = MERRA2DatasetProcessor(
            output_dir=output_dir,
            cache_dir=f"{cache_dir}_{period_id}",  # Separate cache for each process
            earthdata_username=os.getenv("EARTHDATA_USERNAME"),
            earthdata_password=os.getenv("EARTHDATA_PASSWORD"),
        )

        output_path = processor.process_timerange(
            start_date=start_date,
            end_date=end_date,
            temporal_resolution=temporal_resolution,
            output_filename=f"merra2_prithvi_{period_id}.npz",
        )

        return period_id, True, str(output_path)

    except Exception as e:
        return period_id, False, str(e)


def batch_process_months(
    year: int,
    months: List[int],
    temporal_resolution: str = "3H",
    output_dir: str = "./batch_output",
    cache_dir: str = "./batch_cache",
    max_workers: int = 2,
) -> List[Path]:
    """
    Process multiple months in parallel.

    Args:
        year: Year to process
        months: List of month numbers (1-12)
        temporal_resolution: Temporal resolution
        output_dir: Output directory
        cache_dir: Cache directory
        max_workers: Maximum number of parallel workers

    Returns:
        List of output file paths
    """
    print(f"=== Batch Processing: {year} - Months {months} ===\n")

    # Create argument tuples for each month
    tasks = []
    for month in months:
        start_date = f"{year}-{month:02d}-01"
        # Get last day of month
        last_day = pd.Timestamp(year, month, 1).days_in_month
        end_date = f"{year}-{month:02d}-{last_day}"

        tasks.append((start_date, end_date, temporal_resolution, output_dir, cache_dir))

    print(f"📋 Processing {len(tasks)} months with {max_workers} parallel workers")
    print(f"⏰ Estimated time: {len(tasks) * 15 // max_workers} minutes")

    successful_outputs = []
    failed_tasks = []

    # Process in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(process_single_period, task): task for task in tasks}

        # Collect results
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            period_id, success, result = future.result()

            if success:
                print(f"✅ {period_id}: Success - {result}")
                successful_outputs.append(Path(result))
            else:
                print(f"❌ {period_id}: Failed - {result}")
                failed_tasks.append((task, result))

    print(f"\n📊 Batch processing summary:")
    print(f"   Successful: {len(successful_outputs)}")
    print(f"   Failed: {len(failed_tasks)}")

    if failed_tasks:
        print(f"\n❌ Failed tasks:")
        for task, error in failed_tasks:
            start_date, end_date, _, _, _ = task
            print(f"   {start_date} to {end_date}: {error}")

    return successful_outputs


def process_multiple_resolutions(
    start_date: str,
    end_date: str,
    resolutions: List[str] = ["1H", "3H", "Monthly"],
    output_dir: str = "./multi_res_output",
) -> List[Path]:
    """
    Process the same time period at multiple temporal resolutions.

    Args:
        start_date: Start date
        end_date: End date
        resolutions: List of temporal resolutions
        output_dir: Output directory

    Returns:
        List of output file paths
    """
    print(f"=== Multi-Resolution Processing: {start_date} to {end_date} ===\n")

    outputs = []

    for resolution in resolutions:
        print(f"🔄 Processing at {resolution} resolution...")

        try:
            processor = MERRA2DatasetProcessor(
                output_dir=output_dir,
                cache_dir=f"./cache_{resolution}",
                earthdata_username=os.getenv("EARTHDATA_USERNAME"),
                earthdata_password=os.getenv("EARTHDATA_PASSWORD"),
            )

            output_path = processor.process_timerange(
                start_date=start_date, end_date=end_date, temporal_resolution=resolution
            )

            print(f"✅ {resolution}: {output_path}")
            outputs.append(output_path)

        except Exception as e:
            print(f"❌ {resolution}: Failed - {e}")

    return outputs


def validate_batch_outputs(output_paths: List[Path]) -> None:
    """
    Validate multiple processed datasets.

    Args:
        output_paths: List of dataset file paths
    """
    print(f"\n=== Validating {len(output_paths)} Datasets ===\n")

    valid_count = 0

    for output_path in output_paths:
        print(f"🔍 Validating {output_path.name}...")

        try:
            # Load basic info
            info = MERRA2DataLoader.load_dataset_info(output_path)

            # Validate compatibility
            validation = validate_dataset_compatibility(output_path)

            if validation["valid_dataset"]:
                print(f"✅ Valid dataset")
                valid_count += 1
            else:
                print(f"⚠️  Issues found:")
                for category in ["surface", "static", "vertical"]:
                    missing_key = f"missing_{category}_vars"
                    if validation[missing_key]:
                        print(f"   Missing {category} vars: {validation[missing_key]}")

            # Print size info
            total_size = output_path.stat().st_size / (1024**3)  # GB
            print(f"   Size: {total_size:.2f} GB")

            if "surface_shape" in info:
                time_steps = info["surface_shape"][0]
                print(f"   Time steps: {time_steps}")

        except Exception as e:
            print(f"❌ Error validating {output_path.name}: {e}")

        print()

    print(f"📊 Validation summary: {valid_count}/{len(output_paths)} datasets are valid")


def create_combined_dataset(
    output_paths: List[Path], combined_output_path: str = "./combined_dataset.npz"
) -> Path:
    """
    Combine multiple processed datasets into a single file.

    Args:
        output_paths: List of dataset paths to combine
        combined_output_path: Output path for combined dataset

    Returns:
        Path to combined dataset
    """
    print(f"\n=== Combining {len(output_paths)} Datasets ===\n")

    import json
    from datetime import datetime

    import numpy as np

    # Load all datasets
    all_surface_data = []
    all_vertical_data = []
    all_times = []
    reference_static = None
    reference_vars = None
    reference_coords = None

    for i, path in enumerate(output_paths):
        print(f"📖 Loading {path.name}...")

        data = np.load(path, allow_pickle=True)

        if i == 0:
            # Use first dataset as reference for static data and variables
            reference_static = data["static"]
            reference_vars = {
                "surface_vars": data["surface_vars"].tolist(),
                "static_vars": data["static_vars"].tolist(),
                "vertical_vars": data["vertical_vars"].tolist(),
            }
            reference_coords = data["coordinates"].item()

        # Accumulate time-varying data
        if data["surface"] is not None:
            all_surface_data.append(data["surface"])
        if data["vertical"] is not None:
            all_vertical_data.append(data["vertical"])

        # Accumulate time coordinates
        coords = data["coordinates"].item()
        all_times.extend(coords["time"])

    # Combine arrays
    print("🔗 Combining arrays...")
    combined_surface = np.concatenate(all_surface_data, axis=0) if all_surface_data else None
    combined_vertical = np.concatenate(all_vertical_data, axis=0) if all_vertical_data else None

    # Create combined coordinates
    combined_coords = reference_coords.copy()
    combined_coords["time"] = all_times

    # Save combined dataset
    print(f"💾 Saving combined dataset to {combined_output_path}...")

    combined_path = Path(combined_output_path)
    combined_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        combined_path,
        surface=combined_surface,
        static=reference_static,
        vertical=combined_vertical,
        coordinates=combined_coords,
        **reference_vars,
    )

    # Save metadata
    metadata = {
        **reference_vars,
        "coordinates": combined_coords,
        "creation_time": datetime.now().isoformat(),
        "format_version": "1.0",
        "source_files": [str(p) for p in output_paths],
        "combined": True,
    }

    metadata_path = combined_path.with_suffix(".metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"✅ Combined dataset created: {combined_path}")
    if combined_surface is not None:
        print(f"   Surface shape: {combined_surface.shape}")
    if combined_vertical is not None:
        print(f"   Vertical shape: {combined_vertical.shape}")

    return combined_path


def main():
    """Demonstrate batch processing capabilities."""

    print("=== MERRA-2 Dataset Processor - Batch Processing Example ===\n")

    # Check credentials
    if not os.getenv("EARTHDATA_USERNAME") or not os.getenv("EARTHDATA_PASSWORD"):
        print("⚠️  WARNING: NASA Earthdata credentials not found!")
        print("Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables")
        print("Continuing with demo mode...\n")

    # Example 1: Process multiple months of 2020
    print("Example 1: Processing multiple months of 2020 (3-hourly)")
    output_paths_3h = batch_process_months(
        year=2020,
        months=[1, 2, 3],  # January, February, March
        temporal_resolution="3H",
        output_dir="./batch_output_3h",
        max_workers=2,
    )

    # Example 2: Process same period at multiple resolutions
    print("\nExample 2: Processing January 2020 at multiple resolutions")
    multi_res_paths = process_multiple_resolutions(
        start_date="2020-01-01",
        end_date="2020-01-31",
        resolutions=["3H", "Monthly"],  # Skip 1H for demo
        output_dir="./multi_res_output",
    )

    # Combine all outputs for validation
    all_outputs = output_paths_3h + multi_res_paths

    if all_outputs:
        # Validate all outputs
        validate_batch_outputs(all_outputs)

        # Create combined dataset (only from 3-hourly data)
        if output_paths_3h:
            combined_path = create_combined_dataset(output_paths_3h, "./combined_q1_2020_3h.npz")

    print("\n🎉 Batch processing example completed!")
    print("\nGenerated datasets can be used for:")
    print("- Large-scale training with multiple months of data")
    print("- Multi-resolution analysis and comparison")
    print("- Time series analysis across different periods")


if __name__ == "__main__":
    main()

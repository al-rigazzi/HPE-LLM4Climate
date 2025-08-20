#!/usr/bin/env python3
"""
Data preparation script for multimodal climate-text training.

This script helps prepare your climate data and text pairs for training.
Adapt this script to your specific data format and requirements.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
from tqdm import tqdm


def create_dummy_data(output_dir: str, num_samples: int = 1000):
    """
    Create dummy training data for testing the training pipeline.
    Replace this with your actual data preparation logic.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create train/val split
    train_samples = []
    val_samples = []

    print(f"Creating {num_samples} dummy samples...")

    for i in tqdm(range(num_samples)):
        # Create dummy climate data [time=2, channels=160, height=64, width=64]
        climate_data = torch.randn(2, 160, 64, 64)

        # Save climate data
        climate_file = f"climate_{i:05d}.pt"
        torch.save(climate_data, output_path / climate_file)

        # Create dummy text pairs
        location = np.random.choice([
            "New York", "London", "Tokyo", "Sydney", "Mumbai",
            "São Paulo", "Cairo", "Moscow", "Lagos", "Jakarta"
        ])

        weather_condition = np.random.choice([
            "heavy rainfall", "drought conditions", "extreme heat",
            "cold temperatures", "moderate weather", "stormy conditions",
            "high humidity", "low precipitation", "wind patterns", "snow"
        ])

        input_text = (
            f"Analyze the climate data for {location}. "
            f"What are the expected weather patterns and climate risks?"
        )

        target_text = (
            f"Based on the climate data analysis for {location}, "
            f"the region is experiencing {weather_condition}. "
            f"This indicates potential impacts on local agriculture and "
            f"water resources. Climate adaptation measures should focus on "
            f"sustainable resource management and infrastructure resilience."
        )

        sample = {
            "climate_file": climate_file,
            "text": input_text,
            "target": target_text,
            "location": location,
            "weather_condition": weather_condition
        }

        # 80/20 train/val split
        if i < int(0.8 * num_samples):
            train_samples.append(sample)
        else:
            val_samples.append(sample)

    # Save train index
    train_dir = output_path / "train"
    train_dir.mkdir(exist_ok=True)
    with open(train_dir / "index.json", 'w') as f:
        json.dump(train_samples, f, indent=2)

    # Save val index
    val_dir = output_path / "val"
    val_dir.mkdir(exist_ok=True)
    with open(val_dir / "index.json", 'w') as f:
        json.dump(val_samples, f, indent=2)

    # Move climate files to appropriate directories
    for sample in train_samples:
        src = output_path / sample["climate_file"]
        dst = train_dir / sample["climate_file"]
        if src.exists():
            shutil.move(str(src), str(dst))

    for sample in val_samples:
        src = output_path / sample["climate_file"]
        dst = val_dir / sample["climate_file"]
        if src.exists():
            shutil.move(str(src), str(dst))

    print(f"Created {len(train_samples)} training samples in {train_dir}")
    print(f"Created {len(val_samples)} validation samples in {val_dir}")


def validate_data_format(data_dir: str) -> bool:
    """Validate that the data directory has the correct format."""
    data_path = Path(data_dir)

    # Check if train and val directories exist
    train_dir = data_path / "train"
    val_dir = data_path / "val"

    if not train_dir.exists() or not val_dir.exists():
        print("Error: Missing train or val directories")
        return False

    # Check if index files exist
    train_index = train_dir / "index.json"
    val_index = val_dir / "index.json"

    if not train_index.exists() or not val_index.exists():
        print("Error: Missing index.json files")
        return False

    # Validate index content
    try:
        with open(train_index, 'r') as f:
            train_data = json.load(f)

        with open(val_index, 'r') as f:
            val_data = json.load(f)

        # Check required fields
        required_fields = ["climate_file", "text", "target"]

        for sample in train_data[:5]:  # Check first 5 samples
            for field in required_fields:
                if field not in sample:
                    print(f"Error: Missing field '{field}' in training data")
                    return False

            # Check if climate file exists
            climate_file = train_dir / sample["climate_file"]
            if not climate_file.exists():
                print(f"Error: Climate file not found: {climate_file}")
                return False

        print(f"Data validation passed!")
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")

        return True

    except Exception as e:
        print(f"Error validating data: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--output_dir", type=str, default="data/training",
                       help="Output directory for training data")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of dummy samples to create")
    parser.add_argument("--validate_only", action="store_true",
                       help="Only validate existing data format")

    args = parser.parse_args()

    if args.validate_only:
        validate_data_format(args.output_dir)
    else:
        create_dummy_data(args.output_dir, args.num_samples)
        validate_data_format(args.output_dir)

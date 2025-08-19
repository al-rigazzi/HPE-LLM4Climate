#!/usr/bin/env python3
"""
Basic Usage Example for MERRA-2 Dataset Processor

This script demonstrates basic usage of the MERRA-2 dataset processor
for creating datasets compatible with PrithviWxC_Encoder.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from merra2_dataset_processor import MERRA2DatasetProcessor
from data_loader import PrithviMERRA2Dataset, MERRA2DataLoader, validate_dataset_compatibility


def main():
    """Demonstrate basic usage of the dataset processor."""
    
    print("=== MERRA-2 Dataset Processor - Basic Usage Example ===\n")
    
    # Configuration
    start_date = "2020-01-01"
    end_date = "2020-01-07"  # One week for quick testing
    temporal_resolution = "3H"
    output_dir = "./example_output"
    cache_dir = "./example_cache"
    
    print(f"Processing data from {start_date} to {end_date}")
    print(f"Temporal resolution: {temporal_resolution}")
    print(f"Output directory: {output_dir}")
    
    # Check for NASA Earthdata credentials
    if not os.getenv('EARTHDATA_USERNAME') or not os.getenv('EARTHDATA_PASSWORD'):
        print("\n⚠️  WARNING: NASA Earthdata credentials not found!")
        print("Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables")
        print("or the download step will fail.\n")
        
        # For demo purposes, we'll continue with dummy credentials
        print("Continuing with demo mode (downloads will fail)...")
        earthdata_username = "demo_user"
        earthdata_password = "demo_pass"
    else:
        earthdata_username = os.getenv('EARTHDATA_USERNAME')
        earthdata_password = os.getenv('EARTHDATA_PASSWORD')
        print("✅ NASA Earthdata credentials found")
    
    try:
        # Step 1: Create processor
        print("\n1. Creating MERRA-2 dataset processor...")
        processor = MERRA2DatasetProcessor(
            output_dir=output_dir,
            cache_dir=cache_dir,
            earthdata_username=earthdata_username,
            earthdata_password=earthdata_password
        )
        print("✅ Processor created successfully")
        
        # Step 2: Process the data
        print("\n2. Processing MERRA-2 data...")
        print("This step downloads and processes 6 different MERRA-2 collections")
        print("Expected processing time: 5-10 minutes for one week of data")
        
        try:
            output_path = processor.process_timerange(
                start_date=start_date,
                end_date=end_date,
                temporal_resolution=temporal_resolution
            )
            print(f"✅ Data processed successfully: {output_path}")
            
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            print("This is expected if NASA Earthdata credentials are not provided")
            
            # Create a dummy dataset for demonstration
            print("\n📝 Creating dummy dataset for demonstration...")
            create_dummy_dataset(output_dir, start_date, end_date, temporal_resolution)
            output_path = Path(output_dir) / f"merra2_prithvi_{start_date}_{end_date}_{temporal_resolution}.npz"
        
        # Step 3: Load and examine the dataset
        print("\n3. Loading and examining the processed dataset...")
        
        # Load dataset info
        info = MERRA2DataLoader.load_dataset_info(output_path)
        print(f"📊 Dataset info:")
        print(f"   Surface variables: {len(info['surface_vars'])}")
        print(f"   Static variables: {len(info['static_vars'])}")
        print(f"   Vertical variables: {len(info['vertical_vars'])}")
        
        if 'surface_shape' in info:
            print(f"   Surface data shape: {info['surface_shape']}")
        if 'vertical_shape' in info:
            print(f"   Vertical data shape: {info['vertical_shape']}")
        if 'static_shape' in info:
            print(f"   Static data shape: {info['static_shape']}")
        
        # Step 4: Validate compatibility with PrithviWxC_Encoder
        print("\n4. Validating compatibility with PrithviWxC_Encoder...")
        validation = validate_dataset_compatibility(output_path)
        
        if validation['valid_dataset']:
            print("✅ Dataset is fully compatible with PrithviWxC_Encoder")
        else:
            print("⚠️  Dataset has some missing variables:")
            if validation['missing_surface_vars']:
                print(f"   Missing surface vars: {validation['missing_surface_vars']}")
            if validation['missing_static_vars']:
                print(f"   Missing static vars: {validation['missing_static_vars']}")
            if validation['missing_vertical_vars']:
                print(f"   Missing vertical vars: {validation['missing_vertical_vars']}")
        
        # Step 5: Create PyTorch dataset and dataloader
        print("\n5. Creating PyTorch dataset and dataloader...")
        
        # Create dataset
        dataset = PrithviMERRA2Dataset(
            dataset_path=output_path,
            input_time_steps=2,
            time_step_hours=6,
            lead_time_hours=6
        )
        print(f"✅ Dataset created with {len(dataset)} samples")
        
        # Create dataloader
        dataloader = MERRA2DataLoader.create_dataloader(
            dataset_path=output_path,
            batch_size=2,
            shuffle=False,
            num_workers=0
        )
        print("✅ DataLoader created")
        
        # Step 6: Test loading a batch
        print("\n6. Testing batch loading...")
        for i, batch in enumerate(dataloader):
            print(f"📦 Batch {i + 1}:")
            for key, value in batch.items():
                if isinstance(value, type(batch['x'])):  # torch.Tensor
                    print(f"   {key}: {value.shape} ({value.dtype})")
                else:
                    print(f"   {key}: {value}")
            
            if i >= 2:  # Only show first 3 batches
                break
        
        print("\n🎉 Basic usage example completed successfully!")
        print(f"\nProcessed dataset saved to: {output_path}")
        print("You can now use this dataset for training PrithviWxC_Encoder")
        
    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        import traceback
        traceback.print_exc()


def create_dummy_dataset(output_dir, start_date, end_date, temporal_resolution):
    """Create a dummy dataset for demonstration when downloads fail."""
    import numpy as np
    import json
    from datetime import datetime, timedelta
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate dummy data with realistic shapes
    n_times = 56  # One week of 3-hourly data
    n_lat, n_lon = 361, 576  # MERRA-2 spatial dimensions
    n_levels = 14
    
    # Surface variables (20)
    surface_data = np.random.randn(n_times, 20, n_lat, n_lon).astype(np.float32)
    
    # Static variables (4)
    static_data = np.random.rand(4, n_lat, n_lon).astype(np.float32)
    
    # Vertical variables (10 vars * 14 levels)
    vertical_data = np.random.randn(n_times, 10, n_levels, n_lat, n_lon).astype(np.float32)
    
    # Coordinates
    times = [datetime.fromisoformat(start_date) + timedelta(hours=3*i) for i in range(n_times)]
    lats = np.linspace(-90, 90, n_lat)
    lons = np.linspace(-180, 179.375, n_lon)
    levels = [34.0, 39.0, 41.0, 43.0, 44.0, 45.0, 48.0, 51.0, 53.0, 56.0, 63.0, 68.0, 71.0, 72.0]
    
    coordinates = {
        'time': [t.isoformat() for t in times],
        'lat': lats,
        'lon': lons,
        'levels': levels
    }
    
    # Variable names
    surface_vars = [
        "EFLUX", "GWETROOT", "HFLUX", "LAI", "LWGAB", "LWGEM", "LWTUP", 
        "PS", "QV2M", "SLP", "SWGNT", "SWTNT", "T2M", "TQI", "TQL", 
        "TQV", "TS", "U10M", "V10M", "Z0M"
    ]
    static_vars = ["FRACI", "FRLAND", "FROCEAN", "PHIS"]
    vertical_vars = ["CLOUD", "H", "OMEGA", "PL", "QI", "QL", "QV", "T", "U", "V"]
    
    # Save dataset
    output_path = Path(output_dir) / f"merra2_prithvi_{start_date}_{end_date}_{temporal_resolution}.npz"
    
    np.savez_compressed(
        output_path,
        surface=surface_data,
        static=static_data,
        vertical=vertical_data,
        coordinates=coordinates,
        surface_vars=surface_vars,
        static_vars=static_vars,
        vertical_vars=vertical_vars
    )
    
    # Save metadata
    metadata = {
        'surface_vars': surface_vars,
        'static_vars': static_vars,
        'vertical_vars': vertical_vars,
        'coordinates': coordinates,
        'creation_time': datetime.now().isoformat(),
        'format_version': '1.0',
        'note': 'This is a dummy dataset created for demonstration purposes'
    }
    
    metadata_path = output_path.with_suffix('.metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"✅ Dummy dataset created: {output_path}")


if __name__ == "__main__":
    main()

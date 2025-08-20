"""
Data Loader for Processed MERRA-2 Datasets

This module provides utilities for loading and using datasets processed
by the MERRA-2 dataset processor for PrithviWxC_Encoder.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class PrithviMERRA2Dataset(Dataset):
    """
    PyTorch Dataset for loading processed MERRA-2 data for PrithviWxC_Encoder.

    This dataset loads the numpy arrays created by merra2_dataset_processor.py
    and provides them in the format expected by PrithviWxC_Encoder.
    """

    def __init__(self,
                 dataset_path: Union[str, Path],
                 input_time_steps: int = 2,
                 time_step_hours: int = 6,
                 lead_time_hours: int = 6,
                 transform: Optional[callable] = None,
                 normalize: bool = True):
        """
        Initialize the dataset.

        Args:
            dataset_path: Path to the processed .npz dataset file
            input_time_steps: Number of input time steps
            time_step_hours: Hours between input time steps
            lead_time_hours: Hours for lead time prediction
            transform: Optional transform to apply to data
            normalize: Whether to normalize the data
        """
        self.dataset_path = Path(dataset_path)
        self.input_time_steps = input_time_steps
        self.time_step_hours = time_step_hours
        self.lead_time_hours = lead_time_hours
        self.transform = transform
        self.normalize = normalize

        # Load dataset
        self._load_dataset()

        # Calculate valid time indices for sampling
        self._calculate_valid_indices()

    def _load_dataset(self):
        """Load the processed dataset from disk."""
        logger.info(f"Loading dataset from {self.dataset_path}")

        # Load main data
        data = np.load(self.dataset_path, allow_pickle=True)

        self.surface_data = data['surface']      # (time, var, lat, lon)
        self.static_data = data['static']        # (var, lat, lon)
        self.vertical_data = data['vertical']    # (time, var, level, lat, lon)
        self.coordinates = data['coordinates'].item()
        self.surface_vars = data['surface_vars'].tolist()
        self.static_vars = data['static_vars'].tolist()
        self.vertical_vars = data['vertical_vars'].tolist()

        # Load metadata if available
        metadata_path = self.dataset_path.with_suffix('.metadata.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        # Convert time coordinates to useful format
        self.times = self.coordinates['time']
        if isinstance(self.times[0], str):
            # Convert string timestamps to datetime
            import pandas as pd
            self.times = pd.to_datetime(self.times)

        logger.info(f"Loaded dataset with {len(self.times)} time steps")
        logger.info(f"Surface variables: {self.surface_vars}")
        logger.info(f"Static variables: {self.static_vars}")
        logger.info(f"Vertical variables: {self.vertical_vars}")

        if self.surface_data is not None:
            logger.info(f"Surface data shape: {self.surface_data.shape}")
        if self.vertical_data is not None:
            logger.info(f"Vertical data shape: {self.vertical_data.shape}")
        if self.static_data is not None:
            logger.info(f"Static data shape: {self.static_data.shape}")

    def _calculate_valid_indices(self):
        """Calculate valid time indices for creating samples."""
        n_times = len(self.times)

        # Need enough past time steps and lead time
        min_past_steps = (self.input_time_steps - 1) * (self.time_step_hours // 3)  # Assuming 3-hour data
        min_lead_steps = self.lead_time_hours // 3

        self.valid_indices = list(range(min_past_steps, n_times - min_lead_steps))

        logger.info(f"Valid indices: {len(self.valid_indices)} out of {n_times} time steps")

    def __len__(self) -> int:
        """Return number of valid samples."""
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary with tensors in PrithviWxC_Encoder format
        """
        # Get the target time index
        target_idx = self.valid_indices[idx]

        # Calculate input time indices
        time_step_indices = self.time_step_hours // 3  # Convert to 3-hour steps
        input_indices = [
            target_idx - (self.input_time_steps - 1 - i) * time_step_indices
            for i in range(self.input_time_steps)
        ]

        # Calculate lead time index
        lead_step_indices = self.lead_time_hours // 3
        lead_idx = target_idx + lead_step_indices

        # Extract input data
        sample = {}

        # Surface data: (time, var, lat, lon) -> (time, var, lat, lon)
        if self.surface_data is not None:
            surface_input = self.surface_data[input_indices]  # (input_time_steps, var, lat, lon)
            sample['x'] = torch.from_numpy(surface_input).float()

        # Vertical data: (time, var, level, lat, lon) -> (time, var*level, lat, lon)
        if (self.vertical_data is not None and
            self.vertical_data.size > 0 and
            len(self.vertical_data.shape) > 0 and
            self.vertical_data.shape != ()):
            vertical_input = self.vertical_data[input_indices]  # (input_time_steps, var, level, lat, lon)
            # Flatten var and level dimensions
            n_time, n_var, n_level, n_lat, n_lon = vertical_input.shape
            vertical_flat = vertical_input.reshape(n_time, n_var * n_level, n_lat, n_lon)

            # Concatenate with surface data if it exists
            if 'x' in sample:
                sample['x'] = torch.cat([sample['x'], torch.from_numpy(vertical_flat).float()], dim=1)
            else:
                sample['x'] = torch.from_numpy(vertical_flat).float()

        # Static data: (var, lat, lon) -> (var, lat, lon)
        if (self.static_data is not None and
            self.static_data.size > 0 and
            len(self.static_data.shape) > 0 and
            self.static_data.shape != ()):
            sample['static'] = torch.from_numpy(self.static_data).float()

        # Time information
        input_time = (input_indices[-1] - input_indices[0]) * 3  # Hours between first and last input
        lead_time = (lead_idx - input_indices[-1]) * 3  # Hours from last input to target

        sample['input_time'] = torch.tensor([input_time], dtype=torch.float32)
        sample['lead_time'] = torch.tensor([lead_time], dtype=torch.float32)

        # Target data (for training/validation)
        if self.surface_data is not None:
            target_surface = self.surface_data[lead_idx]  # (var, lat, lon)

            if (self.vertical_data is not None and
                self.vertical_data.size > 0 and
                len(self.vertical_data.shape) > 0 and
                self.vertical_data.shape != ()):
                target_vertical = self.vertical_data[lead_idx]  # (var, level, lat, lon)
                # Flatten and concatenate
                target_vertical_flat = target_vertical.reshape(-1, *target_vertical.shape[-2:])
                target_combined = np.concatenate([target_surface, target_vertical_flat], axis=0)
                sample['target'] = torch.from_numpy(target_combined).float()
            else:
                sample['target'] = torch.from_numpy(target_surface).float()

        # Apply transforms if provided
        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_variable_info(self) -> Dict[str, List[str]]:
        """Get information about variables in the dataset."""
        return {
            'surface_vars': self.surface_vars,
            'static_vars': self.static_vars,
            'vertical_vars': self.vertical_vars,
            'coordinates': self.coordinates
        }

    def get_spatial_coords(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get latitude and longitude coordinates."""
        return self.coordinates['lat'], self.coordinates['lon']

    def get_time_range(self) -> Tuple[str, str]:
        """Get the time range of the dataset."""
        return str(self.times[0]), str(self.times[-1])


class MERRA2DataLoader:
    """
    Utility class for creating data loaders for processed MERRA-2 datasets.
    """

    @staticmethod
    def create_dataloader(dataset_path: Union[str, Path],
                         batch_size: int = 1,
                         shuffle: bool = True,
                         num_workers: int = 0,
                         **dataset_kwargs) -> torch.utils.data.DataLoader:
        """
        Create a PyTorch DataLoader for the dataset.

        Args:
            dataset_path: Path to processed dataset
            batch_size: Batch size for data loader
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            **dataset_kwargs: Additional arguments for PrithviMERRA2Dataset

        Returns:
            PyTorch DataLoader
        """
        dataset = PrithviMERRA2Dataset(dataset_path, **dataset_kwargs)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

        return dataloader

    @staticmethod
    def load_dataset_info(dataset_path: Union[str, Path]) -> Dict:
        """
        Load metadata and basic info about a processed dataset.

        Args:
            dataset_path: Path to processed dataset

        Returns:
            Dictionary with dataset information
        """
        dataset_path = Path(dataset_path)

        # Load basic data info
        data = np.load(dataset_path, allow_pickle=True)

        info = {
            'dataset_path': str(dataset_path),
            'surface_vars': data['surface_vars'].tolist() if 'surface_vars' in data else [],
            'static_vars': data['static_vars'].tolist() if 'static_vars' in data else [],
            'vertical_vars': data['vertical_vars'].tolist() if 'vertical_vars' in data else [],
            'coordinates': data['coordinates'].item() if 'coordinates' in data else {},
        }

        # Add shape information
        if 'surface' in data and data['surface'] is not None:
            info['surface_shape'] = data['surface'].shape
        if 'vertical' in data and data['vertical'] is not None:
            info['vertical_shape'] = data['vertical'].shape
        if 'static' in data and data['static'] is not None:
            info['static_shape'] = data['static'].shape

        # Load metadata if available
        metadata_path = dataset_path.with_suffix('.metadata.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                info['metadata'] = json.load(f)

        return info


def load_multiple_datasets(dataset_paths: List[Union[str, Path]],
                          **dataloader_kwargs) -> List[torch.utils.data.DataLoader]:
    """
    Load multiple processed datasets and return data loaders.

    Args:
        dataset_paths: List of paths to processed datasets
        **dataloader_kwargs: Arguments for DataLoader creation

    Returns:
        List of PyTorch DataLoaders
    """
    dataloaders = []

    for path in dataset_paths:
        logger.info(f"Loading dataset from {path}")
        dataloader = MERRA2DataLoader.create_dataloader(path, **dataloader_kwargs)
        dataloaders.append(dataloader)

    return dataloaders


def validate_dataset_compatibility(dataset_path: Union[str, Path]) -> Dict[str, bool]:
    """
    Validate that a processed dataset is compatible with PrithviWxC_Encoder.

    Args:
        dataset_path: Path to processed dataset

    Returns:
        Dictionary with validation results
    """
    import config

    info = MERRA2DataLoader.load_dataset_info(dataset_path)

    # Required variables from PrithviWxC_Encoder
    required_surface = list(config.VARIABLE_INFO['surface_vars'].keys())
    required_static = list(config.VARIABLE_INFO['static_vars'].keys())
    required_vertical = list(config.VARIABLE_INFO['vertical_vars'].keys())

    results = {
        'valid_dataset': True,
        'surface_vars_complete': set(required_surface).issubset(set(info['surface_vars'])),
        'static_vars_complete': set(required_static).issubset(set(info['static_vars'])),
        'vertical_vars_complete': set(required_vertical).issubset(set(info['vertical_vars'])),
        'missing_surface_vars': list(set(required_surface) - set(info['surface_vars'])),
        'missing_static_vars': list(set(required_static) - set(info['static_vars'])),
        'missing_vertical_vars': list(set(required_vertical) - set(info['vertical_vars'])),
    }

    # Overall validity
    results['valid_dataset'] = (
        results['surface_vars_complete'] and
        results['static_vars_complete'] and
        results['vertical_vars_complete']
    )

    return results


# Example usage and testing functions
def example_usage():
    """Example of how to use the data loader."""

    # Example dataset path
    dataset_path = "processed_data/merra2_prithvi_2020-01-01_2020-01-31_3H.npz"

    # Create dataset
    dataset = PrithviMERRA2Dataset(
        dataset_path=dataset_path,
        input_time_steps=2,
        time_step_hours=6,
        lead_time_hours=6
    )

    # Create data loader
    dataloader = MERRA2DataLoader.create_dataloader(
        dataset_path=dataset_path,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )

    # Test loading a batch
    for batch in dataloader:
        print("Batch keys:", batch.keys())
        if 'x' in batch:
            print("Input shape:", batch['x'].shape)
        if 'static' in batch:
            print("Static shape:", batch['static'].shape)
        if 'target' in batch:
            print("Target shape:", batch['target'].shape)
        break  # Just test one batch

    # Get dataset info
    info = MERRA2DataLoader.load_dataset_info(dataset_path)
    print("Dataset info:", info)

    # Validate compatibility
    validation = validate_dataset_compatibility(dataset_path)
    print("Validation results:", validation)


if __name__ == "__main__":
    example_usage()

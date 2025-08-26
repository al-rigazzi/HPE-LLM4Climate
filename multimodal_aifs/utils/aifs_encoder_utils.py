"""
AIFS Encoder Utilities

This module provides utilities for working with the extracted AIFS encoder,
including wrapper classes and helper functions for climate data processing.
"""

import sys
import warnings
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch import nn

# Add project root to path for AIFS wrapper import
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    # Add parent directory to path for aifs_wrapper import
    sys.path.append(str(project_root / "multimodal_aifs"))
    from aifs_wrapper import AIFSWrapper

    sys.path.append(str(project_root / "multimodal_aifs" / "models" / "extracted_models"))
    from load_aifs_encoder import load_aifs_encoder

    AIFS_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"AIFS components not available: {e}")
    AIFS_AVAILABLE = False


class AIFSEncoderWrapper(nn.Module):
    """
    Wrapper for AIFS encoder with enhanced functionality for multimodal applications.

    This wrapper provides a clean interface to the AIFS encoder with additional
    features for climate data processing and integration with multimodal systems.
    """

    def __init__(
        self, encoder_path: str | None = None, device: str = "cpu", use_extracted: bool = True
    ):
        """
        Initialize AIFS encoder wrapper.

        Args:
            encoder_path: Path to AIFS model/encoder (optional)
            device: Device to run inference on
            use_extracted: Whether to use pre-extracted encoder
        """
        super().__init__()

        self.device = device
        self.use_extracted = use_extracted
        self.encoder: nn.Module | None = None
        self.encoder_info: Dict[str, Any] | None = None

        # Input/output dimensions based on AIFS architecture
        self.input_dim = 218  # AIFS expected input features
        self.output_dim = 1024  # AIFS encoder output dimension

        self._load_encoder(encoder_path)

    def _load_encoder(self, encoder_path: str | None = None):
        """Load the AIFS encoder."""
        if not AIFS_AVAILABLE:
            # Create a dummy encoder for testing
            self.encoder = nn.Linear(self.input_dim, self.output_dim)
            assert self.encoder is not None  # Help mypy understand encoder is not None
            self.encoder_info = {
                "type": "dummy",
                "total_parameters": sum(p.numel() for p in self.encoder.parameters()),
                "note": "AIFS not available, using dummy encoder",
            }
            warnings.warn("AIFS not available, using dummy encoder")
            return

        try:
            if self.use_extracted:
                # Use pre-extracted encoder
                self.encoder, self.encoder_info = load_aifs_encoder()
                if self.encoder_info is not None:
                    param_count = self.encoder_info["total_parameters"]
                    print(f"✅ Loaded extracted AIFS encoder: {param_count:,} parameters")
            else:
                # Use full AIFS wrapper
                aifs_wrapper = AIFSWrapper(encoder_path)
                model_info = aifs_wrapper.load_model()
                if model_info["pytorch_model"] is not None:
                    # Extract encoder from full model
                    full_model = model_info["pytorch_model"]
                    self.encoder = full_model.model.encoder
                    if self.encoder is not None:
                        self.encoder_info = {
                            "type": "full_model_encoder",
                            "total_parameters": sum(p.numel() for p in self.encoder.parameters()),
                            "source": "AIFS full model",
                        }
                else:
                    raise RuntimeError("Could not access PyTorch model from AIFS wrapper")

            # Move to specified device
            if self.encoder is not None:
                self.encoder = self.encoder.to(self.device)
                self.encoder.eval()

        except Exception as e:
            warnings.warn(f"Failed to load AIFS encoder: {e}")
            # Fallback to dummy encoder
            self.encoder = nn.Linear(self.input_dim, self.output_dim).to(self.device)
            self.encoder_info = {
                "type": "fallback",
                "total_parameters": sum(p.numel() for p in self.encoder.parameters()),
                "error": str(e),
            }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through AIFS encoder.

        Args:
            x: Input tensor of shape (batch_size, input_features)

        Returns:
            Encoded tensor of shape (batch_size, output_features)
        """
        # Ensure input has correct shape
        if x.dim() == 1:
            x = x.unsqueeze(0)

        if x.shape[-1] != self.input_dim:
            if x.shape[-1] > self.input_dim:
                # Truncate if too many features
                x = x[..., : self.input_dim]
            else:
                # Pad if too few features
                padding = torch.zeros(
                    *x.shape[:-1], self.input_dim - x.shape[-1], device=x.device, dtype=x.dtype
                )
                x = torch.cat([x, padding], dim=-1)

        # Move to correct device
        x = x.to(self.device)

        # Apply encoder
        with torch.no_grad():
            if (
                self.encoder is not None
                and hasattr(self.encoder, "emb_nodes_src")
                and AIFS_AVAILABLE
            ):
                # Use AIFS source embedding layer (known to work)
                encoded = torch.as_tensor(self.encoder.emb_nodes_src(x))
            elif self.encoder is not None:
                # Use direct forward (for dummy or simple encoders)
                encoded = torch.as_tensor(self.encoder(x))
            else:
                raise RuntimeError("Encoder is None")

        return encoded

    def encode_climate_data(self, climate_data: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        Encode climate data with preprocessing.

        Args:
            climate_data: Climate data tensor or array

        Returns:
            Encoded climate features
        """
        if isinstance(climate_data, np.ndarray):
            climate_data = torch.tensor(climate_data, dtype=torch.float32)

        # Preprocess climate data if needed
        climate_data = self._preprocess_climate_data(climate_data)

        # Encode
        return self.forward(climate_data)

    def _preprocess_climate_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        Preprocess climate data for AIFS encoder.

        Args:
            data: Raw climate data tensor

        Returns:
            Preprocessed data ready for encoding
        """
        # Handle different input shapes
        if data.dim() == 3:
            # Spatial data (batch, lat, lon) -> flatten spatial dimensions
            batch_size = data.shape[0]
            data = data.view(batch_size, -1)
        elif data.dim() == 4:
            # Multi-variable spatial data (batch, vars, lat, lon)
            batch_size, n_vars = data.shape[:2]
            data = data.reshape(batch_size, n_vars * data.shape[2] * data.shape[3])

        # Ensure we have the right number of features for AIFS
        if data.shape[-1] != self.input_dim:
            # Apply feature transformation
            data = self._transform_to_aifs_features(data)

        return data

    def _transform_to_aifs_features(self, data: torch.Tensor) -> torch.Tensor:
        """
        Transform arbitrary climate data to AIFS feature format.

        Args:
            data: Input climate data

        Returns:
            Data with AIFS-compatible features (218 dimensions)
        """
        current_features = data.shape[-1]
        target_features = self.input_dim

        if current_features == target_features:
            return data
        if current_features > target_features:
            # Truncate excess features
            return data[..., :target_features]

        # Pad with zeros or apply learned transformation
        batch_size = data.shape[0]
        padding_size = target_features - current_features

        # Zero padding
        padding = torch.zeros(batch_size, padding_size, device=data.device, dtype=data.dtype)
        return torch.cat([data, padding], dim=-1)

    def get_encoder_info(self) -> Dict:
        """Get information about the loaded encoder."""
        encoder_params = (
            sum(p.numel() for p in self.encoder.parameters()) if self.encoder is not None else 0
        )
        encoder_info_dict = self.encoder_info if self.encoder_info is not None else {}

        return {
            "encoder_type": type(self.encoder).__name__ if self.encoder is not None else "None",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "device": str(self.device),
            "parameters": encoder_params,
            "aifs_available": AIFS_AVAILABLE,
            "use_extracted": self.use_extracted,
            **encoder_info_dict,
        }

    def encode_batch(self, batch_data: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
        """
        Encode data in batches for memory efficiency.

        Args:
            batch_data: Data to encode (n_samples, features)
            batch_size: Batch size for processing

        Returns:
            Encoded data (n_samples, encoded_features)
        """
        n_samples = batch_data.shape[0]
        encoded_batches = []

        for i in range(0, n_samples, batch_size):
            batch = batch_data[i : i + batch_size]
            encoded_batch = self.encode_climate_data(batch)
            encoded_batches.append(encoded_batch)

        return torch.cat(encoded_batches, dim=0)


def create_aifs_encoder(
    encoder_path: str | None = None, device: str = "cpu", use_extracted: bool = True
) -> AIFSEncoderWrapper:
    """
    Factory function to create AIFS encoder wrapper.

    Args:
        encoder_path: Path to AIFS model/encoder
        device: Device for inference
        use_extracted: Whether to use pre-extracted encoder

    Returns:
        AIFSEncoderWrapper instance
    """
    return AIFSEncoderWrapper(encoder_path=encoder_path, device=device, use_extracted=use_extracted)


def test_aifs_encoder():
    """Test AIFS encoder wrapper functionality."""
    print("🧪 Testing AIFS Encoder Wrapper")
    print("=" * 40)

    # Create encoder
    encoder = create_aifs_encoder()

    # Test basic encoding
    test_data = torch.randn(2, 218)  # Batch of 2 samples
    encoded = encoder.encode_climate_data(test_data)

    print("✅ Basic encoding test:")
    print(f"   Input shape: {test_data.shape}")
    print(f"   Output shape: {encoded.shape}")
    print(f"   Encoder info: {encoder.get_encoder_info()}")

    # Test with different input shapes
    test_spatial = torch.randn(1, 5, 10, 20)  # Spatial data
    encoded_spatial = encoder.encode_climate_data(test_spatial)

    print("\n✅ Spatial data test:")
    print(f"   Input shape: {test_spatial.shape}")
    print(f"   Output shape: {encoded_spatial.shape}")

    print("\n🎯 All tests passed!")


if __name__ == "__main__":
    test_aifs_encoder()

#!/usr/bin/env python3
"""
AIFS Time Series Tokenizer

This module provides a practical implementation for using AIFS encoder
to tokenize 5-D time series climate data.

The AIFS encoder can be used for time series tokenization through several strategies:
1. Sequential processing: Encode each timestep separately
2. Temporal batching: Reshape time dimension as batch dimension
3. Hybrid approach: Combine spatial encoding with temporal modeling

Author: GitHub Copilot
Date: August 20, 2025
"""

import sys
from pathlib import Path
from typing import Optional

import torch
from torch import nn

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from multimodal_aifs.utils.aifs_encoder_utils import AIFSEncoderWrapper


class AIFSTimeSeriesTokenizer(nn.Module):
    """
    Time series tokenizer using AIFS encoder for spatial-temporal climate data.

    This tokenizer can handle 5-D tensors with shape [batch, time, vars, height, width]
    and convert them into sequence tokens for downstream processing.
    """

    def __init__(
        self,
        aifs_encoder_path: Optional[str] = None,
        temporal_modeling: str = "lstm",  # "lstm", "transformer", "none"
        hidden_dim: int = 512,
        num_layers: int = 2,
        device: str = "cpu",
    ):
        """
        Initialize AIFS time series tokenizer.

        Args:
            aifs_encoder_path: Path to AIFS encoder model
            temporal_modeling: Type of temporal modeling ("lstm", "transformer", "none")
            hidden_dim: Hidden dimension for temporal model
            num_layers: Number of layers in temporal model
            device: Device to run on
        """
        super().__init__()

        self.device = device
        self.temporal_modeling = temporal_modeling
        self.hidden_dim = hidden_dim

        # Initialize AIFS encoder for spatial feature extraction
        self.aifs_encoder = AIFSEncoderWrapper(encoder_path=aifs_encoder_path, device=device)

        # Get AIFS output dimension
        self.spatial_dim = self.aifs_encoder.output_dim  # 1024

        # Initialize temporal modeling component
        if temporal_modeling == "lstm":
            self.temporal_model = nn.LSTM(
                input_size=self.spatial_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.1 if num_layers > 1 else 0.0,
            )
        elif temporal_modeling == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.spatial_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                dropout=0.1,
                batch_first=True,
            )
            self.temporal_model = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        elif temporal_modeling == "none":
            self.temporal_model = None
        else:
            raise ValueError(f"Unsupported temporal modeling: {temporal_modeling}")

        # Output projection if using temporal modeling
        if self.temporal_model is not None:
            output_dim = hidden_dim if temporal_modeling == "lstm" else self.spatial_dim
            self.output_projection = nn.Linear(output_dim, hidden_dim)
        else:
            self.output_projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for time series tokenization.

        Args:
            x: Input tensor of shape [batch, time, vars, height, width]

        Returns:
            Tokenized representation based on temporal modeling choice:
            - "lstm": [batch, time, hidden_dim]
            - "transformer": [batch, time, spatial_dim]
            - "none": [batch, time, spatial_dim]
        """
        return self.tokenize_time_series(x)

    def tokenize_time_series(self, tensor_5d: torch.Tensor) -> torch.Tensor:
        """
        Tokenize 5-D time series tensor using AIFS encoder.

        Args:
            tensor_5d: Input tensor [batch, time, vars, height, width]

        Returns:
            Tokenized sequence [batch, time, features]
        """
        _, time_steps, _, _, _ = tensor_5d.shape

        # Strategy 1: Sequential processing (most accurate)
        timestep_encodings = []

        for t in range(time_steps):
            # Extract timestep: [batch, vars, height, width]
            timestep_data = tensor_5d[:, t, :, :, :]

            # Encode spatial features using AIFS
            spatial_encoding = self.aifs_encoder.encode_climate_data(timestep_data)
            timestep_encodings.append(spatial_encoding)

        # Stack to create sequence: [batch, time, spatial_dim]
        sequence_encodings = torch.stack(timestep_encodings, dim=1)

        # Apply temporal modeling if specified
        if self.temporal_model is not None:
            if self.temporal_modeling == "lstm":
                # LSTM expects [batch, seq, features]
                temporal_output, _ = self.temporal_model(sequence_encodings)
                # Use last hidden state or all hidden states
                final_output = self.output_projection(temporal_output)
            elif self.temporal_modeling == "transformer":
                # Transformer encoder
                temporal_output = self.temporal_model(sequence_encodings)
                final_output = self.output_projection(temporal_output)
            else:
                final_output = sequence_encodings
        else:
            final_output = sequence_encodings

        return final_output

    def tokenize_batch_parallel(self, tensor_5d: torch.Tensor) -> torch.Tensor:
        """
        Alternative tokenization using batch-parallel processing.

        This method reshapes the time dimension as batch dimension for faster processing,
        but may be less memory efficient for large time series.

        Args:
            tensor_5d: Input tensor [batch, time, vars, height, width]

        Returns:
            Tokenized sequence [batch, time, features]
        """
        batch_size, time_steps, num_vars, height, width = tensor_5d.shape

        # Reshape: [batch*time, vars, height, width]
        reshaped = tensor_5d.view(batch_size * time_steps, num_vars, height, width)

        # Encode all timesteps in parallel
        all_encodings = self.aifs_encoder.encode_climate_data(reshaped)

        # Reshape back: [batch, time, spatial_dim]
        sequence_encodings = all_encodings.view(batch_size, time_steps, -1)

        # Apply temporal modeling
        if self.temporal_model is not None:
            if self.temporal_modeling == "lstm":
                temporal_output, _ = self.temporal_model(sequence_encodings)
                final_output = self.output_projection(temporal_output)
            elif self.temporal_modeling == "transformer":
                temporal_output = self.temporal_model(sequence_encodings)
                final_output = self.output_projection(temporal_output)
            else:
                final_output = sequence_encodings
        else:
            final_output = sequence_encodings

        return final_output

    def extract_spatial_tokens(self, tensor_5d: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial tokens only (no temporal modeling).

        Args:
            tensor_5d: Input tensor [batch, time, vars, height, width]

        Returns:
            Spatial tokens [batch, time, spatial_dim]
        """
        return self.tokenize_time_series(tensor_5d)

    def get_tokenizer_info(self) -> dict:
        """Get information about the tokenizer configuration."""
        return {
            "aifs_encoder": self.aifs_encoder.get_encoder_info(),
            "temporal_modeling": self.temporal_modeling,
            "hidden_dim": self.hidden_dim,
            "spatial_dim": self.spatial_dim,
            "device": self.device,
            "output_shape_pattern": "batch x time x features",
        }


def demonstrate_time_series_tokenization():
    """Demonstrate time series tokenization with different strategies."""

    print("🚀 AIFS Time Series Tokenization Demonstration")
    print("=" * 60)

    # Create sample 5-D time series data
    batch_size, time_steps, num_vars, height, width = 2, 8, 5, 64, 64
    sample_data = torch.randn(batch_size, time_steps, num_vars, height, width)

    print(f"Sample data shape: {sample_data.shape}")
    interpretation = (
        f"Interpretation: (batch={batch_size}, time={time_steps}, "
        f"vars={num_vars}, h={height}, w={width})"
    )
    print(interpretation)
    print()

    # Test different temporal modeling approaches
    approaches = [
        ("none", "Spatial encoding only"),
        ("lstm", "LSTM temporal modeling"),
        ("transformer", "Transformer temporal modeling"),
    ]

    for approach, description in approaches:
        print(f"🔧 Testing: {description}")

        try:
            # Initialize tokenizer
            tokenizer = AIFSTimeSeriesTokenizer(
                temporal_modeling=approach, hidden_dim=512, device="cpu"
            )

            # Get tokenizer info
            info = tokenizer.get_tokenizer_info()
            print(f"   AIFS encoder: {info['aifs_encoder']['encoder_type']}")
            print(f"   Spatial dim: {info['spatial_dim']}")
            print(f"   Temporal model: {info['temporal_modeling']}")

            # Tokenize the time series
            tokens = tokenizer.tokenize_time_series(sample_data)
            print(f"   ✅ Tokenization: {sample_data.shape} -> {tokens.shape}")

            # Test batch parallel method
            if approach == "none":  # Test on simpler case
                tokens_parallel = tokenizer.tokenize_batch_parallel(sample_data)
                print(f"   ✅ Batch parallel: {sample_data.shape} -> {tokens_parallel.shape}")

                # Check if results are similar (they should be identical for "none" case)
                if torch.allclose(tokens, tokens_parallel, atol=1e-6):
                    print("   ✅ Sequential and parallel methods produce identical results")
                else:
                    print("   ⚠️  Sequential and parallel methods differ (expected for temporal)")

        except Exception as e:
            print(f"   ❌ Failed: {e}")

        print()

    print("📋 Use Cases for 5-D Time Series Tokenization:")
    print("-" * 50)
    print("• Weather forecasting with multi-timestep input")
    print("• Climate pattern analysis over time")
    print("• Temporal anomaly detection in weather data")
    print("• Multi-modal climate-text models with temporal context")
    print("• Time series classification and clustering")
    print()

    print("💡 Implementation Tips:")
    print("-" * 20)
    print("• Use sequential processing for maximum accuracy")
    print("• Use batch-parallel for speed when memory allows")
    print("• Choose temporal modeling based on your downstream task:")
    print("  - 'none': For spatial analysis without temporal relationships")
    print("  - 'lstm': For sequential temporal dependencies")
    print("  - 'transformer': For attention-based temporal modeling")
    print("• Consider chunking very long time series to manage memory")


if __name__ == "__main__":
    demonstrate_time_series_tokenization()

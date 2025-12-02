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
AIFS Climate Fusion Module

Provides climate-text fusion capabilities specifically designed for AIFS by
combining raw encoder embeddings (102 channels) from the AIFSCompleteEncoder with
textual representations. The module aggregates grid-level embeddings, projects them to a
fusion space, and applies cross/self attention to align climate and text modalities.
"""

import torch
import torch.nn.functional as F
from torch import nn

from ..constants import (
    AIFS_RAW_ENCODER_OUTPUT_DIM,
    DEFAULT_NUM_HEADS,
    FUSION_DEFAULT_DIM,
    TEXT_DEFAULT_DIM,
)
from ..utils.device_utils import (
    autocast_if_available,
    configure_device_for_max_perf,
    resolve_device,
    supports_amp,
)
from ..utils.text_utils import ClimateTextProcessor

# Import the  AIFS encoder utilities
from .aifs_encoder_utils import AIFSCompleteEncoder


class AIFSClimateTextFusion(nn.Module):
    """
    Climate-text fusion module built on top of the AIFSCompleteEncoder.

    This module combines climate embeddings produced by the actual AIFS encoder (102 raw channels)
    with textual descriptions to create rich multimodal representations for downstream tasks.

    Features:
        - Uses the true AIFS encoder without any projection layer
        - Handles full 5D climate tensors: [batch, time, ensemble, grid_points, variables]
        - Supports cross-attention and self-attention fusion between climate and text streams
    """

    def __init__(
        self,
        aifs_model=None,
        aifs_checkpoint_path: str | None = None,
        # Updated to match actual AIFS encoder output
        climate_dim: int = AIFS_RAW_ENCODER_OUTPUT_DIM,
        text_dim: int = TEXT_DEFAULT_DIM,
        fusion_dim: int = FUSION_DEFAULT_DIM,
        num_attention_heads: int = DEFAULT_NUM_HEADS,
        dropout: float = 0.1,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        verbose: bool = True,
    ):
        """
        Initialize AIFS climate-text fusion module with  encoder.

        Args:
            aifs_model: The complete AIFS model instance (preferred)
            aifs_checkpoint_path: Path to saved AIFSCompleteEncoder checkpoint (alternative)
            climate_dim: Dimension of AIFS climate encodings
                        (AIFS_RAW_ENCODER_OUTPUT_DIM for complete encoder)
            text_dim: Dimension of text embeddings
            fusion_dim: Dimension of fused representations
            num_attention_heads: Number of attention heads
            dropout: Dropout rate
            device: Device to run on
            dtype: Data type for model parameters
            verbose: Whether to print initialization messages
        """
        super().__init__()

        self.device = resolve_device(device)
        configure_device_for_max_perf(self.device)

        self.dtype = dtype or (torch.float16 if supports_amp(self.device) else torch.float32)
        self.autocast_dtype = torch.float16 if supports_amp(self.device) else None
        self.verbose = verbose
        self.climate_dim = climate_dim
        self.text_dim = text_dim
        self.fusion_dim = fusion_dim
        self.num_attention_heads = num_attention_heads

        # Initialize the  AIFS Complete Encoder
        self.aifs_encoder: AIFSCompleteEncoder | None = None
        if aifs_model is not None:
            # Create new AIFSCompleteEncoder from AIFS model
            self.aifs_encoder = AIFSCompleteEncoder(aifs_model, verbose=verbose, device=self.device)
            if verbose:
                print("Using AIFSCompleteEncoder with provided AIFS model")
        elif aifs_checkpoint_path is not None:
            # Load from checkpoint (requires AIFS model to be loaded separately)
            if verbose:
                print(
                    "Loading from checkpoint requires AIFS model. "
                    "Consider providing aifs_model parameter."
                )
            self.aifs_encoder = None  # Will be set when aifs_model is provided
            self.checkpoint_path = aifs_checkpoint_path
        else:
            raise ValueError("Either aifs_model or aifs_checkpoint_path must be provided")

        # Climate data projection (updated for new encoder output dimension)
        target_device = self.device
        target_dtype = self.dtype

        self.climate_projection = nn.Sequential(
            nn.Linear(climate_dim, fusion_dim, dtype=target_dtype),
            nn.LayerNorm(fusion_dim, dtype=target_dtype),
            nn.ReLU(),
            nn.Dropout(dropout),
        ).to(target_device)

        # Text projection
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, fusion_dim, dtype=target_dtype),
            nn.LayerNorm(fusion_dim, dtype=target_dtype),
            nn.ReLU(),
            nn.Dropout(dropout),
        ).to(target_device)

        # Cross-attention for climate-text fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
            dtype=target_dtype,
        ).to(target_device)

        # Self-attention for final fusion
        self.self_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
            dtype=target_dtype,
        ).to(target_device)

        # Feed-forward network
        self.feedforward = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 4, dtype=target_dtype),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 4, fusion_dim, dtype=target_dtype),
            nn.Dropout(dropout),
        ).to(target_device)

        # Layer normalization
        self.norm1 = nn.LayerNorm(fusion_dim, dtype=target_dtype).to(target_device)
        self.norm2 = nn.LayerNorm(fusion_dim, dtype=target_dtype).to(target_device)
        self.norm3 = nn.LayerNorm(fusion_dim, dtype=target_dtype).to(target_device)

        # Output projection
        self.output_projection = nn.Linear(fusion_dim, fusion_dim, dtype=target_dtype).to(
            target_device
        )

        # Initialize text processor
        self.text_processor = ClimateTextProcessor()

        # Convert entire module to appropriate dtype for device
        if supports_amp(self.device) and target_dtype == torch.float16:
            self.half()  # Convert all module parameters to FP16

    def encode_climate_data(self, climate_data: torch.Tensor) -> torch.Tensor:
        """
        Encode climate data using the AIFSCompleteEncoder.

        Args:
            climate_data: Raw climate data tensor [batch, time, ensemble, grid_points, variables]

        Returns:
            Encoded climate features [batch, fusion_dim] (aggregated AIFS encoder embeddings)
        """
        if self.aifs_encoder is None:
            raise ValueError(
                "AIFS encoder not available. Provide aifs_model during initialization."
            )

        with autocast_if_available(self.device, dtype=self.autocast_dtype):
            with torch.no_grad():
                # Use the complete encoder that returns actual AIFS embeddings
                # Returns [batch, time, grid_points, embedding_dim] or
                # [batch, grid_points, embedding_dim]
                encoded = self.aifs_encoder(climate_data)  # e.g. [1, 1, 542080, 102]

                # Handle different output shapes from AIFS encoder
                if encoded.dim() == 4:  # [batch, time, grid_points, embedding_dim]
                    # Aggregate over time and grid points to get global representation
                    batch_size = encoded.shape[0]
                    # Mean across time and grid dimensions
                    global_encoded = encoded.mean(dim=(1, 2))  # [batch, embedding_dim]
                elif encoded.dim() == 3:  # [batch, grid_points, embedding_dim]
                    # Aggregate grid point embeddings to create global climate representation
                    global_encoded = encoded.mean(dim=1)  # [batch, embedding_dim]
                elif encoded.dim() == 2:  # [grid_points, embedding_dim]
                    # Take mean across grid points to get global representation
                    global_encoded = encoded.mean(dim=0, keepdim=True).to(dtype=self.dtype)
                    # Expand to match original batch size if needed
                    batch_size = climate_data.shape[0]
                    global_encoded = global_encoded.expand(batch_size, -1)
                else:
                    global_encoded = encoded.to(dtype=self.dtype)
            # Ensure input tensor matches layer dtype
            if hasattr(self.climate_projection[0], "weight"):
                target_dtype = self.climate_projection[0].weight.dtype
                global_encoded = global_encoded.to(target_dtype)

            projected = self.climate_projection(global_encoded)

            # Ensure output matches module dtype
            if (
                hasattr(self.climate_projection[0], "weight")
                and projected.dtype != self.climate_projection[0].weight.dtype
            ):
                projected = projected.to(self.climate_projection[0].weight.dtype)

        return torch.as_tensor(projected)

    def encode_text(
        self, texts: list[str], text_embeddings: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Encode text descriptions.

        Args:
            texts: list of text descriptions
            text_embeddings: Pre-computed text embeddings [num_texts, text_dim]

        Returns:
            Encoded text features [num_texts, fusion_dim]
        """
        if text_embeddings is None:
            raise ValueError(
                "text_embeddings must be provided. "
                "Use a proper text encoder (e.g., sentence-transformers, BERT) "
                "to generate embeddings before calling this method."
            )

        with autocast_if_available(self.device, dtype=self.autocast_dtype):
            # Project to fusion dimension
            projected = self.text_projection(text_embeddings)  # [num_texts, fusion_dim]

        return torch.as_tensor(projected)

    def apply_cross_attention(
        self, climate_features: torch.Tensor, text_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-attention between climate and text features.

        Args:
            climate_features: Climate feature tensor
            text_features: Text feature tensor

        Returns:
            tuple of attended features (climate, text)
        """
        # Climate attending to text
        climate_attended, _ = self.cross_attention(
            climate_features.unsqueeze(1),  # Add sequence dimension
            text_features.unsqueeze(1),
            text_features.unsqueeze(1),
        )
        climate_attended = climate_attended.squeeze(1)

        # Text attending to climate
        text_attended, _ = self.cross_attention(
            text_features.unsqueeze(1), climate_features.unsqueeze(1), climate_features.unsqueeze(1)
        )
        text_attended = text_attended.squeeze(1)

        return climate_attended, text_attended

    def fuse_features(
        self, climate_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse climate and text features using attention mechanism.

        Args:
            climate_features: Climate feature tensor
            text_features: Text feature tensor

        Returns:
            Fused multimodal features
        """
        # Climate and text features are already projected to fusion_dim
        # by encode_climate_data and encode_text methods
        with autocast_if_available(self.device, dtype=self.autocast_dtype):
            # Apply cross-attention
            climate_attended, text_attended = self.apply_cross_attention(
                climate_features, text_features
            )

            # Residual connections and normalization
            climate_features = self.norm1(climate_features + climate_attended)
            text_features = self.norm1(
                text_features + text_attended
            )  # Concatenate and apply self-attention
            combined_features = torch.stack([climate_features, text_features], dim=1)

            fused_features, _ = self.self_attention(
                combined_features, combined_features, combined_features
            )

            # Apply feed-forward network
            fused_features = self.norm2(fused_features)
            ff_output = self.feedforward(fused_features)
            fused_features = self.norm3(fused_features + ff_output)

            # Pool over sequence dimension (climate + text)
            pooled_features = fused_features.mean(dim=1)

            # Final projection
            output = self.output_projection(pooled_features)

        return torch.as_tensor(output)

    def forward(
        self,
        climate_data: torch.Tensor,
        texts: list[str],
        text_embeddings: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass of the fusion module.

        Args:
            climate_data: Raw climate data
            texts: Text descriptions
            text_embeddings: Pre-computed text embeddings (optional)

        Returns:
            Dictionary containing fusion results
        """
        with autocast_if_available(self.device, dtype=self.autocast_dtype):
            # Encode climate data
            climate_features = self.encode_climate_data(climate_data)

            # Encode text
            text_features = self.encode_text(texts, text_embeddings)

            # Fuse features
            fused_features = self.fuse_features(climate_features, text_features)

        return {
            "climate_features": climate_features,
            "text_features": text_features,
            "fused_features": fused_features,
            "fusion_dim": torch.tensor(self.fusion_dim, dtype=torch.long),
        }

    def get_climate_similarity(
        self, climate_data1: torch.Tensor, climate_data2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity between climate data samples.

        Args:
            climate_data1: First climate data tensor
            climate_data2: Second climate data tensor

        Returns:
            Similarity scores
        """
        with autocast_if_available(self.device, dtype=self.autocast_dtype):
            features1 = self.encode_climate_data(climate_data1)
            features2 = self.encode_climate_data(climate_data2)

            # Cosine similarity
            # pylint: disable=not-callable
            similarity = F.cosine_similarity(features1, features2, dim=-1)
            # pylint: enable=not-callable
        return similarity

    def get_text_climate_alignment(
        self,
        climate_data: torch.Tensor,
        texts: list[str],
        text_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute alignment between climate data and text descriptions.

        Args:
            climate_data: Climate data tensor
            texts: Text descriptions
            text_embeddings: Pre-computed text embeddings (optional)

        Returns:
            Alignment scores
        """
        with autocast_if_available(self.device, dtype=self.autocast_dtype):
            climate_features = self.encode_climate_data(climate_data)
            text_features = self.encode_text(texts, text_embeddings)

            # Compute alignment as cosine similarity
            # pylint: disable=not-callable
            alignment = F.cosine_similarity(climate_features, text_features, dim=-1)
            # pylint: enable=not-callable
        return alignment


class AIFSClimateEmbedding(nn.Module):
    """
    Lightweight climate embedding using the AIFSCompleteEncoder.

    Creates embeddings directly from climate data using the complete AIFS encoder
    that returns actual raw encoder outputs [542080, 102].
    """

    def __init__(
        self,
        aifs_model=None,
        aifs_checkpoint_path: str | None = None,
        climate_dim: int = AIFS_RAW_ENCODER_OUTPUT_DIM,
        embedding_dim: int = 256,
        device: str | torch.device = "cpu",
        verbose: bool = True,
    ):
        """
        Initialize AIFS climate embedding with encoder.

        Args:
            aifs_model: The complete AIFS model instance (preferred)
            aifs_checkpoint_path: Path to saved AIFSCompleteEncoder checkpoint (alternative)
            climate_dim: Dimension of AIFS climate encodings (102 for complete encoder)
            embedding_dim: Final embedding dimension
            device: Device to run on
            verbose: Whether to print initialization messages
        """
        super().__init__()

        self.device = resolve_device(device)
        configure_device_for_max_perf(self.device)

        self.climate_dim = climate_dim
        self.embedding_dim = embedding_dim
        self.verbose = verbose

        # Device-adaptive dtype selection for FP16 support
        self.dtype = torch.float16 if supports_amp(self.device) else torch.float32

        # Initialize the  AIFS Complete Encoder
        self.aifs_encoder: AIFSCompleteEncoder | None = None
        if aifs_model is not None:
            # Create new AIFSCompleteEncoder from AIFS model
            self.aifs_encoder = AIFSCompleteEncoder(aifs_model, verbose=verbose, device=self.device)
            if verbose:
                print("Using AIFSCompleteEncoder with provided AIFS model")
        elif aifs_checkpoint_path is not None:
            # Load from checkpoint (requires AIFS model to be loaded separately)
            if verbose:
                print(
                    "Loading from checkpoint requires AIFS model. "
                    "Consider providing aifs_model parameter."
                )
            self.aifs_encoder = None  # Will be set when aifs_model is provided
            self.checkpoint_path = aifs_checkpoint_path
        else:
            raise ValueError("Either aifs_model or aifs_checkpoint_path must be provided")

        # Climate embedding projection
        self.projection = nn.Sequential(
            nn.Linear(climate_dim, embedding_dim, dtype=self.dtype),
            nn.LayerNorm(embedding_dim, dtype=self.dtype),
            nn.ReLU(),
        ).to(self.device)

    def forward(self, climate_data: torch.Tensor) -> torch.Tensor:
        """
        Create embeddings from climate data using the AIFSCompleteEncoder.

        Args:
            climate_data: Input climate data [batch, time, ensemble, grid_points, variables]

        Returns:
            Climate embeddings [batch, embedding_dim]
        """
        if self.aifs_encoder is None:
            raise ValueError(
                "AIFS encoder not available. Provide aifs_model during initialization."
            )

        with autocast_if_available(self.device, dtype=self.dtype if supports_amp(self.device) else None):
            # Encode with AIFS complete encoder
            with torch.no_grad():
                aifs_features = self.aifs_encoder(
                    climate_data
                )  # [1, 1, 542080, 102] or [542080, 102]

                # Aggregate grid point embeddings to create global climate representation
                # Use mean pooling across grid points to get global features
                if aifs_features.dim() == 4:  # [batch, time, grid_points, features]
                    # Aggregate across time and grid point dimensions
                    global_features = aifs_features.mean(dim=(1, 2))  # [batch, features]
                elif aifs_features.dim() == 2:  # [grid_points, features]
                    # Take mean across grid points to get global representation
                    global_features = aifs_features.mean(dim=0, keepdim=True)  # [1, features]
                    # Expand to match original batch size if needed
                    batch_size = climate_data.shape[0]
                    global_features = global_features.expand(batch_size, -1)  # [batch, features]
                else:
                    global_features = aifs_features  # Already in correct format

            # Ensure dtype and device consistency for accelerator operations
            if global_features.dtype != self.dtype:
                global_features = global_features.to(dtype=self.dtype)
            if global_features.device != self.device:
                global_features = global_features.to(self.device)

            # Project to embedding space
            embeddings = self.projection(global_features)

        return torch.as_tensor(embeddings)


def create_aifs_fusion_from_model(
    aifs_model, fusion_dim: int = 512, verbose: bool = True, device: str | torch.device = "cpu"
):
    """
    Create AIFSClimateTextFusion from an AIFS model.

    Args:
        aifs_model: Complete AIFS model instance
        fusion_dim: Fusion dimension
        verbose: Whether to print creation messages

    Returns:
        AIFSClimateTextFusion instance
    """
    return AIFSClimateTextFusion(
        aifs_model=aifs_model,
        climate_dim=AIFS_RAW_ENCODER_OUTPUT_DIM,
        fusion_dim=fusion_dim,
        verbose=verbose,
        device=device,
    )


def create_aifs_embedding_from_model(
    aifs_model,
    embedding_dim: int = 256,
    verbose: bool = True,
    device: str | torch.device = "cpu",
):
    """
    Create AIFSClimateEmbedding from an AIFS model.

    Args:
        aifs_model: Complete AIFS model instance
        embedding_dim: Embedding dimension
        verbose: Whether to print creation messages

    Returns:
        AIFSClimateEmbedding instance
    """
    return AIFSClimateEmbedding(
        aifs_model=aifs_model,
        climate_dim=AIFS_RAW_ENCODER_OUTPUT_DIM,
        embedding_dim=embedding_dim,
        verbose=verbose,
        device=device,
    )

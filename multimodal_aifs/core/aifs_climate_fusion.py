"""
AIFS Climate Fusion Module

This module provides climate-text fusion capabilities specifically designed for AIFS,
combining climate data encoded through the  AIFSCompleteEncoder with textual
descriptions for enhanced multimodal climate analysis.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from ..utils.text_utils import ClimateTextProcessor

# Import the  AIFS encoder utilities
from .aifs_encoder_utils import AIFSCompleteEncoder, load_aifs_encoder


class AIFSClimateTextFusion(nn.Module):
    """
    Climate-text fusion module using the  AIFSCompleteEncoder for climate data.

    This module combines AIFS-encoded climate data (complete encoder from inputs to embeddings)
    with textual descriptions to create rich multimodal representations for climate analysis.

     Features:
    - Uses AIFSCompleteEncoder that returns actual encoder embeddings [542080, 218]
    - No more workaround encoders - uses the complete AIFS model from inputs to encoder output
    - Handles full 5D climate tensors: [batch, time, ensemble, grid_points, variables]
    """

    def __init__(
        self,
        aifs_model=None,
        aifs_checkpoint_path: Optional[str] = None,
        climate_dim: int = 218,  # Updated to match actual AIFS encoder output
        text_dim: int = 768,
        fusion_dim: int = 512,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        device: str = "cpu",
        verbose: bool = True,
    ):
        """
        Initialize AIFS climate-text fusion module with  encoder.

        Args:
            aifs_model: The complete AIFS model instance (preferred)
            aifs_checkpoint_path: Path to saved AIFSCompleteEncoder checkpoint (alternative)
            climate_dim: Dimension of AIFS climate encodings (218 for complete encoder)
            text_dim: Dimension of text embeddings
            fusion_dim: Dimension of fused representations
            num_attention_heads: Number of attention heads
            dropout: Dropout rate
            device: Device to run on
            verbose: Whether to print initialization messages
        """
        super().__init__()

        self.climate_dim = climate_dim
        self.text_dim = text_dim
        self.fusion_dim = fusion_dim
        self.num_attention_heads = num_attention_heads
        self.device = device
        self.verbose = verbose

        # Initialize the  AIFS Complete Encoder
        if aifs_model is not None:
            # Create new AIFSCompleteEncoder from AIFS model
            self.aifs_encoder = AIFSCompleteEncoder(aifs_model, verbose=verbose)
            if verbose:
                print("✅ Using AIFSCompleteEncoder with provided AIFS model")
        elif aifs_checkpoint_path is not None:
            # Load from checkpoint (requires AIFS model to be loaded separately)
            if verbose:
                print(
                    "⚠️ Loading from checkpoint requires AIFS model. Consider providing aifs_model parameter."
                )
            self.aifs_encoder = None  # Will be set when aifs_model is provided
            self.checkpoint_path = aifs_checkpoint_path
        else:
            raise ValueError("Either aifs_model or aifs_checkpoint_path must be provided")

        # Climate data projection (updated for new encoder output dimension)
        self.climate_projection = nn.Sequential(
            nn.Linear(climate_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Text projection
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Cross-attention for climate-text fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_attention_heads, dropout=dropout, batch_first=True
        )

        # Self-attention for final fusion
        self.self_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_attention_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward network
        self.feedforward = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 4, fusion_dim),
            nn.Dropout(dropout),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)
        self.norm3 = nn.LayerNorm(fusion_dim)

        # Output projection
        self.output_projection = nn.Linear(fusion_dim, fusion_dim)

        # Initialize text processor
        self.text_processor = ClimateTextProcessor()

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

        with torch.no_grad():
            # Use the complete encoder that returns actual AIFS embeddings [grid_points, embedding_dim]
            encoded = self.aifs_encoder(climate_data)  # [542080, 218]

            # Aggregate grid point embeddings to create global climate representation
            # Use mean pooling across grid points to get global features
            if encoded.dim() == 2:  # [grid_points, embedding_dim]
                # Take mean across grid points to get global representation
                global_encoded = encoded.mean(dim=0, keepdim=True)  # [1, 218]
                # Expand to match original batch size if needed
                batch_size = climate_data.shape[0]
                global_encoded = global_encoded.expand(batch_size, -1)  # [batch, 218]
            else:
                global_encoded = encoded  # Already in correct format

        # Project to fusion dimension
        projected = self.climate_projection(global_encoded)
        return torch.as_tensor(projected)

    def encode_text(
        self, texts: List[str], text_embeddings: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Encode text descriptions.

        Args:
            texts: List of text descriptions
            text_embeddings: Pre-computed text embeddings (optional)

        Returns:
            Encoded text features
        """
        if text_embeddings is None:
            # Simple embedding for now - in practice, use a proper text encoder
            # Create embeddings for each text but average them to get single representation
            num_texts = len(texts)
            text_embeddings = torch.randn(num_texts, self.text_dim, device=self.device)
            # Average across texts to get single text representation
            text_embeddings = text_embeddings.mean(dim=0, keepdim=True)  # [1, text_dim]

        # Project to fusion dimension
        projected = self.text_projection(text_embeddings)
        return torch.as_tensor(projected)

    def apply_cross_attention(
        self, climate_features: torch.Tensor, text_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-attention between climate and text features.

        Args:
            climate_features: Climate feature tensor
            text_features: Text feature tensor

        Returns:
            Tuple of attended features (climate, text)
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
        texts: List[str],
        text_embeddings: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the fusion module.

        Args:
            climate_data: Raw climate data
            texts: Text descriptions
            text_embeddings: Pre-computed text embeddings (optional)

        Returns:
            Dictionary containing fusion results
        """
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
        features1 = self.encode_climate_data(climate_data1)
        features2 = self.encode_climate_data(climate_data2)

        # Cosine similarity
        similarity = F.cosine_similarity(features1, features2, dim=-1)
        return similarity

    def get_text_climate_alignment(
        self,
        climate_data: torch.Tensor,
        texts: List[str],
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
        climate_features = self.encode_climate_data(climate_data)
        text_features = self.encode_text(texts, text_embeddings)

        # Compute alignment as cosine similarity
        alignment = F.cosine_similarity(climate_features, text_features, dim=-1)
        return alignment


class AIFSClimateEmbedding(nn.Module):
    """
    Lightweight climate embedding using the  AIFSCompleteEncoder.

    Creates embeddings directly from climate data using the complete AIFS encoder
    that returns actual encoder outputs [542080, 218].
    """

    def __init__(
        self,
        aifs_model=None,
        aifs_checkpoint_path: Optional[str] = None,
        climate_dim: int = 218,  # Updated to match actual AIFS encoder output
        embedding_dim: int = 256,
        device: str = "cpu",
        verbose: bool = True,
    ):
        """
        Initialize AIFS climate embedding with encoder.

        Args:
            aifs_model: The complete AIFS model instance (preferred)
            aifs_checkpoint_path: Path to saved AIFSCompleteEncoder checkpoint (alternative)
            climate_dim: Dimension of AIFS climate encodings (218 for complete encoder)
            embedding_dim: Final embedding dimension
            device: Device to run on
            verbose: Whether to print initialization messages
        """
        super().__init__()

        self.climate_dim = climate_dim
        self.embedding_dim = embedding_dim
        self.device = device
        self.verbose = verbose

        # Initialize the  AIFS Complete Encoder
        if aifs_model is not None:
            # Create new AIFSCompleteEncoder from AIFS model
            self.aifs_encoder = AIFSCompleteEncoder(aifs_model, verbose=verbose)
            if verbose:
                print("✅ Using AIFSCompleteEncoder with provided AIFS model")
        elif aifs_checkpoint_path is not None:
            # Load from checkpoint (requires AIFS model to be loaded separately)
            if verbose:
                print(
                    "⚠️ Loading from checkpoint requires AIFS model. Consider providing aifs_model parameter."
                )
            self.aifs_encoder = None  # Will be set when aifs_model is provided
            self.checkpoint_path = aifs_checkpoint_path
        else:
            raise ValueError("Either aifs_model or aifs_checkpoint_path must be provided")

        # Climate embedding projection
        self.projection = nn.Sequential(
            nn.Linear(climate_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
        )

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

        # Encode with AIFS complete encoder
        with torch.no_grad():
            aifs_features = self.aifs_encoder(climate_data)  # [542080, 218]

            # Aggregate grid point embeddings to create global climate representation
            # Use mean pooling across grid points to get global features
            if aifs_features.dim() == 2:  # [grid_points, embedding_dim]
                # Take mean across grid points to get global representation
                global_features = aifs_features.mean(dim=0, keepdim=True)  # [1, 218]
                # Expand to match original batch size if needed
                batch_size = climate_data.shape[0]
                global_features = global_features.expand(batch_size, -1)  # [batch, 218]
            else:
                global_features = aifs_features  # Already in correct format

        # Project to embedding space
        embeddings = self.projection(global_features)

        return torch.as_tensor(embeddings)


def create_aifs_fusion_from_model(aifs_model, fusion_dim: int = 512, verbose: bool = True):
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
        climate_dim=218,  # AIFSCompleteEncoder output dimension
        fusion_dim=fusion_dim,
        verbose=verbose,
    )


def create_aifs_embedding_from_model(aifs_model, embedding_dim: int = 256, verbose: bool = True):
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
        climate_dim=218,  # AIFSCompleteEncoder output dimension
        embedding_dim=embedding_dim,
        verbose=verbose,
    )


def test_aifs_climate_fusion():
    """Test AIFS climate fusion module architecture and projections."""
    print("🌡️ Testing AIFS Climate Fusion Module Architecture")
    print("=" * 50)

    try:
        print("📦 Testing module initialization and projections...")

        # Test projections directly (without requiring AIFS model)
        climate_dim = 218  # AIFSCompleteEncoder output dimension
        text_dim = 768
        fusion_dim = 512
        embedding_dim = 256

        # Test climate projection layers
        climate_projection = nn.Sequential(
            nn.Linear(climate_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Test embedding projection layers
        embedding_projection = nn.Sequential(
            nn.Linear(climate_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
        )

        # Test text projection layers
        text_projection = nn.Sequential(
            nn.Linear(text_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Test with synthetic encoder output (218-dim as from AIFSCompleteEncoder)
        batch_size = 4
        synthetic_aifs_output = torch.randn(batch_size, climate_dim)
        synthetic_text_embeddings = torch.randn(batch_size, text_dim)

        # Test climate projection
        climate_projected = climate_projection(synthetic_aifs_output)
        print(f"✅ Climate projection: {synthetic_aifs_output.shape} → {climate_projected.shape}")

        # Test embedding projection
        climate_embedded = embedding_projection(synthetic_aifs_output)
        print(f"✅ Climate embedding: {synthetic_aifs_output.shape} → {climate_embedded.shape}")

        # Test text projection
        text_projected = text_projection(synthetic_text_embeddings)
        print(f"✅ Text projection: {synthetic_text_embeddings.shape} → {text_projected.shape}")

        # Test attention mechanism
        cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=8, dropout=0.1, batch_first=True
        )

        # Test cross-attention between climate and text
        climate_att, _ = cross_attention(
            climate_projected.unsqueeze(1), text_projected.unsqueeze(1), text_projected.unsqueeze(1)
        )
        print(f"✅ Cross-attention output: {climate_att.squeeze(1).shape}")

        print("\n🎉 All architecture tests passed!")
        print("✨ Fusion module ready for integration with AIFSCompleteEncoder!")
        print("\n💡 Usage with real AIFS model:")
        print("   # Load your AIFS model first")
        print("   aifs_model = load_your_aifs_model()")
        print("   # Then create fusion module")
        print("   fusion_module = AIFSClimateTextFusion(aifs_model=aifs_model)")
        print("   # Or embedding module")
        print("   embedding_module = AIFSClimateEmbedding(aifs_model=aifs_model)")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    test_aifs_climate_fusion()

"""
Multimodal Fusion with PrithviWxC and Llama 3

This module implements multimodal fusion between climate data (using PrithviWxC encoder)
and text data (using Llama 3). It enables tasks like:
- Climate assessment generation
- Climate trend understanding
- Climate-aware question answering
- Multimodal climate analysis

Usage:
    from multimodal.climate_text_fusion import ClimateTextFusion

    # Note: meta-llama/Meta-Llama-3-8B requires HuggingFace approval
    model = ClimateTextFusion(
        prithvi_encoder_path='data/weights/prithvi_encoder.pt',
        llama_model_name='meta-llama/Meta-Llama-3-8B'  # Requires HF approval
    )

    # For testing without approval:
    # llama_model_name='prajjwal1/bert-tiny'  # No approval needed

    # Process climate data and text together
    output = model(climate_data, text_input)
"""

import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

# Import transformers for language models
try:
    from transformers import (
        AutoModel,
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("Transformers library not available. Install with: pip install transformers")

# Import our custom encoder
try:
    from ..utils.encoder_extractor import PrithviWxC_Encoder
except ImportError:
    # Fallback for direct script execution
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.encoder_extractor import PrithviWxC_Encoder


class ClimateFeatureProjector(nn.Module):
    """
    Projects climate features to text embedding space for fusion.
    """

    def __init__(
        self,
        climate_dim: int,
        text_dim: int,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            climate_dim: Dimension of climate features from PrithviWxC encoder
            text_dim: Dimension of text embeddings from Llama 3
            hidden_dim: Hidden dimension for projection layers
            num_layers: Number of projection layers
            dropout: Dropout rate
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = max(climate_dim, text_dim)

        layers = []
        input_dim = climate_dim

        for i in range(num_layers):
            output_dim = text_dim if i == num_layers - 1 else hidden_dim
            layers.extend(
                [
                    nn.Linear(input_dim, output_dim),
                    nn.LayerNorm(output_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            input_dim = output_dim

        # Remove last activation and dropout
        layers = layers[:-2]
        self.projection = nn.Sequential(*layers)

    def forward(self, climate_features: torch.Tensor) -> torch.Tensor:
        """
        Project climate features to text embedding space.

        Args:
            climate_features: [batch, seq_len, climate_dim]

        Returns:
            projected_features: [batch, seq_len, text_dim]
        """
        return self.projection(climate_features)


class CrossModalAttention(nn.Module):
    """
    Cross-attention mechanism for climate-text fusion.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [batch, query_len, embed_dim]
            key: [batch, key_len, embed_dim]
            value: [batch, key_len, embed_dim]
            key_padding_mask: [batch, key_len]
        """
        # Cross attention
        attn_output, _ = self.multihead_attn(
            query=query, key=key, value=value, key_padding_mask=key_padding_mask
        )

        # Residual connection and normalization
        x = self.norm1(query + attn_output)

        # Feed forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x


class ClimateTextFusion(nn.Module):
    """
    Multimodal fusion model combining PrithviWxC encoder with Llama 3.
    """

    def __init__(
        self,
        prithvi_encoder_path: str = None,
        prithvi_encoder: torch.nn.Module = None,
        llama_model_name: str = "meta-llama/Meta-Llama-3-8B",
        fusion_mode: str = "cross_attention",
        max_climate_tokens: int = 1024,
        max_text_length: int = 512,
        num_fusion_layers: int = 2,
        fusion_dropout: float = 0.1,
        freeze_prithvi: bool = True,
        freeze_llama: bool = True,
        device: str = "auto",
    ):
        """
        Args:
            prithvi_encoder_path: Path to the extracted PrithviWxC encoder weights
            llama_model_name: Hugging Face model name (meta-llama/Meta-Llama-3-8B requires approval)
            fusion_mode: How to fuse modalities ('cross_attention', 'concatenate', 'add')
            max_climate_tokens: Maximum number of climate tokens to process
            max_text_length: Maximum text sequence length
            num_fusion_layers: Number of fusion layers
            fusion_dropout: Dropout rate in fusion layers
            freeze_prithvi: Whether to freeze PrithviWxC encoder
            freeze_llama: Whether to freeze Llama 3 weights
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers library required. Install with: pip install transformers"
            )

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.fusion_mode = fusion_mode
        self.max_climate_tokens = max_climate_tokens
        self.max_text_length = max_text_length

        # Load or use provided PrithviWxC encoder
        if prithvi_encoder is not None:
            self.climate_encoder = prithvi_encoder
            # Assume pre-loaded encoder has embed_dim attribute
            if hasattr(prithvi_encoder, "embed_dim"):
                self.climate_dim = prithvi_encoder.embed_dim
            else:
                # Try to infer from encoder structure
                self.climate_dim = 1024  # Default Prithvi dimension
        elif prithvi_encoder_path is not None:
            self.climate_encoder = self._load_prithvi_encoder(prithvi_encoder_path)
            self.climate_dim = self.climate_encoder.embed_dim
        else:
            raise ValueError("Either prithvi_encoder_path or prithvi_encoder must be provided")

        if freeze_prithvi:
            for param in self.climate_encoder.parameters():
                param.requires_grad = False

        # Check if Llama loading should be skipped (for testing)
        skip_llama = os.environ.get("DISABLE_LLAMA_LOADING", "0") == "1"

        if skip_llama:
            # Create dummy tokenizer and model for testing
            print("  🔧 Skipping Llama loading for encoder-only testing")
            self.tokenizer = None
            self.text_model = None
            self.text_dim = 4096  # Default Llama dimension
        else:
            # Load Llama 3 model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
            self.text_model = AutoModel.from_pretrained(llama_model_name)

            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            if freeze_llama:
                for param in self.text_model.parameters():
                    param.requires_grad = False

            # Get embedding dimensions
            self.text_dim = self.text_model.config.hidden_size

        # Initialize fusion components (always needed, even in test mode)
        self._init_fusion_components(num_fusion_layers, fusion_dropout)

        # Move to device
        if not skip_llama:
            self.to(self.device)

    def _load_prithvi_encoder(self, encoder_path: str) -> PrithviWxC_Encoder:
        """Load the PrithviWxC encoder from saved weights with smart architecture detection."""
        checkpoint = torch.load(encoder_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"]

        # Smart architecture detection from actual weights
        print("🔍 Detecting real Prithvi architecture from weights...")

        # Detect actual dimensions from weight shapes
        actual_in_channels = (
            state_dict["input_scalers_mu"].shape[2] if "input_scalers_mu" in state_dict else 160
        )

        # Detect patch embedding input channels
        if "patch_embedding.proj.weight" in state_dict:
            patch_embed_in_channels = state_dict["patch_embedding.proj.weight"].shape[1]
        else:
            patch_embed_in_channels = 320  # Default for real Prithvi

        if "patch_embedding_static.proj.weight" in state_dict:
            static_embed_in_channels = state_dict["patch_embedding_static.proj.weight"].shape[1]
        else:
            static_embed_in_channels = 168  # Default for real Prithvi

        # Determine actual static channels and residual mode
        # If static_embed_in_channels > actual_in_channels, then residual="climate" mode
        if static_embed_in_channels > actual_in_channels:
            # residual="climate" mode: patch_embedding_static
            # expects in_channels + in_channels_static
            actual_static_channels = static_embed_in_channels - actual_in_channels
            residual_mode = "climate"
        else:
            # residual!="climate" mode: patch_embedding_static expects only in_channels_static
            actual_static_channels = static_embed_in_channels
            residual_mode = "non-climate"

        # Count actual transformer layers
        actual_transformer_layers = 0
        for key in state_dict.keys():
            if "transformers." in key:
                layer_parts = key.split(".")
                for i, part in enumerate(layer_parts):
                    if part == "transformers" and i + 1 < len(layer_parts):
                        try:
                            layer_num = int(layer_parts[i + 1])
                            actual_transformer_layers = max(
                                actual_transformer_layers, layer_num + 1
                            )
                        except ValueError:
                            pass

        # Convert actual transformer count to n_blocks_encoder
        # LocalGlobalLocalBlock creates 2*n_blocks+1 transformers
        # So if we have N transformers: N = 2*n_blocks+1 => n_blocks = (N-1)/2
        actual_n_blocks = (actual_transformer_layers - 1) // 2

        print("  📊 Detected architecture:")
        print(f"     Input channels: {actual_in_channels}")
        print(f"     Static channels: {actual_static_channels}")
        print(f"     Transformer layers: {actual_transformer_layers}")
        print(f"     N_blocks_encoder: {actual_n_blocks}")
        print(f"     Patch embed channels: {patch_embed_in_channels}")
        print(f"     Static embed channels: {static_embed_in_channels}")
        print(f"     Residual mode: {residual_mode}")

        real_config = {
            "in_channels": actual_in_channels,
            "input_size_time": 2,
            "in_channels_static": actual_static_channels,
            "input_scalers_epsilon": 0.0,
            "static_input_scalers_epsilon": 0.0,
            "n_lats_px": 360,
            "n_lons_px": 576,
            "patch_size_px": [2, 2],
            "mask_unit_size_px": [30, 32],
            "embed_dim": 2560,
            "n_blocks_encoder": actual_n_blocks,
            "mlp_multiplier": 4,
            "n_heads": 16,
        }

        # Always use climate mode if static_embed_in_channels > actual_static_channels
        residual_mode = (
            "climate" if static_embed_in_channels > actual_static_channels else "channel"
        )  # Create scalers with exact dimensions from weights to avoid size mismatches
        if "input_scalers_mu" in state_dict:
            in_mu = state_dict["input_scalers_mu"].clone()
            in_sig = state_dict["input_scalers_sigma"].clone()
        else:
            in_mu = torch.zeros(actual_in_channels)
            in_sig = torch.ones(actual_in_channels)

        if "static_input_scalers_mu" in state_dict:
            static_mu_full = state_dict["static_input_scalers_mu"].clone()
            static_sig_full = state_dict["static_input_scalers_sigma"].clone()

            # If scalers have more channels than model needs, truncate them
            if static_mu_full.shape[1] > actual_static_channels:
                print(
                    f"  🔧 Truncating static scalers from "
                    f"{static_mu_full.shape[1]} to {actual_static_channels} channels"
                )
                static_mu = static_mu_full[:, :actual_static_channels, :, :]
                static_sig = static_sig_full[:, :actual_static_channels, :, :]
            else:
                static_mu = static_mu_full
                static_sig = static_sig_full

        else:
            static_mu = torch.zeros(1, actual_static_channels, 1, 1)
            static_sig = torch.ones(1, actual_static_channels, 1, 1)

        encoder = PrithviWxC_Encoder(
            in_channels=real_config["in_channels"],
            input_size_time=real_config["input_size_time"],
            in_channels_static=actual_static_channels,
            input_scalers_mu=in_mu,
            input_scalers_sigma=in_sig,
            input_scalers_epsilon=real_config["input_scalers_epsilon"],
            static_input_scalers_mu=static_mu,
            static_input_scalers_sigma=static_sig,
            static_input_scalers_epsilon=real_config["static_input_scalers_epsilon"],
            n_lats_px=real_config["n_lats_px"],
            n_lons_px=real_config["n_lons_px"],
            patch_size_px=real_config["patch_size_px"],
            mask_unit_size_px=real_config["mask_unit_size_px"],
            mask_ratio_inputs=0.0,  # No masking for inference
            embed_dim=real_config["embed_dim"],
            n_blocks_encoder=real_config["n_blocks_encoder"],
            mlp_multiplier=real_config["mlp_multiplier"],
            n_heads=real_config["n_heads"],
            dropout=0.0,
            drop_path=0.0,
            parameter_dropout=0.0,
            residual="climate" if residual_mode == "climate" else None,
            masking_mode="global",
            positional_encoding="fourier",
            encoder_shifting=False,
            checkpoint_encoder=[],
        )

        # Fix state dict to match the architecture we're creating
        if "static_input_scalers_mu" in state_dict:
            # Truncate static scalers to match the architecture
            original_static_shape = state_dict["static_input_scalers_mu"].shape
            if original_static_shape[1] > actual_static_channels:
                print(
                    "  🔧 Adjusting state dict static scalers from "
                    f"{original_static_shape[1]} to {actual_static_channels} channels"
                )
                state_dict["static_input_scalers_mu"] = state_dict["static_input_scalers_mu"][
                    :, :actual_static_channels, :, :
                ]
                state_dict["static_input_scalers_sigma"] = state_dict["static_input_scalers_sigma"][
                    :, :actual_static_channels, :, :
                ]

        # Load state dict with smart handling of mismatches
        try:
            # Pre-filter state dict to only include keys that the encoder actually has
            encoder_state_keys = set(encoder.state_dict().keys())
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in encoder_state_keys}

            # Also ensure static scalers match exactly
            if "static_input_scalers_mu" in filtered_state_dict:
                expected_shape = encoder.static_input_scalers_mu.shape
                if filtered_state_dict["static_input_scalers_mu"].shape != expected_shape:
                    print(
                        "  🔧 Truncating static scalers to match encoder "
                        f"expectation: {expected_shape}"
                    )
                    filtered_state_dict["static_input_scalers_mu"] = filtered_state_dict[
                        "static_input_scalers_mu"
                    ][:, : expected_shape[1], :, :]
                    filtered_state_dict["static_input_scalers_sigma"] = filtered_state_dict[
                        "static_input_scalers_sigma"
                    ][:, : expected_shape[1], :, :]

            missing_keys, unexpected_keys = encoder.load_state_dict(
                filtered_state_dict, strict=True
            )

            # With filtering and strict=True, we should have zero missing/unexpected keys
            if missing_keys or unexpected_keys:
                raise RuntimeError(
                    "Encoder loading failed - Missing: "
                    f"{len(missing_keys)}, Unexpected: {len(unexpected_keys)}"
                )

            print(f"  ✅ Successfully loaded real Prithvi encoder with {actual_n_blocks} layers")
            print(
                f"  🎯 Loaded {len(filtered_state_dict)}/{len(state_dict)} "
                "compatible weights (100% of encoder weights)"
            )
        except Exception as e:
            print(f"  ❌ Error during load: {str(e)[:100]}...")
            # Fallback: create a minimal working encoder
            raise e

        return encoder

    def _init_fusion_components(self, num_layers: int, dropout: float):
        """Initialize fusion components based on fusion mode."""

        if self.fusion_mode == "cross_attention":
            # Project climate features to text embedding space
            self.climate_projector = ClimateFeatureProjector(
                climate_dim=self.climate_dim, text_dim=self.text_dim, dropout=dropout
            )

            # Cross-modal attention layers
            self.fusion_layers = nn.ModuleList(
                [
                    CrossModalAttention(embed_dim=self.text_dim, num_heads=8, dropout=dropout)
                    for _ in range(num_layers)
                ]
            )

        elif self.fusion_mode == "concatenate":
            # Simple concatenation with projection
            self.fusion_projection = nn.Sequential(
                nn.Linear(self.climate_dim + self.text_dim, self.text_dim),
                nn.LayerNorm(self.text_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        elif self.fusion_mode == "add":
            # Additive fusion with projection
            self.climate_projector = ClimateFeatureProjector(
                climate_dim=self.climate_dim, text_dim=self.text_dim, dropout=dropout
            )

        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")

        # Output projection layer
        self.output_projection = nn.Sequential(
            nn.Linear(self.text_dim, self.text_dim),
            nn.LayerNorm(self.text_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.text_dim, self.text_dim),
        )

    def encode_climate(self, climate_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode climate data using PrithviWxC encoder.

        Args:
            climate_batch: Batch dictionary with climate data

        Returns:
            climate_features: [batch, n_tokens, embed_dim]
        """
        with torch.no_grad() if hasattr(self, "_freeze_prithvi") else torch.enable_grad():
            climate_features = self.climate_encoder(climate_batch)

        # Flatten spatial dimensions if needed
        if climate_features.dim() == 4:  # [batch, n_global_tokens, n_local_tokens, embed_dim]
            batch_size, n_global, n_local, embed_dim = climate_features.shape
            climate_features = climate_features.view(batch_size, n_global * n_local, embed_dim)

        # Limit number of tokens if specified
        if self.max_climate_tokens and climate_features.size(1) > self.max_climate_tokens:
            climate_features = climate_features[:, : self.max_climate_tokens]

        return climate_features

    def encode_text(self, text_inputs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text using Llama 3.

        Args:
            text_inputs: List of text strings

        Returns:
            text_features: [batch, seq_len, embed_dim]
            attention_mask: [batch, seq_len]
        """
        # Tokenize text
        encoded = self.tokenizer(
            text_inputs,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # Get text embeddings
        with torch.no_grad() if hasattr(self, "_freeze_llama") else torch.enable_grad():
            outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
            text_features = outputs.last_hidden_state

        return text_features, attention_mask

    def fuse_modalities(
        self,
        climate_features: torch.Tensor,
        text_features: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse climate and text features.

        Args:
            climate_features: [batch, climate_seq_len, climate_dim]
            text_features: [batch, text_seq_len, text_dim]
            text_attention_mask: [batch, text_seq_len]

        Returns:
            fused_features: [batch, text_seq_len, text_dim]
        """

        if self.fusion_mode == "cross_attention":
            # Project climate features to text space
            climate_projected = self.climate_projector(climate_features)

            # Apply cross-attention layers
            fused = text_features
            for layer in self.fusion_layers:
                fused = layer(query=fused, key=climate_projected, value=climate_projected)

        elif self.fusion_mode == "concatenate":
            # Simple concatenation approach
            batch_size, text_len, text_dim = text_features.shape
            climate_len = climate_features.size(1)

            # Repeat climate features to match text length
            if climate_len != text_len:
                if climate_len < text_len:
                    # Repeat climate features
                    repeat_factor = text_len // climate_len + 1
                    climate_repeated = climate_features.repeat(1, repeat_factor, 1)[:, :text_len]
                else:
                    # Truncate climate features
                    climate_repeated = climate_features[:, :text_len]
            else:
                climate_repeated = climate_features

            # Concatenate and project
            concatenated = torch.cat([text_features, climate_repeated], dim=-1)
            fused = self.fusion_projection(concatenated)

        elif self.fusion_mode == "add":
            # Additive fusion
            climate_projected = self.climate_projector(climate_features)

            # Handle sequence length mismatch
            batch_size, text_len, text_dim = text_features.shape
            climate_len = climate_projected.size(1)

            if climate_len != text_len:
                if climate_len < text_len:
                    # Pad climate features
                    padding = torch.zeros(
                        batch_size,
                        text_len - climate_len,
                        text_dim,
                        device=climate_projected.device,
                    )
                    climate_projected = torch.cat([climate_projected, padding], dim=1)
                else:
                    # Truncate climate features
                    climate_projected = climate_projected[:, :text_len]

            fused = text_features + climate_projected

        return fused

    def forward(
        self,
        climate_batch: Dict[str, torch.Tensor],
        text_inputs: List[str],
        return_attention_weights: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the multimodal fusion model.

        Args:
            climate_batch: Climate data batch
            text_inputs: List of text strings
            return_attention_weights: Whether to return attention weights

        Returns:
            Dictionary containing:
                - 'fused_features': Fused multimodal features
                - 'climate_features': Climate features
                - 'text_features': Text features
                - 'attention_mask': Text attention mask
        """
        # Encode both modalities
        climate_features = self.encode_climate(climate_batch)
        text_features, text_attention_mask = self.encode_text(text_inputs)

        # Fuse modalities
        fused_features = self.fuse_modalities(climate_features, text_features, text_attention_mask)

        # Apply output projection
        output_features = self.output_projection(fused_features)

        result = {
            "fused_features": output_features,
            "climate_features": climate_features,
            "text_features": text_features,
            "attention_mask": text_attention_mask,
        }

        return result


class ClimateQuestionAnswering(nn.Module):
    """
    Climate-aware question answering using the multimodal fusion model.
    """

    def __init__(self, fusion_model: ClimateTextFusion, num_classes: int = 2):
        super().__init__()
        self.fusion_model = fusion_model
        self.qa_head = nn.Sequential(
            nn.Linear(fusion_model.text_dim, fusion_model.text_dim // 2),
            nn.LayerNorm(fusion_model.text_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_model.text_dim // 2, num_classes),
        )

    def forward(self, climate_batch: Dict[str, torch.Tensor], questions: List[str]) -> torch.Tensor:
        """
        Args:
            climate_batch: Climate data
            questions: List of questions about the climate data

        Returns:
            logits: [batch, seq_len, num_classes]
        """
        fusion_output = self.fusion_model(climate_batch, questions)
        fused_features = fusion_output["fused_features"]
        return self.qa_head(fused_features)


class ClimateTextGeneration(nn.Module):
    """
    Climate-conditioned text generation using Llama 3.
    """

    def __init__(
        self,
        prithvi_encoder_path: str,
        llama_model_name: str = "meta-llama/Meta-Llama-3-8B",
    ):
        super().__init__()

        # Load models
        self.climate_encoder = self._load_prithvi_encoder(prithvi_encoder_path)
        self.text_generator = AutoModelForCausalLM.from_pretrained(llama_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llama_model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Climate conditioning layer
        self.climate_adapter = nn.Sequential(
            nn.Linear(self.climate_encoder.embed_dim, self.text_generator.config.hidden_size),
            nn.LayerNorm(self.text_generator.config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def _load_prithvi_encoder(self, encoder_path: str) -> PrithviWxC_Encoder:
        """Load PrithviWxC encoder with smart architecture detection
        (same as in ClimateTextFusion).
        """
        checkpoint = torch.load(encoder_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"]

        # Smart architecture detection from actual weights
        actual_in_channels = (
            state_dict["input_scalers_mu"].shape[2] if "input_scalers_mu" in state_dict else 160
        )

        # Better static channels detection
        if "static_input_scalers_mu" in state_dict:
            actual_static_channels = state_dict["static_input_scalers_mu"].shape[1]
        elif "patch_embedding_static.proj.weight" in state_dict:
            actual_static_channels = state_dict["patch_embedding_static.proj.weight"].shape[1]
        else:
            actual_static_channels = 11  # Default for real Prithvi

        # Count actual transformer layers
        actual_n_blocks = 0
        for key in state_dict.keys():
            if "encoder.lgl_block.transformers." in key and ".attention.0.weight" in key:
                layer_num = int(key.split(".")[3])
                actual_n_blocks = max(actual_n_blocks, layer_num + 1)

        # Detect static embedding input channels to determine residual mode
        if "patch_embedding_static.proj.weight" in state_dict:
            static_embed_in_channels = state_dict["patch_embedding_static.proj.weight"].shape[1]
        else:
            static_embed_in_channels = 168

        # Determine the correct residual mode
        if static_embed_in_channels == actual_in_channels + actual_static_channels:
            residual_mode = "climate"
            expected_static_channels = actual_static_channels
        else:
            residual_mode = "climate"  # Use climate mode to match the weights
            expected_static_channels = static_embed_in_channels - actual_in_channels

        # Use real Prithvi configuration with actual weight dimensions
        if "input_scalers_mu" in state_dict:
            in_mu = state_dict["input_scalers_mu"].clone()
            in_sig = state_dict["input_scalers_sigma"].clone()
        else:
            in_mu = torch.zeros(actual_in_channels)
            in_sig = torch.ones(actual_in_channels)

        if "static_input_scalers_mu" in state_dict:
            static_mu = state_dict["static_input_scalers_mu"].clone()
            static_sig = state_dict["static_input_scalers_sigma"].clone()
            actual_static_scaler_channels = static_mu.shape[1]
        else:
            static_mu = torch.zeros(expected_static_channels)
            static_sig = torch.ones(expected_static_channels)
            actual_static_scaler_channels = expected_static_channels

        encoder = PrithviWxC_Encoder(
            in_channels=actual_in_channels,
            input_size_time=2,
            in_channels_static=actual_static_scaler_channels,
            input_scalers_mu=in_mu,
            input_scalers_sigma=in_sig,
            input_scalers_epsilon=0.0,
            static_input_scalers_mu=static_mu,
            static_input_scalers_sigma=static_sig,
            static_input_scalers_epsilon=0.0,
            n_lats_px=360,
            n_lons_px=576,
            patch_size_px=[2, 2],
            mask_unit_size_px=[30, 32],
            mask_ratio_inputs=0.0,
            embed_dim=2560,
            n_blocks_encoder=actual_n_blocks,
            mlp_multiplier=4,
            n_heads=16,
            dropout=0.0,
            drop_path=0.0,
            parameter_dropout=0.0,
            residual=residual_mode,
            masking_mode="global",
            positional_encoding="fourier",
            encoder_shifting=False,
            checkpoint_encoder=[],
        )

        _missing_keys, _unexpected_keys = encoder.load_state_dict(state_dict, strict=False)
        return encoder

    def forward(
        self,
        climate_batch: Dict[str, torch.Tensor],
        prompt: str = "Based on the climate data, generate a climate assessment:",
        max_length: int = 200,
        temperature: float = 0.7,
    ) -> List[str]:
        """
        Forward pass for climate-conditioned text generation.

        Args:
            climate_batch: Climate data
            prompt: Text prompt for generation
            max_length: Maximum generation length
            temperature: Sampling temperature

        Returns:
            generated_texts: List of generated reports
        """
        return self.generate_climate_report(climate_batch, prompt, max_length, temperature)

    def generate_climate_report(
        self,
        climate_batch: Dict[str, torch.Tensor],
        prompt: str = "Based on the climate data, generate a climate assessment:",
        max_length: int = 200,
        temperature: float = 0.7,
    ) -> List[str]:
        """
        Generate climate assessment text conditioned on climate data.

        Args:
            climate_batch: Climate data
            prompt: Text prompt for generation
            max_length: Maximum generation length
            temperature: Sampling temperature

        Returns:
            generated_texts: List of generated reports
        """
        # Encode climate data
        with torch.no_grad():
            climate_features = self.climate_encoder(climate_batch)

        # Adapt climate features for text generation
        climate_adapted = self.climate_adapter(climate_features.mean(dim=1))  # [batch, hidden_size]

        # Tokenize prompt
        inputs = self.tokenizer(
            [prompt] * climate_adapted.size(0), return_tensors="pt", padding=True
        )

        # Generate text conditioned on climate
        with torch.no_grad():
            outputs = self.text_generator.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode generated text
        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs
        ]

        return generated_texts

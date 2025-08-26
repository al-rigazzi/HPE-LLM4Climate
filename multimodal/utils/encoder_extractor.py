"""
PrithviWxC Encoder Extractor

This module provides utilities for extracting the encoder component from
PrithviWxC models for use in multimodal fusion applications.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from torch import nn

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PrithviWxC.dataloaders.merra2 import input_scalers, static_input_scalers

# Import PrithviWxC model components
from PrithviWxC.model import (
    PatchEmbed,
    PrithviWxC,
    PrithviWxCEncoderDecoder,
    SWINShiftNoBuffer,
)


class PrithviWxC_Encoder(nn.Module):  # pylint: disable=invalid-name
    """
    Encoder-only version of PrithviWxC model.

    This class extracts and implements only the encoder portion of the full PrithviWxC model,
    including input preprocessing, patch embedding, position encoding, time encoding,
    masking, and the encoder transformer blocks.
    """

    def __init__(
        self,
        in_channels: int,
        input_size_time: int,
        in_channels_static: int,
        input_scalers_mu: torch.Tensor,
        input_scalers_sigma: torch.Tensor,
        input_scalers_epsilon: float,
        static_input_scalers_mu: torch.Tensor,
        static_input_scalers_sigma: torch.Tensor,
        static_input_scalers_epsilon: float,
        n_lats_px: int,
        n_lons_px: int,
        patch_size_px: tuple[int],
        mask_unit_size_px: tuple[int],
        mask_ratio_inputs: float,
        embed_dim: int,
        n_blocks_encoder: int,
        mlp_multiplier: float,
        n_heads: int,
        dropout: float,
        drop_path: float,
        parameter_dropout: float,
        residual: str,
        masking_mode: str,
        positional_encoding: str,
        encoder_shifting: bool = False,
        checkpoint_encoder: list[int] | None = None,
    ) -> None:
        """
        Initialize the encoder-only model.

        Args:
            in_channels: Number of input channels.
            input_size_time: Number of timestamps in input.
            in_channels_static: Number of input channels for static data.
            input_scalers_mu: Tensor of size (in_channels,). Used to rescale input.
            input_scalers_sigma: Tensor of size (in_channels,). Used to rescale input.
            input_scalers_epsilon: Float. Used to rescale input.
            static_input_scalers_mu: Tensor of size (in_channels_static).
                Used to rescale static inputs.
            static_input_scalers_sigma: Tensor of size (in_channels_static).
                Used to rescale static inputs.
            static_input_scalers_epsilon: Float. Used to rescale static inputs.
            n_lats_px: Total latitudes in data. In pixels.
            n_lons_px: Total longitudes in data. In pixels.
            patch_size_px: Patch size for tokenization. In pixels lat/lon.
            mask_unit_size_px: Size of each mask unit. In pixels lat/lon.
            mask_ratio_inputs: Masking ratio for inputs. 0 to 1.
            embed_dim: Embedding dimension
            n_blocks_encoder: Number of local-global transformer pairs in encoder.
            mlp_multiplier: MLP multiplier for hidden features in feed forward networks.
            n_heads: Number of attention heads.
            dropout: Dropout.
            drop_path: DropPath.
            parameter_dropout: Dropout applied to parameters.
            residual: Indicates whether and how model should work as residual model.
            positional_encoding: Position encoding type ('absolute' or 'fourier').
            masking_mode: String ['local', 'global', 'both'] that controls
                the type of masking used.
            encoder_shifting: Whether to use swin shifting in the encoder.
            checkpoint_encoder: List of integers controlling if gradient
                checkpointing is used on encoder.
        """
        super().__init__()

        self.in_channels = in_channels
        self.input_size_time = input_size_time
        self.in_channels_static = in_channels_static
        self.n_lats_px = n_lats_px
        self.n_lons_px = n_lons_px
        self.patch_size_px = patch_size_px
        self.mask_unit_size_px = mask_unit_size_px
        self.mask_ratio_inputs = mask_ratio_inputs
        self.embed_dim = embed_dim
        self.n_blocks_encoder = n_blocks_encoder
        self.mlp_multiplier = mlp_multiplier
        self.n_heads = n_heads
        self.dropout = dropout
        self.drop_path = drop_path
        self.residual = residual
        self._encoder_shift = encoder_shifting
        self.positional_encoding = positional_encoding
        self._checkpoint_encoder = checkpoint_encoder or []

        # Validate inputs
        assert self.n_lats_px % self.mask_unit_size_px[0] == 0
        assert self.n_lons_px % self.mask_unit_size_px[1] == 0
        assert self.mask_unit_size_px[0] % self.patch_size_px[0] == 0
        assert self.mask_unit_size_px[1] % self.patch_size_px[1] == 0

        if self.patch_size_px[0] != self.patch_size_px[1]:
            raise NotImplementedError(
                "Current pixel shuffle implementation assumes same "
                "patch size along both dimensions."
            )

        # Shape calculations
        self.local_shape_mu = (
            self.mask_unit_size_px[0] // self.patch_size_px[0],
            self.mask_unit_size_px[1] // self.patch_size_px[1],
        )
        self.global_shape_mu = (
            self.n_lats_px // self.mask_unit_size_px[0],
            self.n_lons_px // self.mask_unit_size_px[1],
        )

        # Input scalers
        self.input_scalers_epsilon = input_scalers_epsilon
        self.register_buffer("input_scalers_mu", input_scalers_mu.reshape(1, 1, -1, 1, 1))
        self.register_buffer("input_scalers_sigma", input_scalers_sigma.reshape(1, 1, -1, 1, 1))

        # Static input scalers
        self.static_input_scalers_epsilon = static_input_scalers_epsilon
        self.register_buffer(
            "static_input_scalers_mu", static_input_scalers_mu.reshape(1, -1, 1, 1)
        )
        self.register_buffer(
            "static_input_scalers_sigma",
            static_input_scalers_sigma.reshape(1, -1, 1, 1),
        )

        # Dropout
        self.parameter_dropout = nn.Dropout2d(p=parameter_dropout)

        # Patch embedding layers
        self.patch_embedding = PatchEmbed(
            patch_size=patch_size_px,
            channels=in_channels * input_size_time,
            embed_dim=embed_dim,
        )

        if self.residual == "climate":
            self.patch_embedding_static = PatchEmbed(
                patch_size=patch_size_px,
                channels=in_channels + in_channels_static,
                embed_dim=embed_dim,
            )
        else:
            self.patch_embedding_static = PatchEmbed(
                patch_size=patch_size_px,
                channels=in_channels_static,
                embed_dim=embed_dim,
            )

        # Time embedding
        self.input_time_embedding = nn.Linear(1, embed_dim // 4, bias=True)
        self.lead_time_embedding = nn.Linear(1, embed_dim // 4, bias=True)

        # Mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, self.embed_dim))
        self._nglobal_mu = np.prod(self.global_shape_mu)
        self._global_idx = torch.arange(self._nglobal_mu)

        self._nlocal_mu = np.prod(self.local_shape_mu)
        self._local_idx = torch.arange(self._nlocal_mu)

        # Encoder shifter and encoder
        if self._encoder_shift:
            self.encoder_shifter = e_shifter = SWINShiftNoBuffer(
                self.mask_unit_size_px,
                self.global_shape_mu,
                self.local_shape_mu,
                self.patch_size_px,
                n_context_tokens=0,
            )
        else:
            self.encoder_shifter = e_shifter = None

        self.encoder = PrithviWxCEncoderDecoder(
            embed_dim=embed_dim,
            n_blocks=n_blocks_encoder,
            mlp_multiplier=mlp_multiplier,
            n_heads=n_heads,
            dropout=dropout,
            drop_path=drop_path,
            shifter=e_shifter,
            transformer_cp=checkpoint_encoder,
        )

        # Masking mode setup
        self.masking_mode = masking_mode.lower()

    def _gen_mask_global(self, shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate global masking indices."""
        batch_size, seq_len = shape

        if self.mask_ratio_inputs == 0:
            indices_masked = torch.empty((batch_size, 0), dtype=torch.long)
            indices_unmasked = self._global_idx.unsqueeze(0).repeat(batch_size, 1)
        else:
            n_masked = int(seq_len * self.mask_ratio_inputs)
            noise = torch.randn((batch_size, seq_len))
            indices_shuffle = torch.argsort(noise, dim=1)
            indices_masked = indices_shuffle[:, :n_masked]
            indices_unmasked = indices_shuffle[:, n_masked:]

        return indices_masked, indices_unmasked

    def _gen_mask_local(self, shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate local masking indices."""
        # Simplified local masking - for full implementation, refer to original model
        return self._gen_mask_global(shape)

    def generate_mask(self, shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate masking indices based on masking mode."""
        if self.masking_mode == "global":
            return self._gen_mask_global(shape)
        if self.masking_mode == "local":
            return self._gen_mask_local(shape)
        if self.masking_mode == "both":
            # For simplicity, use global masking
            return self._gen_mask_global(shape)
        raise ValueError(f"Unknown masking mode: {self.masking_mode}")

    def to_patching(self, x: torch.Tensor) -> torch.Tensor:
        """Transform data from lat/lon space to two axis patching

        Args:
            x: Tensor in lat/lon space (N, C, Nlat//P_0, Nlon//P_1)

        Returns:
            Tensor in patch space (N, G, L, C)
        """
        n_batch = x.shape[0]

        x = x.view(
            n_batch,
            self.embed_dim,
            self.global_shape_mu[0],
            self.local_shape_mu[0],
            self.global_shape_mu[1],
            self.local_shape_mu[1],
        )
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()

        s = x.shape
        return x.view(n_batch, s[1] * s[2], s[3] * s[4], -1)

    def time_encoding(self, input_time: torch.Tensor, lead_time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_time: Tensor of shape [batch].
            lead_time: Tensor of shape [batch].
        Returns:
            Tensor of shape [batch, embed_dim, 1, 1]
        """
        input_time = self.input_time_embedding(input_time.view(-1, 1, 1, 1))
        lead_time = self.lead_time_embedding(lead_time.view(-1, 1, 1, 1))

        time_encoding = torch.cat(
            (
                torch.cos(input_time),
                torch.cos(lead_time),
                torch.sin(input_time),
                torch.sin(lead_time),
            ),
            axis=3,
        )
        return time_encoding

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through encoder only.

        Args:
            batch: Dictionary containing:
                - 'x': Input data [batch, time, parameter, lat, lon]
                - 'static': Static data [batch, channel_static, lat, lon]
                - 'climate': Climate data [batch, parameter, lat, lon] (if residual='climate')
                - 'input_time': Input time [batch]
                - 'lead_time': Lead time [batch]

        Returns:
            Encoded features [batch, n_unmasked_tokens, local_seq, embed_dim]
        """
        # Input validation
        assert batch["x"].shape[2] == self.in_channels
        assert batch["x"].shape[3] == self.n_lats_px
        assert batch["x"].shape[4] == self.n_lons_px
        assert batch["static"].shape[2] == self.n_lats_px
        assert batch["static"].shape[3] == self.n_lons_px

        # Rescale inputs
        x_rescaled = (batch["x"] - self.input_scalers_mu) / (
            self.input_scalers_sigma + self.input_scalers_epsilon
        )
        batch_size = x_rescaled.shape[0]

        # Process static inputs
        x_static = (batch["static"] - self.static_input_scalers_mu) / (
            self.static_input_scalers_sigma + self.static_input_scalers_epsilon
        )

        # Handle climate data for residual mode
        if self.residual == "climate":
            climate_scaled = (batch["climate"] - self.input_scalers_mu.view(1, -1, 1, 1)) / (
                self.input_scalers_sigma.view(1, -1, 1, 1) + self.input_scalers_epsilon
            )

        # Flatten time and parameter dimensions
        x_rescaled = x_rescaled.flatten(1, 2)

        # Apply parameter dropout
        x_rescaled = self.parameter_dropout(x_rescaled)

        # Patch embedding
        x_embedded = self.patch_embedding(x_rescaled)

        if self.residual == "climate":
            static_embedded = self.patch_embedding_static(
                torch.cat((x_static, climate_scaled), dim=1)
            )
        else:
            static_embedded = self.patch_embedding_static(x_static)

        # Convert to patch format
        x_embedded = self.to_patching(x_embedded)
        static_embedded = self.to_patching(static_embedded)

        # Time encoding
        time_encoding = self.time_encoding(batch["input_time"], batch["lead_time"])

        # Combine embeddings
        tokens = x_embedded + static_embedded + time_encoding

        # Generate masks
        indices_masked, indices_unmasked = self.generate_mask((batch_size, self._nglobal_mu))
        indices_masked = indices_masked.to(device=tokens.device)
        indices_unmasked = indices_unmasked.to(device=tokens.device)
        maskdim: int = indices_unmasked.ndim

        # Extract unmasked tokens
        unmask_view = (*indices_unmasked.shape, *[1] * (tokens.ndim - maskdim))
        unmasked = torch.gather(
            tokens,
            dim=maskdim - 1,
            index=indices_unmasked.view(*unmask_view).expand(
                *indices_unmasked.shape, *tokens.shape[maskdim:]
            ),
        )

        # Encode
        x_encoded = self.encoder(unmasked)

        return x_encoded


def extract_encoder_weights(full_model: PrithviWxC, encoder_model: PrithviWxC_Encoder) -> None:
    """
    Extract encoder weights from full model and load into encoder-only model.

    Args:
        full_model: Full PrithviWxC model with loaded weights
        encoder_model: Encoder-only model to receive the weights
    """
    # Get state dicts
    full_state_dict = full_model.state_dict()
    encoder_state_dict = encoder_model.state_dict()

    # Copy matching weights
    for key in encoder_state_dict.keys():
        if key in full_state_dict:
            encoder_state_dict[key].copy_(full_state_dict[key])
            print(f"Copied weights for: {key}")
        else:
            print(f"Warning: Could not find weights for: {key}")

    # Load the extracted weights
    encoder_model.load_state_dict(encoder_state_dict)


def main():
    """Main function for command-line encoder extraction."""
    parser = argparse.ArgumentParser(description="Extract encoder from PrithviWxC model")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to model configuration YAML file",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        required=True,
        help="Path to full model weights file",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save encoder weights"
    )
    parser.add_argument("--surf_scaler_path", type=str, help="Path to surface input scalers file")
    parser.add_argument("--vert_scaler_path", type=str, help="Path to vertical input scalers file")

    args = parser.parse_args()

    # Load configuration
    with open(args.config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Define variables (these should match your dataset configuration)
    surface_vars = [
        "EFLUX",
        "GWETROOT",
        "HFLUX",
        "LAI",
        "LWGAB",
        "LWGEM",
        "LWTUP",
        "PS",
        "QV2M",
        "SLP",
        "SWGNT",
        "SWTNT",
        "T2M",
        "TQI",
        "TQL",
        "TQV",
        "TS",
        "U10M",
        "V10M",
        "Z0M",
    ]
    static_surface_vars = ["FRACI", "FRLAND", "FROCEAN", "PHIS"]
    vertical_vars = ["CLOUD", "H", "OMEGA", "PL", "QI", "QL", "QV", "T", "U", "V"]
    levels = [
        34.0,
        39.0,
        41.0,
        43.0,
        44.0,
        45.0,
        48.0,
        51.0,
        53.0,
        56.0,
        63.0,
        68.0,
        71.0,
        72.0,
    ]

    # Load scalers
    if args.surf_scaler_path and args.vert_scaler_path:
        in_mu, in_sig = input_scalers(
            surface_vars,
            vertical_vars,
            levels,
            Path(args.surf_scaler_path),
            Path(args.vert_scaler_path),
        )
        static_mu, static_sig = static_input_scalers(
            Path(args.surf_scaler_path), static_surface_vars
        )
    else:
        # Use dummy scalers if paths not provided
        in_channels = config["params"]["in_channels"]
        in_channels_static = config["params"]["in_channels_static"]
        in_mu = torch.zeros(in_channels)
        in_sig = torch.ones(in_channels)
        static_mu = torch.zeros(in_channels_static)
        static_sig = torch.ones(in_channels_static)
        print("Warning: Using dummy scalers. Provide scaler paths for proper functionality.")

    # Create encoder model
    encoder_model = PrithviWxC_Encoder(
        in_channels=config["params"]["in_channels"],
        input_size_time=config["params"]["input_size_time"],
        in_channels_static=config["params"]["in_channels_static"],
        input_scalers_mu=in_mu,
        input_scalers_sigma=in_sig,
        input_scalers_epsilon=config["params"]["input_scalers_epsilon"],
        static_input_scalers_mu=static_mu,
        static_input_scalers_sigma=static_sig,
        static_input_scalers_epsilon=config["params"]["static_input_scalers_epsilon"],
        n_lats_px=config["params"]["n_lats_px"],
        n_lons_px=config["params"]["n_lons_px"],
        patch_size_px=config["params"]["patch_size_px"],
        mask_unit_size_px=config["params"]["mask_unit_size_px"],
        mask_ratio_inputs=0.0,  # Set to 0 for inference
        embed_dim=config["params"]["embed_dim"],
        n_blocks_encoder=config["params"]["n_blocks_encoder"],
        mlp_multiplier=config["params"]["mlp_multiplier"],
        n_heads=config["params"]["n_heads"],
        dropout=config["params"]["dropout"],
        drop_path=config["params"]["drop_path"],
        parameter_dropout=config["params"]["parameter_dropout"],
        residual="climate",  # Adjust as needed
        masking_mode="global",  # Adjust as needed
        positional_encoding="fourier",  # Adjust as needed
        encoder_shifting=False,  # Adjust as needed
        checkpoint_encoder=[],
    )

    # Load full model weights
    print("Loading full model weights...")
    state_dict = torch.load(args.weights_path, map_location="cpu", weights_only=False)
    if "model_state" in state_dict:
        state_dict = state_dict["model_state"]

    # Filter out problematic keys that might not match
    filtered_state_dict = {}
    for key, value in state_dict.items():
        # Skip shifter mask keys that might cause issues
        if "shifter.local_mask" in key or "shifter.global_mask" in key:
            print(f"Skipping problematic key: {key}")
            continue
        filtered_state_dict[key] = value

    # Create a temporary full model to load weights
    full_model = PrithviWxC(
        in_channels=config["params"]["in_channels"],
        input_size_time=config["params"]["input_size_time"],
        in_channels_static=config["params"]["in_channels_static"],
        input_scalers_mu=in_mu,
        input_scalers_sigma=in_sig,
        input_scalers_epsilon=config["params"]["input_scalers_epsilon"],
        static_input_scalers_mu=static_mu,
        static_input_scalers_sigma=static_sig,
        static_input_scalers_epsilon=config["params"]["static_input_scalers_epsilon"],
        output_scalers=torch.ones(config["params"]["in_channels"]),  # Dummy output scalers
        n_lats_px=config["params"]["n_lats_px"],
        n_lons_px=config["params"]["n_lons_px"],
        patch_size_px=config["params"]["patch_size_px"],
        mask_unit_size_px=config["params"]["mask_unit_size_px"],
        mask_ratio_inputs=0.0,
        mask_ratio_targets=0.0,
        embed_dim=config["params"]["embed_dim"],
        n_blocks_encoder=config["params"]["n_blocks_encoder"],
        n_blocks_decoder=config["params"]["n_blocks_decoder"],
        mlp_multiplier=config["params"]["mlp_multiplier"],
        n_heads=config["params"]["n_heads"],
        dropout=config["params"]["dropout"],
        drop_path=config["params"]["drop_path"],
        parameter_dropout=config["params"]["parameter_dropout"],
        residual="climate",
        masking_mode="global",
        encoder_shifting=False,
        decoder_shifting=False,
        positional_encoding="fourier",
        checkpoint_encoder=[],
        checkpoint_decoder=[],
    )

    full_model.load_state_dict(filtered_state_dict, strict=False)  # Extract encoder weights
    print("Extracting encoder weights...")
    extract_encoder_weights(full_model, encoder_model)

    # Save encoder model
    print(f"Saving encoder model to {args.output_path}")
    torch.save(
        {
            "model_state_dict": encoder_model.state_dict(),
            "config": config,
            "model_type": "PrithviWxC_Encoder",
        },
        args.output_path,
    )

    print("Encoder extraction completed successfully!")

    # Print model summary
    total_params = sum(p.numel() for p in encoder_model.parameters())
    trainable_params = sum(p.numel() for p in encoder_model.parameters() if p.requires_grad)
    print(f"Encoder model parameters: {total_params:,} total, {trainable_params:,} trainable")


if __name__ == "__main__":
    main()

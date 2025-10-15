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
AIFS Encoder Utilities

This module provides utilities for extracting, saving, and loading the complete AIFS encoder
that returns encoder embeddings only (not final predictions). This enables the encoder to be
used in multimodal fusion applications.

Key Components:
- AIFSCompleteEncoder: Complete AIFS model from inputs to encoder embeddings
- save_aifs_encoder: Save encoder checkpoint with metadata
- load_aifs_encoder: Load encoder from checkpoint
"""

import os
from datetime import datetime
from typing import Any

import torch
from torch import nn

from ..constants import (
    AIFS_GRID_POINTS,
    AIFS_PROJECTED_ENCODER_OUTPUT_DIM,
    AIFS_RAW_ENCODER_OUTPUT_DIM,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_CHECKPOINT_NAME,
    EXPECTED_INPUT_SHAPE,
    EXPECTED_OUTPUT_SHAPE,
)

try:
    # AIFS dependencies check - these imports verify AIFS availability
    import einops
    from anemoi.models.distributed.shapes import get_shape_shards

    # Verify dependencies are available
    AIFS_AVAILABLE = bool(einops and get_shape_shards)
except ImportError:
    AIFS_AVAILABLE = False


class AIFSCompleteEncoder(nn.Module):
    """
    Complete AIFS encoder that runs the full AIFS model from inputs to ENCODER OUTPUT ONLY.
    This includes ALL internal processing up to the encoder stage:
    1. Input preprocessing and normalization
    2. Edge/node data preparation
    3. Graph transformer encoding
    4. Returns ENCODER EMBEDDINGS (not final predictions)
    """

    def __init__(self, aifs_model, verbose: bool = True, device: str = "cpu"):
        """
        Initialize the complete AIFS encoder.

        Args:
            aifs_model: The full AIFS model instance (AnemoiModelInterface)
            verbose: Whether to print initialization messages
            device: Device to run on ("cpu", "cuda", "mps")
        """
        super().__init__()

        # Store the full AIFS model - access the inner model for encoder components
        self.aifs_interface = aifs_model
        self.aifs_model = aifs_model.model  # Access the actual AnemoiModelEncProcDec
        self.verbose = verbose
        self.device = device

        # AIFS model is too memory-intensive for MPS, always keep it on CPU
        # Only the projection layer will be on the target device
        self.aifs_device = "cpu"
        if self.verbose and device == "mps":
            print("Note: AIFS encoder will run on CPU (too large for MPS memory)")

        # Determine dtype based on device
        # AIFS always uses FP32 on CPU for stability
        self.use_fp16 = device in ["cuda", "mps"]
        self.dtype = torch.float16 if self.use_fp16 else torch.float32

        # Projection layer to transform AIFS encoder output to expected dimension
        # AIFS encoder produces 102 features, but downstream code expects 218
        self.output_projection = nn.Linear(
            AIFS_RAW_ENCODER_OUTPUT_DIM, AIFS_PROJECTED_ENCODER_OUTPUT_DIM, dtype=self.dtype
        )

        # Keep AIFS on CPU (always FP32), move only projection layer to target device
        self.aifs_interface = self.aifs_interface.cpu().float()
        self.aifs_model = self.aifs_interface.model

        # Move projection layer to target device
        self.output_projection = self.output_projection.to(device)
        if self.use_fp16:
            self.output_projection = self.output_projection.half()

        if self.verbose:
            print(f"Inner model type: {type(self.aifs_model)}")
            print(f"Has encoder: {hasattr(self.aifs_model, 'encoder')}")
            print(f"Has trainable_data: {hasattr(self.aifs_model, 'trainable_data')}")
            print(f"AIFS encoder on: {self.aifs_device} (FP32)")
            print(f"Projection layer on: {device} ({'FP16' if self.use_fp16 else 'FP32'})")
            print(
                f"Added projection layer: {AIFS_RAW_ENCODER_OUTPUT_DIM} -> "
                f"{AIFS_PROJECTED_ENCODER_OUTPUT_DIM} features"
            )

    def forward(self, x):
        """
        Forward pass through the complete AIFS model up to ENCODER OUTPUT ONLY

        Args:
            x: Input tensor in AIFS format [batch, time, ensemble, grid, vars]

        Returns:
            Encoder embeddings from the AIFS model (NOT final predictions)
        """
        if not AIFS_AVAILABLE:
            raise RuntimeError(
                "AIFS dependencies not available. Please install anemoi-models and einops."
            )

        # Convert input to appropriate dtype
        if self.use_fp16 and x.dtype != torch.float16:
            x = x.half()
        elif not self.use_fp16 and x.dtype != torch.float32:
            x = x.float()

        # Check input dimensions
        _, _, _, grid_size, _ = x.shape
        # Get expected grid size - handle both AIFS versions
        # Older versions have latlons_data, newer versions (1.1+) do not
        if hasattr(self.aifs_model, "latlons_data"):
            expected_grid_size = self.aifs_model.latlons_data.shape[0]
        else:
            # For newer models without latlons_data, use constant
            expected_grid_size = AIFS_GRID_POINTS

        if grid_size != expected_grid_size:
            raise ValueError(
                f"Grid size mismatch, expected {expected_grid_size} but received {grid_size}"
            )

        # Follow the EXACT same steps as AnemoiModelEncProcDec.forward() but stop at encoder
        with torch.no_grad():
            # AIFS is always on CPU (set in __init__), move input to CPU for processing
            original_device = x.device
            x_cpu = x.cpu().float()  # Convert to FP32 for CPU processing

            # Clear MPS memory if input was on MPS
            if original_device.type == "mps" and hasattr(torch.backends, "mps"):
                torch.mps.empty_cache()
            elif original_device.type == "cuda":
                torch.cuda.empty_cache()

            # Ensure model is in eval mode for inference chunking to work
            # (ANEMOI_INFERENCE_NUM_CHUNKS only applies in eval mode)
            was_training = self.aifs_model.training
            if was_training:
                self.aifs_model.eval()

            # Call the AIFS model on CPU
            try:
                full_output = self.aifs_model(x_cpu)
                if isinstance(full_output, tuple):
                    # Extract the first component (encoder output)
                    data_embeddings = full_output[0]
                else:
                    data_embeddings = full_output
            except RuntimeError as e:
                # Handle memory issues gracefully
                if "out of memory" in str(e).lower():
                    # Clear cache and suggest fallback to CPU
                    if x.device.type == "cuda":
                        torch.cuda.empty_cache()
                    elif x.device.type == "mps":
                        torch.mps.empty_cache()
                    raise RuntimeError(
                        f"AIFS model out of memory on {x.device}. "
                        f"Consider using CPU or smaller batch size: {e}"
                    ) from e
                # Handle unsupported operations
                if "not currently implemented" in str(e) or "not currently supported" in str(e):
                    raise RuntimeError(
                        f"AIFS model operation not supported on {x.device}. "
                        f"Set PYTORCH_ENABLE_MPS_FALLBACK=1 for CPU fallback: {e}"
                    ) from e
                raise RuntimeError(f"Failed to encode with AIFS model: {e}") from e
            except Exception as e:
                raise RuntimeError(f"Failed to encode with AIFS model: {e}") from e
            finally:
                # Restore training mode if it was on
                if was_training:
                    self.aifs_model.train()

            # Move embeddings back to original device and convert dtype if needed
            if original_device.type != "cpu":
                if self.verbose:
                    print(f"Moving AIFS output from CPU to {original_device}")
                # Convert to FP16 when moving to MPS/CUDA (if use_fp16 is True)
                if self.use_fp16:
                    data_embeddings = data_embeddings.half().to(original_device)
                else:
                    data_embeddings = data_embeddings.to(original_device)

            # Apply projection to transform to expected AIFS_PROJECTED_ENCODER_OUTPUT_DIM features
            # AIFS outputs AIFS_RAW_ENCODER_OUTPUT_DIM features, we need to project to
            # AIFS_PROJECTED_ENCODER_OUTPUT_DIM
            if data_embeddings.shape[-1] == 102:
                projected_data_embeddings = self.output_projection(data_embeddings)
            else:
                projected_data_embeddings = data_embeddings

        if self.verbose:
            print("AIFS encoder forward completed")
            print(f"Raw encoder output shape: {data_embeddings.shape} ({data_embeddings.dtype})")
            print(
                "Projected output shape: "
                f"{projected_data_embeddings.shape} ({projected_data_embeddings.dtype})"
            )
            print(
                f"Raw data embeddings range: "
                f"[{data_embeddings.min():.4f}, {data_embeddings.max():.4f}]"
            )
            print(
                f"Projected data embeddings range: "
                f"[{projected_data_embeddings.min():.4f}, {projected_data_embeddings.max():.4f}]"
            )

        # Return the projected encoder embeddings
        # (data embeddings represent the main climate features)
        return projected_data_embeddings


def save_aifs_encoder(
    complete_encoder: AIFSCompleteEncoder,
    output_embeddings: torch.Tensor,
    checkpoint_dir: str = "multimodal_aifs/models/extracted_models",
    filename: str = "aifs_complete_encoder.pth",
    verbose: bool = True,
) -> str:
    """
    Save AIFSCompleteEncoder to checkpoint with metadata.

    Args:
        complete_encoder: The AIFSCompleteEncoder instance to save
        output_embeddings: Sample output embeddings for shape reference
        checkpoint_dir: Directory to save the checkpoint
        filename: Filename for the checkpoint
        verbose: Whether to print progress messages

    Returns:
        Path to the saved checkpoint file
    """
    if verbose:
        print("SAVING AIFSCompleteEncoder CHECKPOINT")
        print("=" * 50)

    # Create directory for encoder checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)

    try:
        # Checkpoint file path
        checkpoint_path = os.path.join(checkpoint_dir, filename)

        # Create comprehensive checkpoint
        checkpoint = {
            "model_state_dict": complete_encoder.state_dict(),
            "model_class": "AIFSCompleteEncoder",
            "input_shape_example": "[1, 2, 1, 542080, 103]",
            "output_shape_example": list(output_embeddings.shape),
            "total_parameters": sum(p.numel() for p in complete_encoder.parameters()),
            "creation_date": datetime.now().isoformat(),
            "description": "AIFSCompleteEncoder - AIFS model from inputs to encoder embeddings",
        }

        torch.save(checkpoint, checkpoint_path)

        if verbose:
            print("AIFSCompleteEncoder checkpoint saved!")
            print(f"Path: {checkpoint_path}")
            print(f"Size: {os.path.getsize(checkpoint_path) / 1024 / 1024:.2f} MB")
            print(f"Parameters: {checkpoint['total_parameters']:,}")
            print(f"Expected output shape: {checkpoint['output_shape_example']}")

        return checkpoint_path

    except Exception as e:
        print(f"Failed to save checkpoint: {e}")
        raise


def load_aifs_encoder(
    checkpoint_path: str, aifs_model, verbose: bool = True, device: str = "cpu"
) -> AIFSCompleteEncoder:
    """
    Load AIFSCompleteEncoder from checkpoint

    Args:
        checkpoint_path: Path to the saved checkpoint
        aifs_model: The AIFS model instance to wrap
        verbose: Whether to print progress messages
        device: Device to run on ("cpu", "cuda", "mps")

    Returns:
        Loaded AIFSCompleteEncoder instance
    """
    if verbose:
        print(f"Loading AIFSCompleteEncoder from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Create new encoder instance
    encoder = AIFSCompleteEncoder(aifs_model, verbose=verbose, device=device)

    # Load the saved state
    encoder.load_state_dict(checkpoint["model_state_dict"])

    if verbose:
        print("AIFSCompleteEncoder loaded successfully!")
        print(f"Parameters: {checkpoint['total_parameters']:,}")
        print(f"Expected output: {checkpoint['output_shape_example']}")

    return encoder


def create_aifs_encoder(
    aifs_model, verbose: bool = True, device: str = "cpu"
) -> AIFSCompleteEncoder:
    """
    Create a new AIFSCompleteEncoder instance.

    Args:
        aifs_model: The AIFS model instance to wrap
        verbose: Whether to print progress messages
        device: Device to run on ("cpu", "cuda", "mps")

    Returns:
        New AIFSCompleteEncoder instance
    """
    if verbose:
        print("Creating AIFSCompleteEncoder...")

    encoder = AIFSCompleteEncoder(aifs_model, verbose=verbose, device=device)

    if verbose:
        print("AIFSCompleteEncoder created successfully")

    return encoder


def get_checkpoint_info(checkpoint_path: str) -> dict[str, Any]:
    """
    Get information about a saved checkpoint without loading the model.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dictionary with checkpoint metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Remove the actual model state dict to save memory
    info = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}

    # Add file information
    info["file_size_mb"] = os.path.getsize(checkpoint_path) / 1024 / 1024
    info["file_path"] = checkpoint_path

    return info


def validate_checkpoint(checkpoint_path: str, aifs_model, verbose: bool = True) -> bool:
    """
    Validate that a checkpoint can be loaded correctly.

    Args:
        checkpoint_path: Path to the checkpoint file
        aifs_model: The AIFS model instance to test with
        verbose: Whether to print validation messages

    Returns:
        True if checkpoint is valid, False otherwise
    """
    try:
        if verbose:
            print(f"Validating checkpoint: {checkpoint_path}")

        # Try to load the encoder
        encoder = load_aifs_encoder(checkpoint_path, aifs_model, verbose=False)

        # Check that it has the expected attributes
        assert hasattr(encoder, "aifs_model"), "Missing aifs_model attribute"
        assert hasattr(encoder, "forward"), "Missing forward method"

        # Check parameter count
        param_count = sum(p.numel() for p in encoder.parameters())
        assert param_count > 0, "No parameters found"

        if verbose:
            print("Checkpoint validation passed!")
            print(f"Parameters: {param_count:,}")

        return True

    except Exception as e:
        if verbose:
            print(f"Checkpoint validation failed: {e}")
        return False


# Configuration and constants - imported from constants.py module


def get_default_checkpoint_path() -> str:
    """Get the default path for AIFS encoder checkpoints."""
    return os.path.join(DEFAULT_CHECKPOINT_DIR, DEFAULT_CHECKPOINT_NAME)


def check_aifs_dependencies() -> bool:
    """Check if AIFS dependencies are available."""
    return AIFS_AVAILABLE


if __name__ == "__main__":
    print("AIFS Encoder Utils")
    print("=" * 30)
    print(f"AIFS dependencies available: {AIFS_AVAILABLE}")
    print(f"Default checkpoint path: {get_default_checkpoint_path()}")
    print(f"Expected input shape: {EXPECTED_INPUT_SHAPE}")
    print(f"Expected output shape: {EXPECTED_OUTPUT_SHAPE}")

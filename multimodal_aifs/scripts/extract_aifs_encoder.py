#!/usr/bin/env python3
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
Extract AIFS Encoder

This script extracts the encoder from the AIFS-Single-1.1 model for use in
multimodal fusion applications.

Usage:
    python multimodal_aifs/scripts/extract_aifs_encoder.py
"""

import sys
from pathlib import Path
from platform import system

import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Flash attention mock for MacOS (must be before importing anemoi)
if system() == "Darwin":
    import types

    flash_attn_mock = types.ModuleType("flash_attn")
    flash_attn_interface_mock = types.ModuleType("flash_attn_interface")

    def flash_attn_func(*args, **kwargs):
        """Memory-efficient flash attention mock using PyTorch's scaled_dot_product_attention."""
        query = args[0] if args else kwargs.get("q")
        if query is None:
            raise ValueError("Query tensor required for flash attention")
        key = args[1] if len(args) > 1 else kwargs.get("k", query)
        value = args[2] if len(args) > 2 else kwargs.get("v", query)

        # Transpose to PyTorch's expected format: [batch, nheads, seqlen, headdim]
        query_t = query.transpose(1, 2)
        key_t = key.transpose(1, 2)
        value_t = value.transpose(1, 2)

        # Use PyTorch's memory-efficient scaled dot product attention
        # This avoids materializing the full attention matrix
        # pylint: disable=not-callable
        output = torch.nn.functional.scaled_dot_product_attention(
            query_t, key_t, value_t, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        # Transpose back to flash attention format: [batch, seqlen, nheads, headdim]
        return output.transpose(1, 2)

    flash_attn_interface_mock.flash_attn_func = flash_attn_func  # type: ignore[attr-defined]
    flash_attn_interface_mock.flash_attn_varlen_func = flash_attn_func  # type: ignore[attr-defined]
    flash_attn_mock.flash_attn_interface = flash_attn_interface_mock  # type: ignore[attr-defined]

    sys.modules["flash_attn"] = flash_attn_mock
    sys.modules["flash_attn.flash_attn_interface"] = flash_attn_interface_mock

from anemoi.inference.runners.simple import SimpleRunner

from multimodal_aifs.core.aifs_encoder_utils import (
    AIFSCompleteEncoder,
    save_aifs_encoder,
)


def extract_aifs_encoder(
    output_dir: str = "multimodal_aifs/models/extracted_models",
    device: str | None = None,
):
    """
    Extract the encoder from AIFS-Single-1.1.

    Args:
        output_dir: Directory to save the extracted encoder
        device: Device to use for extraction (auto-detected if None)
    """
    print("\n" + "=" * 70)
    print("EXTRACTING AIFS-SINGLE-1.1 ENCODER")
    print("=" * 70)

    # Auto-detect device if not specified
    if device is None:
        # For AIFS extraction, CPU is recommended because:
        # - MPS lacks memory-efficient attention kernels (tries to allocate 97GB for 542k grid)
        # - MPS needs CPU fallback for scatter_reduce operations anyway
        # - CPU has proper memory-efficient SDPA implementation
        if torch.cuda.is_available():
            device = "cuda"
            print("\nUsing CUDA with FP16")
        else:
            device = "cpu"
            print("\nUsing CPU (recommended for AIFS due to large grid size)")
    else:
        print(f"\nUsing {device}")

    # Load AIFS-Single-1.1 model
    print("\nLoading AIFS-Single-1.1...")
    checkpoint = {"huggingface": "ecmwf/aifs-single-1.1"}
    runner = SimpleRunner(checkpoint, device=device)
    aifs_model = runner.model

    print(f"✓ Model loaded on {device}")
    print(f"  Model type: {type(aifs_model)}")

    # Create complete encoder with FP16 for memory efficiency
    print("\nCreating complete encoder wrapper (FP16 for memory efficiency)...")
    encoder = AIFSCompleteEncoder(aifs_model, verbose=True, device=device)

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

    print("\n✓ Encoder created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print("  Precision: FP16 (half precision)")

    # Generate sample output embeddings for the checkpoint
    print("\nGenerating sample output for checkpoint...")
    with torch.no_grad():
        # Create dummy input matching AIFS expected shape:
        # [batch, time, ensemble, grid_points, features]
        # AIFS-Single-1.1: 542,080 grid points, 103 input features
        # Use FP16 to reduce memory usage
        dummy_input = torch.randn(1, 2, 1, 542080, 103, device=device, dtype=torch.float16)
        sample_output = encoder(dummy_input)

    print(f"  Sample output shape: {sample_output.shape}")

    # Save encoder
    print(f"\nSaving encoder to {output_dir}...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    save_aifs_encoder(
        complete_encoder=encoder,
        output_embeddings=sample_output,
        checkpoint_dir=str(output_path),
        filename="aifs_single_1.1_encoder.pt",
    )

    print("✓ Encoder saved successfully!")
    print(f"  Location: {output_path / 'aifs_single_1.1_encoder.pt'}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    extract_aifs_encoder()

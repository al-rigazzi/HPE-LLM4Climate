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
"""Analyze whether AIFS encoder outputs retain positional encodings.

This script performs three steps:

1. Identifies the positional/forcing variables injected into the encoder input
   tensor (cos/sin latitude, longitude, julian day, local time, and insolation)
   and retrieves their feature indices directly from the AIFS checkpoint.
2. Prints representative samples and statistics for those positional channels
   using real ECMWF data prepared through ``SimpleRunner.prepare_input_tensor``.
3. Runs the AIFS encoder, then measures Pearson correlation between every
   positional channel and every encoder output dimension to highlight any
   encoder features that resemble positional encodings.

Usage example:

```
ANEMOI_INFERENCE_NUM_CHUNKS=16 \
python multimodal_aifs/scripts/analyze_encoder_positional_correlations.py \
  --zarr-path data/real_ecmwf_latest.zarr \
  --start-time 2025-10-10T00:00:00 \
  --end-time 2025-10-10T06:00:00 \
  --time-steps 2 \
  --device cpu \
  --runner-device cpu \
  --report-path results/positional_correlation_report.json
```
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from platform import system
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

# Ensure local package imports resolve
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from multimodal_aifs.constants import AIFS_INPUT_VARIABLES, ALL_AIFS_VARIABLES
from multimodal_aifs.core.aifs_encoder_utils import AIFSCompleteEncoder
from multimodal_aifs.utils.zarr_data_loader import ZarrClimateLoader

POSITIONAL_FORCINGS = [
    "cos_latitude",
    "sin_latitude",
    "cos_longitude",
    "sin_longitude",
    "cos_julian_day",
    "sin_julian_day",
    "cos_local_time",
    "sin_local_time",
    "insolation",
]


def configure_flash_attention() -> None:
    """Install a flash-attention fallback for macOS environments."""

    if system() != "Darwin":
        return

    import types

    flash_attn_mock = types.ModuleType("flash_attn")
    flash_attn_interface_mock = types.ModuleType("flash_attn_interface")

    def flash_attn_func(*args, **kwargs):
        query = args[0] if args else kwargs.get("q")
        if query is None:
            raise ValueError("Query tensor required for flash attention fallback")

        key = args[1] if len(args) > 1 else kwargs.get("k", query)
        value = args[2] if len(args) > 2 else kwargs.get("v", query)

        query_t = query.transpose(1, 2)
        key_t = key.transpose(1, 2)
        value_t = value.transpose(1, 2)

        output = torch.nn.functional.scaled_dot_product_attention(
            query_t, key_t, value_t, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        return output.transpose(1, 2)

    flash_attn_interface_mock.flash_attn_func = flash_attn_func  # type: ignore[attr-defined]
    flash_attn_interface_mock.flash_attn_varlen_func = flash_attn_func  # type: ignore[attr-defined]
    flash_attn_mock.flash_attn_interface = flash_attn_interface_mock  # type: ignore[attr-defined]

    sys.modules["flash_attn"] = flash_attn_mock
    sys.modules["flash_attn.flash_attn_interface"] = flash_attn_interface_mock


def bypass_environment_validation() -> None:
    """Override AIFS checkpoint environment validation for local analysis."""

    import anemoi.inference.checkpoint as checkpoint_module

    if hasattr(checkpoint_module, "Checkpoint"):

        def patched_validate(self, on_difference=None):  # pylint: disable=unused-argument
            return None

        checkpoint_module.Checkpoint.validate_environment = patched_validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=Path("data/real_ecmwf_latest.zarr"),
        help="Path to ECMWF Zarr dataset in grid_point format.",
    )
    parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help="Optional ISO8601 timestamp for slice start.",
    )
    parser.add_argument(
        "--end-time",
        type=str,
        default=None,
        help="Optional ISO8601 timestamp for slice end.",
    )
    parser.add_argument(
        "--time-offset",
        type=int,
        default=0,
        help="Offset (index) to use when start/end are omitted.",
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        default=2,
        help="Number of timesteps to analyze.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for encoder inference (cpu, cuda, mps).",
    )
    parser.add_argument(
        "--runner-device",
        type=str,
        default="cpu",
        help="Device for SimpleRunner instantiation (keep cpu unless enough GPU memory).",
    )
    parser.add_argument(
        "--permutation-mode",
        choices=("roll", "none"),
        default="none",
        help="Optional permutation (roll) applied before correlation for stress tests.",
    )
    parser.add_argument(
        "--grid-shift",
        type=int,
        default=50000,
        help="Grid shift for roll permutations (when permutation-mode=roll).",
    )
    parser.add_argument(
        "--sample-grid-points",
        type=int,
        default=8,
        help="Number of grid points to print for positional channel samples.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K correlations to report per positional channel.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional JSON file to store metrics and correlations.",
    )
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=1,
        help="Factor to downsample the grid dimension via mean pooling (must divide the grid size).",
    )
    parser.add_argument(
        "--inference-chunks",
        type=int,
        default=16,
        help="Value for ANEMOI_INFERENCE_NUM_CHUNKS to reduce encoder memory (set to 0 to skip).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational logs.",
    )
    return parser.parse_args()


def log(message: str, quiet: bool = False) -> None:
    if not quiet:
        print(message)


def ensure_inference_chunks(chunks: int | None) -> None:
    if chunks and chunks > 0:
        os.environ.setdefault("ANEMOI_INFERENCE_NUM_CHUNKS", str(chunks))


def load_dataset_slice(loader: ZarrClimateLoader, args: argparse.Namespace):
    if args.start_time is not None or args.end_time is not None:
        data = loader.load_time_range(args.start_time, args.end_time)
    else:
        if loader.ds is None:
            raise RuntimeError("Failed to initialize Zarr dataset")
        slice_start = max(args.time_offset, 0)
        slice_end = slice_start + args.time_steps
        if slice_end > loader.ds.sizes.get("time", 0):
            raise ValueError(
                f"Requested time window [{slice_start}, {slice_end}) exceeds dataset length "
                f"({loader.ds.sizes.get('time', 0)})."
            )
        data = loader.ds.isel(time=slice(slice_start, slice_end))
    if data.sizes.get("time", 0) < args.time_steps:
        raise ValueError(
            f"Dataset slice has {data.sizes.get('time', 0)} time steps but --time-steps={args.time_steps}."
        )
    return data


def build_input_state(dataset, time_steps: int) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for var in ALL_AIFS_VARIABLES:
        if var not in dataset.data_vars:
            raise KeyError(f"Variable '{var}' missing from dataset slice")
        values = dataset[var].values
        if values.shape[0] < time_steps:
            raise ValueError(
                f"Variable '{var}' has {values.shape[0]} timesteps, expected >= {time_steps}"
            )
        fields[var] = values[:time_steps]
    date = pd.Timestamp(dataset.time.values[0]).to_pydatetime()
    return {"date": date, "fields": fields}


def prepare_input_tensor(
    runner,
    input_state: dict[str, Any],
    time_steps: int,
    quiet: bool,
) -> np.ndarray:
    if not hasattr(runner, "constant_forcings_inputs"):
        runner.constant_forcings_inputs = runner.create_constant_forcings_inputs(input_state)
    if not hasattr(runner, "dynamic_forcings_inputs"):
        runner.dynamic_forcings_inputs = runner.create_dynamic_forcings_inputs(input_state)
    input_tensor = runner.prepare_input_tensor(input_state)
    input_tensor_np = np.asarray(input_tensor)
    if input_tensor_np.shape[0] < time_steps:
        raise ValueError(
            f"Prepared tensor has {input_tensor_np.shape[0]} timesteps but --time-steps={time_steps}."
        )
    log(f"SimpleRunner tensor shape: {input_tensor_np.shape}", quiet)
    return input_tensor_np[:time_steps]


def summarize_positional_channels(
    input_tensor: np.ndarray,
    mapping: dict[str, int],
    sample_points: int,
    quiet: bool,
) -> dict[str, Any]:
    samples: dict[str, Any] = {}
    time_dim, _, grid_dim = input_tensor.shape
    indices = list(range(min(sample_points, grid_dim)))
    log("\n=== Positional Channel Samples ===", quiet)
    for name in POSITIONAL_FORCINGS:
        idx = mapping.get(name)
        if idx is None:
            log(f"- {name}: not present in checkpoint mapping", quiet)
            continue
        channel = input_tensor[:, idx, :]
        stats = {
            "index": idx,
            "min": float(channel.min()),
            "max": float(channel.max()),
            "mean": float(channel.mean()),
            "std": float(channel.std()),
            "samples": channel[0, indices].tolist(),
        }
        samples[name] = stats
        log(
            f"- {name:>15} (idx={idx:>3}): min={stats['min']:.4f}, max={stats['max']:.4f}, "
            f"mean={stats['mean']:.4f}, std={stats['std']:.4f}",
            quiet,
        )
        log(f"  Time0 first {len(indices)} grid values: {stats['samples']}", quiet)
        if time_dim > 1:
            extra = channel[1, indices].tolist()
            log(f"  Time1 first {len(indices)} grid values: {extra}", quiet)
    return samples


def maybe_roll_tensor(tensor: torch.Tensor, mode: str, shift: int) -> torch.Tensor:
    if mode != "roll":
        return tensor
    grid_dim = 3  # [batch, time, ensemble, grid, vars]
    return torch.roll(tensor, shifts=shift % tensor.shape[grid_dim], dims=grid_dim)


def compute_correlations(
    positional_data: dict[str, np.ndarray],
    encoder_embeddings: torch.Tensor,
    top_k: int,
) -> dict[str, list[dict[str, float]]]:
    # encoder_embeddings: [grid_points, embedding_dim]
    enc_np = encoder_embeddings.detach().cpu().numpy()
    grid_points, embed_dim = enc_np.shape
    result: dict[str, list[dict[str, float]]] = {}

    for name, channel in positional_data.items():
        if channel.ndim == 2:
            pos_vec = channel.mean(axis=0)
        else:
            pos_vec = channel.reshape(grid_points)
        pos_center = pos_vec - pos_vec.mean()
        pos_norm = np.linalg.norm(pos_center)
        if pos_norm < 1e-8:
            continue
        correlations: list[dict[str, float]] = []
        for feat_idx in range(embed_dim):
            enc_vec = enc_np[:, feat_idx]
            enc_center = enc_vec - enc_vec.mean()
            enc_norm = np.linalg.norm(enc_center)
            if enc_norm < 1e-8:
                continue
            corr = float(np.dot(pos_center, enc_center) / (pos_norm * enc_norm))
            correlations.append({"feature": int(feat_idx), "correlation": corr})
        correlations.sort(key=lambda d: abs(d["correlation"]), reverse=True)
        result[name] = correlations[:top_k]
    return result


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    quiet = bool(args.quiet)
    ensure_inference_chunks(args.inference_chunks)

    log("Configuring flash-attention fallback and environment patch...", quiet)
    configure_flash_attention()
    bypass_environment_validation()

    log("Loading AIFS SimpleRunner...", quiet)
    from anemoi.inference.runners.simple import SimpleRunner

    checkpoint = {"huggingface": "ecmwf/aifs-single-1.1"}
    runner = SimpleRunner(checkpoint, device=args.runner_device)
    mapping = runner.checkpoint.variable_to_input_tensor_index

    log("Loading dataset slice...", quiet)
    loader = ZarrClimateLoader(str(args.zarr_path))
    dataset_slice = load_dataset_slice(loader, args)
    log(f"Selected slice dims: {dict(dataset_slice.sizes)}", quiet)

    log("Preparing input tensor via SimpleRunner...", quiet)
    input_state = build_input_state(dataset_slice, args.time_steps)
    input_tensor_np = prepare_input_tensor(runner, input_state, args.time_steps, quiet)

    if input_tensor_np.shape[1] != AIFS_INPUT_VARIABLES:
        raise RuntimeError(
            f"Unexpected feature count {input_tensor_np.shape[1]} (expected {AIFS_INPUT_VARIABLES})."
        )

    positional_channels = {
        name: input_tensor_np[:, mapping[name], :]
        for name in POSITIONAL_FORCINGS
        if name in mapping
    }

    positional_summaries = summarize_positional_channels(
        input_tensor_np,
        mapping,
        args.sample_grid_points,
        quiet,
    )

    log("Converting tensor to AIFS encoder format...", quiet)
    tensor = torch.from_numpy(input_tensor_np).float()
    tensor = tensor.permute(0, 2, 1)  # [time, grid, vars]
    tensor = tensor.unsqueeze(0).unsqueeze(2)  # [batch, time, ensemble, grid, vars]

    tensor = maybe_roll_tensor(tensor, args.permutation_mode, args.grid_shift)

    log("Running AIFS encoder forward pass...", quiet)
    encoder = AIFSCompleteEncoder(runner.model, verbose=not quiet, device=args.device)
    encoder.output_projection = nn.Identity().to(torch.device(args.device))
    log("Configured encoder to return raw 102-D embeddings (projection disabled).", quiet)
    tensor = tensor.to(args.device)
    with torch.no_grad():
        embeddings = encoder(tensor)
    log(f"Encoder output shape: {tuple(embeddings.shape)}", quiet)

    # Flatten to [grid_points, embedding_dim]
    if embeddings.dim() == 4:
        encoder_grid = embeddings.squeeze(0).squeeze(0)
    elif embeddings.dim() == 3:
        encoder_grid = embeddings.squeeze(0)
    else:
        raise RuntimeError(f"Unexpected encoder output dimensions: {embeddings.shape}")

    downsample_factor = max(1, args.downsample_factor)
    if downsample_factor > 1:
        if encoder_grid.shape[0] % downsample_factor != 0:
            raise ValueError(
                f"Grid size {encoder_grid.shape[0]} not divisible by downsample factor {downsample_factor}."
            )
        log(
            f"Downsampling encoder grid by factor {downsample_factor} (mean pooling along grid)...",
            quiet,
        )
        encoder_grid = encoder_grid.reshape(-1, downsample_factor, encoder_grid.shape[-1]).mean(
            dim=1
        )
        # Downsample positional channels to match
        for name, channel in positional_channels.items():
            # channel shape: [time, grid]
            reshaped = channel.reshape(channel.shape[0], downsample_factor, -1)
            positional_channels[name] = reshaped.mean(axis=1)
        log(f"Downsampled encoder shape: {tuple(encoder_grid.shape)}", quiet)

    log("Computing positional correlations...", quiet)
    correlations = compute_correlations(positional_channels, encoder_grid, args.top_k)

    log("\n=== Top correlations per positional variable ===", quiet)
    for name, entries in correlations.items():
        if not entries:
            log(f"- {name}: no valid correlations (zero variance)", quiet)
            continue
        tops = ", ".join(f"feature {e['feature']} (corr={e['correlation']:+.4f})" for e in entries)
        log(f"- {name}: {tops}", quiet)

    result = {
        "input_tensor_shape": input_tensor_np.shape,
        "encoder_output_shape": list(embeddings.shape),
        "positional_indices": {name: mapping.get(name) for name in POSITIONAL_FORCINGS},
        "positional_samples": positional_summaries,
        "top_correlations": correlations,
        "permutation_mode": args.permutation_mode,
        "grid_shift": args.grid_shift if args.permutation_mode == "roll" else None,
        "downsample_factor": downsample_factor,
    }
    return result


def main() -> None:
    args = parse_args()
    result = run_analysis(args)
    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        with args.report_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        log(f"Report written to {args.report_path}", args.quiet)


if __name__ == "__main__":
    main()

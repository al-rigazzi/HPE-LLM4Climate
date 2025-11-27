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
"""Check whether the AIFS encoder preserves positional information.

This script loads a real ECMWF Zarr dataset, runs the AIFS encoder on the
corresponding tensors, applies a permutation (grid roll or random permutation)
across the spatial grid, and compares the resulting embeddings before and after
alignment. When positional encodings are retained, the aligned embeddings will
remain measurably different.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from platform import system
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def configure_flash_attention() -> None:
    """Provide a flash attention fallback for MacOS systems."""

    if system() != "Darwin":
        return

    import types

    import torch.nn.functional as F_local

    flash_attn_mock = types.ModuleType("flash_attn")
    flash_attn_interface_mock = types.ModuleType("flash_attn_interface")

    def flash_attn_func(*args, **kwargs):
        """Use PyTorch scaled dot product attention as a safe fallback."""

        query = args[0] if args else kwargs.get("q")
        if query is None:
            raise ValueError("Query tensor required for flash attention fallback")

        key = args[1] if len(args) > 1 else kwargs.get("k", query)
        value = args[2] if len(args) > 2 else kwargs.get("v", query)

        query_t = query.transpose(1, 2)
        key_t = key.transpose(1, 2)
        value_t = value.transpose(1, 2)

        output = F_local.scaled_dot_product_attention(
            query_t, key_t, value_t, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        return output.transpose(1, 2)

    flash_attn_interface_mock.flash_attn_func = flash_attn_func  # type: ignore[attr-defined]
    flash_attn_interface_mock.flash_attn_varlen_func = flash_attn_func  # type: ignore[attr-defined]
    flash_attn_mock.flash_attn_interface = flash_attn_interface_mock  # type: ignore[attr-defined]

    sys.modules["flash_attn"] = flash_attn_mock
    sys.modules["flash_attn.flash_attn_interface"] = flash_attn_interface_mock


configure_flash_attention()

from anemoi.inference.runners.simple import SimpleRunner  # noqa: E402  (needs flash attn patch)

from multimodal_aifs.constants import (  # noqa: E402
    AIFS_GRID_POINTS,
    AIFS_INPUT_VARIABLES,
    AIFS_RAW_ENCODER_OUTPUT_DIM,
)
from multimodal_aifs.core.aifs_encoder_utils import AIFSCompleteEncoder  # noqa: E402
from multimodal_aifs.utils.zarr_data_loader import ZarrClimateLoader  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether the AIFS encoder retains positional information by comparing "
            "encoder outputs under grid permutations."
        )
    )
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=Path("data/real_ecmwf_latest.zarr"),
        help="Path to the ECMWF Zarr dataset prepared in AIFS grid-point format.",
    )
    parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help="Optional start timestamp (ISO8601) to slice the dataset."
        " When omitted, --time-offset is used instead.",
    )
    parser.add_argument(
        "--end-time",
        type=str,
        default=None,
        help="Optional end timestamp (ISO8601) to slice the dataset."
        " When omitted, --time-steps from the offset will be selected.",
    )
    parser.add_argument(
        "--time-offset",
        type=int,
        default=0,
        help="Time index offset (used when start/end timestamps are not provided).",
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        default=2,
        help="Number of consecutive timesteps to load for the analysis.",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=16,
        help="Number of inference chunks for encoder forward pass (memory optimization)",
    )
    parser.add_argument(
        "--permutation-mode",
        choices=("roll", "random"),
        default="roll",
        help="Permutation strategy to apply across the grid dimension.",
    )
    parser.add_argument(
        "--grid-shift",
        type=int,
        default=32768,
        help="Grid shift (number of points) used when permutation-mode=roll.",
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
        help="Device for loading the AIFS SimpleRunner (keep cpu unless you have ample GPU memory).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ecmwf/aifs-single-1.1",
        help="Hugging Face identifier for the AIFS checkpoint.",
    )
    parser.add_argument(
        "--skip-normalization",
        action="store_true",
        help="Skip normalization inside the Zarr loader (not recommended).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-3,
        help="Relative L2 threshold above which positional encoding is considered present.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional JSON file where metrics will be persisted.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational logs (errors are still reported).",
    )
    return parser.parse_args()


def log(message: str, quiet: bool = False) -> None:
    """Print a message when not in quiet mode."""

    if not quiet:
        print(message)


def load_dataset_slice(loader: ZarrClimateLoader, args: argparse.Namespace):
    """Load the requested time segment from the Zarr dataset."""

    if args.start_time is not None or args.end_time is not None:
        return loader.load_time_range(args.start_time, args.end_time)

    if loader.ds is None:
        raise RuntimeError("Zarr dataset failed to load.")

    slice_start = max(args.time_offset, 0)
    slice_end = slice_start + args.time_steps

    if "time" not in loader.ds.dims:
        raise ValueError("Dataset does not contain a 'time' dimension")

    if slice_end > loader.ds.sizes["time"]:
        raise ValueError(
            f"Requested time window [{slice_start}, {slice_end}) exceeds available timesteps "
            f"({loader.ds.sizes['time']}). Provide --start-time/--end-time instead."
        )

    return loader.ds[loader.available_variables].isel(time=slice(slice_start, slice_end))


def find_grid_dimension(tensor: torch.Tensor) -> int | None:
    """Return the dimension index matching the grid size, if present."""

    for dim, size in enumerate(tensor.shape):
        if size == AIFS_GRID_POINTS:
            return dim
    return None


def roll_permutation(
    tensor: torch.Tensor, grid_dim: int, shift: int
) -> tuple[torch.Tensor, Callable[[torch.Tensor, int], torch.Tensor]]:
    """Apply a circular roll across the grid dimension and return align fn."""

    effective_shift = shift % tensor.shape[grid_dim]
    permuted = torch.roll(tensor, shifts=effective_shift, dims=grid_dim)

    def align_fn(output: torch.Tensor, output_grid_dim: int) -> torch.Tensor:
        return torch.roll(output, shifts=-effective_shift, dims=output_grid_dim)

    return permuted, align_fn


def random_permutation(
    tensor: torch.Tensor, grid_dim: int
) -> tuple[torch.Tensor, Callable[[torch.Tensor, int], torch.Tensor]]:
    """Apply a random permutation across the grid dimension and build the inverse."""

    grid_size = tensor.shape[grid_dim]
    perm = torch.randperm(grid_size, device=tensor.device)
    permuted = tensor.index_select(dim=grid_dim, index=perm)

    inverse = torch.empty_like(perm)
    inverse.scatter_(0, perm, torch.arange(grid_size, device=tensor.device))

    def align_fn(output: torch.Tensor, output_grid_dim: int) -> torch.Tensor:
        return output.index_select(dim=output_grid_dim, index=inverse)

    return permuted, align_fn


def flatten_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten tensor to a 1-D vector for norm comparisons."""

    return tensor.reshape(-1)


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute relative L2 error between tensors."""

    diff = flatten_tensor(a - b)
    ref = flatten_tensor(a)
    denom = torch.linalg.vector_norm(ref).clamp_min(1e-8)
    rel = torch.linalg.vector_norm(diff) / denom
    return float(rel.item())


def summary_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    """Return summary statistics between two tensors."""

    diff = a - b
    return {
        "relative_l2": relative_l2(a, b),
        "mean_abs": float(diff.abs().mean().item()),
        "max_abs": float(diff.abs().max().item()),
    }


def grid_cosine_similarity(
    baseline: torch.Tensor, candidate: torch.Tensor, grid_dim: int
) -> float | None:
    """Compute cosine similarity averaged over grid tokens if possible."""

    if baseline.shape != candidate.shape or baseline.ndim < 2:
        return None

    moved_base = baseline.movedim(grid_dim, -2)
    moved_candidate = candidate.movedim(grid_dim, -2)

    if moved_base.shape[-1] != AIFS_RAW_ENCODER_OUTPUT_DIM:
        return None

    base_tokens = moved_base.reshape(-1, moved_base.shape[-1])
    candidate_tokens = moved_candidate.reshape(-1, moved_candidate.shape[-1])

    cos = F.cosine_similarity(base_tokens, candidate_tokens, dim=-1)
    return float(cos.mean().item())


def compare_embeddings(
    baseline: torch.Tensor,
    permuted: torch.Tensor,
    aligned: torch.Tensor | None,
    grid_dim: int | None,
) -> dict[str, Any]:
    """Compute comparison statistics for raw and aligned embeddings."""

    metrics: dict[str, Any] = {
        "raw": summary_stats(baseline, permuted),
        "aligned": None,
        "aligned_grid_cosine": None,
    }

    if aligned is not None:
        metrics["aligned"] = summary_stats(baseline, aligned)
        if grid_dim is not None:
            metrics["aligned_grid_cosine"] = grid_cosine_similarity(baseline, aligned, grid_dim)

    return metrics


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the positional encoding verification workflow."""

    quiet = bool(args.quiet)
    log("Loading AIFS SimpleRunner...", quiet)

    checkpoint = {"huggingface": args.checkpoint}
    runner = SimpleRunner(checkpoint, device=args.runner_device)
    aifs_model = runner.model

    log("Initializing AIFSCompleteEncoder wrapper...", quiet)
    os.environ.setdefault("ANEMOI_INFERENCE_NUM_CHUNKS", str(args.num_chunks))
    log(
        f"ANEMOI_INFERENCE_NUM_CHUNKS set to {os.environ['ANEMOI_INFERENCE_NUM_CHUNKS']}.",
        quiet,
    )

    encoder = AIFSCompleteEncoder(aifs_model, verbose=not quiet, device=args.device)
    encoder.output_projection = nn.Identity().to(torch.device(args.device))
    log("Configured encoder to return raw 102-D embeddings (projection disabled).", quiet)

    log(f"Loading Zarr dataset from {args.zarr_path}...", quiet)
    loader = ZarrClimateLoader(str(args.zarr_path))

    climate_slice = load_dataset_slice(loader, args)
    log(
        f"Selected data slice with dimensions: {dict(climate_slice.sizes)}",
        quiet,
    )

    use_fp16 = torch.device(args.device).type in ("cuda", "mps")

    log("Converting dataset to AIFS tensor (5D)...", quiet)
    tensor_5d = loader.to_aifs_tensor(
        climate_slice,
        batch_size=1,
        normalize=not args.skip_normalization,
        device=args.device,
        use_fp16=use_fp16,
        runner=runner,
        use_forcing_pipeline=True,
    )

    if tensor_5d.shape[-1] != AIFS_INPUT_VARIABLES:
        raise RuntimeError(
            f"Unexpected variable count {tensor_5d.shape[-1]} (expected {AIFS_INPUT_VARIABLES})."
        )

    if tensor_5d.shape[1] < args.time_steps:
        raise ValueError(
            f"Requested {args.time_steps} timesteps but only {tensor_5d.shape[1]} available in slice."
        )

    tensor_5d = tensor_5d[:, : args.time_steps]

    grid_dim_input = find_grid_dimension(tensor_5d)
    if grid_dim_input is None:
        raise RuntimeError("Failed to locate grid dimension in input tensor.")

    log("Running encoder on baseline tensor...", quiet)
    with torch.no_grad():
        baseline_embeddings = encoder(tensor_5d)

    log("Applying grid permutation and rerunning encoder...", quiet)
    if args.permutation_mode == "roll":
        permuted_tensor, align_fn = roll_permutation(tensor_5d, grid_dim_input, args.grid_shift)
    else:
        permuted_tensor, align_fn = random_permutation(tensor_5d, grid_dim_input)

    with torch.no_grad():
        permuted_embeddings = encoder(permuted_tensor)

    grid_dim_output = find_grid_dimension(baseline_embeddings)
    aligned_embeddings = None
    if grid_dim_output is not None:
        aligned_embeddings = align_fn(permuted_embeddings, grid_dim_output)
    else:
        log(
            "Encoder output does not expose the grid dimension explicitly; aligned metrics skipped.",
            quiet,
        )

    metrics = compare_embeddings(
        baseline_embeddings, permuted_embeddings, aligned_embeddings, grid_dim_output
    )

    positional_signal = None
    if metrics["aligned"] is not None:
        positional_signal = metrics["aligned"]["relative_l2"] > args.threshold

    result = {
        "input_shape": list(tensor_5d.shape),
        "encoder_output_shape": list(baseline_embeddings.shape),
        "grid_dimension_input": grid_dim_input,
        "grid_dimension_output": grid_dim_output,
        "permutation_mode": args.permutation_mode,
        "grid_shift": args.grid_shift if args.permutation_mode == "roll" else None,
        "metrics": metrics,
        "threshold": args.threshold,
        "positional_encoding_detected": positional_signal,
    }

    if aligned_embeddings is None and args.permutation_mode == "random":
        result["note"] = (
            "Encoder output lacks an explicit grid dimension; cannot align random permutation results."
        )

    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        with args.report_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        log(f"Metrics written to {args.report_path}", quiet)

    rel_raw = metrics["raw"]["relative_l2"]
    aligned_rel = None
    if metrics["aligned"] is not None:
        aligned_rel = metrics["aligned"]["relative_l2"]

    log("\n=== Positional Encoding Check Summary ===", quiet)
    log(f"Input tensor shape: {tensor_5d.shape}", quiet)
    log(f"Encoder output shape: {baseline_embeddings.shape}", quiet)
    log(f"Raw relative L2 difference: {rel_raw:.6f}", quiet)
    if aligned_rel is not None:
        log(f"Aligned relative L2 difference: {aligned_rel:.6f}", quiet)
        verdict = (
            "Positional encoding likely retained"
            if positional_signal
            else "Permutation invariance detected"
        )
        log(f"Verdict (threshold={args.threshold}): {verdict}", quiet)
        if metrics["aligned_grid_cosine"] is not None:
            log(
                f"Grid-token cosine similarity (aligned): {metrics['aligned_grid_cosine']:.6f}",
                quiet,
            )
    else:
        log("Aligned comparison unavailable (no grid dimension in encoder output).", quiet)

    return result


def main() -> None:
    """Entry point."""

    args = parse_args()
    run_check(args)


if __name__ == "__main__":
    main()

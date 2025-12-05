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
"""Device utilities for selecting and configuring compute devices."""

from __future__ import annotations

import warnings
from contextlib import contextmanager, nullcontext
from typing import Iterator

import torch


def _is_mps_supported() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def _cuda_supports_bf16() -> bool:
    return torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()


def _format_device(device: torch.device) -> str:
    if device.index is not None:
        return f"{device.type}:{device.index}"
    return device.type


def _select_cuda_device(spec: str = "cuda") -> torch.device:
    if not torch.cuda.is_available():
        return _fallback_device("cuda")
    if ":" in spec:
        return torch.device(spec)
    return torch.device(f"cuda:{torch.cuda.current_device()}")


def _select_mps_device(spec: str = "mps") -> torch.device:
    if not _is_mps_supported():
        return _fallback_device("mps")
    if ":" in spec:
        return torch.device(spec)
    return torch.device("mps:0")


def _canonicalize_device(device: torch.device) -> torch.device:
    if device.type == "cuda":
        return _select_cuda_device(_format_device(device))
    if device.type == "mps":
        return _select_mps_device(_format_device(device))
    return device


def _fallback_device(requested: str) -> torch.device:
    fallback = get_best_device()
    if fallback.type != requested:
        warnings.warn(
            (
                f"Requested device '{requested}' is not available. "
                f"Falling back to '{_format_device(fallback)}'."
            ),
            RuntimeWarning,
            stacklevel=3,
        )
    return fallback


def get_best_device() -> torch.device:
    """
    Select the best available device for computation.

    Priority order:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon GPU)
    3. CPU

    Returns
    -------
    torch.device
        The best available device for computation.
    """

    if torch.cuda.is_available():
        # torch.device("cuda") does not resolve to a concrete index, which breaks
        # APIs such as torch.cuda.set_device. Always bind explicitly to the
        # current CUDA device so downstream code receives a fully qualified
        # device handle.
        default_index = torch.cuda.current_device()
        return torch.device(f"cuda:{default_index}")
    if _is_mps_supported():
        return torch.device("mps:0")
    return torch.device("cpu")


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Normalize user-provided device hints to a concrete :class:`torch.device`."""

    if device is None or (
        isinstance(device, str) and device.lower() in {"auto", "best", "default"}
    ):
        return get_best_device()

    device_str = _format_device(device) if isinstance(device, torch.device) else str(device)
    normalized = device_str.lower()

    if normalized.startswith("cuda"):
        selected = _select_cuda_device(normalized)
    elif normalized.startswith("mps"):
        selected = _select_mps_device(normalized)
    else:
        selected = _canonicalize_device(torch.device(normalized))

    return selected


def configure_device_for_max_perf(device: torch.device) -> None:
    """Enable CUDA performance knobs such as TF32 and cuDNN benchmarking."""

    if device.type != "cuda" or not torch.cuda.is_available():
        return

    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def device_to_str(device: torch.device) -> str:
    """Return a canonical string representation for a :class:`torch.device`."""
    return _format_device(device)


def supports_amp(device: torch.device) -> bool:
    """Return True only when BF16 autocast is supported on the target device."""

    if device.type == "cuda":
        return _cuda_supports_bf16()
    return False


@contextmanager
def autocast_if_available(device: torch.device, dtype: torch.dtype | None = None) -> Iterator[None]:
    """Autocast context manager that no-ops on unsupported devices."""

    requested_dtype = dtype
    if requested_dtype == torch.float16:
        requested_dtype = torch.bfloat16

    dtype_to_use = requested_dtype
    if dtype_to_use is None and device.type == "cuda" and _cuda_supports_bf16():
        dtype_to_use = torch.bfloat16

    if device.type == "cuda" and dtype_to_use == torch.bfloat16:
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            yield
        return

    with nullcontext():
        yield

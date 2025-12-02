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

from contextlib import contextmanager, nullcontext
from typing import Iterator

import torch


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
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Normalize user-provided device hints to a concrete :class:`torch.device`."""

    if isinstance(device, torch.device):
        return device

    if device is None or str(device).lower() in {"auto", "best", "default"}:
        return get_best_device()

    return torch.device(str(device))


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

    if device.index is not None:
        return f"{device.type}:{device.index}"
    return device.type


def supports_amp(device: torch.device) -> bool:
    """Return True when autocast is supported on the target device."""

    return device.type in {"cuda", "mps"}


@contextmanager
def autocast_if_available(
    device: torch.device, dtype: torch.dtype | None = None
) -> Iterator[None]:
    """Autocast context manager that no-ops on unsupported devices."""

    if device.type == "cuda" and torch.cuda.is_available():
        target_dtype = dtype or torch.float16
        with torch.cuda.amp.autocast(dtype=target_dtype):
            yield
        return

    if device.type == "mps" and hasattr(torch, "autocast"):
        target_dtype = dtype or torch.float16
        with torch.autocast(device_type="mps", dtype=target_dtype):
            yield
        return

    with nullcontext():
        yield

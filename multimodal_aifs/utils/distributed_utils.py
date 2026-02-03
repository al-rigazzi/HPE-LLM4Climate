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
Distributed training utilities for multi-node, multi-GPU training.

This module provides utilities for setting up and managing distributed
training across multiple nodes and GPUs using PyTorch's distributed
data parallel (DDP) and NCCL backend.
"""

import os
import socket
from dataclasses import dataclass
from typing import TypeVar

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

T = TypeVar("T", bound=nn.Module)


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""

    backend: str = "nccl"
    init_method: str = "env://"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "29500"


def get_distributed_config_from_env() -> DistributedConfig:
    """
    Get distributed configuration from environment variables.

    This reads SLURM environment variables and sets up the distributed
    configuration accordingly.

    Returns:
        DistributedConfig with values from environment
    """
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))

    master_addr = os.environ.get("MASTER_ADDR", "")
    if not master_addr:
        if "SLURM_NODELIST" in os.environ:
            master_addr = _get_master_addr_from_slurm()
        else:
            master_addr = "localhost"

    master_port = os.environ.get("MASTER_PORT", "29500")

    return DistributedConfig(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        master_addr=master_addr,
        master_port=master_port,
    )


def _get_master_addr_from_slurm() -> str:
    """
    Extract master address from SLURM_NODELIST.

    Returns:
        IP address or hostname of the master node
    """
    nodelist = os.environ.get("SLURM_NODELIST", "")
    if not nodelist:
        return "localhost"

    if "[" in nodelist:
        prefix = nodelist.split("[")[0]
        first_node = nodelist.split("[")[1].split(",")[0].split("-")[0].rstrip("]")
        hostname = f"{prefix}{first_node}"
    else:
        hostname = nodelist.split(",")[0]

    try:
        addr = socket.gethostbyname(hostname)
        return addr
    except socket.gaierror:
        return hostname


def is_distributed() -> bool:
    """Check if distributed training is enabled."""
    return dist.is_initialized()


def should_use_distributed() -> bool:
    """
    Check if distributed training should be used based on environment.

    This checks SLURM/torchrun environment variables before DDP is initialized.
    Use this to decide whether to call setup_distributed().
    """
    # Already initialized
    if dist.is_initialized():
        return True

    # Check for SLURM
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    return world_size > 1


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    if not is_distributed():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """Get the rank of the current process."""
    if not is_distributed():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get the total number of processes."""
    if not is_distributed():
        return 1
    return dist.get_world_size()


def get_local_rank() -> int:
    """Get the local rank (GPU index on this node)."""
    return int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))


def setup_distributed(config: DistributedConfig | None = None) -> DistributedConfig:
    """
    Initialize distributed training.

    Args:
        config: Optional distributed configuration. If None, reads from environment.

    Returns:
        The distributed configuration used
    """
    if config is None:
        config = get_distributed_config_from_env()

    if config.world_size <= 1:
        return config

    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = config.master_port
    os.environ["WORLD_SIZE"] = str(config.world_size)
    os.environ["RANK"] = str(config.rank)
    os.environ["LOCAL_RANK"] = str(config.local_rank)

    if not dist.is_initialized():
        # Use 20-minute timeout for large model loading (default is 10 min)
        from datetime import timedelta
        timeout = timedelta(minutes=20)
        dist.init_process_group(
            backend=config.backend,
            init_method=config.init_method,
            world_size=config.world_size,
            rank=config.rank,
            timeout=timeout,
        )

    if torch.cuda.is_available():
        torch.cuda.set_device(config.local_rank)

    return config


def cleanup_distributed() -> None:
    """Clean up distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_ddp(
    model: T,
    local_rank: int,
    find_unused_parameters: bool = False,
    broadcast_buffers: bool = True,
) -> T | DDP:
    """
    Wrap a model with DistributedDataParallel.

    Args:
        model: The model to wrap
        local_rank: Local GPU rank
        find_unused_parameters: Whether to find unused parameters
        broadcast_buffers: Whether to broadcast buffers

    Returns:
        The wrapped model (or original if not distributed)
    """
    if not is_distributed():
        return model

    if torch.cuda.is_available():
        model = model.to(local_rank)

    return DDP(
        model,
        device_ids=[local_rank] if torch.cuda.is_available() else None,
        output_device=local_rank if torch.cuda.is_available() else None,
        find_unused_parameters=find_unused_parameters,
        broadcast_buffers=broadcast_buffers,
    )


def unwrap_model(model: nn.Module | DDP) -> nn.Module:
    """
    Unwrap a DDP model to get the underlying module.

    Args:
        model: A model that may be wrapped in DDP

    Returns:
        The underlying model
    """
    if isinstance(model, DDP):
        # DDP.module is typed as Any, but we know it's an nn.Module
        module: nn.Module = model.module
        return module
    return model


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-reduce a tensor and compute the mean across all processes.

    Args:
        tensor: Input tensor

    Returns:
        Reduced tensor (mean across all processes)
    """
    if not is_distributed():
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / get_world_size()
    return tensor


def all_gather_object(obj: object) -> list:
    """
    Gather an object from all processes.

    Args:
        obj: Object to gather

    Returns:
        List of objects from all processes
    """
    if not is_distributed():
        return [obj]

    output = [None] * get_world_size()
    dist.all_gather_object(output, obj)
    return output


def broadcast_object(obj: object, src: int = 0) -> object:
    """
    Broadcast an object from a source process to all others.

    Args:
        obj: Object to broadcast (only used on source process)
        src: Source rank

    Returns:
        The broadcasted object
    """
    if not is_distributed():
        return obj

    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def barrier() -> None:
    """Synchronize all processes."""
    if is_distributed():
        dist.barrier()


def print_rank0(message: str) -> None:
    """Print a message only on rank 0."""
    if is_main_process():
        print(message, flush=True)

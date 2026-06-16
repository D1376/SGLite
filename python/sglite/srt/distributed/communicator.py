"""Distributed communication backends and helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from sglite.srt.distributed import DistributedInfo
    from sglite.kernels import PyNCCLCommunicator


@dataclass
class DistributedImpl(ABC):
    """Abstract interface for tensor-parallel communication backends."""

    @abstractmethod
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce the tensor across every tensor-parallel rank and return the result tensor."""
        ...

    @abstractmethod
    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        """Gather the per-rank shards into one concatenated tensor."""
        ...


@dataclass
class TorchDistributedImpl(DistributedImpl):
    """Torch-distributed implementation of collective communication."""

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Use torch.distributed to perform an in-place sum reduction."""
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        """Collect all shards into a new tensor with the first dimension expanded by TP size."""
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        shape = list(x.shape)
        shape[0] = shape[0] * tp_size
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x)
        return out


@dataclass
class PyNCCLDistributedImpl(DistributedImpl):
    """PyNCCL-backed implementation of collective communication."""

    comm: PyNCCLCommunicator

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Delegate the reduction to the PyNCCL communicator."""
        self.comm.all_reduce(x, "sum")
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        """Use PyNCCL to gather each rank's shard into one output tensor."""
        from .info import get_tp_info

        world_size = get_tp_info().size
        output_shape = list(x.shape)
        output_shape[0] *= world_size
        result = x.new_empty(output_shape)
        self.comm.all_gather(result, x)
        return result


class DistributedCommunicator:
    """Facade that dispatches collectives to the active backend."""

    plugins: List[DistributedImpl] = [TorchDistributedImpl()]

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the reduction to the most recently registered backend implementation."""
        return self.plugins[-1].all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the gather to the most recently registered backend implementation."""
        return self.plugins[-1].all_gather(x)


def enable_pynccl_distributed(
    tp_info: DistributedInfo, tp_cpu_group: torch.distributed.ProcessGroup, max_bytes: int
) -> None:
    """
    Enable PyNCCL-based distributed communication for tensor parallelism.
    """
    if tp_info.size == 1:
        return
    from sglite.kernels import init_pynccl

    comm = init_pynccl(
        tp_rank=tp_info.rank,
        tp_size=tp_info.size,
        tp_cpu_group=tp_cpu_group,
        max_size_bytes=max_bytes,
    )

    DistributedCommunicator.plugins.append(PyNCCLDistributedImpl(comm))


def destroy_distributed() -> None:
    """
    Destroy all the distributed communication plugins.
    """
    DistributedCommunicator.plugins = []

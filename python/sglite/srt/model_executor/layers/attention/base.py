"""Base types and shared interfaces for attention."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import torch
    from sglite.srt.request_state import Batch


@dataclass
class BaseAttnMetadata(ABC):
    """Base class for backend-specific attention metadata."""

    @abstractmethod
    def get_last_indices(self, bs: int) -> torch.Tensor:
        """Return the token indices that correspond to the final position of each sequence."""
        ...


class BaseAttnBackend(ABC):
    """Common interface implemented by all attention backends."""

    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        """Run the backend for one layer using the metadata prepared for the current batch."""
        ...

    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> None:
        """Populate any backend-owned metadata tensors needed by `forward`."""
        ...

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        """Allocate the static state needed for CUDA-graph capture and replay."""
        ...

    @abstractmethod
    def prepare_for_capture(self, batch: Batch) -> None:
        """Write batch-specific tensors into the static buffers used during graph capture."""
        ...

    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None:
        """Refresh captured CUDA-graph inputs before replaying the cached graph."""
        ...


class HybridBackend(BaseAttnBackend):
    """Routes prefill and decode work to different backend implementations."""

    def __init__(
        self,
        prefill_backend: BaseAttnBackend,
        decode_backend: BaseAttnBackend,
    ) -> None:
        """Keep separate backends for prefill and decode without changing the caller contract."""
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        """Dispatch the layer to the prefill or decode backend based on the batch phase."""
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.forward(q, k, v, layer_id, batch)

    def prepare_metadata(self, batch: Batch) -> None:
        """Populate metadata using the backend that will handle this batch."""
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.prepare_metadata(batch)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        """Capture graphs only for decode, where the shapes are stable enough to replay."""
        self.decode_backend.init_capture_graph(max_seq_len, bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        """Forward capture preparation to the decode backend."""
        self.decode_backend.prepare_for_capture(batch)

    def prepare_for_replay(self, batch: Batch) -> None:
        """Forward replay preparation to the decode backend."""
        self.decode_backend.prepare_for_replay(batch)

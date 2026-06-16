"""Request and batch containers used by the scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal

import torch

from sglite.sampling_params import SamplingParams

if TYPE_CHECKING:
    from sglite.srt.model_executor.layers.attention import BaseAttnMetadata
    from sglite.srt.mem_cache import BaseCacheHandle


@dataclass(eq=False)
class Req:
    """Represents one in-flight generation request."""
    input_ids: torch.Tensor  # CPU tensor.
    table_idx: int
    cached_len: int
    output_len: int
    uid: int
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle
    _input_buffer: torch.Tensor | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        """Populate derived fields after construction."""
        assert self.input_ids.is_cpu
        # `device_len` advances as tokens are prefetched or decoded; `max_device_len` is fixed.
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len

    @property
    def remain_len(self) -> int:
        """Return how many tokens the request can still generate."""
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        """Return how many tokens have been appended beyond the cached prefix."""
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        """Advance the request by one generated token."""
        self.cached_len = self.device_len
        self.device_len += 1

    def append_host(self, next_token: torch.Tensor) -> None:
        """Append the next decoded token to the CPU-side request buffer."""
        next_token = next_token.view(-1)
        next_len = next_token.numel()
        target_len = max(self.device_len, len(self.input_ids) + next_len)
        if self._input_buffer is None or self._input_buffer.numel() < target_len:
            capacity = max(self.max_device_len, target_len)
            self._input_buffer = torch.empty(
                capacity,
                dtype=self.input_ids.dtype,
                device=self.input_ids.device,
            )
            self._input_buffer[: len(self.input_ids)].copy_(self.input_ids)

        start = target_len - next_len
        self._input_buffer[start:target_len].copy_(next_token)
        self.input_ids = self._input_buffer[:target_len]

    @property
    def can_decode(self) -> bool:
        """Return whether the request still has decode work remaining."""
        return self.remain_len > 0

    def __repr__(self) -> str:
        """Return a debug representation of the request."""
        return (
            f"{type(self)}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )


@dataclass
class Batch:
    """Stores the batched state needed for one model forward pass."""
    reqs: List[Req]
    phase: Literal["prefill", "decode"]
    # Filled by the scheduler immediately before the engine runs the batch.
    input_ids: torch.Tensor = field(init=False)
    positions: torch.Tensor = field(init=False)
    out_loc: torch.Tensor = field(init=False)
    padded_reqs: List[Req] = field(init=False)
    # Attached by the attention backend after page-table mappings are ready.
    attn_metadata: BaseAttnMetadata = field(init=False)

    @property
    def is_prefill(self) -> bool:
        """Return whether the batch is in prefill mode."""
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        """Return whether the batch is in decode mode."""
        return self.phase == "decode"

    @property
    def size(self) -> int:
        """Return the number of live requests in the batch."""
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        """Return the batch size after padding with dummy requests."""
        return len(self.padded_reqs)

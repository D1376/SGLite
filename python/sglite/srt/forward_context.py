"""Forward-pass context shared across model execution components."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglite.srt.model_executor.layers.attention import BaseAttnBackend, BaseAttnMetadata
    from sglite.srt.mem_cache import BaseKVCachePool
    from sglite.srt.model_executor.layers.fused_moe import BaseMoeBackend

    from .request_state import Batch


@dataclass
class Context:
    """Holds runtime objects shared across a single model forward pass."""
    page_size: int
    # The global page table always stores token offsets, even when caches allocate full pages.
    page_table: torch.Tensor = field(init=False)
    attn_backend: BaseAttnBackend = field(init=False)
    moe_backend: BaseMoeBackend = field(init=False)
    kv_cache: BaseKVCachePool = field(init=False)
    _batch: Batch | None = field(default=None, init=False)

    @property
    def batch(self) -> Batch:
        """Return the batch bound to the active model forward."""
        assert self._batch is not None, "No active batch in context"
        return self._batch

    @contextmanager
    def forward_batch(self, batch: Batch):
        """Temporarily bind the runtime context to the batch being executed."""
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            # Layers pull request-local metadata from the active context during this scope.
            self._batch = batch
            yield
        finally:
            self._batch = None


_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context):
    """Register the active runtime context for helper layers and kernels."""
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def reset_global_ctx() -> None:
    """Clear the active runtime context."""
    global _GLOBAL_CTX
    _GLOBAL_CTX = None


def get_global_ctx() -> Context:
    """Return the process-wide runtime context."""
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX

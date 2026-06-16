"""Base types and shared interfaces for memory cache."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

import torch


class BaseKVCachePool(ABC):
    """Interface implemented by concrete KV-cache tensor layouts."""

    @abstractmethod
    def k_cache(self, index: int) -> torch.Tensor:
        """Return the key-cache tensor view for one transformer layer."""
        ...

    @abstractmethod
    def v_cache(self, index: int) -> torch.Tensor:
        """Return the value-cache tensor view for one transformer layer."""
        ...

    @abstractmethod
    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        """Write freshly computed key/value tensors into the cache locations selected by the scheduler."""
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Return the device that owns the underlying cache tensors."""
        ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """Return the storage dtype used by the cache tensors."""
        ...

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Return the number of transformer layers represented by the cache."""
        ...


@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    """Handle that protects or references a matched prefix-cache segment."""
    cached_len: int

    @abstractmethod
    def get_matched_indices(self) -> torch.Tensor:
        """Return the token indices already owned by this cache handle."""
        ...


class SizeInfo(NamedTuple):
    """Summarizes the token and page usage of a prefix-cache entry."""
    evictable_size: int
    protected_size: int

    @property
    def total_size(self) -> int:
        """Return the total number of tokens represented by both protected and evictable entries."""
        return self.evictable_size + self.protected_size


class InsertResult(NamedTuple):
    """Result returned after inserting a prefix into the cache."""
    cached_len: int  # Prefix length already present before insertion.
    handle: BaseCacheHandle  # Handle for the full inserted prefix.


class MatchResult(NamedTuple):
    """Result returned when matching an input prefix against the cache."""
    cuda_handle: BaseCacheHandle
    # TODO: return host-cache match handles here once HiCache joins prefix matching.


class BasePrefixCache(ABC):
    """Interface implemented by prefix-cache eviction and matching policies."""

    @abstractmethod
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """
        Lock or unlock a cache handle.
        This operation will not modify the cache, but change the size info only.
        When a handle is locked, it cannot be evicted.
        Handles must be locked before the previously-returned tensor of `match_prefix` is used.
        Otherwise it may be evicted by calling evict.

        Args:
            handle (BaseCacheHandle): The cache handle to lock or unlock.
            unlock (bool): Whether to unlock the handle. Defaults to False.
        """

    @abstractmethod
    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        """
        Match prefix and return the indices of the matched prefix in the cache.
        This operation will not modify the cache.
        The returned indices is only safe to use when the handle is locked.

        Args:
            input_ids (torch.Tensor): The input ids to match. Shape: (seq_len,)
        Returns:
            MatchResult: The match result containing the cache handles.
        """

    @abstractmethod
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        """
        Insert a new prefix into the cache.
        This operation will modify the cache.
        Args:
            input_ids (torch.Tensor): The input ids to insert. Shape: (seq_len,)
            indices (torch.Tensor): The indices to store the new prefix. Shape: (seq_len,)

        Returns:
            InsertResult: The result of the insertion.
        """

    @abstractmethod
    def evict(self, size: int) -> torch.Tensor:
        """
        Evict some prefixes from the cache to free up space.
        This operation will modify the cache.
        Note that evict 0 is always safe and does nothing.
        Note that the actual evict size may be larger than the requested size.
        Args:
            size (int): The size to evict.

        Returns:
            torch.Tensor: The indices evicted. Shape: (evict_size,)
        Raises:
            RuntimeError: If the requested size is larger than the evictable size.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the prefix cache to an empty state."""

    @property
    @abstractmethod
    def size_info(self) -> SizeInfo:
        """Return current protected and evictable token counts."""

    @abstractmethod
    def check_integrity(self) -> None:
        """Validate internal accounting and raise when the cache is corrupted."""

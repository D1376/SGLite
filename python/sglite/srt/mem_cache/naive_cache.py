"""Naive prefix-cache implementation without reuse or eviction."""

import torch

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo


class NaiveCacheHandle(BaseCacheHandle):
    """Cache handle returned by the always-miss prefix cache."""
    empty_tensor: torch.Tensor  # Shared empty match owned by NaivePrefixCache.

    def __init__(self):
        """Create an always-empty cache handle."""
        super().__init__(cached_len=0)

    def get_matched_indices(self) -> torch.Tensor:
        """Return the empty tensor because the naive cache never matches a prefix."""
        return self.empty_tensor


class NaivePrefixCache(BasePrefixCache):
    """Prefix cache that always misses and never retains prefixes."""

    def __init__(self, device: torch.device):
        """Create the shared empty tensor returned by all cache operations."""
        self.device = device
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        NaiveCacheHandle.empty_tensor = self.empty_tensor
        super().__init__()

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """Ignore lock and unlock requests because nothing is retained in the naive cache."""
        pass

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        """Always report a cache miss."""
        return MatchResult(NaiveCacheHandle())

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        """Pretend to insert nothing and return a fresh empty handle."""
        return InsertResult(0, NaiveCacheHandle())

    def evict(self, size: int) -> torch.Tensor:
        """Return no indices for zero-size eviction and reject real eviction."""
        if size == 0:
            return self.empty_tensor
        raise NotImplementedError("NaiveCacheManager does not support eviction.")

    def reset(self) -> None:
        """Keep the cache empty; no state reset is required."""
        pass

    @property
    def size_info(self) -> SizeInfo:
        """Return zero protected and evictable tokens because the cache never stores data."""
        return SizeInfo(evictable_size=0, protected_size=0)

    def check_integrity(self) -> None:
        """No-op integrity check because there is no internal state to validate."""
        pass

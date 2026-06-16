"""KV-cache allocation, reuse, and eviction logic."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Tuple

import torch
from sglite.srt.request_state import Req
from sglite.srt.mem_cache import BaseCacheHandle, MatchResult, create_prefix_cache
from sglite.srt.utils import div_ceil

if TYPE_CHECKING:
    from .pending_request import PendingReq


class CacheManager:
    """Manages KV-cache allocation, eviction, and prefix reuse."""

    def __init__(self, num_pages: int, page_size: int, page_table: torch.Tensor, type: str):
        """Track free page-aligned slots and own the prefix cache used to recycle them."""
        # Free slots are page starts, for example [0, 2, 4, 6, ...] when page_size = 2.
        device = page_table.device
        self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
        self.prefix_cache = create_prefix_cache(device=device, type=type)
        self.device = device
        self.num_pages = num_pages
        self.page_table = page_table
        self.page_size = page_size

    def match_req(self, req: PendingReq) -> MatchResult:
        """Match the cached prefix of a request, excluding the final token that still needs compute."""
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        return self.prefix_cache.match_prefix(req.input_ids[: input_len - 1])

    @property
    def available_size(self) -> int:
        """Return the token capacity available from free pages plus evictable cached prefixes."""
        return self.prefix_cache.size_info.evictable_size + len(self.free_slots) * self.page_size

    def lock(self, handle: BaseCacheHandle) -> None:
        """Protect a matched cache handle from eviction while a live request depends on it."""
        self.prefix_cache.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        """Release eviction protection for a cache handle once the request stops using it."""
        self.prefix_cache.lock_handle(handle, unlock=True)

    def allocate_paged(self, reqs: List[Req]) -> None:
        """Reserve fresh pages for each request's uncached suffix and write them into the page table."""
        needed_pages = 0
        allocation_info: List[Tuple[int, int, int]] = []
        for req in reqs:
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            if last_page > first_page:
                needed_pages += last_page - first_page
                allocation_info.append((req.table_idx, first_page, last_page))
        if needed_pages > 0:
            allocated = self._page_to_token(self._allocate(needed_pages))
            _write_page_table(self.page_table, allocated, allocation_info, self.page_size)

    def cache_req(self, req: Req, *, finished: bool) -> None:
        """Finalize a request after prefill by merging newly written pages into the prefix cache."""
        # `req.cached_len` is valid for attention. Some of that range may now be covered by
        # another prefix-cache insert, so release duplicate pages after `insert_prefix`.
        insert_ids = req.input_ids[: req.cached_len]
        page_indices = self.page_table[req.table_idx, : req.cached_len]
        old_handle = req.cache_handle
        cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
        # Unlock only after insertion finishes; the old handle protects pages read above.
        self.unlock(old_handle)
        self._free(page_indices[old_handle.cached_len : cached_len])
        if finished:
            # Finished requests do not need tail pages that could not be inserted into the cache.
            self._free(page_indices[new_handle.cached_len :])
        else:
            req.cache_handle = new_handle
            self.lock(new_handle)

    def check_integrity(self) -> None:
        """Verify that free pages plus cached pages still account for the full KV-cache pool."""
        self.prefix_cache.check_integrity()
        cache_pages = self.prefix_cache.size_info.total_size // self.page_size
        if len(self.free_slots) + cache_pages != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_pages({len(self.free_slots)}) +"
                f" cache_pages({cache_pages}) != num_pages({self.num_pages})"
            )
        if self.page_size > 1:
            assert torch.all(self.free_slots % self.page_size == 0)

    @contextmanager
    def lazy_free_region(self):
        """Batch free operations and append them back to the free list once the caller exits."""

        def lazy_free(indices: torch.Tensor) -> None:
            """Record page starts now and return them to the allocator when the context exits."""
            if len(indices) > 0:
                lazy_free_list.append(indices[:: self.page_size])

        lazy_free_list: List[torch.Tensor] = []
        try:
            self._free = lazy_free
            yield
        finally:
            del self._free
            if lazy_free_list:
                self.free_slots = torch.cat([self.free_slots] + lazy_free_list)

    def _allocate(self, needed_pages: int) -> torch.Tensor:
        """Pop pages from the free list, evicting cached prefixes first if capacity is tight."""
        if needed_pages > (free_pages := len(self.free_slots)):
            evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)
            self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size]])
            assert len(self.free_slots) >= needed_pages, "Eviction did not free enough space."
        allocated = self.free_slots[:needed_pages]
        self.free_slots = self.free_slots[needed_pages:]
        return allocated

    def _free(self, indices: torch.Tensor) -> None:
        """Return page starts back to the free list."""
        if len(indices) > 0:
            self.free_slots = torch.cat([self.free_slots, indices[:: self.page_size]])

    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        """Expand page starts into the contiguous token offsets covered by each page."""
        if self.page_size == 1:
            return pages
        # Expand each page start into all token offsets covered by that page.
        offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
        return (pages.unsqueeze(1) + offsets).flatten()


def _write_page_table(
    page_table: torch.Tensor,
    allocated: torch.Tensor,
    allocation_info: List[Tuple[int, int, int]],
    page_size: int,
) -> None:
    """Write newly allocated token slots into the per-request page table."""
    needed_tokens = len(allocated)
    # Stage the scatter indices in pinned CPU memory so both tensors can be copied to the device
    # asynchronously before performing one indexed write into the page table.
    table_idx_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    positions_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    offset = 0
    for table_idx, first_page, last_page in allocation_info:
        first_pos, last_pos = first_page * page_size, last_page * page_size
        length = last_pos - first_pos
        table_idx_host[offset : offset + length].fill_(table_idx)
        torch.arange(first_pos, last_pos, out=positions_host[offset : offset + length])
        offset += length
    assert offset == needed_tokens, "Mismatch in allocated tokens and filled tokens."
    table_idxs = table_idx_host.to(page_table.device, non_blocking=True)
    offsets = positions_host.to(page_table.device, non_blocking=True)
    page_table[table_idxs, offsets] = allocated

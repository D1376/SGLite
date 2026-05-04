"""Request-slot and token-table management."""

import torch


class TableManager:
    """Tracks request slots and the shared token/page table."""
    def __init__(self, max_running_reqs: int, page_table: torch.Tensor) -> None:
        """Create request-slot bookkeeping and the token staging pool."""
        self._max_running_reqs = max_running_reqs
        self._free_slots = list(range(max_running_reqs))
        self.page_table = page_table
        # The dummy request also reads from this pool, so initialize it with token id 0.
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)

    @property
    def available_size(self) -> int:
        """Return the number of free request slots."""
        return len(self._free_slots)

    def allocate(self) -> int:
        """Allocate one request slot."""
        return self._free_slots.pop()

    def free(self, slot: int) -> None:
        """Return one request slot to the free list."""
        self._free_slots.append(slot)

"""Decode-stage scheduling helpers and state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set

from sglite.srt.request_state import Batch, Req


@dataclass
class DecodeManager:
    """Tracks decode-stage requests that are ready to continue."""
    page_size: int
    running_reqs: Set[Req] = field(default_factory=set)

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        """Refresh the decode-ready set after a forward pass."""
        for req in reqs:
            if req.can_decode:
                self.running_reqs.add(req)
            else:
                self.running_reqs.discard(req)

    def remove_req(self, req: Req) -> None:
        """Remove a request from decode scheduling."""
        self.running_reqs.discard(req)

    def abort_req(self, uid: int) -> Req | None:
        """Remove and return the running request for a user id."""
        for req in self.running_reqs:
            if req.uid == uid:
                self.running_reqs.remove(req)
                return req
        return None

    @property
    def inflight_tokens(self) -> int:
        """Return token capacity already committed to active decode requests."""
        # Each running request may keep one partially used page reserved.
        tokens_reserved = (self.page_size - 1) * len(self.running_reqs)
        return sum(req.remain_len for req in self.running_reqs) + tokens_reserved

    def schedule_next_batch(self) -> Batch | None:
        """Build a decode batch from all runnable requests."""
        if not self.runnable:
            return None
        return Batch(reqs=list(self.running_reqs), phase="decode")

    @property
    def runnable(self) -> bool:
        """Return whether this component can schedule more work right now."""
        return len(self.running_reqs) > 0

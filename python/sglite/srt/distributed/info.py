"""Tensor-parallel topology helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedInfo:
    """Describes the current worker's tensor-parallel topology."""
    rank: int
    size: int

    def __post_init__(self):
        """Populate derived fields after construction."""
        assert 0 <= self.rank < self.size

    def is_primary(self) -> bool:
        """Return whether this worker is tensor-parallel rank 0."""
        return self.rank == 0


_TP_INFO: DistributedInfo | None = None


def set_tp_info(rank: int, size: int) -> None:
    """Set the global tensor-parallel topology information."""
    global _TP_INFO
    if _TP_INFO is not None:
        raise RuntimeError("TP info has been set")
    _TP_INFO = DistributedInfo(rank, size)


def get_tp_info() -> DistributedInfo:
    """Return the global tensor-parallel topology information."""
    if _TP_INFO is None:
        raise RuntimeError("TP info has not been set")
    return _TP_INFO


def try_get_tp_info() -> DistributedInfo | None:
    """Return tensor-parallel topology information when it has been initialized."""
    return _TP_INFO


__all__ = ["DistributedInfo", "set_tp_info", "get_tp_info", "try_get_tp_info"]

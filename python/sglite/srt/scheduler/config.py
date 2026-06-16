"""Scheduler configuration models."""

from __future__ import annotations

from dataclasses import dataclass, field

from sglite.srt.model_executor import EngineConfig


def _get_pid_suffix() -> str:
    """Build an IPC suffix that isolates concurrent local launches."""
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    """Configuration used by scheduler workers and their IPC links."""
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    offline_mode: bool = False

    # ZMQ endpoint suffix used to isolate concurrent local launches.
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        """Return the tokenizer-to-scheduler endpoint."""
        return "ipc:///tmp/sglite_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        """Return the scheduler-to-detokenizer endpoint."""
        return "ipc:///tmp/sglite_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        """Return the rank-0 scheduler broadcast endpoint."""
        return "ipc:///tmp/sglite_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        """Return the maximum number of tokens per engine forward."""
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        """Return whether backend workers create the detokenizer link."""
        return True

"""Logging setup helpers."""

from __future__ import annotations

import logging
import os
import sys
from functools import partial
from typing import TYPE_CHECKING, TextIO

_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"
_ANSI_DIM = "\033[2m"
_LEVEL_COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[35m",
}
_BANNER_BLUE = "\033[34m"
_BANNER_LINES = (
    (" ▗▄▄▖ ▗▄▄▖▗▖   ", "▗▄▄▄▖▗▄▄▄▖▗▄▄▄▖"),
    ("▐▌   ▐▌   ▐▌   ", "  █    █  ▐▌   "),
    (" ▝▀▚▖▐▌▝▜▌▐▌   ", "  █    █  ▐▛▀▀▘"),
    ("▗▄▄▞▘▝▚▄▞▘▐▙▄▄▖", "▗▄█▄▖  █  ▐▙▄▄▖"),
)


def _resolve_log_level(level: str | None) -> int:
    """Return the effective log level for the current process."""
    level_name = (level or os.getenv("LOG_LEVEL", "")).upper()
    return _LEVEL_MAP.get(level_name, logging.INFO)


def _is_truthy(value: str) -> bool:
    """Return whether a string flag is enabled."""
    return value.lower() in ("1", "true", "yes")


def _stream_is_tty(stream: TextIO) -> bool:
    """Return whether the stream looks like an interactive terminal."""
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def should_use_color(stream: TextIO | None = None) -> bool:
    """Return whether ANSI styling should be used for the target stream."""
    if os.getenv("NO_COLOR") is not None:
        return False
    return _stream_is_tty(stream or sys.stdout)


def dim_text(text: str, stream: TextIO | None = None) -> str:
    """Return dimmed terminal text when ANSI styling is enabled."""
    if not should_use_color(stream):
        return text
    return f"{_ANSI_DIM}{text}{_ANSI_RESET}"


def _indent_multiline(message: str, indent: int) -> str:
    """Indent continuation lines to keep multi-line logs aligned."""
    if "\n" not in message:
        return message
    padding = " " * indent
    return "\n".join(
        line if index == 0 else f"{padding}{line}"
        for index, line in enumerate(message.splitlines())
    )


class _ConsoleFormatter(logging.Formatter):
    """Formatter with optional ANSI color and aligned multi-line output."""

    def __init__(self, *, suffix: str, use_tp_rank: bool | None, use_color: bool):
        """Initialize the console formatter."""
        super().__init__()
        self._suffix = suffix
        self._use_tp_rank = use_tp_rank
        self._use_color = use_color
        self._tp_info = None

    def _format_suffix(self) -> str:
        """Return the contextual suffix for the current record."""
        from sglite.srt.distributed import try_get_tp_info

        self._tp_info = self._tp_info or try_get_tp_info()
        if self._tp_info is None or self._use_tp_rank is False:
            return self._suffix
        return f"{self._suffix}|core|rank={self._tp_info.rank}"

    def _style_timestamp(self, timestamp: str) -> str:
        """Apply timestamp styling for interactive terminals."""
        if not self._use_color:
            return timestamp
        return f"{_ANSI_BOLD}{timestamp}{_ANSI_RESET}"

    def _style_level(self, level_text: str, level_name: str) -> str:
        """Apply level styling for interactive terminals."""
        if not self._use_color:
            return level_text
        color = _LEVEL_COLORS.get(level_name, "")
        return f"{color}{level_text}{_ANSI_RESET}"

    def format(self, record: logging.LogRecord) -> str:
        """Format one log record for console output."""
        timestamp = self.formatTime(record, "[%Y-%m-%d|%H:%M:%S{suffix}]")
        timestamp = timestamp.format(suffix=self._format_suffix())

        message = record.getMessage()
        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            message = f"{message}\n{exc_text}" if message else exc_text
        if record.stack_info:
            stack_text = self.formatStack(record.stack_info)
            message = f"{message}\n{stack_text}" if message else stack_text

        level_text = f"{record.levelname:<8}"
        plain_prefix = f"{timestamp} {level_text} "
        styled_prefix = (
            f"{self._style_timestamp(timestamp)} "
            f"{self._style_level(level_text, record.levelname)} "
        )
        if message:
            return f"{styled_prefix}{_indent_multiline(message, len(plain_prefix))}"
        return styled_prefix.rstrip()


def _build_suffix(suffix: str, *, strip_file: bool, use_pid: bool | None) -> str:
    """Build the contextual suffix appended to the timestamp."""
    if strip_file:
        suffix = os.path.basename(suffix)

    if suffix:
        suffix = f"|{suffix}"

    if use_pid is None:
        use_pid = _is_truthy(os.getenv("LOG_PID", "0"))

    if use_pid:
        suffix = f"|pid={os.getpid()}{suffix}"

    return suffix


def _build_console_handler(
    *,
    suffix: str,
    strip_file: bool,
    use_pid: bool | None,
    use_tp_rank: bool | None,
    stream: TextIO | None = None,
) -> logging.Handler:
    """Create a console handler with the project formatter."""
    target = stream or sys.stdout
    handler = logging.StreamHandler(target)
    handler.setFormatter(
        _ConsoleFormatter(
            suffix=_build_suffix(suffix, strip_file=strip_file, use_pid=use_pid),
            use_tp_rank=use_tp_rank,
            use_color=should_use_color(target),
        )
    )
    return handler


def init_logger(
    name: str,
    suffix: str = "",
    *,
    strip_file: bool = True,
    level: str | None = None,
    use_pid: bool | None = None,
    use_tp_rank: bool | None = None,
):
    """Initialize the logger for the module with terminal-aware formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(_resolve_log_level(level))
    logger.handlers.clear()
    logger.addHandler(
        _build_console_handler(
            suffix=suffix,
            strip_file=strip_file,
            use_pid=use_pid,
            use_tp_rank=use_tp_rank,
        )
    )
    logger.propagate = False

    tp_info = None

    def _call_rank0(msg, *args, _which, **kwargs):
        """Forward a log call only from tensor-parallel rank 0."""
        from sglite.srt.distributed import get_tp_info

        nonlocal tp_info
        tp_info = tp_info or get_tp_info()
        assert tp_info is not None, "TP info not set yet"
        if tp_info.is_primary():
            getattr(logger, _which)(msg, *args, **kwargs)

    if TYPE_CHECKING:

        class WrapperLogger(logging.Logger):
            """Custom logger to expose rank-0-only helpers."""

            def info_rank0(self, msg, *args, **kwargs): ...
            def warning_rank0(self, msg, *args, **kwargs): ...
            def debug_rank0(self, msg, *args, **kwargs): ...
            def critical_rank0(self, msg, *args, **kwargs): ...

        return WrapperLogger(name)
    logger.info_rank0 = partial(_call_rank0, _which="info")
    logger.debug_rank0 = partial(_call_rank0, _which="debug")
    logger.critical_rank0 = partial(_call_rank0, _which="critical")
    logger.warning_rank0 = partial(_call_rank0, _which="warning")
    return logger


def configure_external_loggers(level: str | None = None) -> None:
    """Route Uvicorn logs through the same console formatter as project logs."""
    log_level = _resolve_log_level(level)
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        logger.handlers.clear()
        logger.addHandler(
            _build_console_handler(
                suffix=logger_name,
                strip_file=False,
                use_pid=False,
                use_tp_rank=False,
            )
        )
        logger.propagate = False


def print_banner(stream: TextIO | None = None) -> None:
    """Print the ANSI SGLite banner once when running in a terminal."""
    target = stream or sys.stdout
    if not should_use_color(target):
        return
    banner = "\n".join(
        f"{_ANSI_BOLD}{_BANNER_BLUE}{blue}{_ANSI_RESET}{_ANSI_BOLD}{plain}{_ANSI_RESET}"
        for blue, plain in _BANNER_LINES
    )
    target.write(f"{banner}\n")
    target.flush()

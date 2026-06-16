"""Tests for terminal-aware logging helpers."""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
LOGGER_MODULE_PATH = REPO_ROOT / "python/sglite/srt/utils/logger.py"


class _FakeStream:
    """Text stream that can pretend to be a TTY."""

    def __init__(self, *, isatty: bool):
        self._isatty = isatty
        self._chunks: list[str] = []

    def write(self, data: str) -> int:
        self._chunks.append(data)
        return len(data)

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return self._isatty

    def getvalue(self) -> str:
        return "".join(self._chunks)


def _load_logger_module(monkeypatch: pytest.MonkeyPatch):
    sglite_pkg = types.ModuleType("sglite")
    sglite_pkg.__path__ = []  # type: ignore[attr-defined]

    distributed_mod = types.ModuleType("sglite.srt.distributed")
    distributed_mod.try_get_tp_info = lambda: None
    distributed_mod.get_tp_info = lambda: None

    monkeypatch.setitem(sys.modules, "sglite", sglite_pkg)
    monkeypatch.setitem(sys.modules, "sglite.srt.distributed", distributed_mod)

    spec = importlib.util.spec_from_file_location("test_logger_module", LOGGER_MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_init_logger_uses_color_for_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_logger_module(monkeypatch)
    stream = _FakeStream(isatty=True)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(module.sys, "stdout", stream)

    logger = module.init_logger("tty_logger", "demo")
    logger.info("hello")

    assert "\033[" in stream.getvalue()


def test_init_logger_uses_plain_text_for_non_tty_or_no_color(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_logger_module(monkeypatch)

    stream = _FakeStream(isatty=False)
    monkeypatch.setattr(module.sys, "stdout", stream)
    logger = module.init_logger("plain_logger", "demo")
    logger.info("hello")
    assert "\033[" not in stream.getvalue()

    no_color_stream = _FakeStream(isatty=True)
    monkeypatch.setattr(module.sys, "stdout", no_color_stream)
    monkeypatch.setenv("NO_COLOR", "1")
    logger = module.init_logger("plain_logger_no_color", "demo")
    logger.info("hello")
    assert "\033[" not in no_color_stream.getvalue()


def test_multiline_messages_indent_continuations(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_logger_module(monkeypatch)
    stream = _FakeStream(isatty=False)
    monkeypatch.setattr(module.sys, "stdout", stream)

    logger = module.init_logger("multiline_logger", "demo")
    logger.info("first line\nsecond line\nthird line")

    lines = stream.getvalue().splitlines()
    assert "first line" in lines[0]
    assert lines[1].startswith(" ")
    assert lines[1].lstrip() == "second line"
    assert lines[2].startswith(" ")
    assert lines[2].lstrip() == "third line"


def test_exception_output_is_indented(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_logger_module(monkeypatch)
    stream = _FakeStream(isatty=False)
    monkeypatch.setattr(module.sys, "stdout", stream)

    logger = module.init_logger("exception_logger", "demo")
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        logger.exception("request failed")

    lines = stream.getvalue().splitlines()
    traceback_line = next(line for line in lines if "Traceback" in line)
    assert traceback_line.startswith(" ")


def test_print_banner_only_renders_for_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_logger_module(monkeypatch)
    monkeypatch.delenv("NO_COLOR", raising=False)

    tty_stream = _FakeStream(isatty=True)
    module.print_banner(tty_stream)
    tty_output = tty_stream.getvalue()
    assert tty_output.count("\n") >= 4
    assert "\033[" in tty_output
    assert "\033[1m\033[34m ▗▄▄▖ ▗▄▄▖▗▖   \033[0m\033[1m▗▄▄▄▖▗▄▄▄▖▗▄▄▄▖\033[0m" in tty_output
    assert "\033[1m\033[34m▗▄▄▞▘▝▚▄▞▘▐▙▄▄▖\033[0m\033[1m▗▄█▄▖  █  ▐▙▄▄▖\033[0m" in tty_output

    plain_stream = _FakeStream(isatty=False)
    module.print_banner(plain_stream)
    assert plain_stream.getvalue() == ""


def test_configure_external_loggers_reuses_project_formatter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_logger_module(monkeypatch)
    stream = _FakeStream(isatty=False)
    monkeypatch.setattr(module.sys, "stdout", stream)

    module.configure_external_loggers()
    try:
        logging.getLogger("uvicorn.error").info("server started")
    finally:
        for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            logger = logging.getLogger(logger_name)
            logger.handlers.clear()
            logger.propagate = True

    output = stream.getvalue()
    assert "uvicorn.error" in output
    assert "server started" in output

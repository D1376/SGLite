"""Tests for API server logging integration."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
API_SERVER_MODULE_PATH = REPO_ROOT / "python/sglite/srt/entrypoints/api.py"


class _DummyLogger:
    """Capture log calls for assertions."""

    def __init__(self) -> None:
        self.info_calls: list[str] = []
        self.debug_calls: list[str] = []

    def info(self, message: str, *args, **_kwargs) -> None:
        self.info_calls.append(message % args if args else message)

    def debug(self, message: str, *args, **_kwargs) -> None:
        self.debug_calls.append(message % args if args else message)


class _DummyFastAPI:
    """Minimal FastAPI stub for decorator registration."""

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def post(self, *_args, **_kwargs):
        return lambda fn: fn

    def api_route(self, *_args, **_kwargs):
        return lambda fn: fn

    def get(self, *_args, **_kwargs):
        return lambda fn: fn


class _DummyStreamingResponse:
    """Simple response placeholder."""

    def __init__(self, *_args, **_kwargs) -> None:
        return None


class _DummyQueue:
    """Minimal queue placeholder."""

    def __init__(self, *_args, **_kwargs) -> None:
        return None

    def stop(self) -> None:
        return None


class _DummyFrontendManager:
    """Frontend manager placeholder that keeps the config reference."""

    def __init__(self, *, config, recv_tokenizer, send_tokenizer) -> None:
        self.config = config
        self.recv_tokenizer = recv_tokenizer
        self.send_tokenizer = send_tokenizer

    def shutdown(self) -> None:
        return None


def _load_api_server_module(monkeypatch: pytest.MonkeyPatch):
    sglite_pkg = types.ModuleType("sglite")
    sglite_pkg.__path__ = []  # type: ignore[attr-defined]
    server_pkg = types.ModuleType("sglite.srt.entrypoints")
    server_pkg.__path__ = []  # type: ignore[attr-defined]

    logger = _DummyLogger()
    configure_calls: list[str] = []
    uvicorn_runs: list[tuple[tuple[object, ...], dict[str, object]]] = []

    fastapi_mod = types.ModuleType("fastapi")
    fastapi_mod.FastAPI = _DummyFastAPI
    fastapi_mod.Request = type("Request", (), {})

    fastapi_responses_mod = types.ModuleType("fastapi.responses")
    fastapi_responses_mod.StreamingResponse = _DummyStreamingResponse

    uvicorn_mod = types.ModuleType("uvicorn")
    uvicorn_mod.run = lambda *args, **kwargs: uvicorn_runs.append((args, kwargs))

    message_mod = types.ModuleType("sglite.srt.messages")
    message_mod.BaseFrontendMsg = type("BaseFrontendMsg", (), {"decoder": object()})
    message_mod.BaseTokenizerMsg = type("BaseTokenizerMsg", (), {"encoder": object()})

    utils_mod = types.ModuleType("sglite.srt.utils")
    utils_mod.ZmqAsyncPullQueue = _DummyQueue
    utils_mod.ZmqAsyncPushQueue = _DummyQueue
    utils_mod.init_logger = lambda *_args, **_kwargs: logger
    utils_mod.configure_external_loggers = lambda: configure_calls.append("called")

    args_mod = types.ModuleType("sglite.srt.entrypoints.args")
    args_mod.ServerArgs = object

    frontend_manager_mod = types.ModuleType("sglite.srt.entrypoints.frontend_manager")
    frontend_manager_mod.FrontendManager = _DummyFrontendManager

    protocol_mod = types.ModuleType("sglite.srt.entrypoints.protocol")
    protocol_mod.GenerateRequest = type("GenerateRequest", (), {})
    protocol_mod.ModelCard = type("ModelCard", (), {})
    protocol_mod.ModelList = type("ModelList", (), {})
    protocol_mod.OpenAICompletionRequest = type("OpenAICompletionRequest", (), {})
    protocol_mod.resolve_completion_prompt = lambda req: req

    cli_mod = types.ModuleType("sglite.srt.entrypoints.cli")
    cli_mod.run_cli = lambda _state: None

    monkeypatch.setitem(sys.modules, "sglite", sglite_pkg)
    monkeypatch.setitem(sys.modules, "sglite.srt.entrypoints", server_pkg)
    monkeypatch.setitem(sys.modules, "fastapi", fastapi_mod)
    monkeypatch.setitem(sys.modules, "fastapi.responses", fastapi_responses_mod)
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn_mod)
    monkeypatch.setitem(sys.modules, "sglite.srt.messages", message_mod)
    monkeypatch.setitem(sys.modules, "sglite.srt.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "sglite.srt.entrypoints.args", args_mod)
    monkeypatch.setitem(sys.modules, "sglite.srt.entrypoints.frontend_manager", frontend_manager_mod)
    monkeypatch.setitem(sys.modules, "sglite.srt.entrypoints.protocol", protocol_mod)
    monkeypatch.setitem(sys.modules, "sglite.srt.entrypoints.cli", cli_mod)

    spec = importlib.util.spec_from_file_location("sglite.srt.entrypoints.api", API_SERVER_MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module._test_logger = logger
    module._configure_calls = configure_calls
    module._uvicorn_runs = uvicorn_runs
    return module


def test_run_api_server_uses_project_logging_for_uvicorn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_api_server_module(monkeypatch)
    start_calls: list[str] = []
    config = types.SimpleNamespace(
        use_dummy_weight=False,
        server_host="127.0.0.1",
        server_port=1376,
        model_path="foo",
        tp_info=types.SimpleNamespace(size=2),
        zmq_frontend_addr="frontend",
        zmq_tokenizer_addr="tokenizer",
        frontend_create_tokenizer_link=False,
    )

    module.run_api_server(config, lambda: start_calls.append("started"), run_cli=False)

    assert start_calls == ["started"]
    assert module._configure_calls == ["called"]
    assert module._uvicorn_runs[0][1]["log_config"] is None
    assert module._uvicorn_runs[0][1]["host"] == "127.0.0.1"
    assert module._uvicorn_runs[0][1]["port"] == 1376
    assert module._test_logger.info_calls == [
        "Frontend ready: mode=server host=127.0.0.1 port=1376 model=foo tp_size=2"
    ]

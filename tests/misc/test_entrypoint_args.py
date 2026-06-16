"""Tests for server argument parsing."""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
ARGS_MODULE_PATH = REPO_ROOT / "python/sglite/srt/entrypoints/args.py"


@dataclass(frozen=True)
class _DistributedInfo:
    rank: int
    size: int

    def __post_init__(self) -> None:
        assert 0 <= self.rank < self.size

    def is_primary(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class _SchedulerConfig:
    model_path: str = ""
    tp_info: _DistributedInfo = field(default_factory=lambda: _DistributedInfo(0, 1))
    dtype: object | None = None
    max_running_req: int = 256
    attention_backend: str = "auto"
    moe_backend: str = "auto"
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    offline_mode: bool = False
    _unique_suffix: str = ".pid=0"


class _DummyLogger:
    def __init__(self) -> None:
        self.info_calls: list[str] = []
        self.debug_calls: list[str] = []

    def info(self, message: str, *args, **_kwargs) -> None:
        self.info_calls.append(message % args if args else message)

    def debug(self, message: str, *args, **_kwargs) -> None:
        self.debug_calls.append(message % args if args else message)


class _SupportedNames:
    def __init__(self, *names: str) -> None:
        self._names = list(names)

    def supported_names(self) -> list[str]:
        return list(self._names)


def _load_server_args_module(monkeypatch: pytest.MonkeyPatch):
    sglite_pkg = types.ModuleType("sglite")
    sglite_pkg.__path__ = []  # type: ignore[attr-defined]

    distributed_mod = types.ModuleType("sglite.srt.distributed")
    distributed_mod.DistributedInfo = _DistributedInfo

    scheduler_mod = types.ModuleType("sglite.srt.scheduler")
    scheduler_mod.SchedulerConfig = _SchedulerConfig

    dummy_logger = _DummyLogger()
    utils_mod = types.ModuleType("sglite.srt.utils")
    utils_mod.init_logger = lambda *_args, **_kwargs: dummy_logger
    utils_mod.cached_load_hf_config = lambda _model_path: types.SimpleNamespace(dtype="float16")

    attention_mod = types.ModuleType("sglite.srt.model_executor.layers.attention")
    attention_mod.validate_attn_backend = lambda backend: backend

    kvcache_mod = types.ModuleType("sglite.srt.mem_cache")
    kvcache_mod.SUPPORTED_CACHE_MANAGER = _SupportedNames("radix", "naive")

    moe_mod = types.ModuleType("sglite.srt.model_executor.layers.fused_moe")
    moe_mod.SUPPORTED_MOE_BACKENDS = _SupportedNames("fused")

    monkeypatch.setitem(sys.modules, "sglite", sglite_pkg)
    monkeypatch.setitem(sys.modules, "sglite.srt.distributed", distributed_mod)
    monkeypatch.setitem(sys.modules, "sglite.srt.scheduler", scheduler_mod)
    monkeypatch.setitem(sys.modules, "sglite.srt.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "sglite.srt.model_executor.layers.attention", attention_mod)
    monkeypatch.setitem(sys.modules, "sglite.srt.mem_cache", kvcache_mod)
    monkeypatch.setitem(sys.modules, "sglite.srt.model_executor.layers.fused_moe", moe_mod)

    spec = importlib.util.spec_from_file_location("test_server_args_module", ARGS_MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module._test_logger = dummy_logger
    return module


@pytest.fixture
def server_args_module(monkeypatch: pytest.MonkeyPatch):
    return _load_server_args_module(monkeypatch)


def test_parse_args_accepts_canonical_flag_names(server_args_module) -> None:
    config, run_cli = server_args_module.parse_args(
        [
            "--model",
            "foo",
            "--dtype",
            "float16",
            "--tp-size",
            "4",
            "--max-running-requests",
            "8",
            "--max-seq-len-override",
            "4096",
            "--mem-frac",
            "0.75",
            "--dummy-weight",
            "--disable-pynccl",
            "--host",
            "0.0.0.0",
            "--port",
            "3000",
            "--graph",
            "16",
            "--tokenizer-count",
            "2",
            "--max-extend-length",
            "2048",
            "--num-pages",
            "512",
            "--page-size",
            "16",
            "--attn-backend",
            "fi",
            "--model-source",
            "huggingface",
            "--cache-type",
            "naive",
            "--moe-backend",
            "fused",
        ]
    )

    assert run_cli is False
    assert config.model_path == "foo"
    assert config.tp_info.size == 4
    assert config.max_running_req == 8
    assert config.max_seq_len_override == 4096
    assert config.memory_ratio == 0.75
    assert config.use_dummy_weight is True
    assert config.use_pynccl is False
    assert config.server_host == "0.0.0.0"
    assert config.server_port == 3000
    assert config.cuda_graph_max_bs == 16
    assert config.num_tokenizer == 2
    assert config.max_extend_tokens == 2048
    assert config.num_page_override == 512
    assert config.page_size == 16
    assert config.attention_backend == "fi"
    assert config.cache_type == "naive"
    assert config.moe_backend == "fused"


@pytest.mark.parametrize(
    ("alias", "value"),
    [
        ("--max-running-reqs", "2"),
        ("--mem-ratio", "0.8"),
        ("--memory-ratio", "0.8"),
        ("--dummy-weights", None),
        ("--no-pynccl", None),
        ("--graph-max-bs", "8"),
        ("--tokenizers", "1"),
        ("--max-prefill-tokens", "1024"),
        ("--max-pages", "64"),
    ],
)
def test_removed_aliases_are_rejected(server_args_module, alias: str, value: str | None) -> None:
    cli_args = ["--model", "foo", "--dtype", "float16", alias]
    if value is not None:
        cli_args.append(value)

    with pytest.raises(SystemExit) as exc_info:
        server_args_module.parse_args(cli_args)

    assert exc_info.value.code == 2


def test_cli_mode_applies_single_request_overrides(server_args_module) -> None:
    config, run_cli = server_args_module.parse_args(
        [
            "--model",
            "foo",
            "--dtype",
            "float16",
            "--max-running-requests",
            "8",
            "--graph",
            "16",
            "--cli",
        ]
    )

    assert run_cli is True
    assert config.max_running_req == 8
    assert config.cuda_graph_max_bs == 16
    assert config.silent_output is True


def test_zero_tokenizers_means_shared_tokenizer_worker(server_args_module) -> None:
    config, _ = server_args_module.parse_args(
        [
            "--model",
            "foo",
            "--dtype",
            "float16",
            "--tokenizer-count",
            "0",
        ]
    )

    assert config.num_tokenizer == 0
    assert config.share_tokenizer is True


def test_parse_args_logs_compact_info_and_debug_dump(server_args_module) -> None:
    config, run_cli = server_args_module.parse_args(["--model", "foo", "--dtype", "float16"])

    assert run_cli is False
    assert config.model_path == "foo"
    debug_output = server_args_module._test_logger.debug_calls[0]
    assert debug_output.startswith("Resolved server arguments:\nServerArgs(")
    assert "model_path='foo'" in debug_output
    assert "server_host='127.0.0.1'" in debug_output
    assert "server_port=1376" in debug_output
    assert server_args_module._test_logger.info_calls == [
        "Launch config: model=foo mode=server tp_size=1 host=127.0.0.1 port=1376 tokenizers=0"
    ]

"""Tests for engine startup logging around loaded model size."""

from __future__ import annotations

import importlib.util
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_MODULE_PATH = REPO_ROOT / "python/sglite/srt/model_executor/engine.py"


class _DummyLogger:
    """Capture rank-0 log calls for assertions."""

    def __init__(self) -> None:
        self.info_rank0_calls: list[str] = []
        self.warning_rank0_calls: list[str] = []
        self.error_calls: list[str] = []

    def info_rank0(self, message: str, *args, **_kwargs) -> None:
        self.info_rank0_calls.append(message % args if args else message)

    def warning_rank0(self, message: str, *args, **_kwargs) -> None:
        self.warning_rank0_calls.append(message % args if args else message)

    def error(self, message: str, *args, **_kwargs) -> None:
        self.error_calls.append(message % args if args else message)


class _DummyContext:
    """Minimal runtime context placeholder."""

    def __init__(self, page_size: int) -> None:
        self.page_size = page_size


class _DummyReq:
    """Simple request placeholder."""

    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class _DummyGraphRunner:
    """Graph runner stub used during engine initialization tests."""

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def destroy_cuda_graphs(self) -> None:
        return None


class _DummySampler:
    """Sampler placeholder."""

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


class _FakeDevice(str):
    """String-like device object that can also act as a context manager."""

    def __new__(cls, value: str):
        return str.__new__(cls, value)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _load_engine_module(monkeypatch: pytest.MonkeyPatch):
    sglite_pkg = types.ModuleType("sglite")
    sglite_pkg.__path__ = []  # type: ignore[attr-defined]
    engine_pkg = types.ModuleType("sglite.srt.model_executor")
    engine_pkg.__path__ = []  # type: ignore[attr-defined]

    logger = _DummyLogger()

    attention_mod = types.ModuleType("sglite.srt.model_executor.layers.attention")
    attention_mod.create_attention_backend = lambda *_args, **_kwargs: object()

    distributed_mod = types.ModuleType("sglite.srt.distributed")
    distributed_mod.destroy_distributed = lambda: None
    distributed_mod.enable_pynccl_distributed = lambda *_args, **_kwargs: None
    distributed_mod.set_tp_info = lambda **_kwargs: None

    kvcache_mod = types.ModuleType("sglite.srt.mem_cache")
    kvcache_mod.create_kvcache_pool = lambda **_kwargs: object()

    layers_mod = types.ModuleType("sglite.srt.model_executor.layers")
    layers_mod.set_rope_device = lambda _device: None

    models_mod = types.ModuleType("sglite.srt.model_executor.models")
    models_mod.create_model = lambda _config: object()

    model_loader_mod = types.ModuleType("sglite.srt.model_executor.model_loader")
    model_loader_mod.load_weight = lambda _model_path, _device: iter(())

    moe_mod = types.ModuleType("sglite.srt.model_executor.layers.fused_moe")
    moe_mod.create_moe_backend = lambda *_args, **_kwargs: object()

    request_state_mod = types.ModuleType("sglite.srt.request_state")
    request_state_mod.Batch = object
    request_state_mod.Req = _DummyReq

    runtime_context_mod = types.ModuleType("sglite.srt.forward_context")
    runtime_context_mod.Context = _DummyContext
    runtime_context_mod.set_global_ctx = lambda _ctx: None

    utils_mod = types.ModuleType("sglite.srt.utils")
    utils_mod.div_even = lambda a, b, allow_replicate=False: a // max(b, 1)
    utils_mod.init_logger = lambda *_args, **_kwargs: logger
    utils_mod.is_sm90_supported = lambda: False
    utils_mod.is_sm100_supported = lambda: False
    utils_mod.torch_dtype = lambda _dtype: nullcontext()

    engine_config_mod = types.ModuleType("sglite.srt.model_executor.config")
    engine_config_mod.EngineConfig = object

    engine_cuda_graph_mod = types.ModuleType("sglite.srt.model_executor.cuda_graph")
    engine_cuda_graph_mod.GraphRunner = _DummyGraphRunner
    engine_cuda_graph_mod.get_free_memory = lambda _device: 0
    engine_cuda_graph_mod.mem_GB = lambda size: f"{size / (1024**3):.2f} GiB"

    engine_sampler_mod = types.ModuleType("sglite.srt.model_executor.sampler")
    engine_sampler_mod.BatchSamplingArgs = object
    engine_sampler_mod.Sampler = _DummySampler

    monkeypatch.setitem(sys.modules, "sglite", sglite_pkg)
    monkeypatch.setitem(sys.modules, "sglite.srt.model_executor", engine_pkg)
    monkeypatch.setitem(sys.modules, "sglite.srt.model_executor.layers.attention", attention_mod)
    monkeypatch.setitem(sys.modules, "sglite.srt.distributed", distributed_mod)
    monkeypatch.setitem(sys.modules, "sglite.srt.mem_cache", kvcache_mod)
    monkeypatch.setitem(sys.modules, "sglite.srt.model_executor.layers", layers_mod)
    monkeypatch.setitem(sys.modules, "sglite.srt.model_executor.models", models_mod)
    monkeypatch.setitem(sys.modules, "sglite.srt.model_executor.model_loader", model_loader_mod)
    monkeypatch.setitem(sys.modules, "sglite.srt.model_executor.layers.fused_moe", moe_mod)
    monkeypatch.setitem(sys.modules, "sglite.srt.request_state", request_state_mod)
    monkeypatch.setitem(sys.modules, "sglite.srt.forward_context", runtime_context_mod)
    monkeypatch.setitem(sys.modules, "sglite.srt.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "sglite.srt.model_executor.config", engine_config_mod)
    monkeypatch.setitem(sys.modules, "sglite.srt.model_executor.cuda_graph", engine_cuda_graph_mod)
    monkeypatch.setitem(sys.modules, "sglite.srt.model_executor.sampler", engine_sampler_mod)

    spec = importlib.util.spec_from_file_location("sglite.srt.model_executor.engine", ENGINE_MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module._test_logger = logger
    return module


def test_load_weight_state_dict_counts_loaded_tensor_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_module(monkeypatch)
    engine = module.Engine.__new__(module.Engine)
    engine.device = "cpu"
    engine.dtype = torch.float16
    engine.model = types.SimpleNamespace(state_dict=lambda: {})

    monkeypatch.setattr(
        module,
        "load_weight",
        lambda *_args: iter(
            [
                ("float_weight", torch.ones(4, dtype=torch.float32)),
                ("int_weight", torch.ones(3, dtype=torch.int32)),
            ]
        ),
    )

    state_dict, local_model_bytes = module.Engine._load_weight_state_dict(
        engine,
        types.SimpleNamespace(use_dummy_weight=False, model_path="foo"),
    )

    assert state_dict["float_weight"].dtype == torch.float16
    assert state_dict["int_weight"].dtype == torch.int32
    assert local_model_bytes == (
        state_dict["float_weight"].numel() * state_dict["float_weight"].element_size()
        + state_dict["int_weight"].numel() * state_dict["int_weight"].element_size()
    )


def test_load_weight_state_dict_counts_dummy_tensor_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_module(monkeypatch)
    engine = module.Engine.__new__(module.Engine)
    engine.device = "cpu"
    engine.dtype = torch.float16
    engine.model = types.SimpleNamespace(
        state_dict=lambda: {
            "weight_a": torch.zeros(2, dtype=torch.float32),
            "weight_b": torch.zeros(3, dtype=torch.float32),
        }
    )

    state_dict, local_model_bytes = module.Engine._load_weight_state_dict(
        engine,
        types.SimpleNamespace(use_dummy_weight=True),
    )

    assert set(state_dict) == {"weight_a", "weight_b"}
    assert all(t.device.type == "cpu" for t in state_dict.values())
    assert local_model_bytes == sum(
        tensor.numel() * tensor.element_size() for tensor in state_dict.values()
    )


def test_log_loaded_model_weights_aggregates_total_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_module(monkeypatch)
    engine = module.Engine.__new__(module.Engine)
    engine.tp_cpu_group = object()
    module._test_logger.info_rank0_calls.clear()

    def fake_all_reduce(tensor: torch.Tensor, *, op, group) -> None:
        assert op == torch.distributed.ReduceOp.SUM
        assert group is engine.tp_cpu_group
        tensor.add_(6 * 1024**3)

    monkeypatch.setattr(module.torch.distributed, "all_reduce", fake_all_reduce)

    module.Engine._log_loaded_model_weights(engine, 2 * 1024**3)

    assert module._test_logger.info_rank0_calls == [
        "Loaded model weights: local=2.00 GiB total=8.00 GiB"
    ]


def test_engine_logs_model_size_before_kv_cache_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_module(monkeypatch)
    order: list[str] = []

    class _FakeModel:
        def load_state_dict(self, _state_dict) -> None:
            order.append("load_state_dict")

    class _OrderGraphRunner:
        def __init__(self, *args, **kwargs) -> None:
            order.append("graph_runner")

        def destroy_cuda_graphs(self) -> None:
            return None

    monkeypatch.setattr(module, "create_model", lambda _config: _FakeModel())
    monkeypatch.setattr(module, "set_rope_device", lambda _device: None)
    monkeypatch.setattr(
        module,
        "create_kvcache_pool",
        lambda **_kwargs: order.append("create_kvcache_pool") or object(),
    )
    monkeypatch.setattr(module, "GraphRunner", _OrderGraphRunner)
    monkeypatch.setattr(module.Engine, "_init_communication", lambda self, _config: object())
    monkeypatch.setattr(
        module.Engine,
        "_sync_get_memory",
        lambda self: (8 * 1024**3, 8 * 1024**3),
    )

    def fake_load_weight_state_dict(self, _config):
        order.append("load_weight_state_dict")
        return {}, 2 * 1024**3

    monkeypatch.setattr(module.Engine, "_load_weight_state_dict", fake_load_weight_state_dict)
    monkeypatch.setattr(
        module.Engine,
        "_log_loaded_model_weights",
        lambda self, _local_model_bytes: order.append("log_loaded_model_weights"),
    )
    monkeypatch.setattr(
        module.Engine,
        "_determine_num_pages",
        lambda self, _old_free_memory, _config: order.append("determine_num_pages") or 2,
    )
    monkeypatch.setattr(module.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(module.torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(module.torch.cuda, "Stream", lambda: object())
    monkeypatch.setattr(module.torch.cuda, "set_stream", lambda _stream: None)

    def fake_device(value: str) -> _FakeDevice:
        if value.startswith("cuda:"):
            return _FakeDevice("cpu")
        return _FakeDevice(value)

    monkeypatch.setattr(module.torch, "device", fake_device)

    config = types.SimpleNamespace(
        tp_info=types.SimpleNamespace(rank=0, size=1),
        dtype=torch.float16,
        page_size=2,
        max_running_req=4,
        attention_backend="fi",
        moe_backend="fused",
        cuda_graph_bs=None,
        cuda_graph_max_bs=None,
        model_config=types.SimpleNamespace(vocab_size=32, is_moe=False),
        max_seq_len=16,
        use_pynccl=True,
        distributed_timeout=60.0,
        distributed_addr="tcp://127.0.0.1:2333",
        max_forward_len=16,
    )

    module.Engine(config)

    assert order.index("load_weight_state_dict") < order.index("load_state_dict")
    assert order.index("load_state_dict") < order.index("log_loaded_model_weights")
    assert order.index("log_loaded_model_weights") < order.index("determine_num_pages")
    assert order.index("determine_num_pages") < order.index("create_kvcache_pool")

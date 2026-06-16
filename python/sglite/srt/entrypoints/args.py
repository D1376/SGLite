"""Command-line parsing for the inference server."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from sglite.srt.distributed import DistributedInfo
from sglite.srt.scheduler import SchedulerConfig
from sglite.srt.utils import init_logger


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _resolve_model_path(model_path: str, *, model_source: str, use_dummy_weight: bool) -> str:
    """Resolve local paths and optionally download remote ModelScope snapshots."""
    if model_path.startswith("~"):
        model_path = os.path.expanduser(model_path)

    if model_source != "modelscope" or os.path.isdir(model_path):
        return model_path

    from modelscope import snapshot_download

    ignore_patterns = []
    if use_dummy_weight:
        ignore_patterns = ["*.bin", "*.safetensors", "*.pt", "*.ckpt"]
    return snapshot_download(model_path, ignore_patterns=ignore_patterns)


def _resolve_dtype(dtype: str | torch.dtype, model_path: str) -> torch.dtype:
    """Resolve `auto` and string dtype options into torch dtypes."""
    if dtype == "auto":
        from sglite.srt.utils import cached_load_hf_config

        dtype = cached_load_hf_config(model_path).dtype
    return DTYPE_MAP[dtype] if isinstance(dtype, str) else dtype


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser shared by server and CLI mode."""
    from sglite.srt.model_executor.layers.attention import validate_attn_backend
    from sglite.srt.mem_cache import SUPPORTED_CACHE_MANAGER
    from sglite.srt.model_executor.layers.fused_moe import SUPPORTED_MOE_BACKENDS

    parser = argparse.ArgumentParser(
        prog="python -m sglite",
        description=(
            "Launch the SGLite OpenAI-compatible inference server or run an "
            "interactive terminal session with a local model directory or a "
            "remote model repository."
        ),
        epilog=(
            "Examples:\n"
            "  python -m sglite --model Qwen/Qwen3-0.6B\n"
            "  python -m sglite --model meta-llama/Llama-3.1-70B-Instruct --tp-size 4 --port 30000\n"
            "  python -m sglite --model Qwen/Qwen3-0.6B --cli"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model-path",
        "--model",
        type=str,
        required=True,
        help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights and activations. 'auto' will use FP16 for FP32/FP16 models and BF16 for BF16 models.",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        "--tp-size",
        type=int,
        default=1,
        help="The tensor parallelism size.",
    )

    parser.add_argument(
        "--max-running-requests",
        type=int,
        dest="max_running_req",
        default=ServerArgs.max_running_req,
        help="The maximum number of running requests.",
    )

    parser.add_argument(
        "--max-seq-len-override",
        type=int,
        default=ServerArgs.max_seq_len_override,
        help="The maximum sequence length override.",
    )

    parser.add_argument(
        "--mem-frac",
        type=float,
        dest="memory_ratio",
        default=ServerArgs.memory_ratio,
        help="The fraction of GPU memory to use for KV cache.",
    )

    parser.add_argument(
        "--dummy-weight",
        action="store_true",
        dest="use_dummy_weight",
        default=ServerArgs.use_dummy_weight,
        help="Use dummy weights for testing.",
    )

    parser.add_argument(
        "--disable-pynccl",
        action="store_false",
        dest="use_pynccl",
        default=ServerArgs.use_pynccl,
        help="Disable PyNCCL for tensor parallelism.",
    )

    parser.add_argument(
        "--host",
        type=str,
        dest="server_host",
        default=ServerArgs.server_host,
        help="The host address for the server.",
    )

    parser.add_argument(
        "--port",
        type=int,
        dest="server_port",
        default=ServerArgs.server_port,
        help="The port number for the server to listen on.",
    )

    parser.add_argument(
        "--cuda-graph-max-bs",
        "--graph",
        type=int,
        default=ServerArgs.cuda_graph_max_bs,
        help="The maximum batch size for CUDA graph capture. None means auto-tuning based on the GPU memory.",
    )

    parser.add_argument(
        "--num-tokenizer",
        "--tokenizer-count",
        type=int,
        default=ServerArgs.num_tokenizer,
        help="The number of tokenizer processes to launch. 0 means the tokenizer is shared with the detokenizer.",
    )

    parser.add_argument(
        "--max-prefill-length",
        "--max-extend-length",
        type=int,
        dest="max_extend_tokens",
        default=ServerArgs.max_extend_tokens,
        help="Chunk Prefill maximum chunk size in tokens.",
    )

    parser.add_argument(
        "--num-pages",
        dest="num_page_override",
        type=int,
        default=ServerArgs.num_page_override,
        help="Set the maximum number of pages for KVCache.",
    )

    parser.add_argument(
        "--page-size",
        type=int,
        default=ServerArgs.page_size,
        help="Set the page size for system management.",
    )

    parser.add_argument(
        "--attn-backend",
        type=validate_attn_backend,
        dest="attention_backend",
        default=ServerArgs.attention_backend,
        help="The attention backend to use. If two backends are specified,"
        " the first one is used for prefill and the second one for decode.",
    )

    parser.add_argument(
        "--model-source",
        type=str,
        default="huggingface",
        choices=["huggingface", "modelscope"],
        help="The source to download model from. Either 'huggingface' or 'modelscope'.",
    )

    parser.add_argument(
        "--cache-type",
        type=str,
        default=ServerArgs.cache_type,
        choices=SUPPORTED_CACHE_MANAGER.supported_names(),
        help="The KV cache management strategy.",
    )

    parser.add_argument(
        "--moe-backend",
        default=ServerArgs.moe_backend,
        choices=["auto"] + SUPPORTED_MOE_BACKENDS.supported_names(),
        help="The MoE backend to use.",
    )

    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run the server in CLI mode.",
    )

    return parser


def _apply_cli_overrides(
    kwargs: Dict[str, Any],
    *,
    run_cli: bool,
    user_set_cuda_graph_max_bs: bool,
    user_set_max_running_req: bool,
) -> None:
    """Apply conservative runtime defaults for single-user interactive CLI mode."""
    if not run_cli:
        return
    if not user_set_cuda_graph_max_bs:
        kwargs["cuda_graph_max_bs"] = 1
    if not user_set_max_running_req:
        kwargs["max_running_req"] = 1
    kwargs["silent_output"] = True


def _normalize_server_kwargs(kwargs: Dict[str, Any]) -> None:
    """Convert parsed CLI values into fields accepted by `ServerArgs`."""
    kwargs["model_path"] = _resolve_model_path(
        kwargs["model_path"],
        model_source=kwargs["model_source"],
        use_dummy_weight=kwargs["use_dummy_weight"],
    )
    del kwargs["model_source"]

    kwargs["dtype"] = _resolve_dtype(kwargs["dtype"], kwargs["model_path"])
    kwargs["tp_info"] = DistributedInfo(0, kwargs["tensor_parallel_size"])
    del kwargs["tensor_parallel_size"]


@dataclass(frozen=True)
class ServerArgs(SchedulerConfig):
    """Resolved server configuration used to launch the runtime."""
    server_host: str = "127.0.0.1"
    server_port: int = 1376
    num_tokenizer: int = 0
    silent_output: bool = False

    @property
    def share_tokenizer(self) -> bool:
        """Return whether tokenization and detokenization share one worker."""
        return self.num_tokenizer == 0

    @property
    def zmq_frontend_addr(self) -> str:
        """Return the frontend reply endpoint."""
        return "ipc:///tmp/sglite_3" + self._unique_suffix

    @property
    def zmq_tokenizer_addr(self) -> str:
        """Return the endpoint used to submit tokenizer work."""
        if self.share_tokenizer:
            return self.zmq_detokenizer_addr
        result = "ipc:///tmp/sglite_4" + self._unique_suffix
        assert result != self.zmq_detokenizer_addr
        return result

    @property
    def tokenizer_create_addr(self) -> bool:
        """Return whether the tokenizer worker owns endpoint creation."""
        return self.share_tokenizer

    @property
    def backend_create_detokenizer_link(self) -> bool:
        """Return whether backend workers should create the detokenizer link."""
        return not self.share_tokenizer

    @property
    def frontend_create_tokenizer_link(self) -> bool:
        """Return whether the frontend should create the tokenizer link."""
        return not self.share_tokenizer

    @property
    def distributed_addr(self) -> str:
        """Return the TCP endpoint used by torch.distributed."""
        return f"tcp://127.0.0.1:{self.server_port + 1}"


def parse_args(args: List[str], run_cli: bool = False) -> Tuple[ServerArgs, bool]:
    """
    Parse command line arguments and return server configuration.

    Args:
        args: Command line arguments (e.g., sys.argv[1:])

    Returns:
        Parsed server arguments and whether CLI mode is enabled.
    """
    parser = _build_parser()
    kwargs = parser.parse_args(args).__dict__.copy()
    user_set_cuda_graph_max_bs = any(
        arg == "--cuda-graph-max-bs" or arg.startswith("--cuda-graph-max-bs=") or arg == "--graph"
        for arg in args
    )
    user_set_max_running_req = any(
        arg == "--max-running-requests" or arg.startswith("--max-running-requests=")
        for arg in args
    )

    run_cli |= kwargs.pop("cli")
    _apply_cli_overrides(
        kwargs,
        run_cli=run_cli,
        user_set_cuda_graph_max_bs=user_set_cuda_graph_max_bs,
        user_set_max_running_req=user_set_max_running_req,
    )
    _normalize_server_kwargs(kwargs)

    result = ServerArgs(**kwargs)
    logger = init_logger(__name__)
    logger.debug("Resolved server arguments:\n%s", result)
    logger.info(
        "Launch config: model=%s mode=%s tp_size=%d host=%s port=%d tokenizers=%d",
        result.model_path,
        "cli" if run_cli else "server",
        result.tp_info.size,
        result.server_host,
        result.server_port,
        result.num_tokenizer,
    )
    return result, run_cli

"""Model executor setup and batch execution."""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, NamedTuple, Tuple

import torch
from sglite.srt.model_executor.layers.attention import create_attention_backend
from sglite.srt.distributed import destroy_distributed, enable_pynccl_distributed, set_tp_info
from sglite.srt.mem_cache import create_kvcache_pool
from sglite.srt.model_executor.layers import set_rope_device
from sglite.srt.model_executor.model_loader import load_weight
from sglite.srt.model_executor.models import create_model
from sglite.srt.model_executor.layers.fused_moe import create_moe_backend
from sglite.srt.request_state import Batch, Req
from sglite.srt.forward_context import Context, set_global_ctx
from sglite.srt.utils import div_even, init_logger, is_sm90_supported, is_sm100_supported, torch_dtype

from .config import EngineConfig
from .cuda_graph import GraphRunner, get_free_memory, mem_GB
from .sampler import BatchSamplingArgs, Sampler

logger = init_logger(__name__)


class ForwardOutput(NamedTuple):
    """Bundles the sampled tokens and transfer event from one forward pass."""
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    """Return the number of bytes occupied by one tensor."""
    return tensor.numel() * tensor.element_size()


class Engine:
    """Owns model state, CUDA resources, and batch execution."""
    def __init__(self, config: EngineConfig):
        """Build CUDA state, model weights, KV cache, and replay helpers."""
        assert not torch.cuda.is_initialized()
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        _adjust_config(config)

        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        torch.cuda.set_device(self.device)
        torch.manual_seed(42)
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype
        # Layers and kernels look up batch-scoped resources through this shared runtime context.
        self.ctx = Context(config.page_size)
        set_global_ctx(self.ctx)

        self.tp_cpu_group = self._init_communication(config)
        init_free_memory = self._sync_get_memory()[1]
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # Model weights.
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_config)
        state_dict, local_model_bytes = self._load_weight_state_dict(config)
        self.model.load_state_dict(state_dict)
        self._log_loaded_model_weights(local_model_bytes)

        # KV cache.
        self.num_pages = self._determine_num_pages(init_free_memory, config)
        num_tokens = self.num_pages * config.page_size
        self.ctx.kv_cache = self.kv_cache = create_kvcache_pool(
            model_config=config.model_config,
            num_pages=self.num_pages + 1,  # Extra page reserved for CUDA graph dummy requests.
            page_size=config.page_size,
            device=self.device,
            dtype=self.dtype,
        )

        # Page table.
        # Keep rows 128-byte aligned and store raw token offsets instead of page ids.
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        aligned_max_seq_len = _align_up_32(self.max_seq_len)
        # Allocate one extra row for the CUDA graph dummy request.
        self.ctx.page_table = self.page_table = torch.zeros(
            (config.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )

        # Attention and MoE backends.
        self.ctx.attn_backend = self.attn_backend = create_attention_backend(
            config.attention_backend, config.model_config
        )
        if config.model_config.is_moe:
            self.ctx.moe_backend = self.moe_backend = create_moe_backend(config.moe_backend)

        self.sampler = Sampler(self.device, config.model_config.vocab_size)

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # CUDA graph replay state.
        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        # CUDA graph capture needs a stable request slot and page-table row for replay.
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
        )

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        """Initialize tensor-parallel communication for this executor rank."""
        if config.tp_info.size == 1 or config.use_pynccl:
            # Keep a CPU-side process group for coordination even when collectives run via PyNCCL.
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            max_bytes = (
                config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
            )
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
        return tp_cpu_group

    def _load_weight_state_dict(self, config: EngineConfig) -> Tuple[Dict[str, torch.Tensor], int]:
        """Load the current rank's weights and return their runtime size in bytes."""
        if config.use_dummy_weight:
            state_dict: Dict[str, torch.Tensor] = {}
            local_model_bytes = 0
            for key, value in self.model.state_dict().items():
                tensor = torch.randn_like(value, device=self.device)
                state_dict[key] = tensor
                local_model_bytes += _tensor_nbytes(tensor)
            return state_dict, local_model_bytes

        state_dict = {}
        local_model_bytes = 0
        for key, value in load_weight(config.model_path, self.device):
            tensor = value.to(self.dtype) if value.dtype.is_floating_point else value
            state_dict[key] = tensor
            local_model_bytes += _tensor_nbytes(tensor)
        return state_dict, local_model_bytes

    def _log_loaded_model_weights(self, local_model_bytes: int) -> None:
        """Log the current-rank and total loaded model weight size."""
        total_model_bytes = torch.tensor([local_model_bytes], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            total_model_bytes,
            op=torch.distributed.ReduceOp.SUM,
            group=self.tp_cpu_group,
        )
        logger.info_rank0(
            "Loaded model weights: local=%s total=%s",
            mem_GB(local_model_bytes),
            mem_GB(int(total_model_bytes.item())),
        )

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        """Choose KV-cache capacity from free memory or an explicit override."""
        new_free_memory = self._sync_get_memory()[1]
        cache_per_page = (
            2
            * config.model_config.head_dim
            * div_even(config.model_config.num_kv_heads, config.tp_info.size, allow_replicate=True)
            * config.page_size
            * self.dtype.itemsize
            * config.model_config.num_layers
        )
        num_pages = config.num_page_override
        if num_pages is None:
            model_memory = old_free_memory - new_free_memory
            available_memory = int(config.memory_ratio * old_free_memory) - model_memory
            num_pages = available_memory // cache_per_page

        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-pages"
        num_tokens = num_pages * config.page_size
        real_kv_size = num_pages * cache_per_page
        logger.info(f"Allocating {num_tokens} tokens for KV cache, K + V = {mem_GB(real_kv_size)}")
        return num_pages

    def _sync_get_memory(self) -> Tuple[int, int]:
        """Return the min and max free memory across tensor-parallel ranks."""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            logger.error(
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        """Temporarily bind the runtime context to the batch being executed."""
        assert torch.cuda.current_stream() == self.stream
        with self.ctx.forward_batch(batch):
            if self.graph_runner.can_use_cuda_graph(batch):
                logits = self.graph_runner.replay(batch)
            else:
                logits = self.model.forward()

        # The model pass has consumed one decode step for each live request in the batch.
        for req in batch.reqs:
            req.complete_one()

        next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def shutdown(self) -> None:
        """Release owned resources and stop background work."""
        self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()


def _align_up_32(num: int) -> int:
    """Round a token count up to the next multiple of 32."""
    return (num + 31) // 32 * 32


def _adjust_config(config: EngineConfig):
    """Resolve automatic backend choices and compatibility overrides in place."""
    def override(attr: str, value: Any):
        """Override one config field in place."""
        object.__setattr__(config, attr, value)

    if config.attention_backend == "auto":
        # TODO: include TRTLLM in auto-selection after page-size normalization moves earlier.
        backend = "fi" if is_sm100_supported() else ("fa,fi" if is_sm90_supported() else "fi")
        override("attention_backend", backend)
        logger.info_rank0(f"Auto-selected attention backend: {config.attention_backend}")

    if "trtllm" in config.attention_backend and config.page_size not in [16, 32, 64]:
        override("page_size", 64)
        logger.warning_rank0("Page size is overridden to 64 for TRTLLM backend")

    if config.model_config.is_moe and config.moe_backend == "auto":
        override("moe_backend", "fused")
        logger.info_rank0(f"Auto-selected MoE backend: {config.moe_backend}")

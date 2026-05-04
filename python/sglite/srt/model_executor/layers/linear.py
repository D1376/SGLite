"""Tensor-parallel linear layer implementations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from sglite.srt.distributed import DistributedCommunicator, get_tp_info
from sglite.srt.utils import div_even

from .base import BaseModule, _STATE_DICT, _concat_prefix
from .quantization import LinearMethodBase, UnquantizedLinearMethod


class _TPLinearBase(BaseModule):
    """Linear layer base with tensor parallelism and optional quantization."""

    def __init__(
        self,
        full_isize: int,
        full_osize: int,
        local_isize: int,
        local_osize: int,
        has_bias: bool,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        """Create local tensor-parallel weights and optional bias."""
        self.full_input_size = full_isize
        self.full_output_size = full_osize
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        self._linear_method = linear_method or UnquantizedLinearMethod()
        self._weights: Dict[str, Any] = self._linear_method.create_weights(
            local_isize, local_osize
        )

        # Keep non-tensor metadata out of state_dict traversal.
        self._weight_metadata: Dict[str, Any] = {}
        for name in list(self._weights.keys()):
            if name.startswith("_"):
                self._weight_metadata[name] = self._weights.pop(name)

        # Expose tensor weights as attributes for BaseModule state_dict traversal.
        for name, tensor in self._weights.items():
            setattr(self, name, tensor)

        self.bias = torch.empty(local_osize) if has_bias else None
        self._weights_processed = False

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        """Load local weights and let the linear method repack if required."""
        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                item = state_dict.pop(_concat_prefix(prefix, name))
                assert isinstance(item, torch.Tensor)
                assert param.shape == item.shape and param.dtype == item.dtype
                setattr(self, name, item)
            elif isinstance(param, BaseModule):
                param.load_state_dict(
                    state_dict, prefix=_concat_prefix(prefix, name), _internal=True
                )

        # Point the method's weight map at the freshly loaded tensors.
        for name in list(self._weights.keys()):
            if hasattr(self, name):
                self._weights[name] = getattr(self, name)

        # Restore metadata before quantized methods repack loaded weights.
        for name, value in self._weight_metadata.items():
            self._weights[name] = value

        if not self._weights_processed:
            processed = self._linear_method.process_weights_after_loading(self._weights)

            self._weight_metadata = {}
            for name, value in processed.items():
                if name.startswith("_"):
                    self._weight_metadata[name] = value
                else:
                    self._weights[name] = value
                    # Runtime tensors created during repacking, such as workspace or g_idx,
                    # must not become state_dict-visible attributes.
                    if hasattr(self, name):
                        setattr(self, name, value)

            # Keep metadata in _weights because apply_weights consumes it.
            self._weights.update(self._weight_metadata)
            self._weights_processed = True

        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the configured dense or quantized linear method."""
        return self._linear_method.apply_weights(self._weights, x, self.bias)


class ReplicatedLinear(_TPLinearBase):
    """Linear layer where weights are replicated across all TP ranks."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        """Create a full local copy of the weight matrix."""
        super().__init__(
            full_isize=input_size,
            full_osize=output_size,
            local_isize=input_size,
            local_osize=output_size,
            has_bias=has_bias,
            linear_method=linear_method,
        )


class MergedColumnParallelLinear(_TPLinearBase):
    """Column-parallel projection with multiple fused output groups."""
    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        has_bias: bool,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        """Shard each merged output group across tensor-parallel ranks."""
        tp_info = get_tp_info()
        tp_output_sizes = [div_even(size, tp_info.size) for size in output_sizes]
        output_size = sum(output_sizes)
        tp_output_size = sum(tp_output_sizes)
        super().__init__(
            input_size, output_size, input_size, tp_output_size, has_bias, linear_method
        )


class QKVParallelLinear(_TPLinearBase):
    """Tensor-parallel projection that produces fused Q, K, and V tensors."""
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        """Shard query heads and replicate KV heads when required."""
        tp_info = get_tp_info()

        local_num_qo = div_even(num_qo_heads, tp_info.size)
        local_num_kv = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        full_isize = hidden_size
        full_osize = (num_qo_heads + 2 * num_kv_heads) * head_dim
        local_isize = hidden_size
        local_osize = (local_num_qo + 2 * local_num_kv) * head_dim
        super().__init__(
            full_isize, full_osize, local_isize, local_osize, has_bias, linear_method
        )


class RowParallelLinear(_TPLinearBase):
    """Row-parallel projection followed by tensor-parallel all-reduce."""
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        """Shard input features across tensor-parallel ranks."""
        tp_info = get_tp_info()
        local_isize = div_even(input_size, tp_info.size)
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(
            input_size, output_size, local_isize, output_size, has_bias, linear_method
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the local projection and reduce partial outputs across ranks."""
        y = self._linear_method.apply_weights(self._weights, x, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y

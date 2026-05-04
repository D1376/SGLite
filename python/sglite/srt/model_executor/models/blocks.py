"""Shared model-building blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglite.srt.model_executor.layers import (
    AttentionLayer,
    BaseModule,
    MergedColumnParallelLinear,
    MoELayer,
    QKVParallelLinear,
    RMSNorm,
    ReplicatedLinear,
    RowParallelLinear,
    gelu_and_mul,
    silu_and_mul,
)
from sglite.srt.model_executor.layers.quantization import LinearMethodBase
from sglite.srt.model_executor.models import ModelConfig
from sglite.srt.utils import nvtx_annotate

if TYPE_CHECKING:
    import torch


class GatedMLP(BaseModule):
    """Feed-forward block with a gated activation path."""
    def __init__(self, config: ModelConfig, linear_method: Optional[LinearMethodBase] = None):
        """Build gate/up and down projections for a gated feed-forward block."""
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            has_bias=False,
            linear_method=linear_method,
        )

        FN_MAP = {"silu": silu_and_mul, "gelu": gelu_and_mul}
        act_fn = FN_MAP.get(config.hidden_act, None)
        if act_fn is None:
            raise ValueError(f"Unsupported activation function: {config.hidden_act}")
        self.act_fn = act_fn
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            has_bias=False,
            linear_method=linear_method,
        )

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated activation followed by the down projection."""
        gate_up = self.gate_up_proj.forward(x)
        del x
        y = self.act_fn(gate_up)
        del gate_up
        return self.down_proj.forward(y)


class MoEMLP(BaseModule):
    """Feed-forward block that routes tokens through experts."""
    def __init__(self, config: ModelConfig):
        """Build the router and expert dispatch layer."""
        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
        )
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            has_bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Route flattened tokens through experts and restore batch shape."""
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate.forward(hidden_states)
        final_hidden_states = self.experts.forward(
            hidden_states=hidden_states, router_logits=router_logits
        )
        final_hidden_states = final_hidden_states.view(num_tokens, hidden_dim)
        return final_hidden_states


class RopeAttn(BaseModule):
    """Attention block wrapper that combines QKV projection, RoPE, and output projection."""
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        has_attn_bias: bool = False,
        has_qk_norm: bool = False,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        """Build QKV projection, optional Q/K norm, attention, and output projection."""
        head_dim = config.head_dim
        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            has_bias=has_attn_bias,
            linear_method=linear_method,
        )
        self.has_qk_norm = has_qk_norm
        if has_qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None
        self.attn = AttentionLayer(
            layer_id=layer_id,
            head_dim=head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            rotary_config=config.rotary_config,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
        )
        self.o_proj = RowParallelLinear(
            head_dim * config.num_qo_heads,
            config.hidden_size,
            has_bias=False,
            linear_method=linear_method,
        )

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply QKV projection, attention backend, and output projection."""
        qkv = self.qkv_proj.forward(x)
        del x
        o = self.attn.forward(qkv)
        return self.o_proj.forward(o)


__all__ = ["GatedMLP", "RopeAttn", "MoEMLP"]

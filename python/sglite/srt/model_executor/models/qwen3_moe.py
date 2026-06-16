"""Model definitions for the Qwen3 MoE architecture."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from sglite.srt.model_executor.layers import (
    BaseModule,
    ModuleList,
    ParallelLMHead,
    RMSNormFused,
    VocabParallelEmbedding,
)
from sglite.srt.forward_context import get_global_ctx
from sglite.srt.utils import nvtx_annotate

from .base import BaseLLMModel
from .blocks import MoEMLP as Qwen3MLP
from .blocks import RopeAttn as Qwen3Attn

if TYPE_CHECKING:
    from .config import ModelConfig


class Qwen3DecoderLayer(BaseModule):
    """Decoder block for the Qwen3 MoE architecture."""
    def __init__(self, config: ModelConfig, layer_id: int):
        """Build attention, MoE MLP, and RMSNorm blocks for one Qwen3 MoE layer."""
        self.self_attn = Qwen3Attn(config, layer_id, has_qk_norm=True)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one Qwen3 MoE decoder layer and carry the residual stream forward."""
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class Qwen3Model(BaseModule):
    """Transformer body for the Qwen3 MoE architecture."""
    def __init__(self, config: ModelConfig):
        """Build embeddings, decoder layers, and final normalization."""
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = ModuleList(
            [Qwen3DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input ids and run all Qwen3 MoE decoder layers."""
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.modules:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class Qwen3MoeForCausalLM(BaseLLMModel):
    """Causal language model wrapper for the Qwen3 MoE architecture."""
    def __init__(self, config: ModelConfig):
        """Build the Qwen3 MoE transformer body and LM head."""
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        """Run the current runtime batch and project hidden states to logits."""
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["Qwen3MoeForCausalLM"]

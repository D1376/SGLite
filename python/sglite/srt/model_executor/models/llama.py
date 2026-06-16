"""Model definitions for the Llama architecture."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
from sglite.srt.model_executor.layers import (
    BaseModule,
    ModuleList,
    ParallelLMHead,
    RMSNormFused,
    VocabParallelEmbedding,
)
from sglite.srt.model_executor.layers.quantization import LinearMethodBase
from sglite.srt.forward_context import get_global_ctx
from sglite.srt.utils import nvtx_annotate

from .base import BaseLLMModel
from .blocks import GatedMLP as LlamaMLP
from .blocks import RopeAttn as LlamaAttn

if TYPE_CHECKING:
    from .config import ModelConfig


class LlamaDecoderLayer(BaseModule):
    """Layer implementation for Llama decoder."""
    def __init__(self, config: ModelConfig, layer_id: int, linear_method: Optional[LinearMethodBase] = None):
        """Build attention, MLP, and RMSNorm blocks for one Llama layer."""
        self.self_attn = LlamaAttn(config, layer_id, linear_method=linear_method)
        self.mlp = LlamaMLP(config, linear_method=linear_method)
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
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one Llama decoder layer and carry the residual stream forward."""
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class LlamaModel(BaseModule):
    """Model definition for Llama."""
    def __init__(self, config: ModelConfig, linear_method: Optional[LinearMethodBase] = None):
        """Build embeddings, decoder layers, and final normalization."""
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = ModuleList(
            [LlamaDecoderLayer(config, layer_id, linear_method=linear_method) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input ids and run all Llama decoder layers."""
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.modules:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class LlamaForCausalLM(BaseLLMModel):
    """Causal language model wrapper for Llama."""
    def __init__(self, config: ModelConfig):
        """Build the Llama transformer body and LM head."""
        linear_method = config.quantization_config.get_linear_method() if config.quantization_config is not None else None
        self.model = LlamaModel(config, linear_method=linear_method)
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


__all__ = ["LlamaForCausalLM"]

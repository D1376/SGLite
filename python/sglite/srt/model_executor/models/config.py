"""Model configuration loading and normalization."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

from sglite.srt.utils.logger import init_logger

if TYPE_CHECKING:
    from transformers import PretrainedConfig

logger = init_logger(__name__)


@dataclass(frozen=True)
class RotaryConfig:
    """Normalized rotary embedding settings."""
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None


def load_quantization_config(
    model_path: str,
    hf_config: Optional[Any] = None,
    force_awq_triton: bool = False,
) -> Optional[Any]:
    """Detect and load AWQ quantization config from HF config or model directory."""
    config_dict = None

    if hf_config is not None:
        quant_cfg = getattr(hf_config, "quantization_config", None)
        if quant_cfg is not None:
            if hasattr(quant_cfg, "to_dict"):
                config_dict = quant_cfg.to_dict()
            elif isinstance(quant_cfg, dict):
                config_dict = quant_cfg

    if config_dict is None:
        local_path = model_path
        if not os.path.isdir(model_path):
            try:
                from huggingface_hub import snapshot_download

                local_path = snapshot_download(repo_id=model_path, local_files_only=True)
            except Exception:
                return None

        for config_name in ["quant_config.json", "quantize_config.json"]:
            config_path = os.path.join(local_path, config_name)
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_dict = json.load(f)
                break

    if config_dict is None:
        return None

    quant_method = config_dict.get("quant_method", "").lower()
    is_awq = quant_method == "awq" or (not quant_method and ("w_bit" in config_dict or "bits" in config_dict))

    if is_awq:
        return _create_awq_config(config_dict, force_awq_triton)

    return None


def _create_awq_config(config_dict: Dict[str, Any], force_awq_triton: bool = False) -> Any:
    """Select AWQ Marlin (SM80+) or AWQ Triton based on GPU capability and config."""
    from sglite.srt.model_executor.layers.quantization import AWQConfig
    from sglite.srt.model_executor.layers.quantization.awq_marlin import AWQMarlinConfig

    if not force_awq_triton:
        if AWQMarlinConfig.is_awq_marlin_compatible(config_dict):
            try:
                marlin_config = AWQMarlinConfig.from_config(config_dict)
                logger.info("Using AWQ Marlin kernel for quantized inference.")
                return marlin_config
            except Exception as e:
                logger.debug("AWQ Marlin setup failed, falling back to Triton: %s", e)

    awq_config = AWQConfig.from_config(config_dict)
    logger.info("Using AWQ Triton kernel for quantized inference.")
    return awq_config


@dataclass(frozen=True)
class ModelConfig:
    """Normalized decoder-only model settings used by the executor."""
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    model_type: str
    architectures: list[str]
    quantization_config: Optional[Any] = None

    @property
    def is_moe(self) -> bool:
        """Return whether this architecture uses mixture-of-experts layers."""
        return "moe" in self.model_type

    @property
    def is_quantized(self) -> bool:
        """Return whether the model should use a quantized linear method."""
        return self.quantization_config is not None

    @classmethod
    def from_hf(
        cls,
        config: PretrainedConfig,
        model_path: Optional[str] = None,
    ) -> ModelConfig:
        """Build normalized executor config from Hugging Face metadata."""
        if hasattr(config, "text_config") and config.text_config is not None:
            top = config
            config = config.text_config
            for attr in ("architectures", "rope_theta", "rope_scaling"):
                if not getattr(config, attr, None) and getattr(top, attr, None):
                    setattr(config, attr, getattr(top, attr))

        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        model_type = getattr(config, "model_type", "llama")
        num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", 0))
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 0)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 0)
        norm_topk_prob = getattr(config, "norm_topk_prob", False)
        architectures = getattr(config, "architectures", ["LlamaForCausalLM"])

        rope_scaling = getattr(config, "rope_scaling", None)
        rope_theta = getattr(config, "rope_theta", None) or (
            rope_scaling.get("rope_theta") if isinstance(rope_scaling, dict) else None
        )

        quant_config = None
        if model_path is not None:
            quant_config = load_quantization_config(model_path, hf_config=config)

        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=head_dim,
                max_position=config.max_position_embeddings,
                base=rope_theta,
                scaling=rope_scaling,
            ),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            model_type=model_type,
            architectures=architectures,
            quantization_config=quant_config,
        )

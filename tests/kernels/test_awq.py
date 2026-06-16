"""Tests for AWQ."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock


def test_awq_config_from_config():
    from sglite.srt.model_executor.layers.quantization.awq import AWQConfig

    config_dict = {
        "quant_method": "awq",
        "w_bit": 4,
        "q_group_size": 128,
        "zero_point": True,
    }
    cfg = AWQConfig.from_config(config_dict)
    assert cfg.weight_bits == 4
    assert cfg.group_size == 128
    assert cfg.zero_point is True
    assert cfg.pack_factor == 8


def test_awq_config_from_config_rejects_unsupported_bits():
    from sglite.srt.model_executor.layers.quantization.awq import AWQConfig

    with pytest.raises(ValueError):
        AWQConfig.from_config(
            {
                "quant_method": "awq",
                "w_bit": 3,
                "q_group_size": 128,
                "zero_point": True,
            }
        )


def test_awq_marlin_config_is_compatible():
    from sglite.srt.model_executor.layers.quantization.awq_marlin import AWQMarlinConfig

    cfg_dict = {
        "quant_method": "awq",
        "w_bit": 4,
        "q_group_size": 128,
        "zero_point": True,
    }
    assert AWQMarlinConfig.is_awq_marlin_compatible(cfg_dict) is True


def test_awq_marlin_config_rejects_wrong_quant_method():
    from sglite.srt.model_executor.layers.quantization.awq_marlin import AWQMarlinConfig

    cfg_dict = {
        "quant_method": "gptq",
        "w_bit": 4,
        "q_group_size": 128,
        "zero_point": True,
    }
    assert AWQMarlinConfig.is_awq_marlin_compatible(cfg_dict) is False


def test_is_awq_weight_positive_cases():
    from sglite.srt.model_executor.model_loader.weight_loader import _is_awq_weight

    assert _is_awq_weight("model.layers.0.self_attn.q_proj.qweight") is True
    assert _is_awq_weight("model.layers.0.self_attn.q_proj.qzeros") is True
    assert _is_awq_weight("model.layers.0.self_attn.q_proj.scales") is True


def test_is_awq_weight_negative_cases():
    from sglite.srt.model_executor.model_loader.weight_loader import _is_awq_weight

    assert _is_awq_weight("model.layers.0.self_attn.q_proj.weight") is False
    assert _is_awq_weight("model.norm.weight") is False


def test_load_quantization_config_returns_none_for_empty_config():
    from sglite.srt.model_executor.models.config import load_quantization_config

    hf_config = MagicMock()
    hf_config.quantization_config = None
    result = load_quantization_config("/nonexistent/path", hf_config=hf_config)
    assert result is None


def test_load_quantization_config_gptq_not_detected_as_awq():
    from sglite.srt.model_executor.models.config import load_quantization_config

    hf_config = MagicMock()
    gptq_mock = MagicMock()
    gptq_mock.to_dict.return_value = {"quant_method": "gptq", "bits": 4}
    hf_config.quantization_config = gptq_mock
    result = load_quantization_config("/nonexistent/path", hf_config=hf_config)
    assert result is None

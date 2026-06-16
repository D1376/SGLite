"""Tests for sampling params."""

from __future__ import annotations

from sglite.sampling_params import SamplingParams


def test_sampling_params_defaults_and_is_greedy_match_existing_behavior():
    params = SamplingParams()

    assert params.temperature == 0.0
    assert params.top_k == -1
    assert params.top_p == 1.0
    assert params.ignore_eos is False
    assert params.max_tokens == 1024
    assert params.is_greedy is True

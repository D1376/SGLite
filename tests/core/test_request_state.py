"""Tests for request."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from sglite.srt.request_state import Req
from sglite.sampling_params import SamplingParams


@dataclass(frozen=True)
class DummyCacheHandle:
    cached_len: int

    def get_matched_indices(self) -> torch.Tensor:
        return torch.empty(0, dtype=torch.int32)


def _make_req(*, cached_len: int = 1, output_len: int = 2) -> Req:
    return Req(
        input_ids=torch.tensor([10, 11, 12], dtype=torch.int32),
        table_idx=3,
        cached_len=cached_len,
        output_len=output_len,
        uid=7,
        sampling_params=SamplingParams(max_tokens=output_len),
        cache_handle=DummyCacheHandle(cached_len),
    )


def test_req_exposes_lengths_and_decode_state():
    req = _make_req(cached_len=1, output_len=2)

    assert req.device_len == 3
    assert req.max_device_len == 5
    assert req.remain_len == 2
    assert req.extend_len == 2
    assert req.can_decode is True


def test_req_complete_one_advances_cached_and_device_lengths():
    req = _make_req(cached_len=1, output_len=2)

    req.complete_one()

    assert req.cached_len == 3
    assert req.device_len == 4
    assert req.remain_len == 1


def test_req_append_host_appends_next_token():
    req = _make_req(cached_len=1, output_len=0)

    req.append_host(torch.tensor([99], dtype=torch.int32))

    assert req.input_ids.tolist() == [10, 11, 12, 99]


def test_req_append_host_reuses_buffer_after_first_append():
    req = _make_req(cached_len=1, output_len=3)

    req.complete_one()
    req.append_host(torch.tensor([99], dtype=torch.int32))
    first_data_ptr = req.input_ids.data_ptr()

    req.complete_one()
    req.append_host(torch.tensor([100], dtype=torch.int32))

    assert req.input_ids.tolist() == [10, 11, 12, 99, 100]
    assert req.input_ids.data_ptr() == first_data_ptr


def test_req_can_decode_is_false_when_no_tokens_remain():
    req = _make_req(cached_len=1, output_len=0)

    assert req.remain_len == 0
    assert req.can_decode is False


def test_req_rejects_invalid_cached_length():
    with pytest.raises(AssertionError):
        _make_req(cached_len=3, output_len=1)

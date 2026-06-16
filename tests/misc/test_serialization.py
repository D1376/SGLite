"""Tests for serialization."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List

import torch
from sglite.srt.messages import BatchBackendMsg, UserMsg
from sglite.srt.messages.serialization import deserialize_type, serialize_type
from sglite.sampling_params import SamplingParams
from sglite.srt.utils import init_logger

logger = init_logger(__name__)


@dataclass
class A:
    x: int
    y: str
    z: List[A]
    w: torch.Tensor


def test_serialize_deserialize():
    t = torch.tensor([1, 2, 3], dtype=torch.int32)
    x = A(10, "hello", [A(20, "world", [], t)], t)
    data = serialize_type(x)
    logger.info(data)
    y = deserialize_type({"A": A}, data)
    logger.info(y)

    u = BatchBackendMsg([UserMsg(uid=0, input_ids=t, sampling_params=SamplingParams())])
    result = u.decoder(u.encoder())
    logger.info(u)
    logger.info(result)


if __name__ == "__main__":
    test_serialize_deserialize()

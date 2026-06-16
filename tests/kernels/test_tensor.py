"""Tests for tensor."""

from __future__ import annotations

from sglite.kernels import test_tensor as _test_tensor
import torch


def main():
    x = torch.empty((12, 2048), dtype=torch.int32, device="cpu")[:, :1024]
    y = torch.empty((12, 1024), dtype=torch.int64, device="cuda:1")
    _test_tensor(x, y)


if __name__ == "__main__":
    main()

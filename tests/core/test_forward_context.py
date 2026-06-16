"""Tests for context."""

from __future__ import annotations

import pytest

from sglite.srt.request_state import Batch
from sglite.srt.forward_context import (
    Context,
    get_global_ctx,
    reset_global_ctx as clear_global_ctx,
    set_global_ctx,
)


@pytest.fixture(autouse=True)
def reset_global_ctx():
    clear_global_ctx()
    yield
    clear_global_ctx()


def test_context_batch_requires_active_forward_batch():
    ctx = Context(page_size=4)

    with pytest.raises(AssertionError, match="No active batch in context"):
        _ = ctx.batch


def test_forward_batch_rejects_nesting():
    ctx = Context(page_size=4)
    batch = Batch(reqs=[], phase="prefill")

    with ctx.forward_batch(batch):
        assert ctx.batch is batch
        with pytest.raises(AssertionError, match="Nested forward_batch is not allowed"):
            with ctx.forward_batch(batch):
                pass


def test_forward_batch_clears_state_after_exception():
    ctx = Context(page_size=4)
    batch = Batch(reqs=[], phase="decode")

    with pytest.raises(RuntimeError, match="boom"):
        with ctx.forward_batch(batch):
            raise RuntimeError("boom")

    with pytest.raises(AssertionError, match="No active batch in context"):
        _ = ctx.batch


def test_reset_global_ctx_clears_existing_context():
    ctx = Context(page_size=8)

    set_global_ctx(ctx)
    assert get_global_ctx() is ctx

    clear_global_ctx()

    with pytest.raises(AssertionError, match="Global context is not set"):
        get_global_ctx()

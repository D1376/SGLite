"""Tests for benchmark client summary helpers."""

from __future__ import annotations

import json

import pytest

from sglite.benchmark.client import RawResult, process_benchmark_results, save_benchmark_result


def test_process_benchmark_results_excludes_stream_stop_chunk():
    raw_data = [
        RawResult(
            input_len=5,
            output_len=3,
            message="",
            tics=[0.0, 0.10, 0.20, 0.30, 0.35],
            actual_output_len=3,
        )
    ]

    summary = process_benchmark_results(raw_data)

    assert summary["actual_output_tokens"] == 3
    assert summary["output_budget_tokens"] == 3
    assert summary["output_length"]["p50"] == 3
    assert summary["ttft_ms"]["avg"] == pytest.approx(100.0)
    assert summary["tpot_ms"]["count"] == 2
    assert summary["e2e_seconds"]["p50"] == pytest.approx(0.35)


def test_save_benchmark_result_writes_summary_json(tmp_path):
    path = save_benchmark_result(
        "online_streaming",
        {"model": "test-model"},
        {"actual_output_tokens": 3},
        tmp_path,
    )

    payload = json.loads(path.read_text())
    assert payload["kind"] == "online_streaming"
    assert payload["config"]["model"] == "test-model"
    assert payload["summary"]["actual_output_tokens"] == 3
    assert payload["created_at"]

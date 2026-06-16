"""Shared benchmark helpers and result processing."""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

if TYPE_CHECKING:
    from openai import AsyncOpenAI as OpenAI
    from tqdm.asyncio import tqdm


_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class Unset:
    """Sentinel type for omitted optional values."""


UNSET = Unset()


def init_logger(name: str, level: str | None = None) -> logging.Logger:
    """Initialize a benchmark logger without importing the SGLite runtime."""
    level_name = (level or os.getenv("LOG_LEVEL", "")).upper()
    logger = logging.getLogger(name)
    logger.setLevel(_LEVEL_MAP.get(level_name, logging.INFO))
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s|%(name)s] %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d|%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = init_logger(__name__)

DEFAULT_OFFLINE_MODELS = (
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-14B-AWQ",
)


@dataclass(frozen=True)
class BenchmarkTrace:
    """Represents one timestamped event in a benchmark trace."""

    timestamp: float  # unit (second)
    message: str
    output_length: int  # output length in tokens
    input_length: int | None = None  # input length in tokens, optional


@dataclass(frozen=True)
class BenchOneResult:
    """Stores timing data for a single benchmarked request."""

    tics: List[float]
    input_len: int
    output_len: int

    def as_json(self) -> List[float]:
        """Execute as json for the bench one result."""
        return [self.input_len, self.output_len] + self.tics

    @staticmethod
    def from_json(raw: List[float]) -> BenchOneResult:
        # check raw[0] and raw[1] are integers
        """Execute from json for the bench one result."""
        assert raw[0].is_integer() and raw[1].is_integer()
        return BenchOneResult(
            tics=raw[2:], input_len=int(raw[0]), output_len=int(raw[1])
        )


@dataclass(frozen=True)
class RawResult:
    """Raw per-request output collected during benchmarking."""

    input_len: int | None
    output_len: int
    message: str
    tics: List[float]
    actual_output_len: int | None = None
    content_tics: List[float] = field(default_factory=list)
    generated_text: str = ""


@dataclass(frozen=True)
class SyntheticWorkload:
    """Tokenized synthetic benchmark workload."""

    prompt_token_ids: List[List[int]]
    output_lengths: List[int]

    @property
    def input_lengths(self) -> List[int]:
        """Return input token lengths for all prompts."""
        return [len(ids) for ids in self.prompt_token_ids]


@dataclass
class Counter:
    """Tracks current and historical maxima for a progress metric."""

    current: int = 0
    history_max: int = 0

    def inc(self, n=1):
        """Execute inc for the counter."""
        self.current += n
        self.history_max = max(self.history_max, self.current)

    def dec(self, n=1):
        """Execute dec for the counter."""
        self.current -= n
        assert self.current >= 0


@dataclass
class Console:
    """Owns progress bars and counters for benchmark reporting."""

    input_pbar: tqdm
    output_pbar: tqdm
    prefill_pbar: tqdm
    decode_pbar: tqdm
    disabled: bool
    inflight_counter: Counter = field(default_factory=Counter)
    queue_counter: Counter = field(default_factory=Counter)

    def update_input(self, n=1):
        """Execute update input for the console."""
        self.input_pbar.update(n)
        self.input_pbar.refresh()
        self.inflight_counter.inc(n)
        self.queue_counter.inc(n)

    def update_output(self, n=1):
        """Execute update output for the console."""
        self.output_pbar.update(n)
        self.output_pbar.refresh()
        self.inflight_counter.dec(n)

    def update_prefill(self, n=1):
        """Execute update prefill for the console."""
        self.prefill_pbar.update(n)
        self.prefill_pbar.refresh()
        self.queue_counter.dec(n)

    def update_decode(self, n=1):
        """Execute update decode for the console."""
        self.decode_pbar.update(n)

    @contextmanager
    def inflight(self, n=1):
        """Execute inflight for the console."""
        self.update_input(n)
        yield
        self.update_output(n)

    @contextmanager
    def log_stats(self):
        """Execute log stats for the console."""
        yield
        self.input_pbar.close()
        self.output_pbar.close()
        self.prefill_pbar.close()
        self.decode_pbar.close()
        if not self.disabled:
            max_inflight = self.inflight_counter.history_max
            max_queue = self.queue_counter.history_max
            logger.info(
                f"Max inflight requests: {max_inflight}, Max queued requests: {max_queue}"
            )


@dataclass(frozen=True)
class BenchmarkResult:
    """Container for a full benchmark run."""

    raw_data: List[BenchOneResult]

    def as_json(self) -> List[List[float]]:
        """Execute as json for the benchmark result."""
        return [r.as_json() for r in self.raw_data]

    @staticmethod
    def from_json(raw: List[List[float]]) -> BenchmarkResult:
        """Execute from json for the benchmark result."""
        return BenchmarkResult(raw_data=[BenchOneResult.from_json(r) for r in raw])


def make_console(
    num_requests: int, sum_output_length: int, use_pbar: bool = True
) -> Console:
    """Create the progress-bar console used for benchmark reporting."""
    from tqdm.asyncio import tqdm

    BAR_FORMAT_0 = (
        "{desc:<10} {percentage:3.0f}%|{bar}|"
        " {n_fmt:>5}/{total_fmt} "
        "[{rate_fmt:>12} {elapsed:>8}/{remaining:<8}]"
    )
    BAR_FORMAT_1 = BAR_FORMAT_0
    n_fmt_align = 5
    prefill_tokens = num_requests
    decode_tokens = sum_output_length - prefill_tokens

    if len(str(decode_tokens)) > n_fmt_align:
        n_fmt_align = len(str(decode_tokens))
        BAR_FORMAT_0 = BAR_FORMAT_0.replace(
            "{n_fmt:>5}", "{n_fmt:>" + str(n_fmt_align) + "}"
        )
        BAR_FORMAT_1 = BAR_FORMAT_0

    if len(str(prefill_tokens)) < len(str(decode_tokens)):
        old_align_str = "{n_fmt:>" + str(n_fmt_align) + "}"
        n_fmt_align += len(str(decode_tokens)) - len(str(prefill_tokens))
        BAR_FORMAT_0 = BAR_FORMAT_0.replace(
            old_align_str, "{n_fmt:>" + str(n_fmt_align) + "}"
        )

    disabled = not use_pbar
    input_pbar = tqdm(
        total=num_requests,
        desc="Requests sent",
        position=0,
        bar_format=BAR_FORMAT_0,
        disable=disabled,
    )
    output_pbar = tqdm(
        total=num_requests,
        desc="Requests done",
        position=1,
        bar_format=BAR_FORMAT_0,
        disable=disabled,
    )
    prefill_pbar = tqdm(
        total=prefill_tokens,
        desc="Prefill token",
        position=2,
        bar_format=BAR_FORMAT_0,
        disable=disabled,
    )
    decode_pbar = tqdm(
        total=decode_tokens,
        desc="Decode token ",
        position=3,
        bar_format=BAR_FORMAT_1,
        disable=disabled,
    )
    return Console(
        input_pbar=input_pbar,
        output_pbar=output_pbar,
        prefill_pbar=prefill_pbar,
        decode_pbar=decode_pbar,
        disabled=disabled,
    )


def length_stats(lengths: List[int]) -> Dict[str, int | None]:
    """Return compact percentile stats for a list of token lengths."""
    if not lengths:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p90": None,
            "p99": None,
            "max": None,
        }

    arr = sorted(lengths)
    n = len(arr)

    def percentile(q: float) -> int:
        return arr[min(int(q * n), n - 1)]

    return {
        "count": n,
        "min": arr[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "max": arr[-1],
    }


def build_generation_summary(
    *,
    input_lengths: List[int],
    output_lengths: List[int],
    output_budget_lengths: List[int],
    elapsed_seconds: float,
) -> Dict[str, Any]:
    """Build the shared summary shape for generation benchmarks."""
    num_requests = len(input_lengths)
    total_output_tokens = sum(output_lengths)
    return {
        "requests": num_requests,
        "elapsed_seconds": elapsed_seconds,
        "request_throughput_per_s": (
            num_requests / elapsed_seconds if elapsed_seconds > 0 else 0.0
        ),
        "input_tokens": sum(input_lengths),
        "actual_output_tokens": total_output_tokens,
        "output_budget_tokens": sum(output_budget_lengths),
        "output_tok_per_s": total_output_tokens / elapsed_seconds
        if elapsed_seconds > 0
        else 0.0,
        "input_length": length_stats(input_lengths),
        "output_length": length_stats(output_lengths),
    }


def generate_synthetic_token_workload(
    *,
    seed: int,
    num_seqs: int,
    min_input_len: int,
    max_input_len: int,
    min_output_len: int,
    max_output_len: int,
    token_id_max: int = 10000,
) -> SyntheticWorkload:
    """Generate the shared token-id workload used by offline benchmarks."""
    rng = random.Random(seed)
    prompt_token_ids = []
    output_lengths = []
    for _ in range(num_seqs):
        input_len = rng.randint(min_input_len, max_input_len)
        prompt_token_ids.append(
            [rng.randint(0, token_id_max) for _ in range(input_len)]
        )
        output_lengths.append(rng.randint(min_output_len, max_output_len))
    return SyntheticWorkload(
        prompt_token_ids=prompt_token_ids, output_lengths=output_lengths
    )


def generate_online_text_workload(
    tokenizer: Any,
    *,
    seed: int,
    num_requests: int,
    max_input_len: int,
    min_output_len: int,
    max_output_len: int,
) -> Tuple[List[str], List[int], List[int]]:
    """Generate shared text prompts and output budgets for online benchmarks."""
    rng = random.Random(seed)
    prompts = []
    input_lengths = []
    for _ in range(num_requests):
        length = rng.randint(1, max_input_len)
        prompts.append(generate_prompt(tokenizer, length, rng=rng))
        input_lengths.append(length)
    output_lengths = [
        rng.randint(min_output_len, max_output_len) for _ in range(num_requests)
    ]
    return prompts, input_lengths, output_lengths


def _flatten_items(prefix: str, value: Any) -> List[Tuple[str, Any]]:
    """Flatten nested dictionaries for table output."""
    if isinstance(value, dict):
        rows: List[Tuple[str, Any]] = []
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            rows.extend(_flatten_items(key, v))
        return rows
    return [(prefix, value)]


def _format_summary_value(value: Any) -> str:
    """Format a summary value for console display."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) >= 100:
            return f"{value:.2f}"
        if abs(value) >= 1:
            return f"{value:.4f}"
        return f"{value:.6f}"
    return str(value)


def print_summary_table(title: str, summary: Dict[str, Any]) -> None:
    """Print a compact benchmark summary table."""
    rows = _flatten_items("", summary)
    if not rows:
        print(f"\n{title}\n(no data)")
        return

    width = max(len(key) for key, _ in rows)
    print(f"\n{title}")
    print("-" * max(len(title), width + 10))
    for key, value in rows:
        print(f"{key:<{width}} : {_format_summary_value(value)}")


def save_benchmark_result(
    kind: str,
    config: Dict[str, Any],
    summary: Dict[str, Any],
    output_dir: str | Path,
) -> Path:
    """Persist one benchmark summary JSON file and return its path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now().astimezone()
    stamp = created_at.strftime("%Y%m%d-%H%M%S")
    path = output_path / f"{stamp}-{kind}.json"
    suffix = 1
    while path.exists():
        path = output_path / f"{stamp}-{kind}-{suffix}.json"
        suffix += 1

    payload = {
        "kind": kind,
        "created_at": created_at.isoformat(),
        "config": config,
        "summary": summary,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Saved result: {path}")
    return path


def save_benchmark_suite_result(
    kind: str,
    results: List[Dict[str, Any]],
    output_dir: str | Path,
) -> Path:
    """Persist a benchmark suite with multiple run summaries in one JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now().astimezone()
    stamp = created_at.strftime("%Y%m%d-%H%M%S")
    path = output_path / f"{stamp}-{kind}.json"
    suffix = 1
    while path.exists():
        path = output_path / f"{stamp}-{kind}-{suffix}.json"
        suffix += 1

    payload = {
        "kind": kind,
        "created_at": created_at.isoformat(),
        "results": results,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Saved result: {path}")
    return path


def clear_benchmark_cache() -> None:
    """Release local and CUDA allocator cache between large model benchmark runs."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        ipc_collect = getattr(torch.cuda, "ipc_collect", None)
        if ipc_collect is not None:
            ipc_collect()


def generate_prompt(tokenizer: Any, n: int, *, rng: random.Random | None = None) -> str:
    """Generate a prompt of approximately `n` tokens using the provided tokenizer."""
    rand = rng or random
    vocab_size = tokenizer.vocab_size // 2
    token_ids = [rand.randint(0, vocab_size) for _ in range(n)]

    for _ in range(64):
        prompt = tokenizer.decode(token_ids)
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) == n:
            return prompt
        if len(token_ids) < n:
            need = n - len(token_ids)
            token_ids.extend([rand.randint(0, vocab_size) for _ in range(need)])
        else:
            token_ids = token_ids[:n]

    raise ValueError("Failed to generate a message of the desired length.")


async def benchmark_one(
    client: OpenAI,
    prompt: str,
    output_length: int,
    model: str,
    *,
    pbar: Console | bool = True,
    extra_body: Dict[str, Any] | None = None,
    input_length: int | None = None,  # a hack to force input length
    use_input_length_override: bool = False,
) -> RawResult:
    """Benchmark a single request against an OpenAI-compatible endpoint."""
    if isinstance(pbar, bool):
        pbar = make_console(1, output_length, use_pbar=pbar)
    with pbar.inflight(1):
        kwargs = {
            "ignore_eos": True,
            "top_k": 1,
        }
        # this is an internal kwargs that might work for our system
        if use_input_length_override and input_length is not None:
            kwargs["input_length_override"] = input_length
        kwargs.update(extra_body or {})  # can override kwargs
        tics = [time.perf_counter()]
        response = await client.chat.completions.create(
            model=model,
            stream=True,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=output_length,
            temperature=0.0,
            extra_body=kwargs,
        )
        content_tics: List[float] = []
        text_parts: List[str] = []
        async for chunk in response:
            now = time.perf_counter()
            tics.append(now)
            choices = getattr(chunk, "choices", [])
            finish_reason = choices[0].finish_reason if choices else None
            if finish_reason is not None:
                continue

            delta = choices[0].delta if choices else None
            content = getattr(delta, "content", None)
            if not content:
                continue

            text_parts.append(content)
            content_tics.append(now)
            if len(content_tics) == 1:
                pbar.update_prefill()
            else:
                pbar.update_decode()
        generated_text = "".join(text_parts)
        return RawResult(
            input_len=input_length,
            output_len=output_length,
            message=prompt,
            tics=tics,
            actual_output_len=len(content_tics),
            content_tics=content_tics,
            generated_text=generated_text,
        )


async def benchmark_one_batch(
    client: OpenAI,
    prompts: List[str],
    output_lengths: List[int] | int,
    model: str,
    *,
    extra_body: Dict[str, Any] | None = None,
    input_lengths: List[int | None] | None = None,
    pbar: Console | bool = True,
) -> List[RawResult]:
    """Benchmark a batch of prompts concurrently against an OpenAI-compatible endpoint."""
    if isinstance(output_lengths, int):
        output_lengths = [output_lengths] * len(prompts)
    if isinstance(pbar, bool):
        pbar = make_console(len(prompts), sum(output_lengths), use_pbar=pbar)
    if input_lengths is None:
        empty_input_lengths: List[int | None] = [None] * len(prompts)
        input_lengths = empty_input_lengths  # work-around for typing bug

    tasks = [
        benchmark_one(
            client=client,
            prompt=prompt,
            output_length=output_length,
            model=model,
            pbar=pbar,
            extra_body=extra_body,
            input_length=input_length,
        )
        for prompt, output_length, input_length in zip(
            prompts, output_lengths, input_lengths, strict=True
        )
    ]
    with pbar.log_stats():
        return await asyncio.gather(*tasks)


async def benchmark_trace(
    client: OpenAI,
    msgs: List[BenchmarkTrace],
    model: str,
    *,
    pbar: Console | bool = True,
) -> List[RawResult]:
    """Replay a recorded arrival trace against an OpenAI-compatible endpoint."""
    if isinstance(pbar, bool):
        sum_output_len = sum(msg.output_length for msg in msgs)
        pbar = make_console(len(msgs), sum_output_len, use_pbar=pbar)
    start = time.perf_counter()
    offset = min(msg.timestamp for msg in msgs) - 1

    async def benchmark_timed(msg: BenchmarkTrace):
        """Execute benchmark timed for the object."""
        target = start + msg.timestamp - offset
        await asyncio.sleep(max(0, target - time.perf_counter()))
        return await benchmark_one(
            client,
            msg.message,
            msg.output_length,
            model,
            pbar=pbar,
            input_length=msg.input_length,
        )

    tasks = [benchmark_timed(msg) for msg in msgs]
    with pbar.log_stats():
        return await asyncio.gather(*tasks)


def _count_generated_tokens(result: RawResult, tokenizer: Any = UNSET) -> int:
    """Count generated output tokens without persisting generated text."""
    if not isinstance(tokenizer, Unset) and result.generated_text:
        return len(tokenizer.encode(result.generated_text, add_special_tokens=False))
    if result.actual_output_len is not None:
        return result.actual_output_len
    return max(0, len(result.tics) - 2)


def _summarize_online_results(
    raw_data: List[RawResult],
    tokenizer: Any = UNSET,
) -> Dict[str, Any]:
    """Aggregate streaming benchmark timings into a serializable summary."""
    if not raw_data:
        raise ValueError("No benchmark results to process")

    ttft_times: List[float] = []
    tpot_times: List[float] = []
    e2e_times: List[float] = []
    input_lengths: List[int] = []
    output_budgets: List[int] = []
    actual_output_lengths: List[int] = []

    min_time = min(min(r.tics) for r in raw_data)
    max_time = max(max(r.tics) for r in raw_data)
    duration_seconds = max_time - min_time
    if duration_seconds <= 0:
        raise ValueError("Benchmark duration must be positive")

    for result in raw_data:
        tics = result.tics
        actual_output_len = _count_generated_tokens(result, tokenizer)

        output_budgets.append(result.output_len)
        actual_output_lengths.append(actual_output_len)
        if result.input_len is not None:
            input_lengths.append(result.input_len)

        token_tics = result.content_tics or tics[1 : 1 + actual_output_len]
        if token_tics:
            ttft_times.append(token_tics[0] - tics[0])
        for i in range(len(token_tics) - 1):
            tpot_times.append(token_tics[i + 1] - token_tics[i])
        e2e_times.append(tics[-1] - tics[0])

    total_output_tokens = sum(actual_output_lengths)
    total_output_budget = sum(output_budgets)
    num_requests = len(raw_data)

    return {
        "requests": num_requests,
        "elapsed_seconds": duration_seconds,
        "request_throughput_per_s": num_requests / duration_seconds,
        "actual_output_tokens": total_output_tokens,
        "output_budget_tokens": total_output_budget,
        "output_tok_per_s": total_output_tokens / duration_seconds,
        "input_tokens": sum(input_lengths)
        if len(input_lengths) == num_requests
        else None,
        "input_length": length_stats(input_lengths),
        "output_length": length_stats(actual_output_lengths),
        "ttft_ms": _time_stats(ttft_times, scale=1000),
        "tpot_ms": _time_stats(tpot_times, scale=1000),
        "e2e_seconds": _time_stats(e2e_times),
    }


def _time_stats(
    times: List[float], scale: float = 1.0
) -> Dict[str, float | int | None]:
    """Return avg and percentile stats for timing values."""
    if not times:
        return {
            "count": 0,
            "avg": None,
            "p50": None,
            "p90": None,
            "p99": None,
            "max": None,
        }

    arr = sorted(times)
    n = len(arr)

    def percentile(q: float) -> float:
        return scale * arr[min(int(n * q), n - 1)]

    return {
        "count": n,
        "avg": scale * sum(arr) / n,
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "max": scale * arr[-1],
    }


def process_benchmark_results(
    raw_data: List[RawResult],
    tokenizer: Any = UNSET,
) -> Dict[str, Any]:
    """Aggregate raw benchmark timings into summary metrics."""
    summary = _summarize_online_results(raw_data, tokenizer)
    logger.info(
        "Num requests: #%s, Actual output tokens: #%s, Output budget tokens: #%s",
        summary["requests"],
        summary["actual_output_tokens"],
        summary["output_budget_tokens"],
    )
    logger.info(
        "TTFT avg/p50/p90/p99/max: %s/%s/%s/%s/%s ms",
        _format_summary_value(summary["ttft_ms"]["avg"]),
        _format_summary_value(summary["ttft_ms"]["p50"]),
        _format_summary_value(summary["ttft_ms"]["p90"]),
        _format_summary_value(summary["ttft_ms"]["p99"]),
        _format_summary_value(summary["ttft_ms"]["max"]),
    )
    logger.info(
        "TPOT avg/p50/p90/p99/max: %s/%s/%s/%s/%s ms",
        _format_summary_value(summary["tpot_ms"]["avg"]),
        _format_summary_value(summary["tpot_ms"]["p50"]),
        _format_summary_value(summary["tpot_ms"]["p90"]),
        _format_summary_value(summary["tpot_ms"]["p99"]),
        _format_summary_value(summary["tpot_ms"]["max"]),
    )
    logger.info(
        "Throughput: %s token/s, %s req/s",
        _format_summary_value(summary["output_tok_per_s"]),
        _format_summary_value(summary["request_throughput_per_s"]),
    )

    return summary


def read_qwen_trace(
    file_path: str,
    tokenizer: Any,
    n: int | None = None,
    dummy: bool = False,
) -> List[BenchmarkTrace]:
    """Load a Qwen benchmark trace file."""
    from pydantic import BaseModel

    class JSONInput(BaseModel):
        chat_id: int
        parent_chat_id: int
        timestamp: float
        input_length: int
        output_length: int
        type: str  # unused
        turn: int  # unused
        hash_ids: List[int]  # unused

    with open(file_path, "r") as f:
        lines = f.readlines()
        if n is not None:
            lines = lines[:n]
    objs = [JSONInput.model_validate_json(line) for line in lines]
    if dummy:
        prompt = generate_prompt(tokenizer, max(obj.input_length for obj in objs))
        ids = tokenizer.encode(prompt, add_special_tokens=False)

        def _get_prompt(obj: JSONInput) -> str:
            return tokenizer.decode(ids[: obj.input_length])

    else:

        def _get_prompt(obj: JSONInput) -> str:
            return generate_prompt(tokenizer, obj.input_length)

    return [
        BenchmarkTrace(
            timestamp=obj.timestamp,
            message=_get_prompt(obj),
            input_length=obj.input_length,
            output_length=obj.output_length,
        )
        for obj in objs
    ]


def read_mooncake_trace(
    file_path: str,
    tokenizer: Any,
    n: int | None = None,
    dummy: bool = False,
) -> List[BenchmarkTrace]:
    """Load a Mooncake benchmark trace file."""
    from pydantic import BaseModel

    class JSONInput(BaseModel):
        timestamp: int
        input_length: int
        output_length: int
        hash_ids: List[int]  # unused for now

    with open(file_path, "r") as f:
        lines = f.readlines()
        if n is not None:
            lines = lines[:n]
    objs = [JSONInput.model_validate_json(line) for line in lines]
    if dummy:
        prompt = generate_prompt(tokenizer, max(obj.input_length for obj in objs))
        ids = tokenizer.encode(prompt, add_special_tokens=False)

        def _get_prompt(obj: JSONInput) -> str:
            return tokenizer.decode(ids[: obj.input_length])

    else:

        def _get_prompt(obj: JSONInput) -> str:
            return generate_prompt(tokenizer, obj.input_length)

    return [
        BenchmarkTrace(
            timestamp=obj.timestamp / 1000,
            message=_get_prompt(obj),
            input_length=obj.input_length,
            output_length=obj.output_length,
        )
        for obj in objs
    ]


def scale_traces(
    traces: List[BenchmarkTrace],
    scale: float,
) -> List[BenchmarkTrace]:
    """Rescale trace timestamps by the requested factor."""
    min_tic = min(trace.timestamp for trace in traces)
    return sorted(
        [
            BenchmarkTrace(
                timestamp=(trace.timestamp - min_tic) * scale,
                message=trace.message,
                input_length=trace.input_length,
                output_length=trace.output_length,
            )
            for trace in traces
        ],
        key=lambda x: x.timestamp,
    )


async def get_model_name(client: OpenAI) -> str:
    """Return model name."""
    async for model in client.models.list():
        return model.id
    raise ValueError("No models available")

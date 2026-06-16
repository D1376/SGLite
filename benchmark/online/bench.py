"""Unified online benchmark runner with managed backend server startup."""

from __future__ import annotations

import argparse
import asyncio
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Sequence, TextIO

if TYPE_CHECKING:
    from openai import AsyncOpenAI as OpenAI

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from common import (  # noqa: E402
    benchmark_one,
    benchmark_one_batch,
    generate_online_text_workload,
    generate_prompt,
    get_model_name,
    init_logger,
    print_summary_table,
    process_benchmark_results,
    save_benchmark_result,
    save_benchmark_suite_result,
)

logger = init_logger(__name__)
RESULTS_DIR = BENCHMARK_ROOT / "results"

DEFAULT_MODEL = "Qwen/Qwen3-32B"
DEFAULT_BACKENDS = ("sglite", "sglang", "vllm")
DEFAULT_TP_SIZES = (1, 2)
DEFAULT_BATCH_SIZES = (256,)
DEFAULT_BATCH_TIMEOUT_SECONDS = 7200.0
DEFAULT_SERVER_CONTEXT_OVERHEAD = 256
DEFAULT_PORTS = {
    "sglite": 1376,
    "sglang": 30000,
    "vllm": 8000,
}
POLL_INTERVAL_SECONDS = 5.0
WAIT_LOG_INTERVAL_SECONDS = 30.0


@dataclass(frozen=True)
class ServerHandle:
    """Tracks one managed backend server process and its log file."""

    process: subprocess.Popen
    log_path: Path
    log_file: TextIO


def _parse_args(default_backends: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model to serve and benchmark.",
    )
    parser.add_argument(
        "--backend",
        action="append",
        choices=DEFAULT_BACKENDS,
        dest="backends",
        help=(
            "Backend to benchmark. Can be passed multiple times. "
            f"Default: {', '.join(default_backends)}."
        ),
    )
    parser.add_argument(
        "--tp-size",
        action="append",
        type=int,
        dest="tp_sizes",
        help="Tensor parallel size to benchmark. Can be passed multiple times.",
    )
    parser.add_argument(
        "--batch-size",
        action="append",
        type=int,
        dest="batch_sizes",
        help="Concurrent request count to benchmark. Can be passed multiple times.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-input-len", type=int, default=8192)
    parser.add_argument("--min-output-len", type=int, default=16)
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument(
        "--server-max-model-len",
        type=int,
        default=None,
        help=(
            "Maximum context length passed to managed servers. Defaults to "
            "--max-input-len + --max-output-len + --server-context-overhead."
        ),
    )
    parser.add_argument(
        "--server-context-overhead",
        type=int,
        default=DEFAULT_SERVER_CONTEXT_OVERHEAD,
        help="Extra context tokens reserved for chat templates and protocol overhead.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server bind host.")
    parser.add_argument(
        "--api-key",
        default="",
        help="OpenAI-compatible API key value to send to the local server.",
    )
    parser.add_argument(
        "--sglite-port",
        type=int,
        default=DEFAULT_PORTS["sglite"],
        help="Port used for managed SGLite server runs.",
    )
    parser.add_argument(
        "--sglang-port",
        type=int,
        default=DEFAULT_PORTS["sglang"],
        help="Port used for managed SGLang server runs.",
    )
    parser.add_argument(
        "--vllm-port",
        type=int,
        default=DEFAULT_PORTS["vllm"],
        help="Port used for managed vLLM server runs.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=1800.0,
        help="Seconds to wait for each managed server to become ready.",
    )
    parser.add_argument(
        "--health-check-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for one /v1/models readiness probe.",
    )
    parser.add_argument(
        "--probe-timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for the short generation probe after readiness.",
    )
    parser.add_argument(
        "--batch-timeout",
        type=float,
        default=DEFAULT_BATCH_TIMEOUT_SECONDS,
        help="Seconds to wait for one online request batch. Use 0 to disable.",
    )
    parser.add_argument(
        "--shutdown-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for graceful server shutdown before killing it.",
    )
    parser.add_argument(
        "--server-log-dir",
        type=Path,
        default=RESULTS_DIR / "logs",
        help="Directory for managed backend server logs.",
    )
    parser.add_argument(
        "--extra-sglite-arg",
        action="append",
        default=[],
        help="Extra SGLite server argument token. Repeat for multiple tokens.",
    )
    parser.add_argument(
        "--extra-sglang-arg",
        action="append",
        default=[],
        help="Extra SGLang server argument token. Repeat for multiple tokens.",
    )
    parser.add_argument(
        "--extra-vllm-arg",
        action="append",
        default=[],
        help="Extra vLLM server argument token. Repeat for multiple tokens.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="progress",
        default=True,
        help="Disable progress bars during request benchmarks.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue the suite after one backend/tp run fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the managed server commands without launching them.",
    )

    args = parser.parse_args()
    args.backends = tuple(args.backends or default_backends)
    args.tp_sizes = tuple(args.tp_sizes or DEFAULT_TP_SIZES)
    args.batch_sizes = tuple(args.batch_sizes or DEFAULT_BATCH_SIZES)
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if not args.backends:
        raise ValueError("At least one backend is required")
    if any(tp_size <= 0 for tp_size in args.tp_sizes):
        raise ValueError("--tp-size values must be positive")
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        raise ValueError("--batch-size values must be positive")
    if args.max_input_len <= 0:
        raise ValueError("--max-input-len must be positive")
    if args.min_output_len <= 0:
        raise ValueError("--min-output-len must be positive")
    if args.max_output_len < args.min_output_len:
        raise ValueError("--max-output-len must be >= --min-output-len")
    if args.server_context_overhead < 0:
        raise ValueError("--server-context-overhead must be >= 0")
    if args.server_max_model_len is not None and args.server_max_model_len <= 0:
        raise ValueError("--server-max-model-len must be positive")
    if (
        args.server_max_model_len is not None
        and args.server_max_model_len < args.max_input_len + args.max_output_len
    ):
        raise ValueError(
            "--server-max-model-len must be >= --max-input-len + --max-output-len"
        )
    if args.startup_timeout <= 0:
        raise ValueError("--startup-timeout must be positive")
    if args.health_check_timeout <= 0:
        raise ValueError("--health-check-timeout must be positive")
    if args.probe_timeout <= 0:
        raise ValueError("--probe-timeout must be positive")
    if args.batch_timeout < 0:
        raise ValueError("--batch-timeout must be >= 0")
    if args.shutdown_timeout <= 0:
        raise ValueError("--shutdown-timeout must be positive")


def _port_for(args: argparse.Namespace, backend: str) -> int:
    return {
        "sglite": args.sglite_port,
        "sglang": args.sglang_port,
        "vllm": args.vllm_port,
    }[backend]


def _extra_args_for(args: argparse.Namespace, backend: str) -> list[str]:
    return {
        "sglite": args.extra_sglite_arg,
        "sglang": args.extra_sglang_arg,
        "vllm": args.extra_vllm_arg,
    }[backend]


def _max_batch_size(args: argparse.Namespace) -> int:
    return max(args.batch_sizes)


def _server_max_model_len(args: argparse.Namespace) -> int:
    return (
        args.server_max_model_len
        or args.max_input_len + args.max_output_len + args.server_context_overhead
    )


def _connect_host(host: str) -> str:
    if host in {"0.0.0.0", "::", ""}:
        return "127.0.0.1"
    return host


def _is_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((_connect_host(host), port), timeout=1.0):
            return True
    except OSError:
        return False


def _ensure_port_free(host: str, port: int, *, backend: str) -> None:
    if _is_port_open(host, port):
        raise RuntimeError(
            f"Port {host}:{port} is already accepting connections before starting "
            f"{backend}. Stop the existing server or choose another port."
        )


async def _wait_for_port_closed(host: str, port: int, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _is_port_open(host, port):
            return
        await asyncio.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for port {host}:{port} to close")


def _server_command(
    args: argparse.Namespace,
    *,
    backend: str,
    tp_size: int,
    port: int,
) -> list[str]:
    if backend == "sglite":
        command = [
            sys.executable,
            "-m",
            "sglite",
            "--model",
            args.model,
            "--host",
            args.host,
            "--port",
            str(port),
            "--tp-size",
            str(tp_size),
            "--max-seq-len-override",
            str(_server_max_model_len(args)),
            "--max-running-requests",
            str(_max_batch_size(args)),
        ]
    elif backend == "sglang":
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            args.model,
            "--host",
            args.host,
            "--port",
            str(port),
            "--tp-size",
            str(tp_size),
            "--context-length",
            str(_server_max_model_len(args)),
            "--max-running-requests",
            str(_max_batch_size(args)),
        ]
    elif backend == "vllm":
        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            args.model,
            "--host",
            args.host,
            "--port",
            str(port),
            "--tensor-parallel-size",
            str(tp_size),
            "--max-model-len",
            str(_server_max_model_len(args)),
            "--max-num-seqs",
            str(_max_batch_size(args)),
        ]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    command.extend(_extra_args_for(args, backend))
    return command


def _base_url(args: argparse.Namespace, port: int) -> str:
    return f"http://{args.host}:{port}/v1"


def _new_server_log_path(log_dir: Path, backend: str, tp_size: int) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return log_dir / f"{stamp}-online-{backend}-tp{tp_size}.log"


def _start_server(
    *,
    command: list[str],
    log_dir: Path,
    backend: str,
    tp_size: int,
) -> ServerHandle:
    log_path = _new_server_log_path(log_dir, backend, tp_size)
    log_file = log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except Exception:
        log_file.close()
        raise
    return ServerHandle(process=process, log_path=log_path, log_file=log_file)


def _signal_process_group(process: subprocess.Popen, sig: signal.Signals) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        return
    except AttributeError:
        if sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()


def _stop_server(handle: ServerHandle, *, timeout: float) -> None:
    process = handle.process
    try:
        if process.poll() is not None:
            _signal_process_group(process, signal.SIGTERM)
        else:
            _signal_process_group(process, signal.SIGTERM)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _signal_process_group(process, signal.SIGKILL)
                process.wait()
    finally:
        handle.log_file.close()


async def _wait_for_server(
    *,
    client: OpenAI,
    handle: ServerHandle,
    backend: str,
    base_url: str,
    timeout: float,
    health_check_timeout: float,
) -> str:
    start = time.monotonic()
    deadline = time.monotonic() + timeout
    next_log_at = start
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        return_code = handle.process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"{backend} server exited with code {return_code} before it was "
                f"ready. See log: {handle.log_path}"
            )
        try:
            served_model = await asyncio.wait_for(
                get_model_name(client),
                timeout=health_check_timeout,
            )
            logger.info(
                "%s server ready at %s with model %s",
                backend,
                base_url,
                served_model,
            )
            return served_model
        except Exception as e:
            last_error = e
            now = time.monotonic()
            if now >= next_log_at:
                logger.info(
                    "Waiting for %s server readiness: elapsed=%.0fs timeout=%.0fs "
                    "health_timeout=%.0fs log=%s last_error=%r",
                    backend,
                    now - start,
                    timeout,
                    health_check_timeout,
                    handle.log_path,
                    last_error,
                )
                next_log_at = now + WAIT_LOG_INTERVAL_SECONDS
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

    raise TimeoutError(
        f"Timed out waiting for {backend} server at {base_url}. "
        f"Last error: {last_error!r}. See log: {handle.log_path}"
    )


async def _probe_generation(
    *,
    client: OpenAI,
    prompt: str,
    served_model: str,
    backend: str,
    timeout: float,
) -> None:
    result = await asyncio.wait_for(
        benchmark_one(
            client,
            prompt,
            2,
            served_model,
            pbar=False,
            input_length=100,
        ),
        timeout=timeout,
    )
    if not result.content_tics:
        raise RuntimeError(f"{backend} server did not stream generated content")
    logger.info("%s generation probe succeeded", backend)


async def _run_batch(
    *,
    args: argparse.Namespace,
    client: OpenAI,
    tokenizer,
    backend: str,
    tp_size: int,
    port: int,
    base_url: str,
    command: list[str],
    log_path: Path,
    served_model: str,
    prompts: list[str],
    input_lengths: list[int],
    output_lengths: list[int],
    batch_size: int,
) -> dict:
    batch_task = benchmark_one_batch(
        client,
        prompts[:batch_size],
        output_lengths[:batch_size],
        served_model,
        input_lengths=input_lengths[:batch_size],
        pbar=args.progress,
    )
    if args.batch_timeout > 0:
        results = await asyncio.wait_for(batch_task, timeout=args.batch_timeout)
    else:
        results = await batch_task
    summary = process_benchmark_results(results, tokenizer=tokenizer)
    config = {
        "backend": backend,
        "model": args.model,
        "served_model": served_model,
        "tp_size": tp_size,
        "seed": args.seed,
        "requests": batch_size,
        "batch_size": batch_size,
        "port": port,
        "base_url": base_url,
        "server_command": command,
        "server_log": str(log_path),
        "server_max_model_len": _server_max_model_len(args),
        "max_running_requests": _max_batch_size(args),
        "batch_timeout": args.batch_timeout,
        "streaming": True,
        "max_input_len": args.max_input_len,
        "min_output_len": args.min_output_len,
        "max_output_len": args.max_output_len,
        "temperature": 0.0,
        "ignore_eos": True,
        "top_k": 1,
    }
    title = (
        f"{backend} Online Streaming Benchmark "
        f"({args.model}, tp_size={tp_size}, batch_size={batch_size})"
    )
    print_summary_table(title, summary)
    save_benchmark_result(
        f"online_{backend}_tp{tp_size}_bs{batch_size}",
        config,
        summary,
        RESULTS_DIR,
    )
    return {"config": config, "summary": summary}


async def _run_backend_tp(
    *,
    args: argparse.Namespace,
    backend: str,
    tp_size: int,
    tokenizer,
    probe_prompt: str,
    prompts: list[str],
    input_lengths: list[int],
    output_lengths: list[int],
) -> list[dict]:
    from openai import AsyncOpenAI as OpenAI

    port = _port_for(args, backend)
    base_url = _base_url(args, port)
    command = _server_command(args, backend=backend, tp_size=tp_size, port=port)
    _ensure_port_free(args.host, port, backend=backend)

    logger.info(
        "Starting %s online benchmark server: tp_size=%d command=%s",
        backend,
        tp_size,
        shlex.join(command),
    )
    handle = _start_server(
        command=command,
        log_dir=args.server_log_dir,
        backend=backend,
        tp_size=tp_size,
    )
    logger.info("%s server log: %s", backend, handle.log_path)

    try:
        async with OpenAI(base_url=base_url, api_key=args.api_key) as client:
            served_model = await _wait_for_server(
                client=client,
                handle=handle,
                backend=backend,
                base_url=base_url,
                timeout=args.startup_timeout,
                health_check_timeout=args.health_check_timeout,
            )
            await _probe_generation(
                client=client,
                prompt=probe_prompt,
                served_model=served_model,
                backend=backend,
                timeout=args.probe_timeout,
            )

            records = []
            for batch_size in args.batch_sizes:
                logger.info(
                    "Running %s benchmark: model=%s tp_size=%d batch_size=%d",
                    backend,
                    args.model,
                    tp_size,
                    batch_size,
                )
                records.append(
                    await _run_batch(
                        args=args,
                        client=client,
                        tokenizer=tokenizer,
                        backend=backend,
                        tp_size=tp_size,
                        port=port,
                        base_url=base_url,
                        command=command,
                        log_path=handle.log_path,
                        served_model=served_model,
                        prompts=prompts,
                        input_lengths=input_lengths,
                        output_lengths=output_lengths,
                        batch_size=batch_size,
                    )
                )
            return records
    finally:
        logger.info("Stopping %s server for tp_size=%d", backend, tp_size)
        _stop_server(handle, timeout=args.shutdown_timeout)
        try:
            await _wait_for_port_closed(args.host, port, timeout=args.shutdown_timeout)
        except TimeoutError:
            logger.warning(
                "Port %s:%d is still accepting connections after stopping %s",
                args.host,
                port,
                backend,
            )


async def _run_suite(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    logger.info("Loading tokenizer for %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    probe_prompt = generate_prompt(tokenizer, 100)
    max_batch_size = _max_batch_size(args)
    logger.info(
        "Generating shared online workload: requests=%d max_input_len=%d "
        "output_len=[%d,%d] seed=%d",
        max_batch_size,
        args.max_input_len,
        args.min_output_len,
        args.max_output_len,
        args.seed,
    )
    prompts, input_lengths, output_lengths = generate_online_text_workload(
        tokenizer,
        seed=args.seed,
        num_requests=max_batch_size,
        max_input_len=args.max_input_len,
        min_output_len=args.min_output_len,
        max_output_len=args.max_output_len,
    )
    logger.info(
        "Generated one shared online workload: requests=%d model=%s seed=%d",
        len(prompts),
        args.model,
        args.seed,
    )

    suite_records: list[dict] = []
    for backend in args.backends:
        for tp_size in args.tp_sizes:
            try:
                suite_records.extend(
                    await _run_backend_tp(
                        args=args,
                        backend=backend,
                        tp_size=tp_size,
                        tokenizer=tokenizer,
                        probe_prompt=probe_prompt,
                        prompts=prompts,
                        input_lengths=input_lengths,
                        output_lengths=output_lengths,
                    )
                )
            except Exception as e:
                logger.exception(
                    "Online benchmark failed: backend=%s tp_size=%d",
                    backend,
                    tp_size,
                )
                if not args.keep_going:
                    raise
                suite_records.append(
                    {
                        "config": {
                            "backend": backend,
                            "model": args.model,
                            "tp_size": tp_size,
                            "port": _port_for(args, backend),
                            "base_url": _base_url(args, _port_for(args, backend)),
                            "server_max_model_len": _server_max_model_len(args),
                            "max_running_requests": _max_batch_size(args),
                            "batch_timeout": args.batch_timeout,
                            "server_command": _server_command(
                                args,
                                backend=backend,
                                tp_size=tp_size,
                                port=_port_for(args, backend),
                            ),
                        },
                        "error": str(e),
                    }
                )

    save_benchmark_suite_result("online_streaming_suite", suite_records, RESULTS_DIR)


def _print_dry_run(args: argparse.Namespace) -> None:
    print("Managed online benchmark commands:")
    for backend in args.backends:
        for tp_size in args.tp_sizes:
            port = _port_for(args, backend)
            command = _server_command(args, backend=backend, tp_size=tp_size, port=port)
            print(
                f"{backend} tp_size={tp_size} port={port}: "
                f"{shlex.join(command)}"
            )


def main(default_backends: Sequence[str] = DEFAULT_BACKENDS) -> None:
    try:
        args = _parse_args(default_backends)
        if args.dry_run:
            _print_dry_run(args)
            return
        asyncio.run(_run_suite(args))
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"Error in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

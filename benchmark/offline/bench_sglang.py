"""Offline benchmark runner for SGLang on the shared synthetic workload."""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from common import (  # noqa: E402
    DEFAULT_OFFLINE_MODELS,
    build_generation_summary,
    clear_benchmark_cache,
    generate_synthetic_token_workload,
    print_summary_table,
    save_benchmark_result,
    save_benchmark_suite_result,
)

RESULTS_DIR = BENCHMARK_ROOT / "results"
DEFAULT_TP_SIZES = (1, 2)
DEFAULT_DISABLE_PIECEWISE_CUDA_GRAPH = True


def _create_engine(
    sgl: Any,
    model: str,
    context_length: int,
    tp_size: int,
    disable_piecewise_cuda_graph: bool,
):
    """Create an SGLang engine across versions with slightly different kwargs."""
    base_kwargs = {
        "model_path": model,
        "log_level": "error",
        "disable_piecewise_cuda_graph": disable_piecewise_cuda_graph,
    }
    legacy_kwargs = {"model_path": model, "log_level": "error"}
    attempts = [
        dict(base_kwargs, tp_size=tp_size, context_length=context_length),
        dict(base_kwargs, tp_size=tp_size),
        dict(base_kwargs, tensor_parallel_size=tp_size, context_length=context_length),
        dict(base_kwargs, tensor_parallel_size=tp_size),
        dict(legacy_kwargs, tp_size=tp_size, context_length=context_length),
        dict(legacy_kwargs, tp_size=tp_size),
        dict(
            legacy_kwargs, tensor_parallel_size=tp_size, context_length=context_length
        ),
        dict(legacy_kwargs, tensor_parallel_size=tp_size),
    ]
    if tp_size == 1:
        attempts.extend(
            [
                dict(base_kwargs, context_length=context_length),
                dict(base_kwargs),
                dict(legacy_kwargs, context_length=context_length),
                dict(legacy_kwargs),
            ]
        )

    last_error = None
    for kwargs in attempts:
        try:
            return sgl.Engine(**kwargs)
        except TypeError as e:
            last_error = e
    assert last_error is not None
    raise last_error


def _completion_lens(outputs: Any) -> list[int]:
    """Extract SGLang completion token lengths from common output shapes."""
    if isinstance(outputs, list):
        return [_completion_lens(output)[0] for output in outputs]

    if not isinstance(outputs, dict):
        raise RuntimeError(f"Unexpected SGLang output type: {type(outputs)!r}")

    meta_info = outputs.get("meta_info")
    if isinstance(meta_info, dict):
        completion_tokens = meta_info.get("completion_tokens")
        if isinstance(completion_tokens, list):
            return [int(x) for x in completion_tokens]
        if isinstance(completion_tokens, int):
            return [completion_tokens]
    elif isinstance(meta_info, list):
        return [int(meta["completion_tokens"]) for meta in meta_info]

    output_ids = outputs.get("output_ids")
    if isinstance(output_ids, list):
        if not output_ids:
            return [0]
        if isinstance(output_ids[0], list):
            return [len(ids) for ids in output_ids]
        return [len(output_ids)]

    raise RuntimeError("SGLang output did not include completion token metadata")


def run_model(
    *,
    sgl: Any,
    model: str,
    workload,
    sampling_params,
    seed: int,
    num_seqs: int,
    min_input_len: int,
    max_input_len: int,
    min_output_len: int,
    max_output_len: int,
    ignore_eos: bool,
    temperature: float,
    context_length: int,
    tp_size: int,
    disable_piecewise_cuda_graph: bool,
) -> dict[str, Any]:
    print(f"\nRunning SGLang offline benchmark for {model} (tp_size={tp_size})")
    engine = _create_engine(
        sgl, model, context_length, tp_size, disable_piecewise_cuda_graph
    )
    try:
        engine.generate(
            prompt="Benchmark: ",
            sampling_params={"temperature": 0.1, "max_new_tokens": 16},
        )

        start = time.perf_counter()
        bench_results = engine.generate(
            input_ids=workload.prompt_token_ids,
            sampling_params=sampling_params,
        )
        elapsed_seconds = time.perf_counter() - start
    finally:
        shutdown = getattr(engine, "shutdown", None)
        if shutdown is not None:
            shutdown()
        clear_benchmark_cache()

    output_lens = _completion_lens(bench_results)
    summary = build_generation_summary(
        input_lengths=workload.input_lengths,
        output_lengths=output_lens,
        output_budget_lengths=workload.output_lengths,
        elapsed_seconds=elapsed_seconds,
    )
    config = {
        "backend": "sglang",
        "model": model,
        "seed": seed,
        "requests": num_seqs,
        "min_input_len": min_input_len,
        "max_input_len": max_input_len,
        "min_output_len": min_output_len,
        "max_output_len": max_output_len,
        "ignore_eos": ignore_eos,
        "temperature": temperature,
        "context_length": context_length,
        "tp_size": tp_size,
        "disable_piecewise_cuda_graph": disable_piecewise_cuda_graph,
    }
    print_summary_table(
        f"SGLang Offline Synthetic Benchmark ({model}, tp_size={tp_size})", summary
    )
    return {"config": config, "summary": summary}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model to benchmark. Can be passed multiple times.",
    )
    parser.add_argument(
        "--tp-size",
        action="append",
        dest="tp_sizes",
        type=int,
        help="Tensor parallel size to benchmark. Can be passed multiple times.",
    )
    parser.add_argument(
        "--enable-piecewise-cuda-graph",
        action="store_false",
        dest="disable_piecewise_cuda_graph",
        default=DEFAULT_DISABLE_PIECEWISE_CUDA_GRAPH,
        help="Opt back into SGLang piecewise CUDA graph capture.",
    )
    parser.add_argument("--run-single", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--result-path", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def _run_single(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import sglang as sgl
    except ImportError as e:
        raise SystemExit(
            "SGLang is not installed. Install it in the benchmark environment."
        ) from e

    seed = 0
    models = tuple(args.models or DEFAULT_OFFLINE_MODELS)
    tp_sizes = tuple(args.tp_sizes or DEFAULT_TP_SIZES)
    num_seqs = 256
    min_input_len = 100
    max_input_len = 1024
    min_output_len = 100
    max_output_len = 1024
    ignore_eos = True
    temperature = 0.6
    context_length = 4096

    workload = generate_synthetic_token_workload(
        seed=seed,
        num_seqs=num_seqs,
        min_input_len=min_input_len,
        max_input_len=max_input_len,
        min_output_len=min_output_len,
        max_output_len=max_output_len,
    )
    sampling_params = [
        {
            "temperature": temperature,
            "ignore_eos": ignore_eos,
            "max_new_tokens": output_len,
        }
        for output_len in workload.output_lengths
    ]

    if len(models) != 1 or len(tp_sizes) != 1:
        raise ValueError("--run-single expects exactly one --model and one --tp-size")

    return run_model(
        sgl=sgl,
        model=models[0],
        workload=workload,
        sampling_params=sampling_params,
        seed=seed,
        num_seqs=num_seqs,
        min_input_len=min_input_len,
        max_input_len=max_input_len,
        min_output_len=min_output_len,
        max_output_len=max_output_len,
        ignore_eos=ignore_eos,
        temperature=temperature,
        context_length=context_length,
        tp_size=tp_sizes[0],
        disable_piecewise_cuda_graph=args.disable_piecewise_cuda_graph,
    )


def _run_one_subprocess(
    args: argparse.Namespace, model: str, tp_size: int, result_path: Path
) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-single",
        "--model",
        model,
        "--tp-size",
        str(tp_size),
        "--result-path",
        str(result_path),
    ]
    if not args.disable_piecewise_cuda_graph:
        cmd.append("--enable-piecewise-cuda-graph")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = _parse_args()
    if args.run_single:
        result = _run_single(args)
        if args.result_path is not None:
            args.result_path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n"
            )
        else:
            save_benchmark_result(
                "offline_sglang", result["config"], result["summary"], RESULTS_DIR
            )
        return

    models = tuple(args.models or DEFAULT_OFFLINE_MODELS)
    tp_sizes = tuple(args.tp_sizes or DEFAULT_TP_SIZES)
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for model in models:
            for tp_size in tp_sizes:
                result_path = tmp_path / f"offline-sglang-{len(results)}.json"
                _run_one_subprocess(args, model, tp_size, result_path)
                results.append(json.loads(result_path.read_text()))

    save_benchmark_suite_result("offline_sglang", results, RESULTS_DIR)


if __name__ == "__main__":
    main()

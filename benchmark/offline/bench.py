"""Offline benchmark runner for synthetic prompt workloads."""

# Adapted from: https://github.com/GeeeekExplorer/nano-vllm/blob/main/bench.py

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

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


def run_model(
    *,
    model: str,
    workload,
    seed: int,
    num_seqs: int,
    min_input_len: int,
    max_input_len: int,
    min_output_len: int,
    max_output_len: int,
    ignore_eos: bool,
    temperature: float,
    max_seq_len_override: int,
    max_extend_tokens: int,
    cuda_graph_max_bs: int,
    page_size: int,
) -> dict:
    from sglite.llm import LLM
    from sglite.sampling_params import SamplingParams

    sampling_params = [
        SamplingParams(
            temperature=temperature,
            ignore_eos=ignore_eos,
            max_tokens=output_len,
        )
        for output_len in workload.output_lengths
    ]

    print(f"\nRunning SGLite offline benchmark for {model}")
    llm = LLM(
        model,
        max_seq_len_override=max_seq_len_override,
        max_extend_tokens=max_extend_tokens,
        cuda_graph_max_bs=cuda_graph_max_bs,
        page_size=page_size,
    )
    try:
        llm.generate(["Benchmark: "], SamplingParams(temperature=0.1))

        start = time.perf_counter()
        bench_results = llm.generate(workload.prompt_token_ids, sampling_params)
        elapsed_seconds = time.perf_counter() - start
    finally:
        del llm
        clear_benchmark_cache()

    output_lens = [len(result["token_ids"]) for result in bench_results]
    summary = build_generation_summary(
        input_lengths=workload.input_lengths,
        output_lengths=output_lens,
        output_budget_lengths=workload.output_lengths,
        elapsed_seconds=elapsed_seconds,
    )
    config = {
        "backend": "sglite",
        "model": model,
        "seed": seed,
        "requests": num_seqs,
        "min_input_len": min_input_len,
        "max_input_len": max_input_len,
        "min_output_len": min_output_len,
        "max_output_len": max_output_len,
        "ignore_eos": ignore_eos,
        "temperature": temperature,
        "max_seq_len_override": max_seq_len_override,
        "max_extend_tokens": max_extend_tokens,
        "cuda_graph_max_bs": cuda_graph_max_bs,
        "page_size": page_size,
    }
    print_summary_table(f"Offline Synthetic Benchmark ({model})", summary)
    return {"config": config, "summary": summary}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model to benchmark. Can be passed multiple times.",
    )
    parser.add_argument("--run-single", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--result-path", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def _run_single(args: argparse.Namespace) -> dict:
    seed = 0
    models = tuple(args.models or DEFAULT_OFFLINE_MODELS)
    num_seqs = 256
    min_input_len = 100
    max_input_len = 1024
    min_output_len = 100
    max_output_len = 1024
    ignore_eos = True
    temperature = 0.6
    max_seq_len_override = 4096
    max_extend_tokens = 16384
    cuda_graph_max_bs = 256
    page_size = 256

    workload = generate_synthetic_token_workload(
        seed=seed,
        num_seqs=num_seqs,
        min_input_len=min_input_len,
        max_input_len=max_input_len,
        min_output_len=min_output_len,
        max_output_len=max_output_len,
    )

    if len(models) != 1:
        raise ValueError("--run-single expects exactly one --model")

    return run_model(
        model=models[0],
        workload=workload,
        seed=seed,
        num_seqs=num_seqs,
        min_input_len=min_input_len,
        max_input_len=max_input_len,
        min_output_len=min_output_len,
        max_output_len=max_output_len,
        ignore_eos=ignore_eos,
        temperature=temperature,
        max_seq_len_override=max_seq_len_override,
        max_extend_tokens=max_extend_tokens,
        cuda_graph_max_bs=cuda_graph_max_bs,
        page_size=page_size,
    )


def _run_one_subprocess(model: str, result_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-single",
            "--model",
            model,
            "--result-path",
            str(result_path),
        ],
        check=True,
    )


def main():
    args = _parse_args()
    if args.run_single:
        result = _run_single(args)
        if args.result_path is not None:
            args.result_path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n"
            )
        else:
            save_benchmark_result(
                "offline_synthetic", result["config"], result["summary"], RESULTS_DIR
            )
        return

    models = tuple(args.models or DEFAULT_OFFLINE_MODELS)
    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for model in models:
            result_path = tmp_path / f"offline-sglite-{len(results)}.json"
            _run_one_subprocess(model, result_path)
            results.append(json.loads(result_path.read_text()))

    save_benchmark_suite_result("offline_synthetic", results, RESULTS_DIR)


if __name__ == "__main__":
    main()

"""Offline benchmark runner for vLLM on the shared synthetic workload."""

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


def _generate_with_token_ids(llm, prompt_token_ids, sampling_params):
    """Run vLLM generation with token IDs across old and new input APIs."""
    try:
        return llm.generate(
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            use_tqdm=True,
        )
    except TypeError:
        prompts = [{"prompt_token_ids": ids} for ids in prompt_token_ids]
        return llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)


def _completion_len(output) -> int:
    completion = output.outputs[0]
    token_ids = getattr(completion, "token_ids", None)
    if token_ids is None:
        raise RuntimeError("vLLM output did not include completion token IDs")
    return len(token_ids)


def run_model(
    *,
    llm_cls,
    sampling_params_cls,
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
    max_model_len: int,
    max_num_seqs: int,
    tp_size: int,
) -> dict[str, Any]:
    print(f"\nRunning vLLM offline benchmark for {model} (tp_size={tp_size})")
    llm = llm_cls(
        model=model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        tensor_parallel_size=tp_size,
    )
    try:
        llm.generate(
            ["Benchmark: "],
            sampling_params_cls(temperature=0.1, max_tokens=16),
            use_tqdm=False,
        )

        start = time.perf_counter()
        bench_results = _generate_with_token_ids(
            llm, workload.prompt_token_ids, sampling_params
        )
        elapsed_seconds = time.perf_counter() - start
    finally:
        del llm
        clear_benchmark_cache()

    output_lens = [_completion_len(result) for result in bench_results]
    summary = build_generation_summary(
        input_lengths=workload.input_lengths,
        output_lengths=output_lens,
        output_budget_lengths=workload.output_lengths,
        elapsed_seconds=elapsed_seconds,
    )
    config = {
        "backend": "vllm",
        "model": model,
        "seed": seed,
        "requests": num_seqs,
        "min_input_len": min_input_len,
        "max_input_len": max_input_len,
        "min_output_len": min_output_len,
        "max_output_len": max_output_len,
        "ignore_eos": ignore_eos,
        "temperature": temperature,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "tp_size": tp_size,
    }
    print_summary_table(
        f"vLLM Offline Synthetic Benchmark ({model}, tp_size={tp_size})", summary
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
    parser.add_argument("--run-single", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--result-path", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def _run_single(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        raise SystemExit(
            "vLLM is not installed. Install it in the benchmark environment."
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
    max_model_len = 4096
    max_num_seqs = 256

    workload = generate_synthetic_token_workload(
        seed=seed,
        num_seqs=num_seqs,
        min_input_len=min_input_len,
        max_input_len=max_input_len,
        min_output_len=min_output_len,
        max_output_len=max_output_len,
    )
    sampling_params = [
        SamplingParams(
            temperature=temperature, ignore_eos=ignore_eos, max_tokens=output_len
        )
        for output_len in workload.output_lengths
    ]

    if len(models) != 1 or len(tp_sizes) != 1:
        raise ValueError("--run-single expects exactly one --model and one --tp-size")

    return run_model(
        llm_cls=LLM,
        sampling_params_cls=SamplingParams,
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
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        tp_size=tp_sizes[0],
    )


def _run_one_subprocess(model: str, tp_size: int, result_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-single",
            "--model",
            model,
            "--tp-size",
            str(tp_size),
            "--result-path",
            str(result_path),
        ],
        check=True,
    )


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
                "offline_vllm", result["config"], result["summary"], RESULTS_DIR
            )
        return

    models = tuple(args.models or DEFAULT_OFFLINE_MODELS)
    tp_sizes = tuple(args.tp_sizes or DEFAULT_TP_SIZES)
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for model in models:
            for tp_size in tp_sizes:
                result_path = tmp_path / f"offline-vllm-{len(results)}.json"
                _run_one_subprocess(model, tp_size, result_path)
                results.append(json.loads(result_path.read_text()))

    save_benchmark_suite_result("offline_vllm", results, RESULTS_DIR)


if __name__ == "__main__":
    main()

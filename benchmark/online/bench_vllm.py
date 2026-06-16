"""Run the managed online benchmark for vLLM only."""

from bench import main


if __name__ == "__main__":
    main(default_backends=("vllm",))

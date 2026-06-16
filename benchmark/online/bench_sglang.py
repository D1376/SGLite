"""Run the managed online benchmark for SGLang only."""

from bench import main


if __name__ == "__main__":
    main(default_backends=("sglang",))

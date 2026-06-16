"""Package entry point for launching SGLite."""

from .srt.entrypoints import launch_server


def main() -> None:
    """Launch SGLite from `python -m sglite`."""
    launch_server()


if __name__ == "__main__":
    main()

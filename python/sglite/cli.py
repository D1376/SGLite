"""CLI helpers for SGLite."""

from .srt.entrypoints import launch_server

if __name__ == "__main__":
    launch_server(run_cli=True)

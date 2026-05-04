"""Environment variable helpers and runtime defaults."""

from __future__ import annotations

import os
from functools import partial
from typing import Callable, Generic, TypeVar


class BaseEnv:
    """Base class for environment-backed settings."""
    def _init(self, name: str) -> None:
        """Read this setting from the named environment variable."""
        raise NotImplementedError


T = TypeVar("T")


class EnvVar(BaseEnv, Generic[T]):
    """Typed environment variable wrapper with best-effort parsing."""
    def __init__(self, default_value: T, fn: Callable[[str], T]):
        """Store the default value and parser for one environment setting."""
        self.value = default_value
        self.fn = fn
        super().__init__()

    def _init(self, name: str) -> None:
        """Apply an environment override when it parses successfully."""
        env_value = os.getenv(name)
        if env_value is not None:
            try:
                self.value = self.fn(env_value)
            except Exception:
                # Keep the default when an override is malformed rather than failing at import time.
                pass

    def __bool__(self):
        """Return the parsed value in boolean contexts."""
        return self.value

    def __str__(self):
        """Return the string form of the env var."""
        return str(self.value)


_TO_BOOL = lambda x: x.lower() in ("1", "true", "yes")


def _PARSE_MEM_BYTES(mem: str) -> int:
    """Parse memory sizes such as `512M`, `1G`, or raw byte counts."""
    mem = mem.strip().upper()
    if not mem[-1].isalpha():
        return int(mem)
    if mem.endswith("B"):
        mem = mem[:-1]
    UNIT_MAP = {"K": 1024, "M": 1024**2, "G": 1024**3}
    return int(float(mem[:-1]) * UNIT_MAP[mem[-1]])


SGLITE_ENV_PREFIX = "SGLITE_"
EnvInt = partial(EnvVar[int], fn=int)
EnvFloat = partial(EnvVar[float], fn=float)
EnvBool = partial(EnvVar[bool], fn=_TO_BOOL)
EnvOption = partial(EnvVar[bool | None], fn=_TO_BOOL, default_value=None)
EnvMem = partial(EnvVar[int], fn=_PARSE_MEM_BYTES)


class EnvClassSingleton:
    """Singleton container that materializes SGLite environment settings."""
    _instance: EnvClassSingleton | None = None

    # Interactive CLI defaults.
    CLI_MAX_TOKENS = EnvInt(2048)
    CLI_TOP_K = EnvInt(-1)
    CLI_TOP_P = EnvFloat(1.0)
    CLI_TEMPERATURE = EnvFloat(0.6)

    # Backend runtime controls.
    FLASHINFER_USE_TENSOR_CORES = EnvOption()
    DISABLE_OVERLAP_SCHEDULING = EnvBool(False)
    PYNCCL_MAX_BUFFER_SIZE = EnvMem(1024**3)

    def __new__(cls):
        """Create the singleton environment container."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize every declared setting from its `SGLITE_` variable."""
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr_value = getattr(self, attr_name)
            assert isinstance(attr_value, BaseEnv)
            attr_value._init(f"{SGLITE_ENV_PREFIX}{attr_name}")


ENV = EnvClassSingleton()

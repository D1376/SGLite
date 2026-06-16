"""Small name-to-object registry helper."""

from typing import Callable, Generic, Iterable, List, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Simple registry that maps string keys to implementations."""
    def __init__(self, type: str):
        """Create an empty registry for one implementation type."""
        self._registry = {}
        self._type = type

    def register(self, name: str) -> Callable[[T], None]:
        """Register a named implementation in the registry."""
        if name in self._registry:
            raise KeyError(f"{self._type} '{name}' is already registered.")

        def decorator(item: T) -> None:
            """Store the decorated implementation under its registry name."""
            self._registry[name] = item

        return decorator

    def __getitem__(self, name: str) -> T:
        """Return one registered implementation by name."""
        if name not in self._registry:
            raise KeyError(f"Unsupported {self._type}: {name}")
        return self._registry[name]

    def supported_names(self) -> List[str]:
        """Return registered implementation names."""
        return list(self._registry.keys())

    def assert_supported(self, names: str | Iterable[str]) -> None:
        """Raise if any requested names are not registered."""
        if isinstance(names, str):
            names = [names]
        for name in names:
            if name not in self._registry:
                from argparse import ArgumentTypeError

                raise ArgumentTypeError(
                    f"Unsupported {self._type}: {name}. "
                    f"Supported items: {self.supported_names()}"
                )

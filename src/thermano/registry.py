"""Tiny name -> factory registry.

Lets configuration files select components by name (``model.type: en_deeponet``,
``material.type: hastelloy_x``) without ``if/elif`` switches in the pipelines.
New architectures or material laws are added by decorating a class; no other
code has to change.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class Registry:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(obj: Callable[..., Any]) -> Callable[..., Any]:
            if name in self._items:
                raise KeyError(f"{self.kind} '{name}' is already registered")
            self._items[name] = obj
            return obj

        return decorator

    def build(self, name: str, **kwargs: Any) -> Any:
        if name not in self._items:
            raise KeyError(
                f"Unknown {self.kind} '{name}'. Available: {sorted(self._items)}"
            )
        return self._items[name](**kwargs)

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items


MODELS = Registry("model")
ACTIVATIONS = Registry("activation")
MATERIALS = Registry("material")

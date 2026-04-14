"""Transform registry: resolve per-implementation reshape callables by name."""

from __future__ import annotations

import importlib.util
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any


def validate_transform_name(name: str) -> None:
    """Raise ValueError unless `name` is a `module.function` path."""
    if '.' not in name:
        msg = f'Transform "{name}" must use "module.function" form'
        raise ValueError(msg)


def resolve_transform(name: str, impl_dir: Path) -> Callable[..., Any]:
    """Resolve a transform name to a callable loaded from the implementation dir."""
    module_name, func_name = name.rsplit('.', 1)
    module_path = impl_dir / f'{module_name}.py'
    if not module_path.exists():
        msg = f'Transform module not found: {module_path}'
        raise FileNotFoundError(msg)
    return _load_callable(module_path, func_name)


_module_cache: dict[str, Any] = {}
_module_cache_lock = threading.Lock()


def _load_callable(module_path: Path, func_name: str) -> Callable[..., Any]:
    """Load a function from a Python file, caching the module."""
    key = str(module_path)
    if key not in _module_cache:
        with _module_cache_lock:
            if key not in _module_cache:
                spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
                if spec is None or spec.loader is None:
                    msg = f'Cannot load transform module from {module_path}'
                    raise ImportError(msg)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                _module_cache[key] = module
    func = getattr(_module_cache[key], func_name, None)
    if func is None:
        msg = f'Transform function "{func_name}" not found in {module_path}'
        raise ValueError(msg)
    return func

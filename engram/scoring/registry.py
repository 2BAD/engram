"""Scorer registry: resolve scorer name strings to callable functions."""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from engram.scoring import scorers as builtin_scorers

_BUILTIN_SCORERS: dict[str, Any] = {
    'exact_match': builtin_scorers.exact_match,
    'fuzzy_match': builtin_scorers.fuzzy_match,
    'set_match': builtin_scorers.set_match,
    'contains': builtin_scorers.contains,
    'contains_all': builtin_scorers.contains_all,
    'contains_any': builtin_scorers.contains_any,
    'numeric_tolerance': builtin_scorers.numeric_tolerance,
}

_FACTORY_SCORERS = {'fuzzy_match', 'numeric_tolerance'}

# Pattern for parameterized scorers like "numeric_tolerance(0.1)" or "fuzzy_match(0.9)"
_PARAM_PATTERN = re.compile(r'^(\w+)\((.+)\)$')


def validate_scorer_name(name: str) -> None:
    """Raise ValueError unless `name` is a built-in, a parameterized built-in, or a `module.func` path."""
    match = _PARAM_PATTERN.match(name)
    if match:
        base_name = match.group(1)
        if base_name not in _BUILTIN_SCORERS:
            msg = f'Unknown scorer "{base_name}"'
            raise ValueError(msg)
        return

    if name in _BUILTIN_SCORERS:
        return

    if '.' in name:
        # Custom scorer path; resolution is deferred to score time.
        return

    available = ', '.join(sorted(_BUILTIN_SCORERS.keys()))
    msg = f'Unknown scorer "{name}". Built-ins: {available}. Custom scorers must use "module.function" form.'
    raise ValueError(msg)


def resolve_scorer(name: str, workflow_dir: Path | None = None) -> Callable[[Any, Any], bool]:
    """
    Resolve a scorer name string to a callable.

    Handles:
    - Built-in scorers: "exact_match"
    - Parameterized built-ins: "numeric_tolerance(0.1)"
    - Custom scorers: "scorers.my_function" (loaded from workflow_dir/scorers.py)
    """
    # Check for parameterized built-in
    match = _PARAM_PATTERN.match(name)
    if match:
        base_name, args_str = match.group(1), match.group(2)
        factory = _BUILTIN_SCORERS.get(base_name)
        if factory is None:
            msg = f'Unknown scorer "{base_name}"'
            raise ValueError(msg)
        args = [_parse_arg(a.strip()) for a in args_str.split(',')]
        return factory(*args)

    # Check for plain built-in
    if name in _BUILTIN_SCORERS:
        scorer = _BUILTIN_SCORERS[name]
        # fuzzy_match and numeric_tolerance are factories; call with no args for defaults
        if name in _FACTORY_SCORERS:
            return scorer()
        return scorer

    # Check for custom scorer (format: "scorers.function_name")
    if '.' in name and workflow_dir is not None:
        module_name, func_name = name.rsplit('.', 1)
        module_path = workflow_dir / f'{module_name}.py'
        if module_path.exists():
            return _load_custom_scorer(module_path, func_name)

    msg = f'Unknown scorer "{name}"'
    raise ValueError(msg)


def _parse_arg(value: str) -> float | str:
    """Parse a scorer argument string to a number or string."""
    try:
        return float(value)
    except ValueError:
        return value.strip('"').strip("'")


def _load_custom_scorer(module_path: Path, func_name: str) -> Callable[[Any, Any], bool]:
    """Load a custom scorer function from a Python file."""
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        msg = f'Cannot load scorer module from {module_path}'
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    func = getattr(module, func_name, None)
    if func is None:
        msg = f'Scorer function "{func_name}" not found in {module_path}'
        raise ValueError(msg)
    return func

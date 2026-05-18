"""Scorer registry: resolve scorer name strings to callable functions."""

from __future__ import annotations

import importlib.util
import inspect
import re
import threading
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
    'json_match': builtin_scorers.json_match,
    'regex': builtin_scorers.regex,
}

_FACTORY_SCORERS = {'fuzzy_match', 'numeric_tolerance', 'json_match', 'regex'}

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


def validate_scorer_spec(spec: str | dict[str, Any]) -> None:
    """
    Validate a scorer spec in either form.

    The legacy string form (``"exact_match"`` or ``"fuzzy_match(0.9)"``) handles factories whose args
    fit on one line. The dict form (``{type: llm_judge, criteria: ..., model: ..., threshold: 0.7}``)
    is for scorers whose configuration is multi-line or named — the ``type`` value names the scorer
    and the remaining keys become kwargs passed to the factory at resolve time.
    """
    if isinstance(spec, dict):
        if 'type' not in spec:
            msg = f'Scorer dict must include a "type" key, got keys: {sorted(spec)}'
            raise ValueError(msg)
        validate_scorer_name(spec['type'])
        return
    if isinstance(spec, str):
        validate_scorer_name(spec)
        return
    msg = f'Scorer spec must be a string or dict, got {type(spec).__name__}'
    raise TypeError(msg)


def resolve_scorer(spec: str | dict[str, Any], workflow_dir: Path | None = None) -> Callable[..., bool]:
    """
    Resolve a scorer spec (string or dict) to a callable.

    String form handles:
    - Built-in scorers: ``"exact_match"``
    - Parameterized built-ins: ``"numeric_tolerance(0.1)"``
    - Custom scorers: ``"scorers.my_function"`` (loaded from ``workflow_dir/scorers.py``)

    Dict form handles factory scorers whose args don't fit a one-line string. ``type`` names the
    scorer; remaining keys are passed as kwargs to the factory.
    """
    if isinstance(spec, dict):
        return _resolve_dict_spec(spec)
    return _resolve_string_spec(spec, workflow_dir)


def _resolve_string_spec(name: str, workflow_dir: Path | None) -> Callable[..., bool]:
    """Resolve the legacy string form. Same behavior as pre-dict-form ``resolve_scorer``."""
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


def _resolve_dict_spec(spec: dict[str, Any]) -> Callable[..., bool]:
    """Resolve a dict-form scorer spec by calling the named factory with remaining keys as kwargs."""
    kwargs = dict(spec)
    scorer_type = kwargs.pop('type', None)
    if scorer_type is None:
        msg = f'Scorer dict must include a "type" key, got keys: {sorted(spec)}'
        raise ValueError(msg)

    factory = _BUILTIN_SCORERS.get(scorer_type)
    if factory is None:
        msg = f'Dict-form scorer "{scorer_type}" is not a known built-in'
        raise ValueError(msg)

    if scorer_type in _FACTORY_SCORERS:
        return factory(**kwargs)

    if kwargs:
        msg = f'Scorer "{scorer_type}" does not accept kwargs: {sorted(kwargs)}'
        raise ValueError(msg)
    return factory


def scorer_accepts_input_data(fn: Callable[..., Any]) -> bool:
    """
    Whether a scorer accepts an ``input_data`` keyword argument.

    Scorers default to the two-arg ``(predicted, expected) -> bool`` contract.
    Scorers that need the original input (e.g. an LLM judge grading a summary
    against its source) declare a third ``input_data`` parameter; the engine
    inspects each scorer with this helper and passes the input through when set.
    Inspection failures (built-ins, C extensions) fall back to two-arg calling.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return 'input_data' in sig.parameters


def _parse_arg(value: str) -> bool | float | str:
    """Parse a scorer argument string to a boolean, number, or string."""
    lower = value.lower()
    if lower == 'true':
        return True
    if lower == 'false':
        return False
    try:
        return float(value)
    except ValueError:
        return value.strip('"').strip("'")


_module_cache: dict[str, Any] = {}
_module_cache_lock = threading.Lock()


def _load_custom_scorer(module_path: Path, func_name: str) -> Callable[[Any, Any], bool]:
    """Load a custom scorer function from a Python file, caching the module."""
    key = str(module_path)
    if key not in _module_cache:
        with _module_cache_lock:
            if key not in _module_cache:
                spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
                if spec is None or spec.loader is None:
                    msg = f'Cannot load scorer module from {module_path}'
                    raise ImportError(msg)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                _module_cache[key] = module
    func = getattr(_module_cache[key], func_name, None)
    if func is None:
        msg = f'Scorer function "{func_name}" not found in {module_path}'
        raise ValueError(msg)
    return func

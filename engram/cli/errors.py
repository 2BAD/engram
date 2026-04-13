"""
Global error handling for the CLI.

Catches common exceptions and prints friendly messages instead of raw
tracebacks.  Set ENGRAM_TRACEBACK=1 to see the full traceback for any error.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from typing import Any

import httpx
import yaml


def run_with_error_handling(app: Any) -> None:
    """Invoke *app* (a Typer/Click callable) with user-friendly error handling."""
    try:
        app()
    except SystemExit:
        raise
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as exc:
        if _want_traceback():
            raise
        message, show_hint = _format_exception(exc)
        _exit_with_error(message, hint=show_hint)


def _exit_with_error(message: str, *, hint: bool = False) -> None:
    """Print an error message to stderr and exit 1."""
    print(f'Error: {message}', file=sys.stderr)
    if hint:
        print('Set ENGRAM_TRACEBACK=1 for the full traceback.', file=sys.stderr)
    sys.exit(1)


def _want_traceback() -> bool:
    return os.environ.get('ENGRAM_TRACEBACK', '') not in ('', '0')


def _format_yaml_error(exc: yaml.YAMLError) -> str:
    mark: yaml.error.Mark | None = getattr(exc, 'problem_mark', None)
    if mark is not None:
        location = f' at line {mark.line + 1}, column {mark.column + 1}'
        if mark.name and mark.name != '<unicode string>':
            location = f' in {mark.name}{location}'
    else:
        location = ''
    problem = getattr(exc, 'problem', str(exc))
    return f'invalid YAML{location}: {problem}'


_FORMATTERS: list[tuple[type | tuple[type, ...], Callable[[Any], str]]] = [
    (yaml.YAMLError, _format_yaml_error),
    (json.JSONDecodeError, lambda e: f'invalid JSON at line {e.lineno}, column {e.colno}: {e.msg}'),
    ((FileNotFoundError, PermissionError, ValueError, TypeError), str),
    (KeyError, lambda e: f'missing required config field: {e}'),
    (httpx.ConnectError, lambda _: 'could not connect to remote server (check your internet connection)'),
    (httpx.TimeoutException, lambda _: 'request timed out (check your internet connection and try again)'),
    (httpx.HTTPStatusError, lambda e: f'HTTP {e.response.status_code} from {e.request.url.host}'),
]


def _format_exception(exc: Exception) -> tuple[str, bool]:
    """Return (message, show_hint) for a caught exception."""
    for exc_type, formatter in _FORMATTERS:
        if isinstance(exc, exc_type):
            return formatter(exc), False
    return f'{type(exc).__name__}: {exc}', True

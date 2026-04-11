"""Built-in scorer functions."""

from __future__ import annotations

from collections.abc import Callable
from difflib import SequenceMatcher
from typing import Any


def exact_match(predicted: Any, expected: Any) -> bool:
    """Case-insensitive string equality after stripping whitespace."""
    return str(predicted).strip().lower() == str(expected).strip().lower()


def fuzzy_match(threshold: float = 0.8) -> Callable[[Any, Any], bool]:
    """String similarity above a threshold using SequenceMatcher."""

    def _scorer(predicted: Any, expected: Any) -> bool:
        ratio = SequenceMatcher(None, str(predicted).lower(), str(expected).lower()).ratio()
        return ratio >= threshold

    return _scorer


def set_match(predicted: Any, expected: Any) -> bool:
    """Unordered set equality for list-valued fields."""
    if isinstance(predicted, str):
        predicted = [s.strip() for s in predicted.split(',')]
    if isinstance(expected, str):
        expected = [s.strip() for s in expected.split(',')]
    return {str(v).strip().lower() for v in predicted} == {str(v).strip().lower() for v in expected}


def contains(predicted: Any, expected: Any) -> bool:
    """Check if predicted string contains the expected substring (case-insensitive)."""
    return str(expected).strip().lower() in str(predicted).strip().lower()


def contains_all(predicted: Any, expected: Any) -> bool:
    """Check if predicted string contains all expected substrings (case-insensitive)."""
    haystack = str(predicted).strip().lower()
    needles = _to_string_list(expected)
    return all(n in haystack for n in needles)


def contains_any(predicted: Any, expected: Any) -> bool:
    """Check if predicted string contains at least one expected substring (case-insensitive)."""
    haystack = str(predicted).strip().lower()
    needles = _to_string_list(expected)
    return any(n in haystack for n in needles)


def _to_string_list(value: Any) -> list[str]:
    """Normalize a list or comma-separated string into lowered, stripped strings."""
    if isinstance(value, str):
        return [s.strip().lower() for s in value.split(',') if s.strip()]
    return [str(v).strip().lower() for v in value]


def numeric_tolerance(tolerance: float = 0.1) -> Callable[[Any, Any], bool]:
    """Check if predicted value is within a percentage tolerance of expected."""

    def _scorer(predicted: Any, expected: Any) -> bool:
        try:
            p, e = float(predicted), float(expected)
        except (ValueError, TypeError):
            return False
        if e == 0:
            return p == 0
        return abs(p - e) / abs(e) <= tolerance

    return _scorer

"""Built-in scorer functions."""

from __future__ import annotations

import json
import re
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
        except ValueError, TypeError:
            return False
        if e == 0:
            return p == 0
        return abs(p - e) / abs(e) <= tolerance

    return _scorer


def json_match(ignore_extra: bool = False) -> Callable[[Any, Any], bool]:
    """
    Structural JSON comparison ignoring key order.

    Parses string values as JSON. When *ignore_extra* is True, extra keys in
    predicted dicts are allowed as long as all expected keys match recursively.
    """

    def _scorer(predicted: Any, expected: Any) -> bool:
        p = _to_json_value(predicted)
        e = _to_json_value(expected)
        if ignore_extra:
            return _json_contains(p, e)
        return p == e

    return _scorer


def _to_json_value(value: Any) -> Any:
    """Parse a JSON string into a Python value, or return as-is if already parsed."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _json_contains(haystack: Any, needle: Any) -> bool:
    """Check that *haystack* contains all keys/values from *needle* (recursive for dicts)."""
    if isinstance(needle, dict) and isinstance(haystack, dict):
        return all(k in haystack and _json_contains(haystack[k], v) for k, v in needle.items())
    if isinstance(needle, list) and isinstance(haystack, list):
        if len(needle) != len(haystack):
            return False
        return all(_json_contains(h, n) for h, n in zip(haystack, needle, strict=True))
    return haystack == needle


_REGEX_FLAGS: dict[str, re.RegexFlag] = {
    'i': re.IGNORECASE,
    'm': re.MULTILINE,
    's': re.DOTALL,
}


def regex(flags: str = '') -> Callable[[Any, Any], bool]:
    """
    Match predicted output against a regex pattern from expected.

    Uses re.search so the pattern can match anywhere in the string.
    Add ^ and $ anchors in the pattern for full-match semantics.
    *flags* is a string of single-character flags: i (ignore case), m (multiline), s (dotall).
    """
    compiled_flags = re.RegexFlag(0)
    for char in flags.lower():
        if char in _REGEX_FLAGS:
            compiled_flags |= _REGEX_FLAGS[char]

    def _scorer(predicted: Any, expected: Any) -> bool:
        return re.search(str(expected), str(predicted), compiled_flags) is not None

    return _scorer

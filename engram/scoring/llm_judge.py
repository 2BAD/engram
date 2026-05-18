"""LLM-as-judge scorer: grade open-ended outputs with a configurable judge model."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from engram.analysis.analyzer import LLMCallResult, call_anthropic_messages

if TYPE_CHECKING:
    from engram.models.input import InputData


# Public so the scoring engine can pull it off resolved judge scorers without an attribute
# lookup string. Engine treats absence as "not a judge" and skips that scorer for cost rollup.
JUDGE_CALLS_ATTR = '_judge_calls'


_JUDGE_SYSTEM_PROMPT = (
    'You evaluate a predicted output against criteria. Score how well it meets them.\n'
    '\n'
    'Respond with ONLY a single JSON object, no markdown, no prose:\n'
    '{"score": <float between 0 and 1>, "reason": "<one sentence>"}\n'
    '\n'
    'The score is your confidence that the predicted output satisfies the criteria. '
    '1.0 means clearly yes, 0.0 means clearly no, intermediate values for partial credit.'
)


def llm_judge(
    criteria: str,
    *,
    model: str = 'claude-sonnet-4-6',
    threshold: float = 0.7,
    reference_free: bool = False,
    max_tokens: int = 256,
) -> Callable[..., bool]:
    """
    Build a scorer that grades the predicted output against criteria via an LLM judge.

    The returned scorer runs the judge at temperature 0 for stability and binarizes the judge's
    0-1 score against the threshold. When reference_free is true the expected label is omitted
    from the judge prompt — useful for "is this coherent?" style criteria with no ground-truth
    answer. The scorer declares an input_data kwarg so the scoring engine threads the dataset's
    InputData through; criteria can then reference the original source (e.g. judging a summary
    against the article it was drawn from). JSON parse failures conservatively score False —
    a judge that can't return parseable JSON shouldn't silently pass items.
    """
    judge_calls: list[LLMCallResult] = []

    def _scorer(predicted: Any, expected: Any, *, input_data: InputData | None = None) -> bool:
        user_message = _build_user_message(criteria, predicted, expected, input_data, reference_free)
        call = call_anthropic_messages(
            model,
            _JUDGE_SYSTEM_PROMPT,
            user_message,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        judge_calls.append(call)
        score = _parse_score(call.text)
        if score is None:
            return False
        return score >= threshold

    # Engine reads this attribute after scoring to aggregate judge cost into EvalReport.
    setattr(_scorer, JUDGE_CALLS_ATTR, judge_calls)
    return _scorer


def _build_user_message(
    criteria: str,
    predicted: Any,
    expected: Any,
    input_data: InputData | None,
    reference_free: bool,
) -> str:
    parts = [f'Criteria:\n{criteria}', f'Predicted output:\n{predicted}']
    if not reference_free:
        parts.append(f'Expected output:\n{expected}')
    if input_data is not None and input_data.text is not None:
        parts.append(f'Source input:\n{input_data.text}')
    return '\n\n'.join(parts)


def _parse_score(text: str) -> float | None:
    """Extract a 0-1 score from the judge response, tolerating markdown fences."""
    obj = _parse_json(text)
    if not isinstance(obj, dict):
        return None
    raw = obj.get('score')
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)) and 0 <= raw <= 1:
        return float(raw)
    return None


def _parse_json(text: str) -> Any:
    """Parse a JSON object from `text`, falling back to extraction from a markdown fence."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return None

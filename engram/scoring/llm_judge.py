"""LLM-as-judge scorer: grade open-ended outputs with a configurable judge model."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from engram.analysis.analyzer import LLMCallResult, call_anthropic_messages

if TYPE_CHECKING:
    from engram.models.input import InputData


@dataclass
class JudgeState:
    """
    Per-scorer state the engine reads (`calls`) and writes (cache config).

    Each call to ``llm_judge(...)`` returns a fresh scorer with its own state. The scoring
    engine looks up `_judge_state` on each resolved scorer to (1) point the cache at the
    experiment dir and (2) sum the call log into EvalReport after scoring runs.
    """

    calls: list[LLMCallResult] = field(default_factory=list)
    cache_dir: Path | None = None
    cache_disabled: bool = False


# Attribute name used by the engine to find each judge scorer's JudgeState. Centralised so
# tests and the engine reference the same string; absence on a scorer means "not a judge".
JUDGE_STATE_ATTR = '_judge_state'


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

    Judge responses are cached on disk under experiments/{id}/judge_cache/ when the engine sets
    state.cache_dir, keyed by a hash of (model, system prompt, user message). Re-scoring an
    experiment without touching criteria/model is then free; pass --no-judge-cache to bypass.
    """
    state = JudgeState()

    def _scorer(predicted: Any, expected: Any, *, input_data: InputData | None = None) -> bool:
        user_message = _build_user_message(criteria, predicted, expected, input_data, reference_free)
        cache_key = _cache_key(model, _JUDGE_SYSTEM_PROMPT, user_message)
        call = _load_cached(state, cache_key)
        if call is None:
            call = call_anthropic_messages(
                model,
                _JUDGE_SYSTEM_PROMPT,
                user_message,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            _save_cached(state, cache_key, call)
        state.calls.append(call)
        score = _parse_score(call.text)
        if score is None:
            return False
        return score >= threshold

    setattr(_scorer, JUDGE_STATE_ATTR, state)
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


def compute_judge_config_hash(scorers: dict[str, str | dict[str, Any]]) -> str:
    """
    Fingerprint the llm_judge specs in a workflow's scorers, or '' when no judges are used.

    Lets `engram compare` warn when two experiments were judged with different criteria/model/
    threshold, since judging accuracy can move purely because the rubric changed. Non-judge
    scorer changes are out of scope here — they're surfaced by the existing config-snapshot diff.
    """
    judge_specs = {
        field_name: spec
        for field_name, spec in scorers.items()
        if isinstance(spec, dict) and spec.get('type') == 'llm_judge'
    }
    if not judge_specs:
        return ''
    canonical = json.dumps(judge_specs, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def _cache_key(model: str, system_prompt: str, user_message: str) -> str:
    """Content-address the LLM call so identical prompts hit the same cache entry across runs."""
    payload = f'{model}\x00{system_prompt}\x00{user_message}'.encode()
    return hashlib.sha256(payload).hexdigest()


def _load_cached(state: JudgeState, key: str) -> LLMCallResult | None:
    if state.cache_disabled or state.cache_dir is None:
        return None
    cache_path = state.cache_dir / f'{key}.json'
    if not cache_path.exists():
        return None
    try:
        raw = json.loads(cache_path.read_text())
        return LLMCallResult(**raw)
    except (OSError, json.JSONDecodeError, TypeError):
        # Corrupt or stale-format entry: ignore, the next save overwrites it.
        return None


def _save_cached(state: JudgeState, key: str, call: LLMCallResult) -> None:
    if state.cache_disabled or state.cache_dir is None:
        return
    state.cache_dir.mkdir(parents=True, exist_ok=True)
    (state.cache_dir / f'{key}.json').write_text(json.dumps(asdict(call)))


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

"""Run result models."""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class TokenUsage:
    """Token counts from a single run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class RunResult:
    """Result of triggering a single workflow run."""

    input_file: str
    output: dict[str, Any] = field(default_factory=dict)
    status: Literal['succeeded', 'failed', 'timeout'] = 'succeeded'
    usage: TokenUsage = field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    error: str = ''
    trace_id: str = ''
    # Index into the repeat group for this input. 0 for single-repeat runs (the default).
    # Older results.json files without this field load with repeat_index=0 unchanged.
    repeat_index: int = 0

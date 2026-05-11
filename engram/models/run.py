"""Run result models."""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class TokenUsage:
    """
    Token counts from a single run.

    ``prompt_tokens`` is the total input including any cache reads and creation;
    ``cache_read_tokens`` and ``cache_creation_tokens`` are subsets of it. This
    convention lets cost compute as a single sum across four rates regardless
    of provider (Anthropic and OpenAI report cache differently in their raw
    APIs; runners normalize before storing).
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


@dataclass
class RunResult:
    """Result of triggering a single workflow run."""

    input_file: str
    output: dict[str, Any] = field(default_factory=dict)
    status: Literal['succeeded', 'failed', 'timeout'] = 'succeeded'
    usage: TokenUsage = field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    # Per-bucket breakdown of cost_usd. Sum equals cost_usd for runs scored after this was added;
    # older runs may have zeros here while cost_usd carries the legacy total.
    cost_input_usd: float = 0.0
    cost_cache_read_usd: float = 0.0
    cost_cache_creation_usd: float = 0.0
    cost_output_usd: float = 0.0
    latency_ms: float = 0.0
    error: str = ''
    trace_id: str = ''
    # Runner-injected key used to correlate out-of-band enrichment (e.g. Dynamiq's finalize
    # backfills cost by matching this against ``trace.input._engram_id``). Empty when the
    # runner doesn't need post-hoc correlation.
    correlation_id: str = ''
    # Index into the repeat group for this input. 0 for single-repeat runs (the default).
    repeat_index: int = 0

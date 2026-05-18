"""LLM call orchestration, cost estimation, and caching for experiment analysis."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from engram.cost.pricing import find_rate, load_pricing

if TYPE_CHECKING:
    from engram.models.analysis import AnalysisConfig


@dataclass
class AnalysisResult:
    """Result of an LLM analysis call."""

    markdown: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class LLMCallResult:
    """Raw text + usage + cost from one LLM messages.create call."""

    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


def estimate_analysis_cost(
    config: AnalysisConfig,
    system_prompt: str,
    user_message: str,
) -> tuple[float, int, int]:
    """
    Estimate the cost of an analysis call.

    Returns (estimated_cost_usd, estimated_input_tokens, estimated_output_tokens).
    Uses a rough 4-chars-per-token heuristic consistent with cost/estimator.py.
    """
    pricing = load_pricing()
    input_rate, output_rate = find_rate(pricing, config.model)

    input_tokens = max(1, (len(system_prompt) + len(user_message)) // 4)
    output_tokens = 1500  # ~600 words of markdown

    cost = input_tokens * input_rate + output_tokens * output_rate
    return cost, input_tokens, output_tokens


def load_cached(exp_dir: Path, filename: str = 'analysis.md') -> str | None:
    """Load a cached result file if it exists."""
    path = exp_dir / filename
    if path.exists():
        return path.read_text()
    return None


def save_cached(exp_dir: Path, result: AnalysisResult, filename: str = 'analysis.md') -> None:
    """Cache an analysis result in the experiment directory."""
    (exp_dir / filename).write_text(result.markdown)


def call_llm(
    config: AnalysisConfig,
    system_prompt: str,
    user_message: str,
) -> AnalysisResult:
    """Send the analysis request to the configured LLM."""
    model = config.model

    if model.startswith('claude'):
        return _call_anthropic(model, system_prompt, user_message)

    # Future: elif model.startswith(('gpt-', 'o1', 'o3', 'o4')):
    #     return _call_openai(model, system_prompt, user_message)

    msg = f'Unsupported analysis model: {model}. Only claude-* models are currently supported.'
    raise ValueError(msg)


def call_anthropic_messages(
    model: str,
    system_prompt: str,
    user_message: str,
    *,
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> LLMCallResult:
    """
    Call the Anthropic Messages API and return text, usage, and cost.

    Lower-level than call_llm: leaves response shaping to the caller so the explain/suggest path
    can wrap text as markdown while the LLM judge scorer parses the same text as JSON.
    """
    import anthropic  # noqa: PLC0415

    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        msg = 'ANTHROPIC_API_KEY not set. Add it to .env or export it in your shell.'
        raise ValueError(msg)

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{'role': 'user', 'content': user_message}],
    )

    text = getattr(response.content[0], 'text', '') if response.content else ''
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    pricing = load_pricing()
    input_rate, output_rate = find_rate(pricing, model)
    cost = input_tokens * input_rate + output_tokens * output_rate

    return LLMCallResult(text=text, input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost)


def _call_anthropic(model: str, system_prompt: str, user_message: str) -> AnalysisResult:
    """Call the Anthropic Messages API and wrap as an AnalysisResult."""
    call = call_anthropic_messages(model, system_prompt, user_message)
    return AnalysisResult(
        markdown=call.text,
        model=model,
        input_tokens=call.input_tokens,
        output_tokens=call.output_tokens,
        cost_usd=call.cost_usd,
    )

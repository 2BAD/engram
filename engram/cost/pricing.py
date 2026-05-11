"""Pricing data: fetch and cache model pricing from LiteLLM."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from engram.models.run import TokenUsage

LITELLM_PRICING_URL = 'https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json'
CACHE_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days


def load_pricing(cache_dir: Path | None = None, overrides: dict[str, dict[str, float]] | None = None) -> dict[str, Any]:
    """Load pricing data, using cache if available and fresh."""
    if cache_dir is None:
        cache_dir = Path.home() / '.engram' / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / 'pricing.json'
    meta_file = cache_dir / 'pricing_meta.json'

    # Check cache freshness
    if cache_file.exists() and meta_file.exists():
        meta = json.loads(meta_file.read_text())
        if time.time() - meta.get('fetched_at', 0) < CACHE_TTL_SECONDS:
            pricing = json.loads(cache_file.read_text())
            return _apply_overrides(pricing, overrides)

    # Fetch fresh data
    pricing = _fetch_pricing()
    cache_file.write_text(json.dumps(pricing))
    meta_file.write_text(json.dumps({'fetched_at': time.time()}))

    return _apply_overrides(pricing, overrides)


def find_rate(pricing: dict[str, Any], model: str) -> tuple[float, float]:
    """
    Find input and output token rates for a model.

    Falls back to zero if no match. Tries the model name as-is, then a
    dash/underscore-insensitive normalization, then (for litellm-style names
    like ``anthropic/claude-...``) the suffix after the slash.
    """
    data = _find_model_data(pricing, model)
    return data.get('input_cost_per_token', 0.0), data.get('output_cost_per_token', 0.0)


def find_cache_rates(pricing: dict[str, Any], model: str) -> tuple[float, float]:
    """
    Cache-creation and cache-read rates for a model.

    Returns ``(creation_rate, read_rate)``. Either rate falls back to the
    model's regular ``input_cost_per_token`` when missing, so cost math stays
    correct for models without prompt-caching pricing data.
    """
    data = _find_model_data(pricing, model)
    input_rate = data.get('input_cost_per_token', 0.0)
    creation = data.get('cache_creation_input_token_cost', input_rate)
    read = data.get('cache_read_input_token_cost', input_rate)
    return creation, read


def compute_cost(pricing: dict[str, Any], model: str, usage: TokenUsage) -> float:
    """Total billed cost for a single run. Use compute_cost_components when you need the breakdown."""
    return sum(compute_cost_components(pricing, model, usage).values())


def compute_cost_without_cache(pricing: dict[str, Any], model: str, usage: TokenUsage) -> float:
    """
    Counterfactual cost: every prompt token billed at the full input rate, no cache discount or premium.

    Used to quantify how much prompt caching is saving on this run. Since engram's ``prompt_tokens`` is
    the inclusive total (uncached + cache reads + cache creation), this is just the simple two-rate sum.
    """
    input_rate, output_rate = find_rate(pricing, model)
    return usage.prompt_tokens * input_rate + usage.completion_tokens * output_rate


def compute_cost_components(pricing: dict[str, Any], model: str, usage: TokenUsage) -> dict[str, float]:
    """
    Per-bucket cost for a single run.

    Splits ``prompt_tokens`` into three buckets (non-cached input, cache reads, cache creation)
    and prices each at its own rate; output tokens at the output rate. The returned dict carries
    one key per bucket, so callers can sum aggregates per component across many runs.
    """
    input_rate, output_rate = find_rate(pricing, model)
    creation_rate, read_rate = find_cache_rates(pricing, model)
    non_cached = max(0, usage.prompt_tokens - usage.cache_read_tokens - usage.cache_creation_tokens)
    return {
        'input_usd': non_cached * input_rate,
        'cache_creation_usd': usage.cache_creation_tokens * creation_rate,
        'cache_read_usd': usage.cache_read_tokens * read_rate,
        'output_usd': usage.completion_tokens * output_rate,
    }


def _find_model_data(pricing: dict[str, Any], model: str) -> dict[str, Any]:
    if model in pricing:
        return pricing[model]

    normalized = model.lower().replace('-', '_')
    for key, data in pricing.items():
        if key.lower().replace('-', '_') == normalized:
            return data

    if '/' in model:
        suffix = model.split('/', 1)[1]
        if suffix in pricing:
            return pricing[suffix]
        normalized_suffix = suffix.lower().replace('-', '_')
        for key, data in pricing.items():
            if key.lower().replace('-', '_') == normalized_suffix:
                return data

    return {}


def _fetch_pricing() -> dict[str, Any]:
    """Fetch pricing data from LiteLLM GitHub."""
    response = httpx.get(LITELLM_PRICING_URL, timeout=30)
    response.raise_for_status()
    return response.json()


def _apply_overrides(
    pricing: dict[str, Any],
    overrides: dict[str, dict[str, float]] | None,
) -> dict[str, Any]:
    """Merge project-level pricing overrides into the pricing data."""
    if not overrides:
        return pricing
    merged = dict(pricing)
    for model, rates in overrides.items():
        merged[model] = {**merged.get(model, {}), **rates}
    return merged

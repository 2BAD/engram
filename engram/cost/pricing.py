"""Pricing data: fetch and cache model pricing from LiteLLM."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx

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

    Returns (input_cost_per_token, output_cost_per_token).
    Falls back to zero if model not found.
    """
    model_data = pricing.get(model, {})
    if not model_data:
        # Try normalizing the model name
        normalized = model.lower().replace('-', '_')
        for key, data in pricing.items():
            if key.lower().replace('-', '_') == normalized:
                model_data = data
                break

    input_cost = model_data.get('input_cost_per_token', 0.0)
    output_cost = model_data.get('output_cost_per_token', 0.0)
    return input_cost, output_cost


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

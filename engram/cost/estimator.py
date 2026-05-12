"""Cost estimation: estimate cost before running."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from engram.config.loader import load_implementation, load_project
from engram.cost.pricing import find_cache_rates, find_rate, load_pricing
from engram.datasets.loader import load_dataset_inputs
from engram.tracking.index import read_index

# Below this threshold Anthropic silently ignores cache_control markers and OpenAI's auto-cache doesn't
# activate, so the estimator projects no cache savings even when prompt_cache is on. (Anthropic Haiku
# actually needs 2048 tokens; we use the lower Sonnet/Opus number conservatively here, accepting that
# Haiku users may see slightly inflated savings in the estimate.)
_MIN_CACHEABLE_TOKENS = 1024


def estimate_cost(
    root: Path,
    implementation_name: str,
    dataset_name: str,
) -> dict[str, Any]:
    """
    Estimate cost for running an implementation against a dataset.

    Input-side cost is always computed. Output-side cost (and the total) are None when no
    prior run is available to calibrate ``avg_output_tokens``.
    """
    impl_config = load_implementation(root, implementation_name)
    impl_dir = root / 'implementations' / implementation_name

    project = load_project(root)
    pricing = load_pricing(overrides=project.pricing_overrides)

    model = impl_config.runner_config.get('model', '')
    input_rate, output_rate = find_rate(pricing, model)
    cache_creation_rate, cache_read_rate = find_cache_rates(pricing, model)

    prompt_tokens = _count_prompt_tokens(impl_dir)
    inputs = load_dataset_inputs(root, dataset_name)

    calibration = _load_calibration(root, implementation_name, dataset_name)
    avg_output_tokens = calibration['avg_output_tokens']
    hit_rate_used = calibration['cache_hit_rate']
    output_estimable = avg_output_tokens is not None

    text_inputs = [inp for inp in inputs if not inp.is_binary]
    prompt_cache_on = _truthy(impl_config.runner_config.get('prompt_cache'))
    cache_active = prompt_cache_on and prompt_tokens >= _MIN_CACHEABLE_TOKENS and len(text_inputs) > 1

    template_rate_per_call = _expected_template_rate(
        cache_active=cache_active,
        n_calls=len(text_inputs),
        input_rate=input_rate,
        cache_read_rate=cache_read_rate,
        cache_creation_rate=cache_creation_rate,
        empirical_hit_rate=hit_rate_used,
    )

    examples = []
    total_template_cost = 0.0
    total_variable_input_cost = 0.0
    total_output_cost = 0.0
    for inp in text_inputs:
        variable_tokens = _rough_token_count(inp.text or inp.text_for_display)
        template_cost = prompt_tokens * template_rate_per_call
        variable_input_cost = variable_tokens * input_rate
        output_cost = avg_output_tokens * output_rate if output_estimable else 0.0

        total_template_cost += template_cost
        total_variable_input_cost += variable_input_cost
        total_output_cost += output_cost

        examples.append(
            {
                'input_file': inp.filename,
                'estimated_input_tokens': prompt_tokens + variable_tokens,
                'estimated_output_tokens': avg_output_tokens,
                'estimated_cost_usd': (
                    round(template_cost + variable_input_cost + output_cost, 6) if output_estimable else None
                ),
            }
        )

    total_input_cost = total_template_cost + total_variable_input_cost

    warnings: list[str] = []
    if any(inp.is_binary for inp in inputs):
        warnings.append('Dataset contains binary inputs (images/PDFs) whose token cost cannot be reliably estimated.')
    if not output_estimable:
        warnings.append(
            'No prior run for this implementation/dataset; output token count cannot be estimated. '
            'Run once (e.g. `engram run -n 1`) to calibrate.'
        )
    if prompt_cache_on and not cache_active:
        reason = (
            f'system prompt is below the {_MIN_CACHEABLE_TOKENS}-token minimum'
            if prompt_tokens < _MIN_CACHEABLE_TOKENS
            else 'fewer than 2 text examples in the dataset'
        )
        warnings.append(f'prompt_cache is enabled but no savings projected: {reason}.')

    result: dict[str, Any] = {
        'implementation': implementation_name,
        'dataset': dataset_name,
        'model': model,
        'input_rate_per_token': input_rate,
        'output_rate_per_token': output_rate,
        'prompt_template_tokens': prompt_tokens,
        'avg_output_tokens': avg_output_tokens,
        'total_examples': len(inputs),
        'estimated_input_cost_usd': round(total_input_cost, 4),
        'estimated_output_cost_usd': round(total_output_cost, 4) if output_estimable else None,
        'total_estimated_cost_usd': round(total_input_cost + total_output_cost, 4) if output_estimable else None,
        'warnings': warnings,
        'examples': examples,
    }
    if cache_active:
        # Savings are input-side only, so they're reportable without output calibration; the
        # without-cache total is only meaningful when output is known.
        uncached_template_cost = prompt_tokens * input_rate * len(text_inputs)
        savings = uncached_template_cost - total_template_cost
        result['estimated_savings_usd'] = round(savings, 4)
        if output_estimable:
            without_cache_total = uncached_template_cost + total_variable_input_cost + total_output_cost
            result['estimated_cost_without_cache_usd'] = round(without_cache_total, 4)
        if hit_rate_used is not None:
            result['cache_hit_rate_used'] = round(hit_rate_used, 4)
    return result


def _truthy(value: object) -> bool:
    """YAML-style truthiness for runner_config booleans."""
    return isinstance(value, str) and value.strip().lower() in {'true', '1', 'yes', 'on'}


def _count_prompt_tokens(impl_dir: Path) -> int:
    """Rough token count for all prompt files (fixed cost per run)."""
    prompts_dir = impl_dir / 'prompts'
    if not prompts_dir.exists():
        return 0
    total = 0
    for f in prompts_dir.iterdir():
        if f.is_file():
            total += _rough_token_count(f.read_text())
    return total


def _rough_token_count(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return max(1, len(text) // 4)


def _load_calibration(root: Path, implementation_name: str, dataset_name: str) -> dict[str, Any]:
    """Pull avg_output_tokens and cache_hit_rate from the latest matching index entry; either may be None."""
    index = read_index(root)
    matching = [
        entry
        for entry in index
        if entry.get('implementation') == implementation_name and entry.get('dataset') == dataset_name
    ]
    if not matching:
        return {'avg_output_tokens': None, 'cache_hit_rate': None}

    latest = matching[-1]
    cost = latest.get('cost') or {}
    return {
        'avg_output_tokens': latest.get('avg_output_tokens'),
        'cache_hit_rate': cost.get('cache_hit_rate'),
    }


def _expected_template_rate(
    cache_active: bool,
    n_calls: int,
    input_rate: float,
    cache_read_rate: float,
    cache_creation_rate: float,
    empirical_hit_rate: float | None,
) -> float:
    """
    Per-token rate for the cached prompt template, averaged across one run of the dataset.

    Without cache: pay the full input rate every call.
    With cache + no history: model one creation up front and (n-1) reads (the idealised pattern).
    With cache + history: blend read and creation rates by the empirical hit rate.
    """
    if not cache_active:
        return input_rate

    if empirical_hit_rate is not None:
        h = max(0.0, min(1.0, empirical_hit_rate))
        return h * cache_read_rate + (1 - h) * cache_creation_rate

    return (cache_creation_rate + (n_calls - 1) * cache_read_rate) / n_calls

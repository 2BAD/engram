"""Cost estimation: estimate cost before running."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from engram.config.loader import load_implementation, load_project
from engram.cost.pricing import find_rate, load_pricing
from engram.datasets.loader import load_dataset_inputs
from engram.tracking.index import read_index

DEFAULT_OUTPUT_TOKENS = 500


def estimate_cost(
    root: Path,
    implementation_name: str,
    dataset_name: str,
) -> dict[str, Any]:
    """
    Estimate cost for running an implementation against a dataset.

    Returns a dict with per-example and total estimates.
    """
    impl_config = load_implementation(root, implementation_name)
    impl_dir = root / 'implementations' / implementation_name

    # Load pricing, applying any project-level overrides from engram.yaml.
    project = load_project(root)
    pricing = load_pricing(overrides=project.pricing_overrides)

    model = impl_config.runner_config.get('model', '')
    input_rate, output_rate = find_rate(pricing, model)

    # Load prompt tokens (fixed cost per run)
    prompt_tokens = _count_prompt_tokens(impl_dir)

    # Load dataset inputs
    inputs = load_dataset_inputs(root, dataset_name)

    # Estimate output tokens from historical data or default
    avg_output_tokens = _estimate_output_tokens(root, implementation_name, dataset_name)

    # Estimate per-example cost
    examples = []
    total_cost = 0.0
    for inp in inputs:
        input_tokens = prompt_tokens + _rough_token_count(inp.text or inp.text_for_display)
        example_cost = (input_tokens * input_rate) + (avg_output_tokens * output_rate)
        total_cost += example_cost
        examples.append(
            {
                'input_file': inp.filename,
                'estimated_input_tokens': input_tokens,
                'estimated_output_tokens': avg_output_tokens,
                'estimated_cost_usd': round(example_cost, 6),
            }
        )

    return {
        'implementation': implementation_name,
        'dataset': dataset_name,
        'model': model,
        'input_rate_per_token': input_rate,
        'output_rate_per_token': output_rate,
        'prompt_template_tokens': prompt_tokens,
        'avg_output_tokens': avg_output_tokens,
        'total_examples': len(inputs),
        'total_estimated_cost_usd': round(total_cost, 4),
        'examples': examples,
    }


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


def _estimate_output_tokens(root: Path, implementation_name: str, dataset_name: str) -> int:
    """Estimate output tokens from historical experiments or use default."""
    index = read_index(root)

    # Find past experiments with same implementation and dataset
    matching = [
        entry
        for entry in index
        if entry.get('implementation') == implementation_name and entry.get('dataset') == dataset_name
    ]

    if not matching:
        return DEFAULT_OUTPUT_TOKENS

    # Use the most recent experiment's average
    latest = matching[-1]
    # If we had token data in the index, we'd use it; for now, use default
    return latest.get('avg_output_tokens', DEFAULT_OUTPUT_TOKENS)

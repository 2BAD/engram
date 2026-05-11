"""Tests for cost estimation and pricing."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from engram.cli import app
from engram.cost.estimator import _rough_token_count, estimate_cost
from engram.cost.pricing import (
    _apply_overrides,
    compute_cost,
    compute_cost_components,
    find_cache_rates,
    find_rate,
)
from engram.models.run import TokenUsage

# --- Pricing ---


def test_find_rate_exact():
    pricing = {
        'claude-sonnet-4-5-20250514': {
            'input_cost_per_token': 0.000003,
            'output_cost_per_token': 0.000015,
        }
    }
    input_rate, output_rate = find_rate(pricing, 'claude-sonnet-4-5-20250514')
    assert input_rate == 0.000003
    assert output_rate == 0.000015


def test_find_rate_normalized():
    pricing = {
        'claude_sonnet_4_5_20250514': {
            'input_cost_per_token': 0.000003,
            'output_cost_per_token': 0.000015,
        }
    }
    input_rate, _output_rate = find_rate(pricing, 'claude-sonnet-4-5-20250514')
    assert input_rate == 0.000003


def test_find_rate_unknown():
    input_rate, output_rate = find_rate({}, 'unknown-model')
    assert input_rate == 0.0
    assert output_rate == 0.0


def test_apply_overrides():
    pricing = {'model-a': {'input_cost_per_token': 0.001}}
    overrides = {'model-a': {'input_cost_per_token': 0.002}}
    merged = _apply_overrides(pricing, overrides)
    assert merged['model-a']['input_cost_per_token'] == 0.002


def test_apply_overrides_none():
    pricing = {'model-a': {'input_cost_per_token': 0.001}}
    assert _apply_overrides(pricing, None) is pricing


def test_find_rate_strips_provider_prefix():
    """A litellm-style `provider/model` key falls back to the unprefixed entry."""
    pricing = {
        'claude-sonnet-4-5-20250514': {
            'input_cost_per_token': 0.000003,
            'output_cost_per_token': 0.000015,
        }
    }
    input_rate, output_rate = find_rate(pricing, 'anthropic/claude-sonnet-4-5-20250514')
    assert input_rate == 0.000003
    assert output_rate == 0.000015


# --- Cache rates ---


_CACHE_FAKE_PRICING = {
    'claude-sonnet-4-5-20250514': {
        'input_cost_per_token': 0.000003,
        'output_cost_per_token': 0.000015,
        'cache_creation_input_token_cost': 0.00000375,
        'cache_read_input_token_cost': 0.0000003,
    },
    'gpt-test': {
        'input_cost_per_token': 0.000001,
        'output_cost_per_token': 0.000004,
        'cache_read_input_token_cost': 0.0000001,
    },
    'plain': {
        'input_cost_per_token': 0.000001,
        'output_cost_per_token': 0.000004,
    },
}


def test_find_cache_rates_explicit():
    creation, read = find_cache_rates(_CACHE_FAKE_PRICING, 'claude-sonnet-4-5-20250514')
    assert creation == 0.00000375
    assert read == 0.0000003


def test_find_cache_rates_falls_back_to_input_rate():
    """Models without prompt-caching rates fall back to the regular input rate, so cost math is conservative."""
    creation, read = find_cache_rates(_CACHE_FAKE_PRICING, 'plain')
    assert creation == 0.000001
    assert read == 0.000001


def test_find_cache_rates_partial():
    """A model with only cache_read pricing keeps the input fallback for creation."""
    creation, read = find_cache_rates(_CACHE_FAKE_PRICING, 'gpt-test')
    assert creation == 0.000001
    assert read == 0.0000001


# --- compute_cost ---


def test_compute_cost_no_cache_matches_legacy_formula():
    """With zero cache tokens, compute_cost reduces to prompt * input + completion * output."""
    usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    cost = compute_cost(_CACHE_FAKE_PRICING, 'claude-sonnet-4-5-20250514', usage)
    # 100 * 0.000003 + 50 * 0.000015 = 0.00105
    assert cost == pytest.approx(0.00105)


def test_compute_cost_splits_cache_buckets():
    """Cache reads price at the read rate, creation at the creation rate, remainder at the input rate."""
    usage = TokenUsage(
        prompt_tokens=1000,  # inclusive total
        completion_tokens=200,
        total_tokens=1200,
        cache_read_tokens=800,
        cache_creation_tokens=100,
    )
    # non_cached = 1000 - 800 - 100 = 100
    # 100 * 3e-6  + 100 * 3.75e-6 + 800 * 3e-7 + 200 * 1.5e-5
    # = 3e-4    + 3.75e-4    + 2.4e-4   + 3e-3
    expected = 100 * 3e-6 + 100 * 3.75e-6 + 800 * 3e-7 + 200 * 1.5e-5
    cost = compute_cost(_CACHE_FAKE_PRICING, 'claude-sonnet-4-5-20250514', usage)
    assert cost == pytest.approx(expected)


def test_compute_cost_cache_savings_are_significant():
    """Sanity check: with 90% cache hit rate, total cost drops to roughly a third."""
    no_cache = TokenUsage(prompt_tokens=1000, completion_tokens=100, total_tokens=1100)
    with_cache = TokenUsage(prompt_tokens=1000, completion_tokens=100, total_tokens=1100, cache_read_tokens=900)
    plain = compute_cost(_CACHE_FAKE_PRICING, 'claude-sonnet-4-5-20250514', no_cache)
    cached = compute_cost(_CACHE_FAKE_PRICING, 'claude-sonnet-4-5-20250514', with_cache)
    assert cached < plain * 0.5


def test_compute_cost_unknown_model_zero():
    usage = TokenUsage(prompt_tokens=100, completion_tokens=50, cache_read_tokens=10)
    assert compute_cost({}, 'unknown', usage) == 0.0


def test_compute_cost_components_sums_to_total():
    """compute_cost_components returns a dict whose values sum to the scalar compute_cost result."""
    usage = TokenUsage(
        prompt_tokens=1000, completion_tokens=200, total_tokens=1200, cache_read_tokens=600, cache_creation_tokens=100
    )
    components = compute_cost_components(_CACHE_FAKE_PRICING, 'claude-sonnet-4-5-20250514', usage)
    assert set(components.keys()) == {'input_usd', 'cache_creation_usd', 'cache_read_usd', 'output_usd'}
    total = compute_cost(_CACHE_FAKE_PRICING, 'claude-sonnet-4-5-20250514', usage)
    assert sum(components.values()) == pytest.approx(total)
    # Spot-check: 300 non-cached input * 3e-6 = 9e-4
    assert components['input_usd'] == pytest.approx(300 * 3e-6)
    # 600 cache reads * 3e-7 = 1.8e-4
    assert components['cache_read_usd'] == pytest.approx(600 * 3e-7)


# --- Token counting ---


def test_rough_token_count():
    assert _rough_token_count('hello world') >= 1
    assert _rough_token_count('a' * 400) == 100


# --- Estimator ---


def _setup_estimator_project(tmp_path: Path) -> None:
    """Create a minimal project for cost estimation tests."""
    (tmp_path / 'engram.yaml').write_text('name: test\n')

    wf_dir = tmp_path / 'workflows' / 'classify'
    wf_dir.mkdir(parents=True)
    (wf_dir / 'workflow.yaml').write_text('name: classify\noutput:\n  fields:\n    topic:\n      type: enum\n')

    impl_dir = tmp_path / 'implementations' / 'classify-api'
    impl_dir.mkdir(parents=True)
    (impl_dir / 'implementation.yaml').write_text(
        'workflow: classify\nplatform: api\nrunner: anthropic\n'
        'runner_config:\n  model: claude-sonnet-4-5-20250514\n  api_key_env: KEY\n'
    )
    prompts_dir = impl_dir / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'system.md').write_text('You are a classifier. Return JSON with topic field.')

    ds_dir = tmp_path / 'datasets' / 'test-ds'
    ds_dir.mkdir(parents=True)
    (ds_dir / 'dataset.yaml').write_text('name: test-ds\n')
    inputs_dir = ds_dir / 'inputs'
    inputs_dir.mkdir()
    (inputs_dir / '001.txt').write_text('Short input text')
    (inputs_dir / '002.txt').write_text('Another short input')

    (tmp_path / 'experiments').mkdir()


def test_estimate_cost(tmp_path: Path):
    _setup_estimator_project(tmp_path)

    fake_pricing = {
        'claude-sonnet-4-5-20250514': {
            'input_cost_per_token': 0.000003,
            'output_cost_per_token': 0.000015,
        }
    }

    with patch('engram.cost.estimator.load_pricing', return_value=fake_pricing):
        result = estimate_cost(tmp_path, 'classify-api', 'test-ds')

    assert result['model'] == 'claude-sonnet-4-5-20250514'
    assert result['total_examples'] == 2
    assert result['total_estimated_cost_usd'] > 0
    assert len(result['examples']) == 2
    assert result['examples'][0]['estimated_cost_usd'] > 0


def test_estimate_applies_project_pricing_overrides(tmp_path: Path):
    """Overrides declared in engram.yaml are merged into the pricing table."""
    _setup_estimator_project(tmp_path)

    # Override rates for the model the project uses: 10x input, 10x output.
    (tmp_path / 'engram.yaml').write_text(
        'name: test\n'
        'pricing_overrides:\n'
        '  claude-sonnet-4-5-20250514:\n'
        '    input_cost_per_token: 0.00003\n'
        '    output_cost_per_token: 0.00015\n'
    )

    baseline_pricing = {
        'claude-sonnet-4-5-20250514': {
            'input_cost_per_token': 0.000003,
            'output_cost_per_token': 0.000015,
        }
    }

    with patch('engram.cost.estimator.load_pricing') as mock_load:
        # Simulate load_pricing actually applying overrides by echoing the call.
        def _apply(overrides=None):
            merged = dict(baseline_pricing)
            if overrides:
                for model, rates in overrides.items():
                    merged[model] = {**merged[model], **rates}
            return merged

        mock_load.side_effect = _apply
        result = estimate_cost(tmp_path, 'classify-api', 'test-ds')

    # load_pricing must have been called with the project overrides.
    call_kwargs = mock_load.call_args.kwargs
    assert call_kwargs['overrides']['claude-sonnet-4-5-20250514']['input_cost_per_token'] == 0.00003
    # And the overridden rates flow through to the final estimate.
    assert result['input_rate_per_token'] == 0.00003
    assert result['output_rate_per_token'] == 0.00015


def test_estimate_uses_historical_calibration(tmp_path: Path):
    """With a prior experiment in the index, output tokens come from history, not the 500 default."""
    _setup_estimator_project(tmp_path)

    # Seed the index with a prior experiment recording 200 avg output tokens.
    index_entry = {
        'id': 'prior-exp',
        'implementation': 'classify-api',
        'dataset': 'test-ds',
        'timestamp': '2026-04-04T12:00:00Z',
        'avg_output_tokens': 200,
    }
    (tmp_path / 'experiments' / 'experiments.jsonl').write_text(json.dumps(index_entry) + '\n')

    fake_pricing = {
        'claude-sonnet-4-5-20250514': {
            'input_cost_per_token': 0.000003,
            'output_cost_per_token': 0.000015,
        }
    }

    with patch('engram.cost.estimator.load_pricing', return_value=fake_pricing):
        result = estimate_cost(tmp_path, 'classify-api', 'test-ds')

    assert result['avg_output_tokens'] == 200
    assert result['examples'][0]['estimated_output_tokens'] == 200


def _setup_cache_estimator_project(tmp_path: Path) -> None:
    """Estimator project with a large enough system prompt to trigger cache projection (>=1024 tokens)."""
    (tmp_path / 'engram.yaml').write_text('name: test\n')

    wf_dir = tmp_path / 'workflows' / 'classify'
    wf_dir.mkdir(parents=True)
    (wf_dir / 'workflow.yaml').write_text('name: classify\noutput:\n  fields:\n    topic:\n      type: enum\n')

    impl_dir = tmp_path / 'implementations' / 'classify-api'
    impl_dir.mkdir(parents=True)
    (impl_dir / 'implementation.yaml').write_text(
        'workflow: classify\nplatform: api\nrunner: anthropic\n'
        'runner_config:\n'
        '  model: claude-sonnet-4-5-20250514\n'
        '  api_key_env: KEY\n'
        '  prompt_cache: "true"\n'
    )
    prompts_dir = impl_dir / 'prompts'
    prompts_dir.mkdir()
    # ~5000 chars / 4 ≈ 1250 tokens, comfortably above the 1024-token cache threshold.
    (prompts_dir / 'system.md').write_text('x' * 5000)

    ds_dir = tmp_path / 'datasets' / 'test-ds'
    ds_dir.mkdir(parents=True)
    (ds_dir / 'dataset.yaml').write_text('name: test-ds\n')
    inputs_dir = ds_dir / 'inputs'
    inputs_dir.mkdir()
    (inputs_dir / '001.txt').write_text('short')
    (inputs_dir / '002.txt').write_text('short')
    (inputs_dir / '003.txt').write_text('short')

    (tmp_path / 'experiments').mkdir()


_CACHE_ESTIMATOR_PRICING = {
    'claude-sonnet-4-5-20250514': {
        'input_cost_per_token': 0.000003,
        'output_cost_per_token': 0.000015,
        'cache_creation_input_token_cost': 0.00000375,
        'cache_read_input_token_cost': 0.0000003,
    }
}


def test_estimate_projects_cache_savings(tmp_path: Path):
    """With prompt_cache enabled and a 1024+ token prompt, cache projection lowers the total."""
    _setup_cache_estimator_project(tmp_path)

    with patch('engram.cost.estimator.load_pricing', return_value=_CACHE_ESTIMATOR_PRICING):
        result = estimate_cost(tmp_path, 'classify-api', 'test-ds')

    # Cached total must be lower than the uncached comparison value the estimator emits.
    assert 'estimated_cost_without_cache_usd' in result
    assert result['total_estimated_cost_usd'] < result['estimated_cost_without_cache_usd']
    # The estimator must apply cache rates to the template tokens (read rate ≈ 10% of input rate),
    # so 2 of 3 runs hit the cache. Output-token cost is unaffected and dilutes the ratio.
    template_uncached = 1250 * 0.000003 * 3
    template_cached = 1250 * 0.00000375 + 1250 * 0.0000003 * 2
    expected_savings = template_uncached - template_cached
    actual_savings = result['estimated_cost_without_cache_usd'] - result['total_estimated_cost_usd']
    assert actual_savings == pytest.approx(expected_savings, rel=0.1)


def test_estimate_skips_cache_projection_below_threshold(tmp_path: Path):
    """A prompt under 1024 tokens leaves caching off in the estimate and warns the user."""
    _setup_estimator_project(tmp_path)  # short prompt, well below 1024 tokens

    # Override the impl to enable the flag.
    impl_yaml = tmp_path / 'implementations' / 'classify-api' / 'implementation.yaml'
    impl_yaml.write_text(impl_yaml.read_text().rstrip() + '\n  prompt_cache: "true"\n')

    with patch('engram.cost.estimator.load_pricing', return_value=_CACHE_ESTIMATOR_PRICING):
        result = estimate_cost(tmp_path, 'classify-api', 'test-ds')

    assert 'estimated_cost_without_cache_usd' not in result
    assert any('prompt_cache is enabled but no savings projected' in w for w in result['warnings'])


def test_estimate_command_json_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """With --json, estimate emits the full structured cost breakdown."""
    _setup_estimator_project(tmp_path)

    fake_pricing = {
        'claude-sonnet-4-5-20250514': {
            'input_cost_per_token': 0.000003,
            'output_cost_per_token': 0.000015,
        }
    }
    monkeypatch.chdir(tmp_path)

    with patch('engram.cost.estimator.load_pricing', return_value=fake_pricing):
        result = CliRunner().invoke(app, ['--json', 'estimate', 'classify-api', '--dataset', 'test-ds'])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload['implementation'] == 'classify-api'
    assert payload['dataset'] == 'test-ds'
    assert payload['model'] == 'claude-sonnet-4-5-20250514'
    assert payload['total_examples'] == 2
    assert payload['total_estimated_cost_usd'] > 0
    assert 'examples' in payload
    assert len(payload['examples']) == 2

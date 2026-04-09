"""Tests for cost estimation and pricing."""

import json
from pathlib import Path
from unittest.mock import patch

from engram.cost.estimator import _rough_token_count, estimate_cost
from engram.cost.pricing import _apply_overrides, find_rate

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

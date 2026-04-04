"""Tests for cost estimation and pricing."""

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

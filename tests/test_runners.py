"""Tests for runners."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import anthropic
import pytest

from engram.models.implementation import ConfigManagement, ImplementationConfig
from engram.runners.anthropic_agent import AnthropicAgentRunner
from engram.runners.anthropic_api import AnthropicApiRunner, _parse_json_output
from engram.runners.registry import get_runner

# --- Registry ---


def test_get_runner_anthropic():
    runner = get_runner('anthropic')
    assert isinstance(runner, AnthropicApiRunner)


def test_get_runner_agent():
    runner = get_runner('anthropic-agent')
    assert isinstance(runner, AnthropicAgentRunner)


def test_get_runner_unknown():
    with pytest.raises(ValueError, match='Unknown runner'):
        get_runner('nonexistent')


# --- JSON parsing ---


def test_parse_json_direct():
    assert _parse_json_output('{"topic": "A"}') == {'topic': 'A'}


def test_parse_json_markdown_fences():
    text = 'Here is the result:\n```json\n{"topic": "A"}\n```\n'
    assert _parse_json_output(text) == {'topic': 'A'}


def test_parse_json_invalid():
    assert _parse_json_output('not json at all') is None


def test_parse_json_array_rejected():
    assert _parse_json_output('[1, 2, 3]') is None


# --- Anthropic API Runner ---


def _make_impl_config(**overrides) -> ImplementationConfig:
    defaults = {
        'workflow': 'classify',
        'platform': 'api',
        'runner': 'anthropic',
        'runner_config': {
            'api_key_env': 'ANTHROPIC_API_KEY',
            'model': 'claude-sonnet-4-5-20250514',
            'max_tokens': '4096',
        },
        'config_management': ConfigManagement(mode='local'),
    }
    defaults.update(overrides)
    return ImplementationConfig(**defaults)


_FAKE_PRICING = {
    'claude-sonnet-4-5-20250514': {
        'input_cost_per_token': 0.000003,
        'output_cost_per_token': 0.000015,
    },
}


def test_api_runner_trigger(tmp_path: Path):
    # Set up prompt file
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'system.md').write_text('You are a classifier. Return JSON.')

    impl_config = _make_impl_config()

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"topic": "A", "sentiment": "Positive"}')]
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 50

    with (
        patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}),
        patch('engram.runners.anthropic_api.anthropic.Anthropic') as mock_cls,
        patch('engram.runners.anthropic_api.load_pricing', return_value=_FAKE_PRICING),
    ):
        mock_cls.return_value.messages.create.return_value = mock_response

        runner = AnthropicApiRunner()
        result = runner.trigger('some input', impl_config, tmp_path)

    assert result.status == 'succeeded'
    assert result.output == {'topic': 'A', 'sentiment': 'Positive'}
    assert result.usage.prompt_tokens == 100
    assert result.usage.completion_tokens == 50
    assert result.latency_ms > 0
    # 100 * 0.000003 + 50 * 0.000015 = 0.0003 + 0.00075 = 0.00105
    assert result.cost_usd == pytest.approx(0.00105)


def test_api_runner_trigger_unknown_model_zero_cost(tmp_path: Path):
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'system.md').write_text('prompt')

    impl_config = _make_impl_config(
        runner_config={
            'api_key_env': 'ANTHROPIC_API_KEY',
            'model': 'claude-unknown-model',
            'max_tokens': '4096',
        },
    )

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"topic": "A"}')]
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 50

    with (
        patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}),
        patch('engram.runners.anthropic_api.anthropic.Anthropic') as mock_cls,
        patch('engram.runners.anthropic_api.load_pricing', return_value=_FAKE_PRICING),
    ):
        mock_cls.return_value.messages.create.return_value = mock_response
        result = AnthropicApiRunner().trigger('input', impl_config, tmp_path)

    assert result.status == 'succeeded'
    assert result.cost_usd == 0.0


def test_api_runner_trigger_parse_failure_still_records_cost(tmp_path: Path):
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'system.md').write_text('prompt')

    impl_config = _make_impl_config()

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='not valid json at all')]
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 50

    with (
        patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}),
        patch('engram.runners.anthropic_api.anthropic.Anthropic') as mock_cls,
        patch('engram.runners.anthropic_api.load_pricing', return_value=_FAKE_PRICING),
    ):
        mock_cls.return_value.messages.create.return_value = mock_response
        result = AnthropicApiRunner().trigger('input', impl_config, tmp_path)

    assert result.status == 'failed'
    assert 'Failed to parse JSON' in result.error
    # Cost is charged even when parsing fails — the API call succeeded.
    assert result.cost_usd == pytest.approx(0.00105)


def test_api_runner_trigger_api_error(tmp_path: Path):
    """A raised anthropic.APIError yields a failed RunResult with the error captured."""
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'system.md').write_text('prompt')

    impl_config = _make_impl_config()

    with (
        patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}),
        patch('engram.runners.anthropic_api.anthropic.Anthropic') as mock_cls,
        patch('engram.runners.anthropic_api.load_pricing', return_value=_FAKE_PRICING),
    ):
        err = anthropic.APIError('rate limit exceeded', request=MagicMock(), body=None)
        mock_cls.return_value.messages.create.side_effect = err
        result = AnthropicApiRunner().trigger('input', impl_config, tmp_path)

    assert result.status == 'failed'
    assert 'rate limit exceeded' in result.error
    assert result.cost_usd == 0.0
    assert result.usage.prompt_tokens == 0


def test_api_runner_trigger_missing_system_prompt(tmp_path: Path):
    """Runner works when prompts/system.md is absent (empty system prompt)."""
    impl_config = _make_impl_config()

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"topic": "A"}')]
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 5

    with (
        patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}),
        patch('engram.runners.anthropic_api.anthropic.Anthropic') as mock_cls,
        patch('engram.runners.anthropic_api.load_pricing', return_value=_FAKE_PRICING),
    ):
        mock_cls.return_value.messages.create.return_value = mock_response
        result = AnthropicApiRunner().trigger('input', impl_config, tmp_path)
        # Verify the call went through with an empty system string.
        call_kwargs = mock_cls.return_value.messages.create.call_args.kwargs
        assert call_kwargs['system'] == ''

    assert result.status == 'succeeded'
    assert result.output == {'topic': 'A'}


def test_api_runner_trigger_empty_content(tmp_path: Path):
    """A response with no content blocks is treated as a parse failure, cost still recorded."""
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'system.md').write_text('prompt')

    impl_config = _make_impl_config()

    mock_response = MagicMock()
    mock_response.content = []
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 50

    with (
        patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}),
        patch('engram.runners.anthropic_api.anthropic.Anthropic') as mock_cls,
        patch('engram.runners.anthropic_api.load_pricing', return_value=_FAKE_PRICING),
    ):
        mock_cls.return_value.messages.create.return_value = mock_response
        result = AnthropicApiRunner().trigger('input', impl_config, tmp_path)

    assert result.status == 'failed'
    assert 'Failed to parse JSON' in result.error
    assert result.cost_usd == pytest.approx(0.00105)


def test_api_runner_trigger_non_dict_json(tmp_path: Path):
    """An array-shaped JSON response fails at trigger level (scorers expect a dict)."""
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'system.md').write_text('prompt')

    impl_config = _make_impl_config()

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='[1, 2, 3]')]
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 5

    with (
        patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}),
        patch('engram.runners.anthropic_api.anthropic.Anthropic') as mock_cls,
        patch('engram.runners.anthropic_api.load_pricing', return_value=_FAKE_PRICING),
    ):
        mock_cls.return_value.messages.create.return_value = mock_response
        result = AnthropicApiRunner().trigger('input', impl_config, tmp_path)

    assert result.status == 'failed'
    assert 'Failed to parse JSON' in result.error


def test_api_runner_configure_pricing_overrides_rates(tmp_path: Path):
    """configure_pricing pre-loads the table, and overrides take effect on cost."""
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'system.md').write_text('prompt')

    impl_config = _make_impl_config()

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"topic": "A"}')]
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 50

    # Doubled rates vs _FAKE_PRICING: 0.000006 input, 0.00003 output.
    overridden = {
        'claude-sonnet-4-5-20250514': {
            'input_cost_per_token': 0.000006,
            'output_cost_per_token': 0.00003,
        },
    }

    with (
        patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}),
        patch('engram.runners.anthropic_api.anthropic.Anthropic') as mock_cls,
        patch('engram.runners.anthropic_api.load_pricing', return_value=overridden) as mock_load,
    ):
        mock_cls.return_value.messages.create.return_value = mock_response
        runner = AnthropicApiRunner()
        runner.configure_pricing({'claude-sonnet-4-5-20250514': {'input_cost_per_token': 0.000006}})
        result = runner.trigger('input', impl_config, tmp_path)

    # configure_pricing forwarded the overrides to load_pricing.
    assert mock_load.call_args.kwargs['overrides']['claude-sonnet-4-5-20250514']['input_cost_per_token'] == 0.000006
    # And the cached table is used for the cost calculation:
    # 100 * 0.000006 + 50 * 0.00003 = 0.0006 + 0.0015 = 0.0021
    assert result.cost_usd == pytest.approx(0.0021)


def test_api_runner_snapshot(tmp_path: Path):
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'system.md').write_text('You are a classifier.')

    impl_config = _make_impl_config()
    runner = AnthropicApiRunner()
    snap = runner.snapshot_config(impl_config, tmp_path)

    assert snap.models == ['claude-sonnet-4-5-20250514']
    assert 'system.md' in snap.prompts
    assert 'api_key_env' not in snap.runner_config


# --- Agent Runner ---


def test_agent_runner_trigger(tmp_path: Path):
    # Write a simple agent file
    agent_code = """
def classify(input_data):
    return {
        'output': {'topic': 'A'},
        'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
        'cost_usd': 0.001,
    }
"""
    (tmp_path / 'agent.py').write_text(agent_code)

    impl_config = ImplementationConfig(
        workflow='classify',
        platform='agent',
        runner='anthropic-agent',
        runner_config={'entry_point': 'agent.py:classify', 'model': 'claude-sonnet-4-5-20250514'},
    )

    runner = AnthropicAgentRunner()
    result = runner.trigger('some input', impl_config, tmp_path)

    assert result.status == 'succeeded'
    assert result.output == {'topic': 'A'}
    assert result.usage.total_tokens == 15
    assert result.cost_usd == 0.001


def test_agent_runner_bad_entry_point():
    impl_config = ImplementationConfig(
        workflow='classify',
        platform='agent',
        runner='anthropic-agent',
        runner_config={'entry_point': 'no_colon'},
    )

    runner = AnthropicAgentRunner()
    result = runner.trigger('input', impl_config, Path('/tmp'))

    assert result.status == 'failed'
    assert 'Invalid entry_point' in result.error


def test_agent_runner_missing_file(tmp_path: Path):
    impl_config = ImplementationConfig(
        workflow='classify',
        platform='agent',
        runner='anthropic-agent',
        runner_config={'entry_point': 'missing.py:func'},
    )

    runner = AnthropicAgentRunner()
    result = runner.trigger('input', impl_config, tmp_path)

    assert result.status == 'failed'
    assert 'not found' in result.error

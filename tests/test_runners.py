"""Tests for runners."""

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import anthropic
import litellm
import openai
import pytest

from engram.models.implementation import ConfigManagement, ImplementationConfig
from engram.models.input import InputData
from engram.runners.anthropic_agent import AnthropicAgentRunner
from engram.runners.anthropic_api import AnthropicApiRunner, _parse_json_output
from engram.runners.errors import MissingAPIKeyError
from engram.runners.litellm_api import LiteLLMRunner
from engram.runners.openai_api import OpenAIApiRunner
from engram.runners.registry import get_runner

# --- Registry ---


def test_get_runner_anthropic():
    runner = get_runner('anthropic')
    assert isinstance(runner, AnthropicApiRunner)


def test_get_runner_agent():
    runner = get_runner('anthropic-agent')
    assert isinstance(runner, AnthropicAgentRunner)


def test_get_runner_openai():
    runner = get_runner('openai')
    assert isinstance(runner, OpenAIApiRunner)


def test_get_runner_litellm():
    runner = get_runner('litellm')
    assert isinstance(runner, LiteLLMRunner)


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


def _make_impl_config(**overrides: object) -> ImplementationConfig:
    defaults: dict[str, object] = {
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
    return ImplementationConfig(
        workflow=str(defaults['workflow']),
        platform=str(defaults['platform']),
        runner=str(defaults['runner']),
        runner_config=cast('dict[str, str]', defaults.get('runner_config', {})),
        config_management=cast('ConfigManagement', defaults.get('config_management', ConfigManagement())),
    )


_FAKE_PRICING = {
    'claude-sonnet-4-5-20250514': {
        'input_cost_per_token': 0.000003,
        'output_cost_per_token': 0.000015,
    },
}


def test_api_runner_prompt_cache_flag_sends_cache_control(tmp_path: Path):
    """With prompt_cache=true the system prompt is sent as a cache-marked block."""
    (tmp_path / 'prompts').mkdir()
    (tmp_path / 'prompts' / 'system.md').write_text('long stable system prompt')

    impl_config = _make_impl_config(
        runner_config={
            'api_key_env': 'ANTHROPIC_API_KEY',
            'model': 'claude-sonnet-4-5-20250514',
            'max_tokens': '4096',
            'prompt_cache': 'true',
        },
    )

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
        AnthropicApiRunner().trigger(InputData(filename='test', text='input'), impl_config, tmp_path)

        call_kwargs = mock_cls.return_value.messages.create.call_args.kwargs
        assert call_kwargs['system'] == [
            {
                'type': 'text',
                'text': 'long stable system prompt',
                'cache_control': {'type': 'ephemeral'},
            },
        ]


def test_api_runner_prompt_cache_off_sends_plain_string(tmp_path: Path):
    """Without the flag the system prompt is still passed as a plain string (no behavior change)."""
    (tmp_path / 'prompts').mkdir()
    (tmp_path / 'prompts' / 'system.md').write_text('prompt')

    impl_config = _make_impl_config()  # no prompt_cache in runner_config
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
        AnthropicApiRunner().trigger(InputData(filename='test', text='input'), impl_config, tmp_path)

        call_kwargs = mock_cls.return_value.messages.create.call_args.kwargs
        assert call_kwargs['system'] == 'prompt'


def test_api_runner_records_cost_without_cache(tmp_path: Path):
    """RunResult.cost_without_cache_usd is the counterfactual price at full input rate."""
    (tmp_path / 'prompts').mkdir()
    (tmp_path / 'prompts' / 'system.md').write_text('prompt')

    impl_config = _make_impl_config()

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"topic": "A"}')]
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 50
    mock_response.usage.cache_creation_input_tokens = 200
    mock_response.usage.cache_read_input_tokens = 700

    fake_pricing = {
        'claude-sonnet-4-5-20250514': {
            'input_cost_per_token': 0.000003,
            'output_cost_per_token': 0.000015,
            'cache_creation_input_token_cost': 0.00000375,
            'cache_read_input_token_cost': 0.0000003,
        },
    }

    with (
        patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}),
        patch('engram.runners.anthropic_api.anthropic.Anthropic') as mock_cls,
        patch('engram.runners.anthropic_api.load_pricing', return_value=fake_pricing),
    ):
        mock_cls.return_value.messages.create.return_value = mock_response
        result = AnthropicApiRunner().trigger(InputData(filename='test', text='input'), impl_config, tmp_path)

    # prompt_total = 1000 (100 + 200 + 700). At full input rate: 1000 * 3e-6 + 50 * 1.5e-5 = 0.00375
    assert result.cost_without_cache_usd == pytest.approx(1000 * 3e-6 + 50 * 1.5e-5)
    # And actual cost is lower because most tokens hit the cache-read rate.
    assert result.cost_usd < result.cost_without_cache_usd


def test_api_runner_extracts_cache_tokens_and_prices_them(tmp_path: Path):
    """Anthropic reports cache_creation + cache_read additively; runner normalizes and the cost helper prices them."""
    (tmp_path / 'prompts').mkdir()
    (tmp_path / 'prompts' / 'system.md').write_text('long stable system prompt')

    impl_config = _make_impl_config()

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"topic": "A"}')]
    mock_response.usage.input_tokens = 100  # non-cached input
    mock_response.usage.output_tokens = 50
    mock_response.usage.cache_creation_input_tokens = 200
    mock_response.usage.cache_read_input_tokens = 700

    fake_pricing = {
        'claude-sonnet-4-5-20250514': {
            'input_cost_per_token': 0.000003,
            'output_cost_per_token': 0.000015,
            'cache_creation_input_token_cost': 0.00000375,
            'cache_read_input_token_cost': 0.0000003,
        },
    }

    with (
        patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}),
        patch('engram.runners.anthropic_api.anthropic.Anthropic') as mock_cls,
        patch('engram.runners.anthropic_api.load_pricing', return_value=fake_pricing),
    ):
        mock_cls.return_value.messages.create.return_value = mock_response
        result = AnthropicApiRunner().trigger(InputData(filename='test', text='input'), impl_config, tmp_path)

    # prompt_tokens is the inclusive total (engram convention), even though Anthropic
    # reports the three buckets separately.
    assert result.usage.prompt_tokens == 1000
    assert result.usage.cache_creation_tokens == 200
    assert result.usage.cache_read_tokens == 700
    # 100 non-cached * 3e-6 + 200 creation * 3.75e-6 + 700 read * 3e-7 + 50 output * 1.5e-5
    expected = 100 * 3e-6 + 200 * 3.75e-6 + 700 * 3e-7 + 50 * 1.5e-5
    assert result.cost_usd == pytest.approx(expected)


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
        result = runner.trigger(InputData(filename='test', text='some input'), impl_config, tmp_path)

    assert result.status == 'succeeded'
    assert result.output == {'topic': 'A', 'sentiment': 'Positive'}
    assert result.usage.prompt_tokens == 100
    assert result.usage.completion_tokens == 50
    assert result.latency_ms > 0
    # 100 * 0.000003 + 50 * 0.000015 = 0.0003 + 0.00075 = 0.00105
    assert result.cost_usd == pytest.approx(0.00105)


def test_api_runner_defaults_temperature_to_zero(tmp_path: Path):
    """When runner_config has no temperature, the API call must receive temperature=0.0."""
    (tmp_path / 'prompts').mkdir()
    (tmp_path / 'prompts' / 'system.md').write_text('prompt')

    impl_config = _make_impl_config()  # no temperature in runner_config

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"topic": "A"}')]
    mock_response.usage.input_tokens = 1
    mock_response.usage.output_tokens = 1

    with (
        patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}),
        patch('engram.runners.anthropic_api.anthropic.Anthropic') as mock_cls,
        patch('engram.runners.anthropic_api.load_pricing', return_value=_FAKE_PRICING),
    ):
        mock_cls.return_value.messages.create.return_value = mock_response
        AnthropicApiRunner().trigger(InputData(filename='test', text='input'), impl_config, tmp_path)

        call_kwargs = mock_cls.return_value.messages.create.call_args.kwargs
        assert call_kwargs['temperature'] == 0.0


def test_api_runner_forwards_explicit_temperature(tmp_path: Path):
    """An explicit temperature in runner_config is forwarded as a float."""
    (tmp_path / 'prompts').mkdir()
    (tmp_path / 'prompts' / 'system.md').write_text('prompt')

    impl_config = _make_impl_config(
        runner_config={
            'api_key_env': 'ANTHROPIC_API_KEY',
            'model': 'claude-sonnet-4-5-20250514',
            'max_tokens': '4096',
            'temperature': '0.7',
        },
    )

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"topic": "A"}')]
    mock_response.usage.input_tokens = 1
    mock_response.usage.output_tokens = 1

    with (
        patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}),
        patch('engram.runners.anthropic_api.anthropic.Anthropic') as mock_cls,
        patch('engram.runners.anthropic_api.load_pricing', return_value=_FAKE_PRICING),
    ):
        mock_cls.return_value.messages.create.return_value = mock_response
        AnthropicApiRunner().trigger(InputData(filename='test', text='input'), impl_config, tmp_path)

        call_kwargs = mock_cls.return_value.messages.create.call_args.kwargs
        assert call_kwargs['temperature'] == 0.7


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
        result = AnthropicApiRunner().trigger(InputData(filename='test', text='input'), impl_config, tmp_path)

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
        result = AnthropicApiRunner().trigger(InputData(filename='test', text='input'), impl_config, tmp_path)

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
        result = AnthropicApiRunner().trigger(InputData(filename='test', text='input'), impl_config, tmp_path)

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
        result = AnthropicApiRunner().trigger(InputData(filename='test', text='input'), impl_config, tmp_path)
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
        result = AnthropicApiRunner().trigger(InputData(filename='test', text='input'), impl_config, tmp_path)

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
        result = AnthropicApiRunner().trigger(InputData(filename='test', text='input'), impl_config, tmp_path)

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
        result = runner.trigger(InputData(filename='test', text='input'), impl_config, tmp_path)

    # configure_pricing forwarded the overrides to load_pricing.
    assert mock_load.call_args.kwargs['overrides']['claude-sonnet-4-5-20250514']['input_cost_per_token'] == 0.000006
    # And the cached table is used for the cost calculation:
    # 100 * 0.000006 + 50 * 0.00003 = 0.0006 + 0.0015 = 0.0021
    assert result.cost_usd == pytest.approx(0.0021)


def test_api_runner_required_env_vars():
    """AnthropicApiRunner reports its api_key_env as required."""
    impl_config = _make_impl_config()
    assert AnthropicApiRunner().required_env_vars(impl_config) == ['ANTHROPIC_API_KEY']


def test_api_runner_missing_key_raises_friendly_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """When ANTHROPIC_API_KEY is unset, trigger raises MissingAPIKeyError (not KeyError)."""
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)

    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'system.md').write_text('prompt')

    impl_config = _make_impl_config()
    with pytest.raises(MissingAPIKeyError) as excinfo:
        AnthropicApiRunner().trigger(InputData(filename='test', text='input'), impl_config, tmp_path)

    assert excinfo.value.env_var == 'ANTHROPIC_API_KEY'
    assert 'ANTHROPIC_API_KEY' in str(excinfo.value)


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
    result = runner.trigger(InputData(filename='test', text='some input'), impl_config, tmp_path)

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
    result = runner.trigger(InputData(filename='test', text='input'), impl_config, Path('/tmp'))

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
    result = runner.trigger(InputData(filename='test', text='input'), impl_config, tmp_path)

    assert result.status == 'failed'
    assert 'not found' in result.error


# --- OpenAI API Runner ---


def _make_openai_impl_config(**overrides: object) -> ImplementationConfig:
    defaults: dict[str, object] = {
        'workflow': 'classify',
        'platform': 'api',
        'runner': 'openai',
        'runner_config': {
            'api_key_env': 'OPENAI_API_KEY',
            'model': 'gpt-5.4-mini',
            'max_tokens': '4096',
        },
        'config_management': ConfigManagement(mode='local'),
    }
    defaults.update(overrides)
    return ImplementationConfig(
        workflow=str(defaults['workflow']),
        platform=str(defaults['platform']),
        runner=str(defaults['runner']),
        runner_config=cast('dict[str, str]', defaults.get('runner_config', {})),
        config_management=cast('ConfigManagement', defaults.get('config_management', ConfigManagement())),
    )


_OPENAI_FAKE_PRICING = {
    'gpt-5.4-mini': {
        'input_cost_per_token': 0.0000001,
        'output_cost_per_token': 0.0000004,
    },
}


def _make_openai_response(content: str, prompt_tokens: int = 100, completion_tokens: int = 50) -> MagicMock:
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = prompt_tokens
    mock_response.usage.completion_tokens = completion_tokens
    mock_response.usage.total_tokens = prompt_tokens + completion_tokens
    return mock_response


def test_openai_runner_extracts_cached_tokens(tmp_path: Path):
    """OpenAI reports cached_tokens as a subset of prompt_tokens via prompt_tokens_details."""
    (tmp_path / 'prompts').mkdir()
    (tmp_path / 'prompts' / 'system.md').write_text('prompt')

    impl_config = _make_openai_impl_config()
    # prompt_tokens (1000) already INCLUDES cached_tokens (900).
    mock_response = _make_openai_response('{"topic": "A"}', prompt_tokens=1000, completion_tokens=50)
    mock_response.usage.prompt_tokens_details.cached_tokens = 900

    fake_pricing = {
        'gpt-5.4-mini': {
            'input_cost_per_token': 0.0000001,
            'output_cost_per_token': 0.0000004,
            'cache_read_input_token_cost': 0.00000001,
        },
    }

    with (
        patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}),
        patch('engram.runners.openai_api.openai.OpenAI') as mock_cls,
        patch('engram.runners.openai_api.load_pricing', return_value=fake_pricing),
    ):
        mock_cls.return_value.chat.completions.create.return_value = mock_response
        result = OpenAIApiRunner().trigger(InputData(filename='test', text='input'), impl_config, tmp_path)

    # prompt_tokens stays as OpenAI reported it (already inclusive of cached).
    assert result.usage.prompt_tokens == 1000
    assert result.usage.cache_read_tokens == 900
    # 100 non-cached * 1e-7 + 900 read * 1e-8 + 50 output * 4e-7 = 0.0000390
    expected = 100 * 1e-7 + 900 * 1e-8 + 50 * 4e-7
    assert result.cost_usd == pytest.approx(expected)


def test_openai_runner_trigger(tmp_path: Path):
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'system.md').write_text('You are a classifier. Return JSON.')

    impl_config = _make_openai_impl_config()
    mock_response = _make_openai_response('{"topic": "A", "sentiment": "Positive"}', 100, 50)

    with (
        patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}),
        patch('engram.runners.openai_api.openai.OpenAI') as mock_cls,
        patch('engram.runners.openai_api.load_pricing', return_value=_OPENAI_FAKE_PRICING),
    ):
        mock_cls.return_value.chat.completions.create.return_value = mock_response
        runner = OpenAIApiRunner()
        result = runner.trigger(InputData(filename='test', text='some input'), impl_config, tmp_path)

    assert result.status == 'succeeded'
    assert result.output == {'topic': 'A', 'sentiment': 'Positive'}
    assert result.usage.prompt_tokens == 100
    assert result.usage.completion_tokens == 50
    assert result.latency_ms > 0
    # 100 * 0.0000001 + 50 * 0.0000004 = 0.00001 + 0.00002 = 0.00003
    assert result.cost_usd == pytest.approx(0.00003)


def test_openai_runner_defaults_temperature_to_zero(tmp_path: Path):
    """When runner_config has no temperature, the API call must receive temperature=0.0."""
    (tmp_path / 'prompts').mkdir()
    (tmp_path / 'prompts' / 'system.md').write_text('prompt')

    impl_config = _make_openai_impl_config()  # no temperature in runner_config
    mock_response = _make_openai_response('{"topic": "A"}')

    with (
        patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}),
        patch('engram.runners.openai_api.openai.OpenAI') as mock_cls,
        patch('engram.runners.openai_api.load_pricing', return_value=_OPENAI_FAKE_PRICING),
    ):
        mock_cls.return_value.chat.completions.create.return_value = mock_response
        OpenAIApiRunner().trigger(InputData(filename='test', text='input'), impl_config, tmp_path)

        call_kwargs = mock_cls.return_value.chat.completions.create.call_args.kwargs
        assert call_kwargs['temperature'] == 0.0


def test_openai_runner_forwards_explicit_temperature(tmp_path: Path):
    """An explicit temperature in runner_config is forwarded as a float."""
    (tmp_path / 'prompts').mkdir()
    (tmp_path / 'prompts' / 'system.md').write_text('prompt')

    impl_config = _make_openai_impl_config(
        runner_config={
            'api_key_env': 'OPENAI_API_KEY',
            'model': 'gpt-5.4-mini',
            'max_tokens': '4096',
            'temperature': '0.5',
        },
    )
    mock_response = _make_openai_response('{"topic": "A"}')

    with (
        patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}),
        patch('engram.runners.openai_api.openai.OpenAI') as mock_cls,
        patch('engram.runners.openai_api.load_pricing', return_value=_OPENAI_FAKE_PRICING),
    ):
        mock_cls.return_value.chat.completions.create.return_value = mock_response
        OpenAIApiRunner().trigger(InputData(filename='test', text='input'), impl_config, tmp_path)

        call_kwargs = mock_cls.return_value.chat.completions.create.call_args.kwargs
        assert call_kwargs['temperature'] == 0.5


def test_openai_runner_requests_json_mode(tmp_path: Path):
    """trigger() must pass response_format={'type': 'json_object'}."""
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'system.md').write_text('prompt')

    impl_config = _make_openai_impl_config()
    mock_response = _make_openai_response('{"topic": "A"}')

    with (
        patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}),
        patch('engram.runners.openai_api.openai.OpenAI') as mock_cls,
        patch('engram.runners.openai_api.load_pricing', return_value=_OPENAI_FAKE_PRICING),
    ):
        mock_cls.return_value.chat.completions.create.return_value = mock_response
        OpenAIApiRunner().trigger(InputData(filename='test', text='input'), impl_config, tmp_path)

        call_kwargs = mock_cls.return_value.chat.completions.create.call_args.kwargs
        assert call_kwargs['response_format'] == {'type': 'json_object'}
        assert call_kwargs['max_completion_tokens'] == 4096
        # System + user messages, in order.
        assert call_kwargs['messages'][0]['role'] == 'system'
        assert call_kwargs['messages'][1]['role'] == 'user'
        assert call_kwargs['messages'][1]['content'] == 'input'


def test_openai_runner_trigger_unknown_model_zero_cost(tmp_path: Path):
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'system.md').write_text('prompt')

    impl_config = _make_openai_impl_config(
        runner_config={
            'api_key_env': 'OPENAI_API_KEY',
            'model': 'gpt-imaginary',
            'max_tokens': '4096',
        },
    )
    mock_response = _make_openai_response('{"topic": "A"}')

    with (
        patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}),
        patch('engram.runners.openai_api.openai.OpenAI') as mock_cls,
        patch('engram.runners.openai_api.load_pricing', return_value=_OPENAI_FAKE_PRICING),
    ):
        mock_cls.return_value.chat.completions.create.return_value = mock_response
        result = OpenAIApiRunner().trigger(InputData(filename='test', text='input'), impl_config, tmp_path)

    assert result.status == 'succeeded'
    assert result.cost_usd == 0.0


def test_openai_runner_trigger_parse_failure_still_records_cost(tmp_path: Path):
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'system.md').write_text('prompt')

    impl_config = _make_openai_impl_config()
    mock_response = _make_openai_response('not json at all', 100, 50)

    with (
        patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}),
        patch('engram.runners.openai_api.openai.OpenAI') as mock_cls,
        patch('engram.runners.openai_api.load_pricing', return_value=_OPENAI_FAKE_PRICING),
    ):
        mock_cls.return_value.chat.completions.create.return_value = mock_response
        result = OpenAIApiRunner().trigger(InputData(filename='test', text='input'), impl_config, tmp_path)

    assert result.status == 'failed'
    assert 'Failed to parse JSON' in result.error
    assert result.cost_usd == pytest.approx(0.00003)


def test_openai_runner_trigger_api_error(tmp_path: Path):
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'system.md').write_text('prompt')

    impl_config = _make_openai_impl_config()

    with (
        patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}),
        patch('engram.runners.openai_api.openai.OpenAI') as mock_cls,
        patch('engram.runners.openai_api.load_pricing', return_value=_OPENAI_FAKE_PRICING),
    ):
        err = openai.APIError('rate limit exceeded', MagicMock(), body=None)
        mock_cls.return_value.chat.completions.create.side_effect = err
        result = OpenAIApiRunner().trigger(InputData(filename='test', text='input'), impl_config, tmp_path)

    assert result.status == 'failed'
    assert 'rate limit exceeded' in result.error
    assert result.cost_usd == 0.0
    assert result.usage.prompt_tokens == 0


def test_openai_runner_trigger_missing_system_prompt(tmp_path: Path):
    """With no prompts/system.md, the runner omits the system message entirely."""
    impl_config = _make_openai_impl_config()
    mock_response = _make_openai_response('{"topic": "A"}')

    with (
        patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}),
        patch('engram.runners.openai_api.openai.OpenAI') as mock_cls,
        patch('engram.runners.openai_api.load_pricing', return_value=_OPENAI_FAKE_PRICING),
    ):
        mock_cls.return_value.chat.completions.create.return_value = mock_response
        result = OpenAIApiRunner().trigger(InputData(filename='test', text='input'), impl_config, tmp_path)

        call_kwargs = mock_cls.return_value.chat.completions.create.call_args.kwargs
        # Only the user message — no system role, since OpenAI rejects empty-string system content.
        assert len(call_kwargs['messages']) == 1
        assert call_kwargs['messages'][0]['role'] == 'user'

    assert result.status == 'succeeded'


def test_openai_runner_trigger_empty_content(tmp_path: Path):
    """A response with no choices is treated as a parse failure, cost still recorded."""
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'system.md').write_text('prompt')

    impl_config = _make_openai_impl_config()
    mock_response = MagicMock()
    mock_response.choices = []
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    mock_response.usage.total_tokens = 150

    with (
        patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}),
        patch('engram.runners.openai_api.openai.OpenAI') as mock_cls,
        patch('engram.runners.openai_api.load_pricing', return_value=_OPENAI_FAKE_PRICING),
    ):
        mock_cls.return_value.chat.completions.create.return_value = mock_response
        result = OpenAIApiRunner().trigger(InputData(filename='test', text='input'), impl_config, tmp_path)

    assert result.status == 'failed'
    assert 'Failed to parse JSON' in result.error
    assert result.cost_usd == pytest.approx(0.00003)


def test_openai_runner_required_env_vars():
    impl_config = _make_openai_impl_config()
    assert OpenAIApiRunner().required_env_vars(impl_config) == ['OPENAI_API_KEY']


def test_openai_runner_missing_key_raises_friendly_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)

    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'system.md').write_text('prompt')

    impl_config = _make_openai_impl_config()
    with pytest.raises(MissingAPIKeyError) as excinfo:
        OpenAIApiRunner().trigger(InputData(filename='test', text='input'), impl_config, tmp_path)

    assert excinfo.value.env_var == 'OPENAI_API_KEY'


def test_openai_runner_snapshot(tmp_path: Path):
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'system.md').write_text('You are a classifier.')

    impl_config = _make_openai_impl_config()
    runner = OpenAIApiRunner()
    snap = runner.snapshot_config(impl_config, tmp_path)

    assert snap.models == ['gpt-5.4-mini']
    assert 'system.md' in snap.prompts
    assert 'api_key_env' not in snap.runner_config


def test_openai_runner_configure_pricing_overrides_rates(tmp_path: Path):
    """configure_pricing pre-loads pricing with overrides, and cost reflects them."""
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'system.md').write_text('prompt')

    impl_config = _make_openai_impl_config()
    mock_response = _make_openai_response('{"topic": "A"}', 100, 50)

    # Doubled rates vs _OPENAI_FAKE_PRICING.
    overridden = {
        'gpt-5.4-mini': {
            'input_cost_per_token': 0.0000002,
            'output_cost_per_token': 0.0000008,
        },
    }

    with (
        patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}),
        patch('engram.runners.openai_api.openai.OpenAI') as mock_cls,
        patch('engram.runners.openai_api.load_pricing', return_value=overridden) as mock_load,
    ):
        mock_cls.return_value.chat.completions.create.return_value = mock_response
        runner = OpenAIApiRunner()
        runner.configure_pricing({'gpt-5.4-mini': {'input_cost_per_token': 0.0000002}})
        result = runner.trigger(InputData(filename='test', text='input'), impl_config, tmp_path)

    assert mock_load.call_args.kwargs['overrides']['gpt-5.4-mini']['input_cost_per_token'] == 0.0000002
    # 100 * 0.0000002 + 50 * 0.0000008 = 0.00002 + 0.00004 = 0.00006
    assert result.cost_usd == pytest.approx(0.00006)


# --- LiteLLM Runner ---


def _make_litellm_impl_config(**overrides: object) -> ImplementationConfig:
    defaults: dict[str, object] = {
        'workflow': 'classify',
        'platform': 'api',
        'runner': 'litellm',
        'runner_config': {
            'api_key_env': 'GEMINI_API_KEY',
            'model': 'gemini/gemini-2.0-flash',
            'max_tokens': '4096',
        },
        'config_management': ConfigManagement(mode='local'),
    }
    defaults.update(overrides)
    return ImplementationConfig(
        workflow=str(defaults['workflow']),
        platform=str(defaults['platform']),
        runner=str(defaults['runner']),
        runner_config=cast('dict[str, str]', defaults.get('runner_config', {})),
        config_management=cast('ConfigManagement', defaults.get('config_management', ConfigManagement())),
    )


_LITELLM_FAKE_PRICING = {
    'gemini/gemini-2.0-flash': {
        'input_cost_per_token': 0.0000001,
        'output_cost_per_token': 0.0000004,
    },
    'claude-sonnet-4-5-20250514': {
        'input_cost_per_token': 0.000003,
        'output_cost_per_token': 0.000015,
    },
}


def _make_litellm_response(content: str, prompt_tokens: int = 100, completion_tokens: int = 50) -> MagicMock:
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = prompt_tokens
    mock_response.usage.completion_tokens = completion_tokens
    mock_response.usage.total_tokens = prompt_tokens + completion_tokens
    return mock_response


def test_litellm_runner_trigger(tmp_path: Path):
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'system.md').write_text('You are a classifier. Return JSON.')

    impl_config = _make_litellm_impl_config()
    mock_response = _make_litellm_response('{"topic": "A", "sentiment": "Positive"}', 100, 50)

    with (
        patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}),
        patch('engram.runners.litellm_api.litellm.completion', return_value=mock_response) as mock_completion,
        patch('engram.runners.litellm_api.load_pricing', return_value=_LITELLM_FAKE_PRICING),
    ):
        result = LiteLLMRunner().trigger(InputData(filename='test', text='some input'), impl_config, tmp_path)

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs['model'] == 'gemini/gemini-2.0-flash'
        assert call_kwargs['api_key'] == 'test-key'
        assert call_kwargs['drop_params'] is True
        assert call_kwargs['response_format'] == {'type': 'json_object'}
        assert call_kwargs['messages'][0]['role'] == 'system'
        assert call_kwargs['messages'][1]['role'] == 'user'

    assert result.status == 'succeeded'
    assert result.output == {'topic': 'A', 'sentiment': 'Positive'}
    assert result.usage.prompt_tokens == 100
    # 100 * 0.0000001 + 50 * 0.0000004 = 0.00003
    assert result.cost_usd == pytest.approx(0.00003)


def test_litellm_runner_without_api_key_env(tmp_path: Path):
    """When api_key_env is absent, no api_key is forwarded — LiteLLM resolves the env var itself."""
    (tmp_path / 'prompts').mkdir()
    (tmp_path / 'prompts' / 'system.md').write_text('prompt')

    impl_config = _make_litellm_impl_config(
        runner_config={'model': 'gemini/gemini-2.0-flash', 'max_tokens': '4096'},
    )
    mock_response = _make_litellm_response('{"topic": "A"}')

    with (
        patch('engram.runners.litellm_api.litellm.completion', return_value=mock_response) as mock_completion,
        patch('engram.runners.litellm_api.load_pricing', return_value=_LITELLM_FAKE_PRICING),
    ):
        result = LiteLLMRunner().trigger(InputData(filename='test', text='input'), impl_config, tmp_path)

        assert 'api_key' not in mock_completion.call_args.kwargs

    assert result.status == 'succeeded'


def test_litellm_runner_prompt_cache_flag_wraps_system_in_cache_block(tmp_path: Path):
    """With prompt_cache=true the system message is sent as a content-block list with cache_control."""
    (tmp_path / 'prompts').mkdir()
    (tmp_path / 'prompts' / 'system.md').write_text('long stable system prompt')

    impl_config = _make_litellm_impl_config(
        runner_config={
            'api_key_env': 'ANTHROPIC_API_KEY',
            'model': 'anthropic/claude-sonnet-4-5-20250514',
            'max_tokens': '4096',
            'prompt_cache': 'true',
        },
    )
    mock_response = _make_litellm_response('{"topic": "A"}')

    with (
        patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}),
        patch('engram.runners.litellm_api.litellm.completion', return_value=mock_response) as mock_completion,
        patch('engram.runners.litellm_api.load_pricing', return_value=_LITELLM_FAKE_PRICING),
    ):
        LiteLLMRunner().trigger(InputData(filename='test', text='input'), impl_config, tmp_path)

        messages = mock_completion.call_args.kwargs['messages']
        assert messages[0] == {
            'role': 'system',
            'content': [
                {
                    'type': 'text',
                    'text': 'long stable system prompt',
                    'cache_control': {'type': 'ephemeral'},
                },
            ],
        }


def test_litellm_runner_extracts_cached_tokens(tmp_path: Path):
    """LiteLLM normalizes to OpenAI shape; cached tokens flow through prompt_tokens_details."""
    (tmp_path / 'prompts').mkdir()
    (tmp_path / 'prompts' / 'system.md').write_text('prompt')

    impl_config = _make_litellm_impl_config(
        runner_config={
            'api_key_env': 'ANTHROPIC_API_KEY',
            'model': 'anthropic/claude-sonnet-4-5-20250514',
            'max_tokens': '4096',
        },
    )
    mock_response = _make_litellm_response('{"topic": "A"}', prompt_tokens=1000, completion_tokens=50)
    mock_response.usage.prompt_tokens_details.cached_tokens = 800

    fake_pricing = {
        'claude-sonnet-4-5-20250514': {
            'input_cost_per_token': 0.000003,
            'output_cost_per_token': 0.000015,
            'cache_read_input_token_cost': 0.0000003,
        },
    }

    with (
        patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}),
        patch('engram.runners.litellm_api.litellm.completion', return_value=mock_response),
        patch('engram.runners.litellm_api.load_pricing', return_value=fake_pricing),
    ):
        result = LiteLLMRunner().trigger(InputData(filename='test', text='input'), impl_config, tmp_path)

    assert result.usage.prompt_tokens == 1000
    assert result.usage.cache_read_tokens == 800
    # find_rate's prefix-strip resolves `anthropic/claude-...` to the unprefixed pricing entry.
    # 200 non-cached * 3e-6 + 800 read * 3e-7 + 50 output * 1.5e-5
    expected = 200 * 3e-6 + 800 * 3e-7 + 50 * 1.5e-5
    assert result.cost_usd == pytest.approx(expected)


def test_litellm_runner_strips_provider_prefix_for_pricing(tmp_path: Path):
    """If the full model key isn't in pricing, try the suffix after the slash."""
    (tmp_path / 'prompts').mkdir()
    (tmp_path / 'prompts' / 'system.md').write_text('prompt')

    impl_config = _make_litellm_impl_config(
        runner_config={
            'api_key_env': 'ANTHROPIC_API_KEY',
            'model': 'anthropic/claude-sonnet-4-5-20250514',
            'max_tokens': '4096',
        },
    )
    mock_response = _make_litellm_response('{"topic": "A"}', 100, 50)

    with (
        patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}),
        patch('engram.runners.litellm_api.litellm.completion', return_value=mock_response),
        patch('engram.runners.litellm_api.load_pricing', return_value=_LITELLM_FAKE_PRICING),
    ):
        result = LiteLLMRunner().trigger(InputData(filename='test', text='input'), impl_config, tmp_path)

    # Pricing lookup fell back to 'claude-sonnet-4-5-20250514' after the prefixed key missed.
    # 100 * 0.000003 + 50 * 0.000015 = 0.00105
    assert result.cost_usd == pytest.approx(0.00105)


def test_litellm_runner_unknown_model_zero_cost(tmp_path: Path):
    (tmp_path / 'prompts').mkdir()
    (tmp_path / 'prompts' / 'system.md').write_text('prompt')

    impl_config = _make_litellm_impl_config(
        runner_config={
            'api_key_env': 'GEMINI_API_KEY',
            'model': 'gemini/nonexistent',
            'max_tokens': '4096',
        },
    )
    mock_response = _make_litellm_response('{"topic": "A"}')

    with (
        patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}),
        patch('engram.runners.litellm_api.litellm.completion', return_value=mock_response),
        patch('engram.runners.litellm_api.load_pricing', return_value=_LITELLM_FAKE_PRICING),
    ):
        result = LiteLLMRunner().trigger(InputData(filename='test', text='input'), impl_config, tmp_path)

    assert result.status == 'succeeded'
    assert result.cost_usd == 0.0


def test_litellm_runner_api_error(tmp_path: Path):
    (tmp_path / 'prompts').mkdir()
    (tmp_path / 'prompts' / 'system.md').write_text('prompt')

    impl_config = _make_litellm_impl_config()
    err = litellm.APIError(
        status_code=429,
        message='rate limit exceeded',
        llm_provider='gemini',
        model='gemini/gemini-2.0-flash',
    )

    with (
        patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}),
        patch('engram.runners.litellm_api.litellm.completion', side_effect=err),
        patch('engram.runners.litellm_api.load_pricing', return_value=_LITELLM_FAKE_PRICING),
    ):
        result = LiteLLMRunner().trigger(InputData(filename='test', text='input'), impl_config, tmp_path)

    assert result.status == 'failed'
    assert 'rate limit exceeded' in result.error
    assert result.cost_usd == 0.0


def test_litellm_runner_parse_failure_still_records_cost(tmp_path: Path):
    (tmp_path / 'prompts').mkdir()
    (tmp_path / 'prompts' / 'system.md').write_text('prompt')

    impl_config = _make_litellm_impl_config()
    mock_response = _make_litellm_response('not json at all', 100, 50)

    with (
        patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}),
        patch('engram.runners.litellm_api.litellm.completion', return_value=mock_response),
        patch('engram.runners.litellm_api.load_pricing', return_value=_LITELLM_FAKE_PRICING),
    ):
        result = LiteLLMRunner().trigger(InputData(filename='test', text='input'), impl_config, tmp_path)

    assert result.status == 'failed'
    assert 'Failed to parse JSON' in result.error
    assert result.cost_usd == pytest.approx(0.00003)


def test_litellm_runner_missing_key_raises_friendly_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)

    (tmp_path / 'prompts').mkdir()
    (tmp_path / 'prompts' / 'system.md').write_text('prompt')

    impl_config = _make_litellm_impl_config()
    with pytest.raises(MissingAPIKeyError) as excinfo:
        LiteLLMRunner().trigger(InputData(filename='test', text='input'), impl_config, tmp_path)

    assert excinfo.value.env_var == 'GEMINI_API_KEY'


def test_litellm_runner_required_env_vars_present():
    impl_config = _make_litellm_impl_config()
    assert LiteLLMRunner().required_env_vars(impl_config) == ['GEMINI_API_KEY']


def test_litellm_runner_required_env_vars_absent():
    """When api_key_env isn't set, no env var is reported — LiteLLM resolves it from the provider prefix."""
    impl_config = _make_litellm_impl_config(
        runner_config={'model': 'gemini/gemini-2.0-flash', 'max_tokens': '4096'},
    )
    assert LiteLLMRunner().required_env_vars(impl_config) == []


def test_litellm_runner_snapshot(tmp_path: Path):
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'system.md').write_text('You are a classifier.')

    impl_config = _make_litellm_impl_config()
    snap = LiteLLMRunner().snapshot_config(impl_config, tmp_path)

    assert snap.models == ['gemini/gemini-2.0-flash']
    assert 'system.md' in snap.prompts
    assert 'api_key_env' not in snap.runner_config

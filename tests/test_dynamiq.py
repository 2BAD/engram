"""Tests for Dynamiq runner and config sync."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

from engram.config.sync import pull_config
from engram.models.implementation import ConfigManagement, ImplementationConfig
from engram.models.input import InputData
from engram.runners.dynamiq import (
    DynamiqRunner,
    _build_result_from_output,
    _build_result_from_trace,
    _unwrap_output,
    classify_node,
    extract_llm_nodes,
    extract_prompt_text,
)
from engram.runners.registry import get_runner


def _make_dynamiq_config(**overrides: object) -> ImplementationConfig:
    defaults: dict[str, object] = {
        'workflow': 'classify',
        'platform': 'hosted',
        'runner': 'dynamiq',
        'runner_config': {
            'app_id': 'test-app-id',
            'access_key_env': 'DYNAMIQ_ACCESS_KEY',
        },
        'config_management': ConfigManagement(
            mode='pull-push',
            workflow_id='test-workflow-id',
            jwt_env='DYNAMIQ_JWT_TOKEN',
        ),
    }
    defaults.update(overrides)
    return ImplementationConfig(
        workflow=str(defaults['workflow']),
        platform=str(defaults['platform']),
        runner=str(defaults['runner']),
        runner_config=defaults.get('runner_config', {}),  # type: ignore[arg-type]
        config_management=defaults.get('config_management', ConfigManagement()),  # type: ignore[arg-type]
    )


# --- Registry ---


def test_get_runner_dynamiq():
    runner = get_runner('dynamiq')
    assert isinstance(runner, DynamiqRunner)


# --- Node extraction ---


def test_classify_llm_node():
    node = {
        'id': 'node-1',
        'name': 'Classifier',
        'type': 'OpenAI',
        'model': 'gpt-4.1-mini',
        'prompt': {
            'messages': [
                {'role': 'system', 'content': 'You are a classifier.'},
                {'role': 'user', 'content': 'Classify this.'},
            ]
        },
    }
    result = classify_node(node)
    assert result is not None
    assert result['model'] == 'gpt-4.1-mini'
    assert result['name'] == 'Classifier'


def test_classify_agent_node():
    node = {
        'id': 'agent-1',
        'name': 'ReAct Agent',
        'type': 'ReActAgent',
        'llm': {'model': 'claude-sonnet-4-5-20250514'},
        'role': 'You are a reasoning agent.',
    }
    result = classify_node(node)
    assert result is not None
    assert result['model'] == 'claude-sonnet-4-5-20250514'
    assert result['prompt'] == 'You are a reasoning agent.'


def test_classify_non_llm_node():
    node = {'id': 'converter-1', 'type': 'JsonConverter'}
    assert classify_node(node) is None


def test_extract_llm_nodes_with_map():
    nodes = [
        {
            'id': 'map-1',
            'type': 'MapNode',
            'node': {
                'id': 'inner-1',
                'name': 'Inner LLM',
                'type': 'Anthropic',
                'model': 'claude-haiku-4-5',
                'prompt': {'messages': [{'role': 'system', 'content': 'test'}]},
            },
        },
        {'id': 'passthrough', 'type': 'JsonConverter'},
    ]
    result = extract_llm_nodes(nodes)
    assert len(result) == 1
    assert result[0]['id'] == 'map-1'
    assert result[0]['model'] == 'claude-haiku-4-5'


def test_extract_prompt_text():
    node = {
        'prompt': {
            'messages': [
                {'role': 'system', 'content': 'Hello'},
                {'role': 'user', 'content': [{'type': 'text', 'text': 'World'}]},
            ]
        }
    }
    assert extract_prompt_text(node) == 'Hello\nWorld'


def test_extract_prompt_text_empty():
    assert extract_prompt_text({}) == ''
    assert extract_prompt_text({'prompt': 'not a dict'}) == ''


# --- Result building ---


def test_unwrap_output_nested():
    assert _unwrap_output({'output': {'topic': 'A', 'sentiment': 'Positive'}}) == {
        'topic': 'A',
        'sentiment': 'Positive',
    }


def test_unwrap_output_flat():
    assert _unwrap_output({'topic': 'A', 'sentiment': 'Positive'}) == {'topic': 'A', 'sentiment': 'Positive'}


def test_unwrap_output_non_dict():
    assert _unwrap_output('not a dict') == {}


def test_build_result_from_output():
    result = _build_result_from_output({'topic': 'A', 'sentiment': 'Positive'}, 500.0)
    assert result.status == 'succeeded'
    assert result.output == {'topic': 'A', 'sentiment': 'Positive'}
    assert result.latency_ms == 500.0


def test_build_result_from_output_nested():
    result = _build_result_from_output({'output': {'topic': 'A'}}, 500.0)
    assert result.output == {'topic': 'A'}


def test_build_result_from_trace():
    trace = {
        'output': {'topic': 'B'},
        'usage': {
            'prompt_tokens': 200,
            'completion_tokens': 100,
            'total_tokens': 300,
            'total_tokens_cost_usd': 0.005,
        },
    }
    result = _build_result_from_trace(trace, 1200.0)
    assert result.status == 'succeeded'
    assert result.output == {'topic': 'B'}
    assert result.usage.total_tokens == 300
    assert result.cost_usd == 0.005


# --- Runner trigger (sync response) ---


def test_dynamiq_runner_trigger_sync(tmp_path: Path):
    impl_config = _make_dynamiq_config()

    mock_app_response = {'data': {'id': 'test-app-id', 'hostname': 'app.example.com'}}
    mock_trigger_response = MagicMock()
    mock_trigger_response.status_code = 200
    mock_trigger_response.json.return_value = {'id': 'trace-123', 'output': {'topic': 'A'}}
    mock_trace = {
        'id': 'trace-123',
        'output': {'topic': 'A'},
        'usage': {
            'prompt_tokens': 100,
            'completion_tokens': 50,
            'total_tokens': 150,
            'total_tokens_cost_usd': 0.0025,
        },
    }

    with (
        patch.dict('os.environ', {'DYNAMIQ_ACCESS_KEY': 'key', 'DYNAMIQ_JWT_TOKEN': 'jwt'}),
        patch('engram.runners.dynamiq.management_api', return_value=mock_app_response),
        patch('engram.runners.dynamiq.httpx.post', return_value=mock_trigger_response),
        patch('engram.runners.dynamiq.get_trace', return_value=mock_trace),
    ):
        runner = DynamiqRunner()
        result = runner.trigger(InputData(filename='test', text='test input'), impl_config, tmp_path)

    assert result.status == 'succeeded'
    assert result.output == {'topic': 'A'}
    assert result.cost_usd == 0.0025
    assert result.usage.total_tokens == 150
    assert result.trace_id == 'trace-123'


def test_dynamiq_runner_trigger_sync_trace_fetch_fails(tmp_path: Path):
    """Falls back to output-only RunResult if the trace fetch fails."""
    impl_config = _make_dynamiq_config()

    mock_app_response = {'data': {'id': 'test-app-id', 'hostname': 'app.example.com'}}
    mock_trigger_response = MagicMock()
    mock_trigger_response.status_code = 200
    mock_trigger_response.json.return_value = {'id': 'trace-123', 'output': {'topic': 'A'}}

    with (
        patch.dict('os.environ', {'DYNAMIQ_ACCESS_KEY': 'key', 'DYNAMIQ_JWT_TOKEN': 'jwt'}),
        patch('engram.runners.dynamiq.management_api', return_value=mock_app_response),
        patch('engram.runners.dynamiq.httpx.post', return_value=mock_trigger_response),
        patch('engram.runners.dynamiq.get_trace', side_effect=httpx.HTTPError('boom')),
    ):
        runner = DynamiqRunner()
        result = runner.trigger(InputData(filename='test', text='test input'), impl_config, tmp_path)

    assert result.status == 'succeeded'
    assert result.output == {'topic': 'A'}
    assert result.cost_usd == 0.0
    assert result.trace_id == 'trace-123'


def test_dynamiq_runner_caches_hostname(tmp_path: Path):
    """Verify hostname is resolved once, not per trigger call."""
    impl_config = _make_dynamiq_config()

    mock_app_response = {'data': {'id': 'test-app-id', 'hostname': 'app.example.com'}}
    mock_trigger_response = MagicMock()
    mock_trigger_response.status_code = 200
    mock_trigger_response.json.return_value = {'id': 'trace-123', 'output': {'topic': 'A'}}
    mock_trace = {'id': 'trace-123', 'output': {'topic': 'A'}, 'usage': {}}

    with (
        patch.dict('os.environ', {'DYNAMIQ_ACCESS_KEY': 'key', 'DYNAMIQ_JWT_TOKEN': 'jwt'}),
        patch('engram.runners.dynamiq.management_api', return_value=mock_app_response) as mock_mgmt,
        patch('engram.runners.dynamiq.httpx.post', return_value=mock_trigger_response),
        patch('engram.runners.dynamiq.get_trace', return_value=mock_trace),
    ):
        runner = DynamiqRunner()
        runner.trigger(InputData(filename='test1', text='input 1'), impl_config, tmp_path)
        runner.trigger(InputData(filename='test2', text='input 2'), impl_config, tmp_path)

    # management_api called once for hostname, not twice
    assert mock_mgmt.call_count == 1


# --- Runner snapshot ---


def test_dynamiq_runner_snapshot(tmp_path: Path):
    impl_config = _make_dynamiq_config()

    mock_deployments = {'data': [{'workflow_id': 'wf-1', 'workflow_version_id': 'v-1'}]}
    mock_workflow = {
        'data': {
            'name': 'Test Workflow',
            'flow': {
                'nodes': [
                    {
                        'id': 'n1',
                        'name': 'Classifier',
                        'type': 'OpenAI',
                        'model': 'gpt-4.1-mini',
                        'prompt': {'messages': [{'role': 'system', 'content': 'Classify.'}]},
                    }
                ]
            },
        }
    }

    def mock_api(_jwt_env, path, params=None):  # noqa: ARG001
        if 'deployments' in path:
            return mock_deployments
        return mock_workflow

    with (
        patch.dict('os.environ', {'DYNAMIQ_JWT_TOKEN': 'jwt'}),
        patch('engram.runners.dynamiq.management_api', side_effect=mock_api),
    ):
        runner = DynamiqRunner()
        snap = runner.snapshot_config(impl_config, tmp_path)

    assert snap.models == ['gpt-4.1-mini']
    assert snap.runner_config['workflow_name'] == 'Test Workflow'
    assert 'Classifier' in snap.prompts


# --- Config sync (pull) ---


def test_pull_config(tmp_path: Path):
    impl_config = _make_dynamiq_config()

    mock_workflow = {
        'data': {
            'name': 'My Workflow',
            'flow': {
                'nodes': [
                    {
                        'id': 'llm-1',
                        'name': 'Agent',
                        'type': 'ReActAgent',
                        'llm': {'model': 'claude-sonnet-4-5-20250514'},
                        'role': 'You classify things.',
                    }
                ]
            },
        }
    }

    with (
        patch.dict('os.environ', {'DYNAMIQ_JWT_TOKEN': 'jwt'}),
        patch('engram.config.sync.management_api', return_value=mock_workflow),
    ):
        manifest = pull_config(tmp_path, impl_config)

    assert manifest['workflow_name'] == 'My Workflow'
    assert len(manifest['nodes']) == 1
    assert manifest['nodes'][0]['model'] == 'claude-sonnet-4-5-20250514'
    assert (tmp_path / 'manifest.json').exists()
    assert (tmp_path / 'prompts' / 'llm-1.role.md').exists()
    assert (tmp_path / 'prompts' / 'llm-1.role.md').read_text() == 'You classify things.'

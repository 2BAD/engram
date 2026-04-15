"""
Dynamiq hosted platform runner.

Triggers workflows via HTTP POST to the app hostname, handles both
sync (output in response) and async (trace polling) execution modes.
Auth is split per the spec: access_key_env in runner_config for triggering,
jwt_env in config_management for the management API.
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from engram.models.config_snapshot import ConfigSnapshot
from engram.models.run import RunResult, TokenUsage
from engram.runners.base import Runner
from engram.runners.dynamiq_api import get_trace, management_api, poll_trace_with_usage
from engram.runners.errors import MissingAPIKeyError

if TYPE_CHECKING:
    from engram.models.implementation import ImplementationConfig
    from engram.models.input import InputData

_TOO_MANY_REQUESTS = 429
_SERVER_ERROR = 500


class DynamiqRunner(Runner):
    """
    Runner that triggers workflows on the Dynamiq hosted platform.

    Resolves the app hostname once and reuses it across all trigger calls
    within the same runner instance.
    """

    def __init__(self) -> None:
        self._hostname: str = ''
        self._app_id: str = ''
        self._access_key: str = ''
        self._jwt_env: str = ''
        self._cache_dir: Path | None = None
        self._initialized = False
        # Correlation UUIDs injected into trigger inputs as ``_engram_id`` so finalize
        # can join traces back to results. Keyed by (input_file, trigger ordinal within
        # that file) because the same filename can appear across repeats.
        self._correlations: dict[tuple[str, int], str] = {}
        self._trigger_counts: dict[str, int] = {}
        self._run_start: datetime | None = None
        self._disable_correlation = False

    def required_env_vars(self, impl_config: ImplementationConfig) -> list[str]:
        rc = impl_config.runner_config
        keys: list[str] = []
        access_key = rc.get('access_key_env')
        if access_key:
            keys.append(access_key)
        jwt_env = impl_config.config_management.jwt_env
        if jwt_env:
            keys.append(jwt_env)
        return keys

    def _ensure_initialized(self, impl_config: ImplementationConfig, impl_dir: Path) -> str | None:
        """Resolve hostname on first call, return error string or None."""
        if self._initialized:
            return None

        rc = impl_config.runner_config
        self._app_id = rc['app_id']
        env_var = rc['access_key_env']
        try:
            self._access_key = os.environ[env_var]
        except KeyError as e:
            raise MissingAPIKeyError(env_var) from e
        self._jwt_env = impl_config.config_management.jwt_env
        self._cache_dir = impl_dir.parent.parent / 'data' / 'cache'
        self._disable_correlation = bool(rc.get('disable_correlation', False))

        app = management_api(self._jwt_env, f'/apps/{self._app_id}')
        self._hostname = app.get('data', app).get('hostname', '')
        self._initialized = True

        if not self._hostname:
            return f'No hostname found for app {self._app_id}'
        return None

    def trigger(self, input_data: InputData, impl_config: ImplementationConfig, impl_dir: Path) -> RunResult:
        """Trigger a Dynamiq workflow and collect the result."""
        error = self._ensure_initialized(impl_config, impl_dir)
        if error:
            return RunResult(input_file='', status='failed', error=error)

        if self._run_start is None:
            self._run_start = datetime.now(UTC)

        rc = impl_config.runner_config
        poll_config = (float(rc.get('poll_timeout', '600')), float(rc.get('poll_interval', '15')))

        # Dynamiq HTTP trigger expects text input.
        text = input_data.text if input_data.text is not None else input_data.text_for_display

        correlation_id = ''
        if not self._disable_correlation:
            correlation_id = uuid.uuid4().hex
            ordinal = self._trigger_counts.get(input_data.filename, 0)
            self._trigger_counts[input_data.filename] = ordinal + 1
            self._correlations[(input_data.filename, ordinal)] = correlation_id

        start = time.monotonic()
        return self._trigger_and_collect(text, start, poll_config, correlation_id)

    def _post_with_retry(
        self, input_data: str, correlation_id: str = '', max_retries: int = 5
    ) -> httpx.Response | RunResult:
        """POST to the Dynamiq trigger endpoint with exponential backoff on 429/5xx."""
        payload: dict[str, Any] = {'input': input_data}
        if correlation_id:
            payload['_engram_id'] = correlation_id
        last_error: str = ''
        for attempt in range(max_retries + 1):
            try:
                resp = httpx.post(
                    f'https://{self._hostname}',
                    headers={'Authorization': f'Bearer {self._access_key}', 'Content-Type': 'application/json'},
                    json={'input': payload},
                    timeout=180,
                )
            except httpx.HTTPError as e:
                last_error = str(e)
                if attempt < max_retries:
                    time.sleep(2**attempt)
                    continue
                return RunResult(input_file='', status='failed', error=last_error)

            if resp.status_code >= _SERVER_ERROR or resp.status_code == _TOO_MANY_REQUESTS:
                last_error = f'HTTP {resp.status_code}: {resp.text[:200]}'
                if attempt < max_retries:
                    time.sleep(2**attempt)
                    continue
                return RunResult(input_file='', status='failed', error=last_error)

            return resp

        return RunResult(input_file='', status='failed', error=last_error)

    def _trigger_and_collect(
        self, input_data: str, start: float, poll_config: tuple[float, float], correlation_id: str = ''
    ) -> RunResult:
        """Send the HTTP trigger and handle sync/async response paths."""
        resp = self._post_with_retry(input_data, correlation_id)
        if isinstance(resp, RunResult):
            resp.latency_ms = (time.monotonic() - start) * 1000
            return resp

        latency = (time.monotonic() - start) * 1000

        if resp.status_code not in (200, 202):
            return RunResult(
                input_file='', status='failed', latency_ms=latency, error=f'HTTP {resp.status_code}: {resp.text[:200]}'
            )

        body = resp.json()
        trace_id = body.get('id', '')

        if 'output' in body:
            if trace_id:
                full_trace = poll_trace_with_usage(self._jwt_env, trace_id, self._cache_dir)
                if full_trace is not None:
                    return _build_result_from_trace(full_trace, latency, trace_id)
            return _build_result_from_output(body['output'], latency, trace_id)

        if not trace_id:
            return RunResult(input_file='', status='failed', latency_ms=latency, error='No trace_id in async response')

        return _await_trace(self._jwt_env, self._app_id, trace_id, start, poll_config, self._cache_dir)

    def snapshot_config(self, impl_config: ImplementationConfig, impl_dir: Path) -> ConfigSnapshot:
        """Snapshot the deployed workflow config from Dynamiq."""
        rc = impl_config.runner_config
        cm = impl_config.config_management
        app_id = rc['app_id']

        try:
            deployments = management_api(cm.jwt_env, f'/apps/{app_id}/deployments')
            deploys = deployments.get('data', [])
            if not deploys:
                return _empty_snapshot(impl_config, impl_dir)

            workflow_id = deploys[0].get('workflow_id', '') or cm.workflow_id
            if not workflow_id:
                return _empty_snapshot(impl_config, impl_dir)

            wf_resp = management_api(cm.jwt_env, f'/workflows/{workflow_id}')
            wf_data = wf_resp.get('data', wf_resp)
            nodes = wf_data.get('flow', {}).get('nodes', [])

            llm_nodes = extract_llm_nodes(nodes)
            models = sorted({n['model'] for n in llm_nodes if n.get('model')})
            prompts = {n['name']: n.get('prompt', '') for n in llm_nodes}

            return ConfigSnapshot(
                implementation=impl_dir.name,
                platform=impl_config.platform,
                runner=impl_config.runner,
                models=models,
                prompts=prompts,
                runner_config={
                    'app_id': app_id,
                    'workflow_id': workflow_id,
                    'workflow_version_id': deploys[0].get('workflow_version_id', ''),
                    'workflow_name': wf_data.get('name', ''),
                },
            )
        except (httpx.HTTPError, KeyError):
            return _empty_snapshot(impl_config, impl_dir)


# -- Trigger helpers --

_TRACE_PAGE_SIZE = 100


def _await_trace(
    jwt_env: str,
    app_id: str,
    trace_id: str,
    start: float,
    poll_config: tuple[float, float],
    cache_dir: Path | None = None,
) -> RunResult:
    """Poll for a trace result, cache the full detail, and build the RunResult."""
    poll_timeout, poll_interval = poll_config
    trace = _poll_single_trace(jwt_env, app_id, trace_id, poll_timeout, poll_interval)
    total_latency = (time.monotonic() - start) * 1000

    if trace is None:
        return RunResult(
            input_file='',
            status='timeout',
            latency_ms=total_latency,
            error='Trace polling timed out',
            trace_id=trace_id,
        )
    if trace['status'] != 'succeeded':
        return RunResult(
            input_file='',
            status='failed',
            latency_ms=total_latency,
            error=f'Trace status: {trace["status"]}',
            trace_id=trace_id,
        )

    # Fetch and cache the full trace detail
    full_trace = get_trace(jwt_env, trace_id, cache_dir)
    return _build_result_from_trace(full_trace, total_latency, trace_id)


def _poll_single_trace(
    jwt_env: str,
    app_id: str,
    trace_id: str,
    timeout: float,
    poll_interval: float,
) -> dict[str, Any] | None:
    """Poll traces API until the specific trace completes or timeout."""
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        page = 1
        while True:
            resp = management_api(
                jwt_env,
                f'/apps/{app_id}/traces',
                {'page': page, 'page_size': _TRACE_PAGE_SIZE, 'sort': '-started_at'},
            )
            data = resp.get('data', [])
            if not data:
                break

            for trace in data:
                if trace['id'] == trace_id and trace['status'] in ('succeeded', 'failed'):
                    return trace

            total = resp.get('pagination', {}).get('total_count', 0)
            if len(data) < _TRACE_PAGE_SIZE or page * _TRACE_PAGE_SIZE >= total:
                break
            page += 1

        time.sleep(poll_interval)

    return None


def _unwrap_output(output: Any) -> dict[str, Any]:
    """Unwrap nested ``{"output": {actual_fields}}`` from Dynamiq responses so scoring sees fields directly."""
    if not isinstance(output, dict):
        return {}
    if list(output.keys()) == ['output'] and isinstance(output['output'], dict):
        return output['output']
    return output


def _build_result_from_output(output: dict[str, Any], latency_ms: float, trace_id: str = '') -> RunResult:
    """Build RunResult from a sync response output."""
    return RunResult(
        input_file='',
        output=_unwrap_output(output),
        status='succeeded',
        latency_ms=latency_ms,
        trace_id=trace_id,
    )


def _build_result_from_trace(trace: dict[str, Any], latency_ms: float, trace_id: str = '') -> RunResult:
    """Build RunResult from a completed trace."""
    usage_data = trace.get('usage', {})
    return RunResult(
        input_file='',
        output=_unwrap_output(trace.get('output', {})),
        status='succeeded',
        usage=TokenUsage(
            prompt_tokens=usage_data.get('prompt_tokens', 0),
            completion_tokens=usage_data.get('completion_tokens', 0),
            total_tokens=usage_data.get('total_tokens', 0),
        ),
        cost_usd=usage_data.get('total_tokens_cost_usd', 0.0),
        latency_ms=latency_ms,
        trace_id=trace_id,
    )


# -- Node extraction (Dynamiq workflow format) --


def extract_llm_nodes(nodes: list[dict]) -> list[dict[str, Any]]:
    """Extract LLM/Agent node configs from a Dynamiq workflow flow."""
    results = []
    for node in nodes:
        node_type = node.get('type', '')

        if 'Map' in node_type:
            inner = node.get('node', {})
            if isinstance(inner, dict):
                extracted = classify_node(inner, map_id=node.get('id', ''))
                if extracted:
                    results.append(extracted)
            continue

        extracted = classify_node(node)
        if extracted:
            results.append(extracted)

    return results


def classify_node(node: dict, map_id: str | None = None) -> dict[str, Any] | None:
    """Classify a Dynamiq workflow node and extract its model and prompt text."""
    node_type = node.get('type', '')
    effective_id = map_id or node.get('id', '')

    if any(t in node_type for t in ('OpenAI', 'Anthropic', 'LLM')):
        return {
            'id': effective_id,
            'name': node.get('name', effective_id),
            'model': node.get('model', ''),
            'type': node_type,
            'prompt': extract_prompt_text(node),
        }

    if 'Agent' in node_type or 'ReAct' in node_type:
        llm = node.get('llm', {})
        model = llm.get('model', '') if isinstance(llm, dict) else node.get('model', '')
        return {
            'id': effective_id,
            'name': node.get('name', effective_id),
            'model': model,
            'type': node_type,
            'prompt': node.get('role', ''),
        }

    return None


def extract_prompt_text(node: dict) -> str:
    """Extract prompt text from a Dynamiq node's message config."""
    prompt = node.get('prompt', {})
    if not isinstance(prompt, dict):
        return ''
    messages = prompt.get('messages', [])
    parts = []
    for msg in messages:
        content = msg.get('content', '')
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get('text'):
                    parts.append(block['text'])
        elif isinstance(content, str) and content:
            parts.append(content)
    return '\n'.join(parts)


def _empty_snapshot(impl_config: ImplementationConfig, impl_dir: Path) -> ConfigSnapshot:
    return ConfigSnapshot(
        implementation=impl_dir.name,
        platform=impl_config.platform,
        runner=impl_config.runner,
    )

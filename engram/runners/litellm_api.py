"""LiteLLM runner: unified access to any provider via litellm.completion."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import litellm

from engram.cost.pricing import compute_cost, load_pricing
from engram.models.config_snapshot import ConfigSnapshot
from engram.models.run import RunResult, TokenUsage
from engram.runners.base import Runner
from engram.runners.errors import MissingAPIKeyError

if TYPE_CHECKING:
    from engram.models.implementation import ImplementationConfig
    from engram.models.input import InputData


class LiteLLMRunner(Runner):
    """Runner that dispatches to any provider via litellm.completion."""

    def __init__(self) -> None:
        self._pricing: dict[str, Any] | None = None

    def configure_pricing(self, overrides: dict[str, dict[str, float]]) -> None:
        """Eager-load the pricing table with project overrides applied."""
        self._pricing = load_pricing(overrides=overrides)

    def required_env_vars(self, impl_config: ImplementationConfig) -> list[str]:
        # Only enforced when explicitly named; otherwise LiteLLM picks the env
        # var from the provider prefix (OPENAI_API_KEY, GEMINI_API_KEY, etc.)
        # and surfaces a friendly error itself if missing.
        key = impl_config.runner_config.get('api_key_env')
        return [key] if key else []

    def trigger(self, input_data: InputData, impl_config: ImplementationConfig, impl_dir: Path) -> RunResult:
        """Send input through litellm.completion and parse the JSON response."""
        rc = impl_config.runner_config
        model = rc['model']
        max_tokens = int(rc.get('max_tokens', '4096'))
        temperature = float(rc.get('temperature', '0'))

        api_key: str | None = None
        env_var = rc.get('api_key_env')
        if env_var:
            try:
                api_key = os.environ[env_var]
            except KeyError as e:
                raise MissingAPIKeyError(env_var) from e

        system_prompt = _load_system_prompt(impl_dir)
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': _build_content(input_data)})

        completion_kwargs: dict[str, Any] = {
            'model': model,
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'response_format': {'type': 'json_object'},
            'num_retries': 5,
            'drop_params': True,
        }
        if api_key is not None:
            completion_kwargs['api_key'] = api_key

        start = time.monotonic()
        try:
            response = litellm.completion(**completion_kwargs)
        except litellm.APIError as e:
            latency = (time.monotonic() - start) * 1000
            return RunResult(input_file='', status='failed', latency_ms=latency, error=str(e))

        latency = (time.monotonic() - start) * 1000

        # LiteLLM normalizes provider usage to the OpenAI shape: cached_tokens
        # lives under prompt_tokens_details and is a subset of prompt_tokens.
        u = getattr(response, 'usage', None)
        cache_read = 0
        if u is not None:
            details = getattr(u, 'prompt_tokens_details', None)
            cache_read = _as_int(getattr(details, 'cached_tokens', 0))
        usage = TokenUsage(
            prompt_tokens=getattr(u, 'prompt_tokens', 0) if u else 0,
            completion_tokens=getattr(u, 'completion_tokens', 0) if u else 0,
            total_tokens=getattr(u, 'total_tokens', 0) if u else 0,
            cache_read_tokens=cache_read,
        )
        cost_usd = self._compute_cost(model, usage)

        choices = getattr(response, 'choices', None) or []
        raw_text = ''
        if choices:
            msg = getattr(choices[0], 'message', None)
            raw_text = getattr(msg, 'content', '') or ''
        output = _parse_json_output(raw_text)

        if output is None:
            return RunResult(
                input_file='',
                output={},
                status='failed',
                usage=usage,
                cost_usd=cost_usd,
                latency_ms=latency,
                error=f'Failed to parse JSON from response: {raw_text[:200]}',
            )

        return RunResult(
            input_file='',
            output=output,
            status='succeeded',
            usage=usage,
            cost_usd=cost_usd,
            latency_ms=latency,
        )

    def _compute_cost(self, model: str, usage: TokenUsage) -> float:
        """Compute USD cost using engram's pricing cache."""
        if self._pricing is None:
            self._pricing = load_pricing()
        return compute_cost(self._pricing, model, usage)

    def snapshot_config(self, impl_config: ImplementationConfig, impl_dir: Path) -> ConfigSnapshot:
        """Capture model, prompts, and runner config."""
        prompts = {}
        prompts_dir = impl_dir / 'prompts'
        if prompts_dir.exists():
            for f in sorted(prompts_dir.iterdir()):
                if f.is_file():
                    prompts[f.name] = f.read_text()

        return ConfigSnapshot(
            implementation=impl_dir.name,
            platform=impl_config.platform,
            runner=impl_config.runner,
            models=[impl_config.runner_config.get('model', '')],
            prompts=prompts,
            runner_config={k: v for k, v in impl_config.runner_config.items() if k != 'api_key_env'},
        )


def _as_int(value: object) -> int:
    """Coerce SDK-reported numeric fields to int; treat anything else (None, MagicMock, ...) as 0."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _build_content(input_data: InputData) -> str | list[dict[str, Any]]:
    """
    Build OpenAI-shape content blocks.

    LiteLLM translates these to each provider's native format. Images use
    data-URI image_url blocks; PDFs use the file content part type.
    """
    if not input_data.is_binary:
        return input_data.text or ''

    if input_data.is_image:
        data_uri = f'data:{input_data.media_type};base64,{input_data.data_base64}'
        return [{'type': 'image_url', 'image_url': {'url': data_uri}}]

    if input_data.is_document:
        return [
            {
                'type': 'file',
                'file': {
                    'filename': input_data.filename,
                    'file_data': f'data:{input_data.media_type};base64,{input_data.data_base64}',
                },
            },
        ]

    return input_data.text_for_display


def _load_system_prompt(impl_dir: Path) -> str:
    prompt_path = impl_dir / 'prompts' / 'system.md'
    if prompt_path.exists():
        return prompt_path.read_text().strip()
    return ''


def _parse_json_output(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from response text, handling markdown fences as a fallback."""
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return None

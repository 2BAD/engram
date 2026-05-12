"""OpenAI Chat Completions API runner."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import openai
from openai.types.chat import ChatCompletionMessageParam

from engram.cost.pricing import compute_cost_components, compute_cost_without_cache, load_pricing
from engram.models.config_snapshot import ConfigSnapshot
from engram.models.run import RunResult, TokenUsage
from engram.runners.base import Runner
from engram.runners.errors import MissingAPIKeyError

if TYPE_CHECKING:
    from engram.models.implementation import ImplementationConfig
    from engram.models.input import InputData


class OpenAIApiRunner(Runner):
    """Runner that calls the OpenAI Chat Completions API directly."""

    def __init__(self) -> None:
        self._pricing: dict[str, Any] | None = None

    def configure_pricing(self, overrides: dict[str, dict[str, float]]) -> None:
        """Eager-load the pricing table with project overrides applied."""
        self._pricing = load_pricing(overrides=overrides)

    def required_env_vars(self, impl_config: ImplementationConfig) -> list[str]:
        key = impl_config.runner_config.get('api_key_env')
        return [key] if key else []

    def trigger(self, input_data: InputData, impl_config: ImplementationConfig, impl_dir: Path) -> RunResult:
        """Send input to the OpenAI API in JSON mode and parse the response."""
        rc = impl_config.runner_config
        env_var = rc['api_key_env']
        try:
            api_key = os.environ[env_var]
        except KeyError as e:
            raise MissingAPIKeyError(env_var) from e
        model = rc['model']
        max_tokens = int(rc.get('max_tokens', '4096'))
        temperature = float(rc.get('temperature', '0'))

        system_prompt = _load_system_prompt(impl_dir)
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': _build_openai_content(input_data)})

        client = openai.OpenAI(api_key=api_key, max_retries=5)

        create_kwargs: dict[str, Any] = {
            'model': model,
            'max_completion_tokens': max_tokens,
            'temperature': temperature,
            'response_format': {'type': 'json_object'},
            'messages': cast('list[ChatCompletionMessageParam]', messages),
        }
        # GPT-5 / o-series only; older models reject these fields.
        create_kwargs.update(_optional_reasoning_kwargs(rc))

        start = time.monotonic()
        try:
            response = client.chat.completions.create(**create_kwargs)
        except openai.APIError as e:
            latency = (time.monotonic() - start) * 1000
            return RunResult(input_file='', status='failed', latency_ms=latency, error=str(e))

        latency = (time.monotonic() - start) * 1000

        # OpenAI's prompt_tokens already includes cached tokens; prompt_tokens_details.cached_tokens
        # is the subset that hit cache. OpenAI does not report a cache-creation count separately.
        u = response.usage
        cache_read = 0
        if u is not None:
            details = getattr(u, 'prompt_tokens_details', None)
            cache_read = _as_int(getattr(details, 'cached_tokens', 0))
        usage = TokenUsage(
            prompt_tokens=u.prompt_tokens if u else 0,
            completion_tokens=u.completion_tokens if u else 0,
            total_tokens=u.total_tokens if u else 0,
            cache_read_tokens=cache_read,
        )
        components = self._compute_cost_components(model, usage)
        cost_usd = sum(components.values())
        cost_no_cache = self._compute_no_cache_cost(model, usage)

        raw_text = response.choices[0].message.content or '' if response.choices else ''
        output = _parse_json_output(raw_text)

        if output is None:
            return RunResult(
                input_file='',
                output={},
                status='failed',
                usage=usage,
                cost_usd=cost_usd,
                cost_input_usd=components['input_usd'],
                cost_cache_read_usd=components['cache_read_usd'],
                cost_cache_creation_usd=components['cache_creation_usd'],
                cost_output_usd=components['output_usd'],
                cost_without_cache_usd=cost_no_cache,
                latency_ms=latency,
                error=f'Failed to parse JSON from response: {raw_text[:200]}',
            )

        return RunResult(
            input_file='',
            output=output,
            status='succeeded',
            usage=usage,
            cost_usd=cost_usd,
            cost_input_usd=components['input_usd'],
            cost_cache_read_usd=components['cache_read_usd'],
            cost_cache_creation_usd=components['cache_creation_usd'],
            cost_output_usd=components['output_usd'],
            cost_without_cache_usd=cost_no_cache,
            latency_ms=latency,
        )

    def _compute_cost_components(self, model: str, usage: TokenUsage) -> dict[str, float]:
        """Per-bucket USD cost using cached LiteLLM pricing data."""
        if self._pricing is None:
            self._pricing = load_pricing()
        return compute_cost_components(self._pricing, model, usage)

    def _compute_no_cache_cost(self, model: str, usage: TokenUsage) -> float:
        """Counterfactual cost: what this call would have cost with no cache discount or premium."""
        if self._pricing is None:
            self._pricing = load_pricing()
        return compute_cost_without_cache(self._pricing, model, usage)

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


def _optional_reasoning_kwargs(rc: dict[str, str]) -> dict[str, str]:
    """GPT-5 / o-series reasoning fields from runner_config, if any."""
    return {k: rc[k] for k in ('reasoning_effort', 'verbosity') if k in rc}


def _as_int(value: object) -> int:
    """Coerce SDK-reported numeric fields to int; treat anything else (None, MagicMock, ...) as 0."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _build_openai_content(input_data: InputData) -> str | list[dict[str, Any]]:
    """
    Build the user message content for the OpenAI Chat Completions API.

    Text inputs are passed as a plain string. Images are encoded as
    data-URI image_url blocks. PDFs use the file content part type.
    """
    if not input_data.is_binary:
        return input_data.text or ''

    if input_data.is_image:
        data_uri = f'data:{input_data.media_type};base64,{input_data.data_base64}'
        return [
            {
                'type': 'image_url',
                'image_url': {'url': data_uri},
            },
        ]

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
    """Load the system prompt from prompts/system.md."""
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

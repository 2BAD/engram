"""Anthropic Messages API runner."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import anthropic
from anthropic.types import MessageParam, TextBlockParam

from engram.cost.pricing import compute_cost_components, compute_cost_without_cache, load_pricing
from engram.models.config_snapshot import ConfigSnapshot
from engram.models.run import RunResult, TokenUsage
from engram.runners.base import Runner
from engram.runners.errors import MissingAPIKeyError

if TYPE_CHECKING:
    from engram.models.implementation import ImplementationConfig
    from engram.models.input import InputData


class AnthropicApiRunner(Runner):
    """Runner that calls the Anthropic Messages API directly."""

    def __init__(self) -> None:
        self._pricing: dict[str, Any] | None = None

    def configure_pricing(self, overrides: dict[str, dict[str, float]]) -> None:
        """Eager-load the pricing table with project overrides applied."""
        self._pricing = load_pricing(overrides=overrides)

    def required_env_vars(self, impl_config: ImplementationConfig) -> list[str]:
        key = impl_config.runner_config.get('api_key_env')
        return [key] if key else []

    def trigger(self, input_data: InputData, impl_config: ImplementationConfig, impl_dir: Path) -> RunResult:
        """Send input to the Anthropic API and parse the JSON response."""
        rc = impl_config.runner_config
        env_var = rc['api_key_env']
        try:
            api_key = os.environ[env_var]
        except KeyError as e:
            raise MissingAPIKeyError(env_var) from e
        model = rc['model']
        max_tokens = int(rc.get('max_tokens', '4096'))
        temperature = float(rc.get('temperature', '0'))
        prompt_cache = _truthy(rc.get('prompt_cache'))

        system_prompt = _load_system_prompt(impl_dir)
        system_param = _system_param(system_prompt, prompt_cache)
        user_content = _build_anthropic_content(input_data)

        client = anthropic.Anthropic(api_key=api_key, max_retries=5)

        create_kwargs: dict[str, Any] = {
            'model': model,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'system': system_param,
            'messages': [cast('MessageParam', {'role': 'user', 'content': user_content})],
        }
        # Extended thinking requires temperature=1 and max_tokens > thinking_budget.
        create_kwargs.update(_optional_thinking_kwargs(rc))

        start = time.monotonic()
        try:
            response = client.messages.create(**create_kwargs)
        except anthropic.APIError as e:
            latency = (time.monotonic() - start) * 1000
            return RunResult(input_file='', status='failed', latency_ms=latency, error=str(e))

        latency = (time.monotonic() - start) * 1000

        # Anthropic reports cache_creation and cache_read tokens *in addition* to
        # input_tokens. Normalize to engram's convention where prompt_tokens is
        # the inclusive total so the cost helper can split it back out.
        cache_creation = _as_int(getattr(response.usage, 'cache_creation_input_tokens', 0))
        cache_read = _as_int(getattr(response.usage, 'cache_read_input_tokens', 0))
        input_total = response.usage.input_tokens + cache_creation + cache_read
        usage = TokenUsage(
            prompt_tokens=input_total,
            completion_tokens=response.usage.output_tokens,
            total_tokens=input_total + response.usage.output_tokens,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_creation,
        )
        components = self._compute_cost_components(model, usage)
        cost_usd = sum(components.values())
        cost_no_cache = self._compute_no_cache_cost(model, usage)

        raw_text = getattr(response.content[0], 'text', '') if response.content else ''
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


def _optional_thinking_kwargs(rc: dict[str, str]) -> dict[str, Any]:
    """Extended-thinking kwarg from runner_config.thinking_budget, if set."""
    raw = rc.get('thinking_budget')
    if not raw:
        return {}
    return {'thinking': {'type': 'enabled', 'budget_tokens': int(raw)}}


def _as_int(value: object) -> int:
    """Coerce SDK-reported numeric fields to int; treat anything else (None, MagicMock, ...) as 0."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _truthy(value: object) -> bool:
    """YAML-style truthiness check for string-valued runner_config booleans."""
    return isinstance(value, str) and value.strip().lower() in {'true', '1', 'yes', 'on'}


def _system_param(system_prompt: str, prompt_cache: bool) -> str | list[TextBlockParam]:
    """
    Build the value for `messages.create(system=...)`.

    Returns a plain string by default and a cache-marked content-block list when prompt_cache is on.
    Anthropic only caches blocks above ~1024 tokens (Sonnet/Opus) or ~2048 (Haiku); below that, the
    cache_control marker is silently ignored. The 25% cache-creation premium is only paid when caching
    actually activates, so enabling the flag on short prompts is harmless aside from a slightly more
    verbose API call.
    """
    if not system_prompt:
        return ''
    if prompt_cache:
        return [cast('TextBlockParam', {'type': 'text', 'text': system_prompt, 'cache_control': {'type': 'ephemeral'}})]
    return system_prompt


def _build_anthropic_content(input_data: InputData) -> str | list[dict[str, Any]]:
    """
    Build the user message content for the Anthropic API.

    Text inputs are passed as a plain string. Binary inputs (images, PDFs) are
    encoded as content blocks per the Anthropic Messages API spec.
    """
    if not input_data.is_binary:
        return input_data.text or ''

    if input_data.is_image:
        return [
            {
                'type': 'image',
                'source': {
                    'type': 'base64',
                    'media_type': input_data.media_type,
                    'data': input_data.data_base64,
                },
            },
        ]

    if input_data.is_document:
        return [
            {
                'type': 'document',
                'source': {
                    'type': 'base64',
                    'media_type': input_data.media_type,
                    'data': input_data.data_base64,
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
    """Extract JSON from response text, handling markdown fences."""
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

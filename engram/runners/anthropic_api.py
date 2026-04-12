"""Anthropic Messages API runner."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anthropic

from engram.cost.pricing import find_rate, load_pricing
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

        system_prompt = _load_system_prompt(impl_dir)
        user_content = _build_anthropic_content(input_data)

        client = anthropic.Anthropic(api_key=api_key)

        start = time.monotonic()
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{'role': 'user', 'content': user_content}],
            )
        except anthropic.APIError as e:
            latency = (time.monotonic() - start) * 1000
            return RunResult(input_file='', status='failed', latency_ms=latency, error=str(e))

        latency = (time.monotonic() - start) * 1000

        usage = TokenUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )
        cost_usd = self._compute_cost(model, usage)

        raw_text = response.content[0].text if response.content else ''
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
        """Compute USD cost from token usage using cached LiteLLM pricing data."""
        if self._pricing is None:
            self._pricing = load_pricing()
        input_rate, output_rate = find_rate(self._pricing, model)
        return usage.prompt_tokens * input_rate + usage.completion_tokens * output_rate

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

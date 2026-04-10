"""OpenAI Chat Completions API runner."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import openai

from engram.cost.pricing import find_rate, load_pricing
from engram.models.config_snapshot import ConfigSnapshot
from engram.models.run import RunResult, TokenUsage
from engram.runners.base import Runner
from engram.runners.errors import MissingAPIKeyError

if TYPE_CHECKING:
    from engram.models.implementation import ImplementationConfig


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

    def trigger(self, input_data: str, impl_config: ImplementationConfig, impl_dir: Path) -> RunResult:
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
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': input_data})

        client = openai.OpenAI(api_key=api_key)

        start = time.monotonic()
        try:
            response = client.chat.completions.create(
                model=model,
                max_completion_tokens=max_tokens,
                temperature=temperature,
                response_format={'type': 'json_object'},
                messages=messages,
            )
        except openai.APIError as e:
            latency = (time.monotonic() - start) * 1000
            return RunResult(input_file='', status='failed', latency_ms=latency, error=str(e))

        latency = (time.monotonic() - start) * 1000

        usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )
        cost_usd = self._compute_cost(model, usage)

        raw_text = response.choices[0].message.content or '' if response.choices else ''
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

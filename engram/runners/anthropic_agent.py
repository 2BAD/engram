"""Anthropic agent runner: executes local Python agent code."""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from engram.models.config_snapshot import ConfigSnapshot
from engram.models.run import RunResult, TokenUsage
from engram.runners.base import Runner

if TYPE_CHECKING:
    from engram.models.implementation import ImplementationConfig
    from engram.models.input import InputData


class AnthropicAgentRunner(Runner):
    """Runner that imports and executes a local Python agent function."""

    def trigger(self, input_data: InputData, impl_config: ImplementationConfig, impl_dir: Path) -> RunResult:
        """Import and call the agent entry point."""
        rc = impl_config.runner_config
        entry_point = rc.get('entry_point', '')

        if ':' not in entry_point:
            return RunResult(
                input_file='',
                status='failed',
                error=f'Invalid entry_point format "{entry_point}", expected "file.py:function"',
            )

        file_part, func_name = entry_point.rsplit(':', 1)
        module_path = impl_dir / file_part

        if not module_path.exists():
            return RunResult(
                input_file='',
                status='failed',
                error=f'Agent file not found: {module_path}',
            )

        try:
            func = _load_function(module_path, func_name)
        except (ImportError, AttributeError) as e:
            return RunResult(input_file='', status='failed', error=f'Failed to load agent: {e}')

        # Agent functions receive text content; binary inputs get a display summary.
        agent_input = input_data.text if input_data.text is not None else input_data.text_for_display

        start = time.monotonic()
        try:
            result = func(agent_input)
        except Exception as e:  # noqa: BLE001
            latency = (time.monotonic() - start) * 1000
            return RunResult(input_file='', status='failed', latency_ms=latency, error=str(e))

        latency = (time.monotonic() - start) * 1000

        if isinstance(result, dict):
            output = result.get('output', result)
            usage_data = result.get('usage', {})
            usage = TokenUsage(
                prompt_tokens=usage_data.get('prompt_tokens', 0),
                completion_tokens=usage_data.get('completion_tokens', 0),
                total_tokens=usage_data.get('total_tokens', 0),
            )
            cost = result.get('cost_usd', 0.0)
        else:
            output = {}
            usage = TokenUsage()
            cost = 0.0

        return RunResult(
            input_file='',
            output=output,
            status='succeeded',
            usage=usage,
            cost_usd=cost,
            latency_ms=latency,
        )

    def snapshot_config(self, impl_config: ImplementationConfig, impl_dir: Path) -> ConfigSnapshot:
        """Capture agent code and runner config."""
        rc = impl_config.runner_config
        entry_point = rc.get('entry_point', '')
        file_part = entry_point.split(':')[0] if ':' in entry_point else ''

        prompts = {}
        agent_file = impl_dir / file_part
        if agent_file.exists():
            prompts[file_part] = agent_file.read_text()

        return ConfigSnapshot(
            implementation=impl_dir.name,
            platform=impl_config.platform,
            runner=impl_config.runner,
            models=[rc.get('model', '')],
            prompts=prompts,
            runner_config={k: v for k, v in rc.items() if k != 'api_key_env'},
        )


def _load_function(module_path: Path, func_name: str) -> Any:
    """Dynamically import a function from a Python file."""
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        msg = f'Cannot load module from {module_path}'
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, func_name)

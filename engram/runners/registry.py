"""Runner registry: name -> runner class, with on-demand module imports."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engram.runners.base import Runner

_RUNNER_SPECS: dict[str, tuple[str, str]] = {
    'anthropic': ('engram.runners.anthropic_api', 'AnthropicApiRunner'),
    'anthropic-agent': ('engram.runners.anthropic_agent', 'AnthropicAgentRunner'),
    'dynamiq': ('engram.runners.dynamiq', 'DynamiqRunner'),
    'openai': ('engram.runners.openai_api', 'OpenAIApiRunner'),
    'litellm': ('engram.runners.litellm_api', 'LiteLLMRunner'),
}

_OPTIONAL_RUNNERS = {'litellm'}


def get_runner(name: str) -> Runner:
    """Get a runner instance by name."""
    validate_runner_name(name)
    module_path, class_name = _RUNNER_SPECS[name]
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        if name in _OPTIONAL_RUNNERS:
            raise ValueError(_optional_install_hint(name)) from exc
        raise
    return getattr(module, class_name)()


def validate_runner_name(name: str) -> None:
    """Raise ValueError if `name` is not a registered runner."""
    if name in _RUNNER_SPECS:
        return
    available = ', '.join(sorted(_RUNNER_SPECS))
    msg = f'Unknown runner "{name}". Available: {available}'
    raise ValueError(msg)


def _optional_install_hint(name: str) -> str:
    if name == 'litellm':
        return (
            'The "litellm" runner needs the optional `litellm` extra. '
            "Install: uv tool install 'engram[litellm]' --prerelease=allow --python 3.13"
        )
    return f'The "{name}" runner is not installed.'

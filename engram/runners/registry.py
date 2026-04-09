"""Runner registry: maps runner name strings to runner classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from engram.runners.anthropic_agent import AnthropicAgentRunner
from engram.runners.anthropic_api import AnthropicApiRunner
from engram.runners.dynamiq import DynamiqRunner
from engram.runners.openai_api import OpenAIApiRunner

if TYPE_CHECKING:
    from engram.runners.base import Runner

_RUNNERS: dict[str, type[Runner]] = {
    'anthropic': AnthropicApiRunner,
    'anthropic-agent': AnthropicAgentRunner,
    'dynamiq': DynamiqRunner,
    'openai': OpenAIApiRunner,
}


def get_runner(name: str) -> Runner:
    """Get a runner instance by name."""
    validate_runner_name(name)
    return _RUNNERS[name]()


def validate_runner_name(name: str) -> None:
    """Raise ValueError if `name` is not a registered runner."""
    if name not in _RUNNERS:
        available = ', '.join(sorted(_RUNNERS.keys()))
        msg = f'Unknown runner "{name}". Available: {available}'
        raise ValueError(msg)

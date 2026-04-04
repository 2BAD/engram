"""Runner registry: maps runner name strings to runner classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from engram.runners.anthropic_agent import AnthropicAgentRunner
from engram.runners.anthropic_api import AnthropicApiRunner

if TYPE_CHECKING:
    from engram.runners.base import Runner

_RUNNERS: dict[str, type[Runner]] = {
    'anthropic': AnthropicApiRunner,
    'anthropic-agent': AnthropicAgentRunner,
}


def get_runner(name: str) -> Runner:
    """Get a runner instance by name."""
    runner_cls = _RUNNERS.get(name)
    if runner_cls is None:
        available = ', '.join(sorted(_RUNNERS.keys()))
        msg = f'Unknown runner "{name}". Available: {available}'
        raise ValueError(msg)
    return runner_cls()

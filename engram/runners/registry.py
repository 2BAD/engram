"""Runner registry: maps runner name strings to runner classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from engram.runners.anthropic_agent import AnthropicAgentRunner
from engram.runners.anthropic_api import AnthropicApiRunner
from engram.runners.dynamiq import DynamiqRunner
from engram.runners.openai_api import OpenAIApiRunner

if TYPE_CHECKING:
    from engram.runners.base import Runner

# LiteLLM is optional. litellm_api imports litellm at module load, so we guard
# that import and only register the runner if it succeeded. validate_runner_name
# returns an install hint when the runner is referenced without the extra.
try:
    from engram.runners.litellm_api import LiteLLMRunner

    _LITELLM_AVAILABLE = True
except ImportError:
    _LITELLM_AVAILABLE = False


_RUNNERS: dict[str, type[Runner]] = {
    'anthropic': AnthropicApiRunner,
    'anthropic-agent': AnthropicAgentRunner,
    'dynamiq': DynamiqRunner,
    'openai': OpenAIApiRunner,
}
if _LITELLM_AVAILABLE:
    _RUNNERS['litellm'] = LiteLLMRunner


def get_runner(name: str) -> Runner:
    """Get a runner instance by name."""
    validate_runner_name(name)
    return _RUNNERS[name]()


def validate_runner_name(name: str) -> None:
    """Raise ValueError if `name` is not a registered runner."""
    if name in _RUNNERS:
        return
    if name == 'litellm' and not _LITELLM_AVAILABLE:
        msg = (
            'The "litellm" runner needs the optional `litellm` extra. '
            "Install: uv tool install 'engram[litellm]' --prerelease=allow --python 3.13"
        )
        raise ValueError(msg)
    available = ', '.join(sorted(_RUNNERS.keys()))
    msg = f'Unknown runner "{name}". Available: {available}'
    raise ValueError(msg)

"""Config snapshot model."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConfigSnapshot:
    """Frozen configuration at trigger time."""

    implementation: str
    platform: str
    runner: str
    models: list[str] = field(default_factory=list)
    prompts: dict[str, str] = field(default_factory=dict)
    runner_config: dict[str, Any] = field(default_factory=dict)
    transform: dict[str, str] = field(default_factory=dict)

"""Project configuration model."""

from dataclasses import dataclass, field


@dataclass
class ProjectConfig:
    """Top-level engram.yaml configuration."""

    name: str
    description: str = ''
    pricing_overrides: dict[str, dict[str, float]] = field(default_factory=dict)

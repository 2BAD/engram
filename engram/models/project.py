"""Project configuration model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engram.models.analysis import AnalysisConfig


@dataclass
class ProjectConfig:
    """Top-level engram.yaml configuration."""

    name: str
    description: str = ''
    pricing_overrides: dict[str, dict[str, float]] = field(default_factory=dict)
    analysis: AnalysisConfig | None = None

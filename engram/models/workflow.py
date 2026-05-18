"""Workflow configuration model."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OutputField:
    """A single field in the workflow output schema."""

    type: str
    values: list[str] | None = None
    description: str = ''


@dataclass
class WorkflowConfig:
    """Workflow definition from workflow.yaml."""

    name: str
    description: str = ''
    input_type: str = 'text'
    input_description: str = ''
    output_fields: dict[str, OutputField] = field(default_factory=dict)
    # Each scorer is either a string ("exact_match", "fuzzy_match(0.9)", "scorers.judge")
    # or a dict {type: ..., ...kwargs} for factories whose config doesn't fit a one-liner.
    scorers: dict[str, str | dict[str, Any]] = field(default_factory=dict)
    confusion_matrices: list[str] = field(default_factory=list)

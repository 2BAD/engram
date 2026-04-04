"""Workflow configuration model."""

from dataclasses import dataclass, field


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
    scorers: dict[str, str] = field(default_factory=dict)
    confusion_matrices: list[str] = field(default_factory=list)

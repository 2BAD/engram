"""Load and parse YAML config files into data models."""

from pathlib import Path

import yaml

from engram.models.dataset import DatasetConfig
from engram.models.implementation import ConfigManagement, ImplementationConfig
from engram.models.project import ProjectConfig
from engram.models.workflow import OutputField, WorkflowConfig


def load_project(root: Path) -> ProjectConfig:
    """Load engram.yaml from the project root."""
    raw = yaml.safe_load((root / 'engram.yaml').read_text())
    return ProjectConfig(
        name=raw['name'],
        description=raw.get('description', ''),
        pricing_overrides=raw.get('pricing_overrides', {}),
    )


def load_workflow(root: Path, name: str) -> WorkflowConfig:
    """Load a workflow definition from workflows/{name}/workflow.yaml."""
    path = root / 'workflows' / name / 'workflow.yaml'
    raw = yaml.safe_load(path.read_text())

    output_fields = {}
    for field_name, field_def in raw.get('output', {}).get('fields', {}).items():
        output_fields[field_name] = OutputField(
            type=field_def['type'],
            values=field_def.get('values'),
            description=field_def.get('description', ''),
        )

    input_config = raw.get('input', {})
    return WorkflowConfig(
        name=raw['name'],
        description=raw.get('description', ''),
        input_type=input_config.get('type', 'text'),
        input_description=input_config.get('description', ''),
        output_fields=output_fields,
        scorers=raw.get('scorers', {}),
        confusion_matrices=raw.get('confusion_matrices', []),
    )


def load_implementation(root: Path, name: str) -> ImplementationConfig:
    """Load an implementation from implementations/{name}/implementation.yaml."""
    path = root / 'implementations' / name / 'implementation.yaml'
    raw = yaml.safe_load(path.read_text())

    cm_raw = raw.get('config_management', {})
    config_management = ConfigManagement(
        mode=cm_raw.get('mode', 'local'),
        workflow_id=cm_raw.get('workflow_id', ''),
        jwt_env=cm_raw.get('jwt_env', ''),
    )

    return ImplementationConfig(
        workflow=raw['workflow'],
        platform=raw['platform'],
        runner=raw['runner'],
        runner_config=raw.get('runner_config', {}),
        config_management=config_management,
    )


def load_dataset(root: Path, name: str) -> DatasetConfig:
    """Load dataset metadata from datasets/{name}/dataset.yaml."""
    path = root / 'datasets' / name / 'dataset.yaml'
    raw = yaml.safe_load(path.read_text())
    return DatasetConfig(
        name=raw['name'],
        description=raw.get('description', ''),
    )

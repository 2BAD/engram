"""Load and parse YAML config files into data models."""

from pathlib import Path

import yaml

from engram.models.dataset import DatasetConfig
from engram.models.implementation import ConfigManagement, ImplementationConfig
from engram.models.project import ProjectConfig
from engram.models.workflow import OutputField, WorkflowConfig
from engram.runners.registry import validate_runner_name
from engram.scoring.registry import validate_scorer_name


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

    scorers = raw.get('scorers', {})
    for field_name, scorer_name in scorers.items():
        try:
            validate_scorer_name(scorer_name)
        except ValueError as e:
            msg = f'workflow "{name}" field "{field_name}": {e}'
            raise ValueError(msg) from e

    input_config = raw.get('input', {})
    return WorkflowConfig(
        name=raw['name'],
        description=raw.get('description', ''),
        input_type=input_config.get('type', 'text'),
        input_description=input_config.get('description', ''),
        output_fields=output_fields,
        scorers=scorers,
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

    runner_name = raw['runner']
    try:
        validate_runner_name(runner_name)
    except ValueError as e:
        msg = f'implementation "{name}": {e}'
        raise ValueError(msg) from e

    return ImplementationConfig(
        workflow=raw['workflow'],
        platform=raw['platform'],
        runner=runner_name,
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

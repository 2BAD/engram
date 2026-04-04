"""Cross-reference validation for project configs."""

from pathlib import Path

import yaml

from engram.config.discovery import discover_implementations, discover_workflows
from engram.config.loader import load_implementation, load_workflow


def validate_project(root: Path) -> list[str]:
    """Validate all project configs and return a list of error messages."""
    errors: list[str] = []

    workflows = discover_workflows(root)
    implementations = discover_implementations(root)

    for impl_name in implementations:
        try:
            impl = load_implementation(root, impl_name)
        except (OSError, KeyError, yaml.YAMLError) as e:
            errors.append(f'implementation {impl_name}: failed to load: {e}')
            continue

        if impl.workflow not in workflows:
            errors.append(f'implementation {impl_name}: references unknown workflow "{impl.workflow}"')
            continue

        try:
            wf = load_workflow(root, impl.workflow)
        except (OSError, KeyError, yaml.YAMLError) as e:
            errors.append(f'workflow {impl.workflow}: failed to load: {e}')
            continue

        for scorer_field in wf.scorers:
            if scorer_field not in wf.output_fields:
                errors.append(f'workflow {wf.name}: scorer bound to unknown field "{scorer_field}"')

        for cm_field in wf.confusion_matrices:
            if cm_field not in wf.output_fields:
                errors.append(f'workflow {wf.name}: confusion matrix references unknown field "{cm_field}"')

    return errors
